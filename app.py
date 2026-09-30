import hashlib
import html
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pypdfium2 as pdfium
import streamlit as st
from dotenv import load_dotenv
from google.genai import errors as google_errors

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from htmlTemplates import css, bot_template, user_template

CHAT_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "models/gemini-embedding-001"
TOP_K = 4
# FAISS filters *after* fetching, so a filtered search has to pull a wide net or a
# vehicle whose pages rank below another manual's would come back with nothing.
FETCH_K = 300
HISTORY_TURNS = 3  # how many previous Q/A pairs to send to the model
EMBED_ATTEMPTS = 4  # the embedding endpoint returns sporadic 500s; retry them
EMBED_BACKOFF = 1.5  # seconds before the first retry, doubled each attempt
EMBED_BATCH = 100  # the endpoint's maximum number of texts per request
EMBED_WORKERS = 8  # batches in flight at once; indexing is almost entirely network wait
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
# Rebuilding the index for an already-seen PDF spends a minute of API calls on a
# result that is identical every time, so indexes are kept on disk, keyed by content.
INDEX_CACHE_DIR = Path(".index_cache")

# The embeddings client wraps every failure - including transient server-side ones -
# in a GoogleGenerativeAIError whose only detail is the message text, so the status
# has to be read back out of the string.
TRANSIENT_EMBED_ERRORS = re.compile(
    r"\b(429|500|502|503|504|INTERNAL|UNAVAILABLE|RESOURCE_EXHAUSTED|DEADLINE_EXCEEDED)\b"
)

# What the user sees when the manuals don't cover the question. Phrased as a next
# step rather than a dead end, so it reads as the document's limit, not a failure.
NO_ANSWER_REPLY = (
    "I couldn't find an answer to that in our knowledge base. "
    "Try rephrasing your question, or adding more detail."
)

ANSWER_SYSTEM_PROMPT = (
    """\
Use the following information to answer the user's question. DO NOT make up answers
that are not based on facts. Explain with detailed answers that are easy to
understand. Provide only relevant information.

Answer in plain text. Do not use any Markdown formatting - no asterisks for bold or
italics, no "#" headings, and no backticks. Start list items with "- ".

ALWAYS reply in exactly this format, whatever the question is:

Probable Cause:
<cause here>

Probable Solution:
<solution here>

Never reply with a plain paragraph. Never use any heading other than these two, and
never leave either section out.

This applies to every question, not only to ones that describe a fault. When the
question is not about a fault - it asks how something works, or what a specification
is - still use the two headings: put what the document says under "Probable Cause:"
and put what the user should do, check, or remember under "Probable Solution:".

The one exception: if the information below does not contain the answer, reply with
exactly this line and nothing else:

"""
    + NO_ANSWER_REPLY
    + """

Never guess to fill the format.

Context:
{context}
"""
)

CONDENSE_SYSTEM_PROMPT = """\
Given the conversation so far and a follow-up message, rewrite the follow-up as a
standalone question that can be understood without the conversation. Return only the
rewritten question. Do not answer it.
"""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ANSWER_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)

condense_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CONDENSE_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)


def as_plain_text(value):
    """Return an exact `str` for a chain output.

    `StrOutputParser` yields `TextAccessor`, a `str` subclass. google-genai
    validates request bodies with Pydantic, and its smart-union matching coerces
    the subclass into an empty `Content()` rather than treating it as text - so
    the query silently leaves as `{}` and the API answers 500 INTERNAL instead of
    a useful 400. Collapsing to an exact `str` keeps the text in the request.
    """
    return str(value)


def retry_transient(call, *args, **kwargs):
    """Run an embedding call, retrying the transient 5xx/429 failures with backoff."""
    delay = EMBED_BACKOFF
    for attempt in range(1, EMBED_ATTEMPTS + 1):
        try:
            return call(*args, **kwargs)
        except GoogleGenerativeAIError as e:
            last_attempt = attempt == EMBED_ATTEMPTS
            if last_attempt or not TRANSIENT_EMBED_ERRORS.search(str(e)):
                raise
            time.sleep(delay)
            delay *= 2


class RetryingEmbeddings(Embeddings):
    """Wraps an embeddings client so transient API failures are retried, not raised."""

    def __init__(self, inner):
        self.inner = inner

    def embed_documents(self, texts, on_progress=None):
        """Embed `texts`, sending the batches concurrently.

        The client batches by 100 but walks the batches one at a time, so a
        400-chunk manual spends ~30s on requests that do not depend on each
        other. Running them together makes the wall clock that of the slowest
        single batch instead of their sum.
        """
        texts = [as_plain_text(t) for t in texts]
        batches = [texts[i : i + EMBED_BATCH] for i in range(0, len(texts), EMBED_BATCH)]
        if not batches:
            return []

        results = [None] * len(batches)
        with ThreadPoolExecutor(max_workers=min(EMBED_WORKERS, len(batches))) as pool:
            pending = {
                pool.submit(retry_transient, self.inner.embed_documents, batch): i
                for i, batch in enumerate(batches)
            }
            for done, future in enumerate(as_completed(pending), start=1):
                results[pending[future]] = future.result()
                if on_progress:
                    on_progress(done, len(batches))
        return [vector for batch in results for vector in batch]

    def embed_query(self, text):
        return retry_transient(self.inner.embed_query, as_plain_text(text))


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return RetryingEmbeddings(GoogleGenerativeAIEmbeddings(model=EMBED_MODEL))


def load_pdfs(uploads):
    """Read the uploaded PDFs into one Document per page, keeping source metadata.

    Uses pdfium rather than pypdf: on the 380-page manual pypdf needs ~8.5s to
    pull the text out, pdfium ~0.6s for the same characters.
    """
    documents = []
    for name, data in uploads:
        pdf = pdfium.PdfDocument(data)
        try:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_textpage().get_text_bounded().strip()
                if not text:
                    continue  # image-only page; nothing to index without OCR
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": name, "page": page_number},
                    )
                )
        finally:
            pdf.close()
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_documents(documents)


def build_vectorstore(chunks, on_progress=None):
    """Embed the chunks and load them into FAISS.

    `FAISS.from_documents` would embed them too, but calling `embed_documents`
    directly is what lets the batches run concurrently and report progress
    while they do.
    """
    embeddings = get_embeddings()
    texts = [c.page_content for c in chunks]
    vectors = embeddings.embed_documents(texts, on_progress=on_progress)
    return FAISS.from_embeddings(
        list(zip(texts, vectors)),
        embeddings,
        metadatas=[c.metadata for c in chunks],
    )


def document_sources(vectorstore):
    """The file names indexed in `vectorstore`, in a stable order."""
    # FAISS keeps the chunk metadata in its docstore; there is no public
    # accessor for the whole collection, so read the mapping directly.
    return sorted({d.metadata.get("source", "?") for d in vectorstore.docstore._dict.values()})


def cache_key(uploads):
    """Fingerprint the uploaded bytes plus every setting that shapes the index."""
    digest = hashlib.sha256()
    digest.update(f"{EMBED_MODEL}|{CHUNK_SIZE}|{CHUNK_OVERLAP}".encode())
    for name, data in sorted(uploads):
        digest.update(name.encode())
        digest.update(data)
    return digest.hexdigest()[:16]


def load_cached_vectorstore(key):
    path = INDEX_CACHE_DIR / key
    if not (path / "index.faiss").exists():
        return None
    try:
        # The pickle being read back is one this app wrote itself, into a
        # directory named after the PDF's own hash.
        return FAISS.load_local(
            str(path), get_embeddings(), allow_dangerous_deserialization=True
        )
    except Exception:
        return None  # stale or half-written cache entry; rebuild it instead


def save_cached_vectorstore(key, vectorstore):
    try:
        vectorstore.save_local(str(INDEX_CACHE_DIR / key))
    except OSError:
        pass  # a read-only deploy still works, it just re-indexes each time


def as_lc_messages(turns):
    messages = []
    for turn in turns[-HISTORY_TURNS:]:
        messages.append(HumanMessage(turn["question"]))
        messages.append(AIMessage(turn["answer"]))
    return messages


def format_context(docs):
    return "\n\n---\n\n".join(
        f"[{d.metadata.get('source', '?')} p.{d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )


NO_ANSWER_PATTERNS = re.compile(
    r"^\W*(i\s+(don'?t|do not)\s+know(\s+the\s+answer)?"
    r"|i\s+(don'?t|do not)\s+have\s+(enough|that)\s+information"
    r"|the\s+(provided\s+)?(information|context|document)\s+does\s+not\s+contain)",
    re.I,
)


def is_no_answer(answer):
    """True when the model said it couldn't answer, in its wording or in ours."""
    text = answer.strip()
    return text.startswith(NO_ANSWER_REPLY[:40]) or bool(NO_ANSWER_PATTERNS.match(text))


def answer_question(question, source=None):
    llm = get_llm()
    history = as_lc_messages(st.session_state.turns)

    # Follow-ups like "and what about the second one?" are meaningless to a retriever,
    # so rewrite them into a standalone query first.
    if history:
        standalone = as_plain_text(
            (condense_prompt | llm | StrOutputParser()).invoke(
                {"chat_history": history, "question": question}
            )
        )
        # A blank rewrite would embed as empty content, so keep the original wording.
        if not standalone.strip():
            standalone = question
    else:
        standalone = question

    # Without this filter a Honda question happily retrieves Toyota pages and the
    # model blends two different vehicles' procedures into one answer.
    docs = st.session_state.vectorstore.similarity_search(
        standalone,
        k=TOP_K,
        fetch_k=FETCH_K,
        filter={"source": source} if source else None,
    )
    if not docs:
        return NO_ANSWER_REPLY, []

    answer = as_plain_text(
        (answer_prompt | llm | StrOutputParser()).invoke(
            {
                "context": format_context(docs),
                "chat_history": history,
                "question": question,
            }
        )
    )

    # The model sometimes falls back to its own "I don't know" wording instead of the
    # line it was given; show the user one phrasing either way, and no page citations
    # for an answer that isn't in those pages.
    if is_no_answer(answer):
        return NO_ANSWER_REPLY, []

    sources = sorted(
        {
            f"{d.metadata.get('source', '?')} (p.{d.metadata.get('page', '?')})"
            for d in docs
        }
    )
    return answer, sources


def friendly_gemini_error(error):
    """Turn a Gemini SDK exception into a short, actionable message for the UI."""
    code = getattr(error, "code", None)
    if code is None:
        # GoogleGenerativeAIError carries no status field, only the message.
        match = re.search(r"\b(4\d\d|5\d\d)\b", str(error))
        code = int(match.group(1)) if match else None
    if code == 429:
        return (
            "Gemini API quota/rate limit hit. Wait a moment and try again, or check "
            "your quota at aistudio.google.com/app/apikey."
        )
    if code in (401, 403):
        return "Gemini rejected the API key. Check GOOGLE_API_KEY in your .env file."
    if code == 503:
        return "Couldn't reach Gemini. Check your internet connection and try again."
    if code is not None and code >= 500:
        return (
            "Gemini had an internal error and kept failing after several retries. "
            "This is usually temporary - wait a minute and try again."
        )
    return f"Gemini request failed: {error}"


def strip_markdown(text):
    """Flatten Markdown the model emitted into plain text.

    The chat bubbles are raw HTML, not `st.markdown`, so any `**bold**` or `##`
    the model still produces would be shown to the user verbatim.
    """
    # Bullets first, so a leading "* " isn't mistaken for an emphasis marker.
    text = re.sub(r"(?m)^(\s*)[*+]\s+", r"\1- ", text)
    text = re.sub(r"(?m)^\s*#{1,6}\s*", "", text)  # headings
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"\1", text, flags=re.S)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.S)
    text = re.sub(r"(?<!\w)\*(\S.*?\S|\S)\*(?!\w)", r"\1", text, flags=re.S)
    text = re.sub(r"(?<!\w)__(.+?)__(?!\w)", r"\1", text, flags=re.S)
    text = re.sub(r"(?<!\w)_(\S.*?\S|\S)_(?!\w)", r"\1", text, flags=re.S)
    text = re.sub(r"```[a-zA-Z0-9_+-]*\n?", "", text)  # fenced code markers
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text


def render_message(template, text, plain=False):
    # The chat templates are injected with unsafe_allow_html, so anything the model
    # or the PDF produced has to be escaped before it goes in.
    if plain:
        text = strip_markdown(text)
    safe = html.escape(text).replace("\n", "<br>")
    st.write(template.replace("{{MSG}}", safe), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="AI Assistant", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    st.session_state.setdefault("turns", [])
    st.session_state.setdefault("vectorstore", None)

    st.header("Chat with AI Assistant")

    if not os.getenv("GOOGLE_API_KEY"):
        st.error("GOOGLE_API_KEY is not set. Add it to your .env file.")
        st.stop()

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your documents here",
            type="pdf",
            accept_multiple_files=True,
        )
        if st.button("Upload"):
            if not pdf_docs:
                st.warning("Select at least one PDF first.")
            else:
                uploads = [(pdf.name, pdf.getvalue()) for pdf in pdf_docs]
                key = cache_key(uploads)

                cached = load_cached_vectorstore(key)
                if cached is not None:
                    st.session_state.vectorstore = cached
                    st.session_state.turns = []
                    st.session_state.pop("selected_source", None)
                    st.success("Your document is ready. Ask a question below.")
                else:
                    status = st.status("Preparing your document", expanded=False)
                    documents = load_pdfs(uploads)
                    if not documents:
                        status.update(label="Couldn't read this document", state="error")
                        st.error(
                            "We couldn't read any text from this file. It looks like a "
                            "scanned copy - please upload a text-based PDF instead."
                        )
                    else:
                        chunks = split_documents(documents)
                        label = "Preparing your document"
                        bar = status.progress(0.0, text=label)

                        def report(done, total):
                            bar.progress(done / total, text=label)

                        try:
                            vectorstore = build_vectorstore(chunks, on_progress=report)
                        except (google_errors.APIError, GoogleGenerativeAIError) as e:
                            status.update(label="Couldn't prepare the document", state="error")
                            st.error(friendly_gemini_error(e))
                        else:
                            save_cached_vectorstore(key, vectorstore)
                            st.session_state.vectorstore = vectorstore
                            st.session_state.turns = []
                            st.session_state.pop("selected_source", None)
                            status.update(label="Ready", state="complete")
                            st.success("Your document is ready. Ask a question below.")

        # Only worth showing once there is more than one manual to choose between.
        if st.session_state.vectorstore is not None:
            available = document_sources(st.session_state.vectorstore)
            if len(available) > 1:
                st.divider()
                st.subheader("Vehicle")
                st.selectbox(
                    "Answer questions about",
                    available,
                    key="selected_source",
                    label_visibility="collapsed",
                )

    for turn in st.session_state.turns:
        render_message(user_template, turn["question"])
        render_message(bot_template, turn["answer"], plain=True)
        if turn["sources"]:
            st.caption("Sources: " + ", ".join(turn["sources"]))

    user_question = st.chat_input("Ask a question:")
    if user_question:
        if st.session_state.vectorstore is None:
            st.error("Please upload documents first.")
        else:
            render_message(user_template, user_question)
            try:
                with st.spinner("Thinking"):
                    answer, sources = answer_question(
                        user_question, st.session_state.get("selected_source")
                    )
            except (google_errors.APIError, GoogleGenerativeAIError) as e:
                st.error(friendly_gemini_error(e))
            else:
                render_message(bot_template, answer, plain=True)
                if sources:
                    st.caption("Sources: " + ", ".join(sources))
                st.session_state.turns.append(
                    {"question": user_question, "answer": answer, "sources": sources}
                )


if __name__ == "__main__":
    main()