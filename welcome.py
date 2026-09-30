"""Landing page: entry point for the Sundeus automotive AI assistants.

Run with `streamlit run welcome.py`. `app.py` stays runnable on its own, so the
chatbot can still be started directly with `streamlit run app.py`.
"""

import streamlit as st

import app

TITLE = "Welcome to ChatGPT for Automotive Problem and Solutions using Generative AI Assistant"

chat_page = st.Page(
    app.main,
    title="AI Assistant for Vehicle Maintenance",
    icon=":material/forum:",
    url_path="chat",
    # Reached from the button on the home page, so keep it out of the nav bar.
    visibility="hidden",
)


def home():
    st.set_page_config(
        page_title="Automotive AI Assistant", page_icon=":material/hub:", layout="wide"
    )

    with st.container(horizontal_alignment="center"):
        st.title(TITLE, anchor=False)
        st.write("")

        st.subheader("AI Assistant for Vehicle Maintenance", anchor=False)
        if st.button("Click Here", type="primary", width="content"):
            st.switch_page(chat_page)
        st.write("")

        # Not built yet. The names stay on the landing page so the line-up is
        # visible; the buttons are disabled until each assistant is wired up.
        for name in ("AI Assistant for Vehicle Design", "Conversational Assistant"):
            st.subheader(name, anchor=False)
            st.button(
                "Click Here",
                key=f"soon-{name}",
                type="primary",
                width="content",
                disabled=True,
                help="Coming soon",
            )
            st.write("")


home_page = st.Page(
    home, title="Home", icon=":material/home:", default=True, url_path="home"
)

st.navigation([home_page, chat_page], position="top").run()
