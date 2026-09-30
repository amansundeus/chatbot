css = '''
<style>
.chat-message{
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user{
    background-color: #2b313e
}
.chat-message.bot{
    background-color: #475063
}
.chat-message .avatar{
    width: 15%;
}
.chat-message .avatar img{
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-for: cover;
    padding-right: 20px;
}
.chat-message .message{
    width: 85%
    height: 200px
    padding: 0 1.5rem;
    color: #fff;
}
'''

bot_template = '''
<div class = "chat-message bot">
    <div class = "avatar">
        <img src = "https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png">
    </div>
    <div class = "message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class = "chat-message user">
    <div class = "avatar">
        <img src = "https://images.pexels.com/photos/356079/pexels-photo-356079.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1">
    </div>
    <div class = "message">{{MSG}}</div>
</div>
'''