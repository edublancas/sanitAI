from flask import Flask
from flask_sock import Sock


from presidioui.dialog import name_dialog, WebsocketDialog


app = Flask(__name__)
sock = Sock(app)


@app.route("/")
def index():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Workflow</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .messages {
            height: calc(100vh - 80px);
            overflow-y: auto;
        }
        .message {
            max-width: 70%;
            margin: 8px;
            padding: 12px;
            border-radius: 20px;
        }
        .bot-message {
            background-color: #e9ecef;
            margin-right: auto;
            border-top-left-radius: 4px;
        }
        .user-message {
            background-color: #0084ff;
            color: white;
            margin-left: auto;
            border-top-right-radius: 4px;
        }
    </style>
    <script>
        let ws = new WebSocket('ws://' + location.host + '/workflow');
        
        ws.onopen = function() {
            console.log('Connected to server');
        };

        ws.onmessage = function(event) {
            const msg = JSON.parse(event.data);
            const messagesDiv = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message bot-message';
            messageDiv.textContent = msg.data.content;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            
            if (msg.data.content.endsWith('?')) {
                document.getElementById('input').disabled = false;
                document.getElementById('input').focus();
            }
        };

        function sendMessage() {
            const input = document.getElementById('input');
            const message = input.value.trim();
            if (message) {
                const messagesDiv = document.getElementById('messages');
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message user-message';
                messageDiv.textContent = message;
                messagesDiv.appendChild(messageDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
                
                ws.send(JSON.stringify({data: message}));
                input.value = '';
                input.disabled = true;
            }
        }
    </script>
</head>
<body class="bg-gray-100">
    <div class="container mx-auto max-w-2xl h-screen flex flex-col">
        <div id="messages" class="messages bg-white p-4 flex-grow overflow-y-auto"></div>
        <div class="bg-white border-t p-4 flex gap-2">
            <input 
                id="input" 
                type="text" 
                class="flex-grow rounded-full px-4 py-2 border focus:outline-none focus:border-blue-500"
                placeholder="Type your message..."
                onkeypress="if(event.key === 'Enter') sendMessage()"
                disabled
            >
            <button 
                onclick="sendMessage()" 
                class="bg-blue-500 text-white rounded-full px-6 py-2 hover:bg-blue-600 focus:outline-none"
            >
                Send
            </button>
        </div>
    </div>
</body>
</html>
"""


@sock.route("/workflow")
def workflow(ws):
    dialog = WebsocketDialog(ws)
    state = name_dialog(dialog)
    # Do something with final state...


if __name__ == "__main__":
    app.run()
