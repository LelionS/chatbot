from flask import Flask, render_template, request, jsonify
from transformers import pipeline, Conversation

app = Flask(__name__)

# Initialize the Hugging Face model
chatbot = pipeline('conversational', model='microsoft/DialoGPT-medium')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['user_message']
    print(f"Received message from user: {user_message}")  # Debug print
    response = generate_response(user_message)
    print(f"Generated response: {response}")  # Debug print
    return jsonify({"response": response})

def generate_response(user_message):
    try:
        conversation = Conversation(user_message)
        chatbot(conversation)
        response_text = conversation.generated_responses[0]
        return response_text
    except Exception as e:
        print(f"Unexpected error: {e}")  # Log unexpected errors
        return "Sorry, I am having trouble responding right now."

if __name__ == '__main__':
    app.run(debug=True)
