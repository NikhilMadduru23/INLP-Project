from flask import Flask, request, jsonify, render_template
from flask_session import Session
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chatbot_cpu import chatbot, history_store, last_known_context  # Import last_known_context

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key')
Session(app)

# Flask routes
@app.route('/')
def index():
    session_id = request.remote_addr  # Use client IP as session ID
    if session_id not in history_store:
        history_store[session_id] = []
    return render_template('index.html', history=history_store[session_id])

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not isinstance(data, dict):
            return jsonify({'response': 'Invalid input format.', 'history': history_store.get(request.remote_addr, [])}), 400
        user_input = data.get('input', '').strip()
        if not isinstance(user_input, str):
            user_input = ''
        session_id = data.get('session_id', request.remote_addr)
        if user_input:
            # Append user input to history
            history_store[session_id].append(f"User: {user_input}")
            # Get bot response
            response = chatbot(user_input, session_id)
            # Append bot response to history
            history_store[session_id].append(f"Bot: {response}")
        else:
            response = "Please provide a valid input."
        return jsonify({'response': response, 'history': history_store[session_id][-10:]})
    except Exception as e:
        return jsonify({'response': f'Error: {str(e)}', 'history': history_store.get(request.remote_addr, [])}), 500

@app.route('/clear', methods=['POST'])
def clear():
    session_id = request.remote_addr  # Use client IP as session ID
    history_store[session_id] = []  # Clear the history for this session
    if session_id in last_known_context:
        last_known_context[session_id] = {"departure_city": None, "destination_city": None, "dates": [], "airlines": [], "times": [], "classes": []}  # Reset context
    return jsonify({'status': 'success', 'history': history_store[session_id]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)