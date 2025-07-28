
import os
import logging
import json
import asyncio
import uuid
from flask import Flask, request, jsonify, session, render_template, Response
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from agent import create_agent_executor, stream_agent_response
from document_processor import process_and_upload_documents
from firestore_manager import save_chat_history, load_chat_history


load_dotenv()


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.secret_key = os.urandom(24) 
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_session_id():
    """Gets or creates a unique session ID."""
    if 'sid' not in session:
        session['sid'] = str(uuid.uuid4())  
    return session['sid']

@app.route('/')
def index():
    session_id = get_session_id()
    chat_history = load_chat_history(session_id)
    logging.info(f"New connection for session: {session_id}, loaded {len(chat_history)} messages.")
    return render_template('index.html', chat_messages=chat_history)

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files: return jsonify({"error": "No files part"}), 400
    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files): return jsonify({"error": "No selected files"}), 400

    try:
        file_paths = []
        for file in files:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            file_paths.append(file_path)
        
        success = process_and_upload_documents(file_paths)
        for file_path in file_paths: os.remove(file_path)

        if not success: return jsonify({"error": "Failed to process and upload documents."}), 500
        
        logging.info("Files processed and uploaded to Vertex AI.")
        return jsonify({"message": f"Successfully added {len(files)} file(s) to the knowledge base."})
    except Exception as e:
        logging.error(f"Error during file upload: {e}", exc_info=True)
        return jsonify({"error": f"An error occurred: {e}"}), 500

# --- THIS IS THE CORRECTED CHAT ROUTE ---
async def stream_and_save(session_id, user_message, api_key):
    """Async generator that loads history, streams the agent's response, and saves updated history."""
    
    # 1. Load the most up-to-date history from Firestore
    chat_history = load_chat_history(session_id)
    
    # 2. Append the new user message
    chat_history.append({"role": "user", "text": user_message})
    
    # 3. Create the agent and stream the response, passing the full history
    agent_executor = create_agent_executor(api_key)
    full_bot_response = ""
    
    async for token in stream_agent_response(agent_executor, user_message, chat_history):
        full_bot_response += token
        yield f"data: {json.dumps({'token': token})}\n\n"
        
    # 4. After streaming, save the full, updated history back to Firestore
    chat_history.append({"role": "model", "text": full_bot_response})
    save_chat_history(session_id, chat_history)
    logging.info("Stream finished, chat history saved to Firestore.")

@app.route('/chat', methods=['GET'])
def chat():
    user_message = request.args.get('message') 
    if not user_message: return Response("error: No message provided", status=400)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: return Response("error: API key not configured.", status=500)
    
    session_id = get_session_id()

    def generate():
        # This sync generator runs our main async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Create an async generator from our main logic function
        async_gen = stream_and_save(session_id, user_message, api_key)
        
        try:
            while True:
                # Pull items from the async generator one by one
                yield loop.run_until_complete(async_gen.__anext__())
        except StopAsyncIteration:
            pass # The stream is finished
        finally:
            loop.close()

    return Response(generate(), mimetype='text/event-stream')