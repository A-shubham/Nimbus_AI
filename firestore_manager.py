# --- firestore_manager.py ---

import logging
from google.cloud import firestore
import config

# Initialize the Firestore client
try:
    db = firestore.Client(project=config.PROJECT_ID)
    logging.info("Firestore client initialized successfully.")
except Exception as e:
    logging.critical(f"Failed to initialize Firestore client: {e}")
    db = None

def save_chat_history(session_id: str, chat_history: list):
    """Saves the entire chat history to a Firestore document."""
    if not db:
        logging.error("Firestore client not available. Cannot save chat history.")
        return
    
    try:
        # Create a reference to a document with the user's session ID as the name
        doc_ref = db.collection('chat_sessions').document(session_id)
        # Set the document's data to be the chat history list
        doc_ref.set({'messages': chat_history})
        logging.info(f"Saved chat history for session: {session_id}")
    except Exception as e:
        logging.error(f"Error saving chat history for session {session_id}: {e}", exc_info=True)

def load_chat_history(session_id: str) -> list:
    """Loads the chat history from a Firestore document."""
    if not db:
        logging.error("Firestore client not available. Cannot load chat history.")
        return []
        
    try:
        doc_ref = db.collection('chat_sessions').document(session_id)
        doc = doc_ref.get()
        if doc.exists:
            logging.info(f"Loaded chat history for session: {session_id}")
            # Return the 'messages' array from the document, or an empty list if not found
            return doc.to_dict().get('messages', [])
        else:
            logging.info(f"No chat history found for new session: {session_id}")
            return []
    except Exception as e:
        logging.error(f"Error loading chat history for session {session_id}: {e}", exc_info=True)
        return []