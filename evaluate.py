# --- evaluate.py (Final Ragas Fix) ---

import os
import asyncio
import logging
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from dotenv import load_dotenv

# --- We no longer need a ragas wrapper, just the models themselves ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import VertexAIEmbeddings

# Import our project modules
import config
from agent import create_agent_executor, stream_agent_response 
from document_processor import process_and_upload_documents
from agent import create_agent_tools 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Our test dataset
eval_dataset = {
    'question': [
        "What is the main feature of Project Nimbus?",
        "When was the project launched?",
        "What is the primary weakness of the system?",
    ],
    'ground_truth': [
        "The core feature of Project Nimbus is an agentic RAG (Retrieval-Augmented Generation) system built on a hybrid architecture.",
        "The project was launched in July 2025.",
        "The provided text does not mention any weaknesses of the system.",
    ]
}

async def main():
    """ Main function to run the evaluation process. """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not found in environment variables.")
        return

    # --- THE FIX: We instantiate the LangChain models directly ---
    # The Ragas `evaluate` function can now accept them without a wrapper.
    ragas_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key)
    ragas_embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
    # -----------------------------------------------------------

    # --- Prepare the Knowledge Base ---
    print("--- Step 1: Preparing the knowledge base ---")
    print("IMPORTANT: This script assumes the Vertex AI index is empty to begin with for an accurate evaluation.")
    test_doc_path = 'test_document.txt'
    if not os.path.exists(test_doc_path):
        print(f"Error: {test_doc_path} not found.")
        return
    process_and_upload_documents([test_doc_path])
    print("Knowledge base is ready.\n")

    # --- Generate Answers and Contexts ---
    print("--- Step 2: Generating answers from the agent ---")
    agent_executor = create_agent_executor(api_key)
    tools = create_agent_tools(api_key)
    retriever_tool = next((tool for tool in tools if tool.name == "Document_Retriever"), None)
    if not retriever_tool:
        print("Error: Document_Retriever tool not found.")
        return

    results = []
    for question in eval_dataset['question']:
        print(f"Asking question: {question}")
        
        full_answer = ""
        async for token in stream_agent_response(agent_executor, question, []):
            full_answer += token
        answer = full_answer.strip()
        
        context = retriever_tool.func(question)
        results.append({
            "question": question,
            "answer": answer,
            "contexts": [context]
        })
    print("Agent has answered all questions.\n")

    # --- Evaluate with RAGas ---
    print("--- Step 3: Evaluating the results with RAGas ---")
    for i, result in enumerate(results):
        result['ground_truth'] = eval_dataset['ground_truth'][i]

    dataset = Dataset.from_list(results)
    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
    
    # --- THE FIX: Pass the direct LangChain models to the evaluate function ---
    result = evaluate(
        dataset=dataset, 
        metrics=metrics,
        llm=ragas_llm, 
        embeddings=ragas_embeddings
    ) 
    
    print("Evaluation complete.\n")

    # --- Print the Report ---
    print("--- Step 4: Final Evaluation Report ---")
    print(result)

if __name__ == "__main__":
    load_dotenv()
    if asyncio.get_event_loop().is_running():
        loop = asyncio.get_event_loop()
        loop.create_task(main())
    else:
        asyncio.run(main())