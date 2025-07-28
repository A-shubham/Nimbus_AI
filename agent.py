# --- agent.py (Final Prompt Fix Version) ---

import logging
import json
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_vertexai.vectorstores import VectorSearchVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain import hub 

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_agent_tools(api_key: str):
    """Creates the tools for the agent."""
    
    def _run_document_retrieval(query: str) -> str:
        logging.info(f"Agent using Vertex AI Document Retrieval Tool for query: '{query}'")
        
        # --- REAL IMPLEMENTATION (Enabled for live demo) ---
        try:
            vector_store = VectorSearchVectorStore.from_components(
                project_id=config.PROJECT_ID,
                region=config.REGION,
                gcs_bucket_name=config.GCS_BUCKET_NAME,
                index_name=config.VECTOR_SEARCH_INDEX_NAME,
                index_id=config.VECTOR_SEARCH_INDEX_ID,
                endpoint_id=config.VECTOR_SEARCH_ENDPOINT_ID,
                embedding=VertexAIEmbeddings(model_name="text-embedding-004"),
            )
            docs = vector_store.similarity_search(query, k=4)
            if not docs:
                return "No relevant information was found in the documents for this query."
            
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            logging.error(f"Error during document retrieval from Vertex AI: {e}", exc_info=True)
            return "Sorry, I encountered an error while searching the documents."
        # --- END OF REAL IMPLEMENTATION ---


    # THE FIX: Removed the extra space in the tool name.
    document_retriever_tool = Tool(
        name="Document_Retriever",
        func=_run_document_retrieval,
        description="""Use this tool to answer any questions about topics the user has provided in documents.
        This is the primary source of information. Use it for any query that asks for information or details."""
    )

    return [document_retriever_tool]

def create_agent_executor(api_key: str):
    tools = create_agent_tools(api_key)
    
    prompt = hub.pull("hwchase17/react-chat")
    
    
    agent_llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest", 
        google_api_key=api_key, 
        temperature=0.2
    )
    
    agent = create_react_agent(agent_llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True, 
        handle_parsing_errors=True,
        max_iterations=5
    )

async def stream_agent_response(agent_executor, query: str, chat_history: list):
    """
    Invokes the agent using the astream() method and yields the final output tokens.
    """
    formatted_chat_history = []
    for msg in chat_history:
        role, content = msg.get("role"), msg.get("text")
        if role == 'user':
            formatted_chat_history.append(HumanMessage(content=content))
        elif role == 'model':
            formatted_chat_history.append(AIMessage(content=content))

    full_response = ""
    async for chunk in agent_executor.astream(
        {"input": query, "chat_history": formatted_chat_history}
    ):
        if "output" in chunk:
            token = chunk["output"][len(full_response):]
            full_response = chunk["output"]
            yield token
