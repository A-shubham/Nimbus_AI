# Nimbus AI Pro - Production-Grade Agentic RAG System

Nimbus AI Pro is a full-stack, conversational AI assistant built on Google Cloud Platform. It allows users to create a persistent, private knowledge base by uploading their own documents and then engage in intelligent, context-aware conversations to retrieve information from them.

This project is designed to be a production-ready application, moving beyond simple scripts to incorporate a scalable cloud architecture, a robust agentic reasoning engine, and a systematic evaluation framework.


---

## Key Features

* **Agentic RAG Architecture:** Utilizes a LangChain ReAct agent that intelligently decides whether to retrieve information from user documents or handle a general conversational query.
* **Persistent Knowledge Base:** Leverages **Google Cloud's Vertex AI Vector Search** to create a scalable and persistent vector database, allowing the knowledge base to grow without performance degradation.
* **Real-Time Streaming:** Provides a modern user experience by streaming the AI's responses token-by-token using Server-Sent Events (SSE) and an asynchronous backend.
* **Persistent Chat Memory:** Employs **Google Firestore** to save and load conversation histories, enabling stateful, multi-turn conversations.
* **Systematic Evaluation:** Includes an offline evaluation pipeline using the **RAGas framework** to quantitatively measure the system's performance on key metrics like faithfulness, context precision, and recall.
* **Cloud-Native & Serverless:** The entire application is containerized with **Docker** and deployed on **Google Cloud Run**, ensuring automatic scaling and cost-efficiency.

---

## Technology Stack

| Category                  | Technologies                                                                                   |
| ------------------------- | ---------------------------------------------------------------------------------------------- |
| **Generative AI & LLMs** | Python, LangChain, RAG Architecture, Agentic Workflows, Prompt Engineering, Gemini               |
| **Cloud & Data** | Google Cloud Platform (GCP), Vertex AI Vector Search, Firestore, Cloud Run, Cloud Storage (GCS)  |
| **Backend & DevOps** | Docker, Flask, Hypercorn (ASGI), REST APIs, Git, Google Cloud Build                              |

---

