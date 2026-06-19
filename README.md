# 🚀 Private RAG Chatbot

A privacy-first Retrieval-Augmented Generation (RAG) chatbot built using Angular, Node.js, LangChain, and Ollama.

This application enables users to chat with a company knowledge base using natural language. It retrieves relevant information from documents and generates context-aware responses using a locally hosted Large Language Model (LLM).

---

## 🌟 Features

- Retrieval-Augmented Generation (RAG)
- Local LLM inference using Ollama
- Real-time streaming responses (SSE)
- Semantic search using embeddings
- Context-aware question answering
- Angular-based chat interface
- LangChain integration
- Prompt guardrails to reduce hallucinations
- Enterprise knowledge base support
- Privacy-focused architecture

---

## 🏗️ Architecture

```text
Knowledge Base
      │
      ▼
Document Loader
      │
      ▼
Text Chunking
      │
      ▼
Embeddings
(nomic-embed-text)
      │
      ▼
MemoryVectorStore
      │
      ▼
Retriever
      │
      ▼
Prompt Template
      │
      ▼
Llama 3.2 (Ollama)
      │
      ▼
Streaming Response
      │
      ▼
Angular Frontend
```

---

## ⚙️ Technology Stack

### Frontend

- Angular 18
- TypeScript
- RxJS

### Backend

- Node.js
- Express.js

### AI Stack

- LangChain
- Ollama
- Llama 3.2
- nomic-embed-text

### Retrieval Layer

- MemoryVectorStore
- RecursiveCharacterTextSplitter

---

## 🔄 How It Works

1. Load company knowledge documents.
2. Split documents into smaller chunks.
3. Generate vector embeddings.
4. Store embeddings in a vector store.
5. Retrieve relevant chunks for user queries.
6. Build a contextual prompt using retrieved information.
7. Generate grounded responses using Llama 3.2.
8. Stream responses back to the Angular UI.

---

## 💬 Example Questions

Try asking:

- Who is the CEO of Alfaiz Technologies?
- What products does the company offer?
- What is the pricing of AlfaizChat?
- What technologies are used internally?
- What are the support SLAs?
- Where are the company offices located?

---

## 📡 API Endpoints

### Health Check

```http
GET /api/health
```

Checks whether the RAG pipeline has been initialized successfully.

### Chat Streaming

```http
POST /api/chat/stream
```

Streams AI-generated responses to the frontend in real time.

---

## 🎯 Business Use Cases

### Enterprise Knowledge Assistant

Provide employees with instant access to internal documentation.

### Customer Support Assistant

Answer product and pricing questions using company knowledge.

### Employee Onboarding

Help new employees understand company processes and policies.

### Internal Documentation Search

Replace manual document searching with natural language queries.

---

## 🛠️ Local Setup

### Clone Repository

```bash
git clone https://github.com/AlfaizMomin/private-rag-chatbot.git
cd private-rag-chatbot
```

### Install Backend Dependencies

```bash
npm install
```

### Start Ollama

```bash
ollama serve
```

### Pull Required Models

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### Start Backend

```bash
npm start
```

### Start Frontend

```bash
ng serve
```

Application will run at:

```text
http://localhost:4200
```

---

## 🧠 Skills Demonstrated

- Angular Development
- Node.js API Development
- Express.js
- LangChain
- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Vector Embeddings
- Prompt Engineering
- LLM Integration
- Server-Sent Events (SSE)
- Full Stack Development
- AI Application Development

---

## 🚀 Future Enhancements

- PDF Upload Support
- DOCX Support
- Persistent Vector Database
- Authentication & RBAC
- Source Citations
- Conversation History
- Multi-Tenant Knowledge Bases
- Analytics Dashboard
- Multi-Language Support

---

## 📋 Project Summary

Built a privacy-first RAG chatbot using Angular, Node.js, LangChain, and Ollama. Implemented document chunking, embedding generation, semantic retrieval, prompt engineering, and real-time streaming responses to provide accurate answers from a company knowledge base.

---

## 👨‍💻 Developer

**Alfaiz Momin**

Full Stack Developer | Angular | Node.js | TypeScript | PostgreSQL | AI Applications | RAG Systems

GitHub: https://github.com/AlfaizMomin
