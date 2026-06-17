# 🟠 RAG Compliance Search — LangChain + Pinecone

> Semantic search across **2M+ pages** of regulatory documents using Retrieval-Augmented Generation. Reduced compliance research time by **60%** for 5,000+ risk professionals.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-1C3C3C?style=flat-square)](https://langchain.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-412991?style=flat-square&logo=openai)](https://openai.com)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-000000?style=flat-square)](https://pinecone.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)

---

## 📋 Overview

This project implements a **production-grade RAG pipeline** for searching regulatory and policy documents — a system built at Citibank to let risk and legal teams query 2M+ pages of internal policy, regulatory filings, and audit documentation using natural language.

**Impact:**
- 🔍 **60% faster** compliance research for 5,000+ professionals
- 📄 **2M+ pages** indexed into a semantic vector store
- ⚡ **<3s** end-to-end query latency (embedding + retrieval + generation)
- 🔒 **Namespace-isolated** access control (user roles map to Pinecone namespaces)

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────┐
│           Query Processing               │
│  • Rewrite ambiguous queries             │
│  • Extract filters (date, doc_type, etc) │
└──────────────────┬───────────────────────┘
                   │
    ┌──────────────▼──────────────┐
    │   OpenAI Embedding API      │
    │   text-embedding-3-large    │
    └──────────────┬──────────────┘
                   │  query_vector [3072-d]
    ┌──────────────▼──────────────┐
    │      Pinecone Index         │
    │   Namespace: {user_role}    │
    │   Top-K = 8 chunks          │
    │   Metadata filter: year, type│
    └──────────────┬──────────────┘
                   │  Retrieved chunks
    ┌──────────────▼──────────────┐
    │   Contextual Compression    │
    │   (filter irrelevant chunks)│
    └──────────────┬──────────────┘
                   │
    ┌──────────────▼──────────────┐
    │    GPT-4 / Claude Haiku     │
    │    Answer + Citations       │
    └──────────────┬──────────────┘
                   │
               ┌───▼────┐
               │Response│  answer + source_docs + confidence
               └────────┘
```

---

## 📂 Project Structure

```
02-rag-compliance-search/
├── README.md
├── requirements.txt
├── rag_pipeline.py          # Core RAG chain with LangChain
├── document_ingestion.py    # PDF ingestion + chunking + embedding + upsert
└── api_server.py            # FastAPI endpoint for the search UI
```

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY=sk-...
export PINECONE_API_KEY=...
export PINECONE_ENV=us-east-1

# Ingest documents (PDF/DOCX folder)
python document_ingestion.py --input-dir ./regulatory_docs --namespace risk_team

# Start the API server
uvicorn api_server:app --host 0.0.0.0 --port 8000

# Query (curl)
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the Basel III capital requirements for credit risk?", "namespace": "risk_team"}'
```

---

## 📊 Performance Benchmarks

| Metric | Value |
|--------|-------|
| Documents Indexed | 2M+ pages |
| Chunk Size | 512 tokens (128 overlap) |
| Embedding Model | text-embedding-3-large |
| Retrieval Latency (p95) | 180ms |
| Generation Latency (p95) | 2.1s |
| End-to-End Latency (p95) | 2.8s |
| Answer Relevance Score | 4.3 / 5.0 (human eval) |
| Namespace Isolation | ✅ Role-based |

---

## 🔑 Key Design Choices

**Chunking strategy:** Recursive character text splitter at 512 tokens with 128-token overlap preserves sentence context across chunk boundaries — critical for regulatory text where clause boundaries matter.

**Namespace isolation:** Pinecone namespaces map to RBAC roles (risk_team, legal_team, audit_team), ensuring users only retrieve documents they're authorized to see.

**Contextual compression:** LangChain's `ContextualCompressionRetriever` filters retrieved chunks with a secondary LLM call before generation, reducing hallucination from off-topic chunks.
