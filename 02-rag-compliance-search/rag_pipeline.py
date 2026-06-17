"""
RAG Compliance Search Pipeline
================================
Production RAG pipeline for semantic search over regulatory documents.
Uses LangChain orchestration, OpenAI embeddings, and Pinecone vector store.

Features:
  - Namespace-isolated retrieval (role-based access control)
  - Contextual compression to filter irrelevant chunks
  - Query rewriting for ambiguous regulatory terms
  - Source citation with page numbers and document metadata
  - Async support for high-concurrency API usage

Author: Akhil Vase | Senior AI/ML Engineer
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import Document
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RAGConfig:
    # OpenAI
    openai_api_key: str = field(default_factory=lambda: os.environ["OPENAI_API_KEY"])
    embedding_model: str = "text-embedding-3-large"
    llm_model: str = "gpt-4o"
    temperature: float = 0.0           # deterministic for compliance use-case

    # Pinecone
    pinecone_api_key: str = field(default_factory=lambda: os.environ["PINECONE_API_KEY"])
    pinecone_index: str = "compliance-docs"
    pinecone_region: str = "us-east-1"
    embedding_dim: int = 3072          # text-embedding-3-large output dim

    # Retrieval
    top_k: int = 8                     # chunks to retrieve per query
    compression_enabled: bool = True   # filter irrelevant chunks before generation
    max_tokens_answer: int = 1024

    # Chunking (used in document_ingestion.py — referenced here for consistency)
    chunk_size: int = 512
    chunk_overlap: int = 128


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a compliance and regulatory research assistant for a financial institution.
Your role is to answer questions accurately using only the provided regulatory documents.

Guidelines:
- Answer ONLY from the provided context. Do not use prior knowledge.
- Cite the exact document, section, and page number for every claim.
- If the context does not contain enough information, say so explicitly.
- Use precise regulatory language. Do not paraphrase in ways that change meaning.
- Flag any ambiguities or conflicting guidance across documents.

Context:
{context}
"""

HUMAN_PROMPT = """\
Question: {question}

Provide a structured answer with:
1. Direct answer to the question
2. Supporting citations (Document: ... | Section: ... | Page: ...)
3. Any caveats or limitations of the available information
"""

QUERY_REWRITE_PROMPT = """\
You are helping refine a regulatory research query for better semantic retrieval.
Rewrite the following query to be more specific and include relevant regulatory terms.
Return only the rewritten query, nothing else.

Original query: {query}
"""


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

class ComplianceRAGPipeline:
    """
    End-to-end RAG pipeline for regulatory document search.

    Usage
    -----
    pipeline = ComplianceRAGPipeline(config)
    result = pipeline.query("What are Basel III tier 1 capital requirements?",
                             namespace="risk_team")
    print(result.answer)
    for doc in result.source_documents:
        print(doc.metadata)
    """

    def __init__(self, config: RAGConfig | None = None):
        self.config = config or RAGConfig()
        self._embeddings = self._build_embeddings()
        self._llm = self._build_llm()
        self._pc = Pinecone(api_key=self.config.pinecone_api_key)
        self._ensure_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        namespace: str,
        chat_history: list[tuple[str, str]] | None = None,
        rewrite_query: bool = True,
        metadata_filter: dict[str, Any] | None = None,
    ) -> "RAGResult":
        """
        Retrieve relevant chunks and generate a grounded, cited answer.

        Parameters
        ----------
        question       : User's natural-language question
        namespace      : Pinecone namespace — maps to user's RBAC role
        chat_history   : List of (human, assistant) turn tuples for conversation
        rewrite_query  : Whether to rewrite ambiguous queries before retrieval
        metadata_filter: Optional Pinecone metadata pre-filter (e.g. {"year": {"$gte": 2020}})

        Returns
        -------
        RAGResult with .answer, .source_documents, .rewritten_query
        """
        effective_query = question
        if rewrite_query:
            effective_query = self._rewrite_query(question)
            if effective_query != question:
                log.info("Query rewritten: '%s' → '%s'", question, effective_query)

        retriever = self._build_retriever(namespace, metadata_filter)
        chain = self._build_chain(retriever)

        response = chain.invoke({
            "question": effective_query,
            "chat_history": chat_history or [],
        })

        return RAGResult(
            answer=response["answer"],
            source_documents=response["source_documents"],
            rewritten_query=effective_query if effective_query != question else None,
        )

    def batch_query(
        self,
        questions: list[str],
        namespace: str,
    ) -> list["RAGResult"]:
        """Run multiple queries; useful for evaluation / benchmarking."""
        return [self.query(q, namespace, rewrite_query=False) for q in questions]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_embeddings(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(
            model=self.config.embedding_model,
            openai_api_key=self.config.openai_api_key,
        )

    def _build_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.config.llm_model,
            temperature=self.config.temperature,
            openai_api_key=self.config.openai_api_key,
            max_tokens=self.config.max_tokens_answer,
        )

    def _ensure_index(self) -> None:
        """Create Pinecone index if it doesn't exist."""
        existing = [i.name for i in self._pc.list_indexes()]
        if self.config.pinecone_index not in existing:
            log.info("Creating Pinecone index '%s' …", self.config.pinecone_index)
            self._pc.create_index(
                name=self.config.pinecone_index,
                dimension=self.config.embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=self.config.pinecone_region),
            )

    def _build_retriever(
        self,
        namespace: str,
        metadata_filter: dict[str, Any] | None,
    ):
        """Build a (optionally compressed) Pinecone retriever for a given namespace."""
        index = self._pc.Index(self.config.pinecone_index)
        vectorstore = LangchainPinecone(
            index=index,
            embedding=self._embeddings,
            text_key="text",
            namespace=namespace,
        )

        base_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": self.config.top_k,
                **({"filter": metadata_filter} if metadata_filter else {}),
            },
        )

        if not self.config.compression_enabled:
            return base_retriever

        compressor = LLMChainExtractor.from_llm(self._llm)
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
        )

    def _build_chain(self, retriever) -> ConversationalRetrievalChain:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template(HUMAN_PROMPT),
        ])

        return ConversationalRetrievalChain.from_llm(
            llm=self._llm,
            retriever=retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": prompt},
            verbose=False,
        )

    def _rewrite_query(self, query: str) -> str:
        """Use LLM to clarify and expand regulatory queries for better retrieval."""
        rewrite_llm = ChatOpenAI(
            model="gpt-4o-mini",  # cheaper model for rewriting
            temperature=0.0,
            openai_api_key=self.config.openai_api_key,
            max_tokens=200,
        )
        response = rewrite_llm.invoke(
            QUERY_REWRITE_PROMPT.format(query=query)
        )
        return response.content.strip()


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RAGResult:
    answer: str
    source_documents: list[Document]
    rewritten_query: str | None = None

    def format_citations(self) -> str:
        """Return a formatted citation block from source document metadata."""
        lines = ["**Sources:**"]
        seen = set()
        for doc in self.source_documents:
            m = doc.metadata
            cite = (
                f"- {m.get('document_title', 'Unknown')} "
                f"| Section {m.get('section', 'N/A')} "
                f"| Page {m.get('page', 'N/A')} "
                f"| Date: {m.get('publication_date', 'N/A')}"
            )
            if cite not in seen:
                lines.append(cite)
                seen.add(cite)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": self.format_citations(),
            "rewritten_query": self.rewritten_query,
            "num_source_chunks": len(self.source_documents),
        }
