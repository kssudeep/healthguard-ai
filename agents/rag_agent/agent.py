"""
agents/rag_agent/agent.py

RAG Agent for HealthGuard AI.
- Hybrid retrieval: FAISS (dense) + BM25 (sparse) with reciprocal rank fusion
- Cross-encoder reranking (ms-marco-MiniLM)
- Knowledge sources: PubMed abstracts, clinical guidelines, drug interactions DB
- Query expansion using vision + NLP findings
- Caches vector store in-memory for speed
"""

from __future__ import annotations
import time
import logging
import json
import os
from pathlib import Path
from dataclasses import asdict

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from core.state.clinical_state import (
    ClinicalState,
    RAGFindings,
    RetrievedDocument,
)
from monitoring.mlflow_tracking.tracker import MLflowTracker
from config.settings import settings

logger = logging.getLogger(__name__)

# ── Model identifiers ─────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Domain-specific for medical text:
MEDICAL_EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K_RETRIEVE = 20     # initial retrieval per source
TOP_K_RERANK = 5        # after reranking

EVIDENCE_LEVELS = {
    "pubmed": "moderate",
    "clinical_guideline": "high",
    "drug_db": "high",
    "textbook": "moderate",
}


# ── Singletons ─────────────────────────────────────────────────────────────────

_vector_store: FAISS = None
_bm25_retriever: BM25Retriever = None
_reranker: CrossEncoder = None
_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        logger.info("[RAGAgent] Loading medical embedding model...")
        _embeddings = HuggingFaceEmbeddings(
            model_name=MEDICAL_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_vector_store() -> FAISS:
    global _vector_store
    if _vector_store is None:
        index_path = settings.FAISS_INDEX_PATH
        if Path(index_path).exists():
            logger.info(f"[RAGAgent] Loading FAISS index from {index_path}")
            _vector_store = FAISS.load_local(
                index_path, get_embeddings(), allow_dangerous_deserialization=True
            )
        else:
            logger.warning("[RAGAgent] No FAISS index found — creating demo index")
            _vector_store = _create_demo_index()
    return _vector_store


def get_bm25_retriever() -> BM25Retriever:
    global _bm25_retriever
    if _bm25_retriever is None:
        corpus_path = settings.BM25_CORPUS_PATH
        if Path(corpus_path).exists():
            with open(corpus_path) as f:
                docs_data = json.load(f)
            documents = [
                Document(page_content=d["content"], metadata=d.get("metadata", {}))
                for d in docs_data
            ]
        else:
            logger.warning("[RAGAgent] No BM25 corpus found — using demo docs")
            documents = _get_demo_documents()
        _bm25_retriever = BM25Retriever.from_documents(documents, k=TOP_K_RETRIEVE)
    return _bm25_retriever


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info("[RAGAgent] Loading cross-encoder reranker...")
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


# ── Demo / fallback data ───────────────────────────────────────────────────────

def _get_demo_documents() -> list[Document]:
    """Minimal demo documents for development without the full corpus."""
    return [
        Document(
            page_content=(
                "Pneumonia is an infection that inflames the air sacs in one or both lungs. "
                "Symptoms include cough with phlegm, fever, chills, and difficulty breathing. "
                "Chest X-ray typically shows consolidation or infiltrates."
            ),
            metadata={"source": "clinical_guideline", "title": "Pneumonia Overview", "doc_id": "CG001"},
        ),
        Document(
            page_content=(
                "Pleural effusion is accumulation of fluid in the pleural space. "
                "Clinical features: dyspnea, pleuritic chest pain, dullness on percussion. "
                "CXR shows blunting of costophrenic angles and fluid opacity."
            ),
            metadata={"source": "clinical_guideline", "title": "Pleural Effusion", "doc_id": "CG002"},
        ),
        Document(
            page_content=(
                "Cardiomegaly on chest X-ray is defined as cardiothoracic ratio > 0.5. "
                "Associated conditions: congestive heart failure, cardiomyopathy, pericardial effusion. "
                "Further investigation: echocardiogram, BNP levels."
            ),
            metadata={"source": "pubmed", "title": "Cardiomegaly on CXR", "doc_id": "PM001"},
        ),
        Document(
            page_content=(
                "Pulmonary atelectasis: collapse or closure of lung resulting in reduced gas exchange. "
                "Causes: obstruction, compression, surfactant deficiency. "
                "CXR: increased opacity, mediastinal shift toward affected side."
            ),
            metadata={"source": "pubmed", "title": "Atelectasis", "doc_id": "PM002"},
        ),
        Document(
            page_content=(
                "Pneumothorax: presence of air in pleural space. Spontaneous or traumatic. "
                "CXR: visible pleural line, absence of lung markings beyond pleural edge. "
                "Treatment: observation (small), needle decompression, chest tube (large/tension)."
            ),
            metadata={"source": "clinical_guideline", "title": "Pneumothorax Management", "doc_id": "CG003"},
        ),
    ]


def _create_demo_index() -> FAISS:
    docs = _get_demo_documents()
    return FAISS.from_documents(docs, get_embeddings())


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    dense_docs: list[Document],
    sparse_docs: list[Document],
    k: int = 60,
) -> list[tuple[Document, float]]:
    """
    Combines dense (FAISS) and sparse (BM25) results using RRF.
    RRF score = Σ 1 / (k + rank_i)
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(dense_docs):
        key = doc.page_content[:100]  # Use content prefix as key
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
        doc_map[key] = doc

    for rank, doc in enumerate(sparse_docs):
        key = doc.page_content[:100]
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
        doc_map[key] = doc

    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [(doc_map[k], scores[k]) for k in sorted_keys]


# ── Reranking ─────────────────────────────────────────────────────────────────

def rerank_documents(
    query: str,
    candidates: list[tuple[Document, float]],
    top_k: int = TOP_K_RERANK,
) -> list[tuple[Document, float, float]]:
    """
    Cross-encoder reranking for precise relevance scoring.
    Returns (doc, rrf_score, rerank_score) tuples.
    """
    reranker = get_reranker()
    pairs = [(query, doc.page_content) for doc, _ in candidates]
    rerank_scores = reranker.predict(pairs)
    combined = [
        (candidates[i][0], candidates[i][1], float(rerank_scores[i]))
        for i in range(len(candidates))
    ]
    combined.sort(key=lambda x: x[2], reverse=True)
    return combined[:top_k]


# ── Query expansion ───────────────────────────────────────────────────────────

def build_enriched_query(state: ClinicalState) -> str:
    """
    Builds a comprehensive RAG query from vision + NLP findings.
    Falls back to symptom text if agents haven't run.
    """
    parts = []

    vision = state.get("vision_findings")
    if vision and vision.top_finding:
        parts.append(f"chest X-ray finding: {vision.top_finding}")
        secondary = [k for k, v in vision.pathologies.items() if v > 0.3 and k != vision.top_finding]
        if secondary:
            parts.append(f"additional findings: {', '.join(secondary[:3])}")

    nlp = state.get("nlp_findings")
    if nlp and nlp.symptoms:
        parts.append(f"patient symptoms: {', '.join(nlp.symptoms[:5])}")
        if nlp.medications:
            parts.append(f"current medications: {', '.join(nlp.medications[:3])}")

    # Use override query if supervisor set one
    if state.get("_rag_query_override"):
        parts.insert(0, state["_rag_query_override"])

    if not parts:
        parts.append(state.get("symptom_text", "chest X-ray analysis"))

    return ". ".join(parts)


# ── Main agent function ───────────────────────────────────────────────────────

def run_rag_agent(state: ClinicalState) -> ClinicalState:
    """
    LangGraph node: RAG Agent.
    Performs hybrid retrieval + reranking over the medical knowledge base.
    Writes RAGFindings back to state.
    """
    start = time.time()
    tracker = MLflowTracker()
    logger.info("[RAGAgent] Starting hybrid retrieval")

    try:
        # 1. Build enriched query
        query = build_enriched_query(state)
        logger.info(f"[RAGAgent] Query: {query[:120]}...")

        # 2. Dense retrieval (FAISS)
        vector_store = get_vector_store()
        dense_results = vector_store.similarity_search(query, k=TOP_K_RETRIEVE)

        # 3. Sparse retrieval (BM25)
        bm25 = get_bm25_retriever()
        sparse_results = bm25.invoke(query)

        # 4. Reciprocal rank fusion
        fused = reciprocal_rank_fusion(dense_results, sparse_results)

        # 5. Cross-encoder reranking
        reranked = rerank_documents(query, fused[:TOP_K_RETRIEVE], top_k=TOP_K_RERANK)

        # 6. Build RetrievedDocument objects
        retrieved_docs = []
        clinical_guidelines = []
        drug_interactions = []

        for doc, rrf_score, rerank_score in reranked:
            meta = doc.metadata
            rd = RetrievedDocument(
                doc_id=meta.get("doc_id", "unknown"),
                source=meta.get("source", "unknown"),
                title=meta.get("title", ""),
                content=doc.page_content,
                relevance_score=rerank_score,
                dense_score=rrf_score,
                sparse_score=rrf_score,
                url=meta.get("url", ""),
            )
            retrieved_docs.append(rd)
            if meta.get("source") == "clinical_guideline":
                clinical_guidelines.append(doc.page_content[:200])
            elif meta.get("source") == "drug_db":
                drug_interactions.append(doc.page_content[:200])

        # 7. Determine evidence level from sources
        sources = {d.source for d in retrieved_docs}
        evidence_level = "high" if "clinical_guideline" in sources else "moderate"

        elapsed = (time.time() - start) * 1000

        findings = RAGFindings(
            retrieved_docs=retrieved_docs,
            query_used=query,
            reranked_top_k=len(reranked),
            clinical_guidelines=clinical_guidelines,
            drug_interactions=drug_interactions,
            evidence_level=evidence_level,
            retrieval_time_ms=elapsed,
        )

        # 8. Log to MLflow
        tracker.log_metrics(state["mlflow_run_id"], {
            "rag_docs_retrieved": len(retrieved_docs),
            "rag_evidence_level": 1 if evidence_level == "high" else 0,
            "rag_retrieval_ms": elapsed,
            "rag_dense_results": len(dense_results),
            "rag_sparse_results": len(sparse_results),
        })

        logger.info(
            f"[RAGAgent] Done. {len(retrieved_docs)} docs retrieved, "
            f"evidence={evidence_level} in {elapsed:.0f}ms"
        )

        return {
            **state,
            "rag_findings": findings,
            "status": "rag_complete",
            "messages": state["messages"] + [{
                "role": "assistant",
                "agent": "rag_agent",
                "content": (
                    f"Retrieved {len(retrieved_docs)} relevant clinical documents. "
                    f"Top source: {retrieved_docs[0].title if retrieved_docs else 'none'}. "
                    f"Evidence level: {evidence_level}."
                ),
            }],
        }

    except Exception as e:
        logger.error(f"[RAGAgent] Error: {e}", exc_info=True)
        return {
            **state,
            "error_log": state["error_log"] + [f"RAGAgent: {str(e)}"],
        }
