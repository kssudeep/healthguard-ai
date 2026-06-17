"""
core/state/clinical_state.py

Shared LangGraph state schema for the HealthGuard AI multi-agent system.
This is the single source of truth for all inter-agent communication.
Each agent reads from and writes to this typed state.
"""

from __future__ import annotations
from typing import Annotated, Any, TypedDict, Literal, Optional
from dataclasses import dataclass, field
import operator


# ── Enums & Literals ────────────────────────────────────────────────────────

AgentName = Literal[
    "supervisor",
    "vision_agent",
    "nlp_agent",
    "rag_agent",
    "critic_agent",
    "synthesizer",
]

WorkflowStatus = Literal[
    "pending",
    "vision_complete",
    "nlp_complete",
    "rag_complete",
    "critic_review",
    "reflection_loop",
    "synthesizing",
    "complete",
    "failed",
]


# ── Sub-state dataclasses ────────────────────────────────────────────────────

@dataclass
class VisionFindings:
    """Outputs from the Vision Agent (DenseNet-121 + GradCAM)."""
    pathologies: dict[str, float] = field(default_factory=dict)
    # e.g. {"Pneumonia": 0.87, "Effusion": 0.43, "Cardiomegaly": 0.12}
    top_finding: str = ""
    confidence: float = 0.0
    gradcam_heatmap_path: str = ""       # saved heatmap PNG path
    image_quality_score: float = 1.0    # 0-1, flags low-quality scans
    dicom_metadata: dict[str, Any] = field(default_factory=dict)
    model_version: str = "densenet121-v2"
    inference_time_ms: float = 0.0


@dataclass
class ClinicalEntity:
    """A single NER-extracted clinical entity."""
    text: str
    label: str        # SYMPTOM | MEDICATION | BODY_PART | DIAGNOSIS | PROCEDURE
    start: int
    end: int
    confidence: float
    negated: bool = False     # "no chest pain" → negated=True


@dataclass
class NLPFindings:
    """Outputs from the NLP Agent (BioBERT NER + SpaCy)."""
    entities: list[ClinicalEntity] = field(default_factory=list)
    symptoms: list[str] = field(default_factory=list)
    medications: list[str] = field(default_factory=list)
    body_parts: list[str] = field(default_factory=list)
    severity_score: float = 0.0   # 0 (mild) – 1 (critical)
    urgency_flag: bool = False
    negated_symptoms: list[str] = field(default_factory=list)
    raw_text: str = ""
    processing_time_ms: float = 0.0


@dataclass
class RetrievedDocument:
    """A single document retrieved from the knowledge base."""
    doc_id: str
    source: str       # "pubmed" | "clinical_guideline" | "drug_db"
    title: str
    content: str
    relevance_score: float       # reranker score
    dense_score: float           # FAISS cosine
    sparse_score: float          # BM25
    url: str = ""


@dataclass
class RAGFindings:
    """Outputs from the RAG Agent (Hybrid FAISS + BM25)."""
    retrieved_docs: list[RetrievedDocument] = field(default_factory=list)
    query_used: str = ""
    reranked_top_k: int = 5
    clinical_guidelines: list[str] = field(default_factory=list)
    drug_interactions: list[str] = field(default_factory=list)
    evidence_level: str = ""    # "high" | "moderate" | "low" | "expert_opinion"
    retrieval_time_ms: float = 0.0


@dataclass
class CriticEvaluation:
    """Outputs from the Critic Agent (confidence + hallucination detection)."""
    overall_confidence: float = 0.0     # 0–1, if < 0.6 → reflection loop
    vision_nlp_agreement: float = 0.0   # do findings align?
    hallucination_flags: list[str] = field(default_factory=list)
    missing_evidence_flags: list[str] = field(default_factory=list)
    reflection_needed: bool = False
    reflection_count: int = 0           # circuit breaker: max 3
    quality_gate_passed: bool = False
    critique_text: str = ""


@dataclass
class ClinicalReport:
    """Final structured report output by the Synthesizer."""
    patient_summary: str = ""
    primary_diagnosis: str = ""
    differential_diagnoses: list[str] = field(default_factory=list)
    supporting_evidence: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    urgency_level: str = ""      # "routine" | "urgent" | "emergency"
    confidence_score: float = 0.0
    disclaimer: str = (
        "This report is AI-generated for research purposes only. "
        "Always consult a licensed medical professional."
    )
    report_id: str = ""
    generated_at: str = ""
    llm_model: str = ""
    total_pipeline_time_ms: float = 0.0


# ── Main LangGraph State ─────────────────────────────────────────────────────

class ClinicalState(TypedDict):
    """
    The complete shared state passed between all LangGraph nodes.

    LangGraph uses TypedDict with Annotated reducers.
    operator.add merges lists across parallel branches.
    """

    # ── Inputs ────────────────────────────────────────────────────────
    session_id: str
    image_path: str           # local path or Azure Blob URL
    symptom_text: str
    patient_age: Optional[int]
    patient_sex: Optional[str]  # "M" | "F" | "unknown"

    # ── Routing & Control ────────────────────────────────────────────
    status: WorkflowStatus
    current_agent: AgentName
    next_agent: AgentName
    error_log: Annotated[list[str], operator.add]   # accumulated errors
    iteration_count: int                             # circuit breaker

    # ── Agent Outputs ─────────────────────────────────────────────────
    vision_findings: Optional[VisionFindings]
    nlp_findings: Optional[NLPFindings]
    rag_findings: Optional[RAGFindings]
    critic_evaluation: Optional[CriticEvaluation]
    final_report: Optional[ClinicalReport]

    # ── Message history (for LLM context) ────────────────────────────
    messages: Annotated[list[dict], operator.add]

    # ── Metadata ──────────────────────────────────────────────────────
    pipeline_start_time: float
    mlflow_run_id: str
    langsmith_trace_id: str
