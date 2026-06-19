"""
agents/critic_agent/agent.py

Critic Agent for HealthGuard AI.
- Evaluates consistency between Vision + NLP + RAG findings
- Detects potential hallucinations in agent outputs
- Computes overall confidence score
- Triggers reflection loop if confidence < threshold
- Circuit breaker: max 3 reflection loops
"""

from __future__ import annotations
import time
import logging
from dataclasses import asdict

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from core.state.clinical_state import (
    ClinicalState,
    CriticEvaluation,
    VisionFindings,
    NLPFindings,
    RAGFindings,
)
from monitoring.mlflow_tracking.tracker import MLflowTracker
from config.settings import settings

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.65
MAX_REFLECTIONS = 3

# ── Hallucination checks ──────────────────────────────────────────────────────

# Known co-occurring pathologies (if Vision says A, NLP/RAG should mention A or B)
PATHOLOGY_SYMPTOM_MAP = {
    "Pneumonia": ["fever", "cough", "shortness of breath", "chest pain", "dyspnea"],
    "Effusion": ["dyspnea", "chest pain", "shortness of breath"],
    "Pneumothorax": ["sudden chest pain", "shortness of breath", "dyspnea"],
    "Cardiomegaly": ["dyspnea", "fatigue", "edema", "shortness of breath"],
    "Atelectasis": ["cough", "shortness of breath", "dyspnea"],
    "Consolidation": ["fever", "cough", "productive cough"],
    "Edema": ["shortness of breath", "dyspnea", "orthopnea"],
}


def check_vision_nlp_agreement(
    vision: VisionFindings,
    nlp: NLPFindings,
) -> tuple[float, list[str]]:
    """
    Checks if NLP symptoms are consistent with Vision findings.
    Returns (agreement_score 0-1, list of discrepancy flags).
    """
    flags = []
    if not vision or not nlp:
        return 0.5, ["Missing vision or NLP findings"]

    top_finding = vision.top_finding
    expected_symptoms = PATHOLOGY_SYMPTOM_MAP.get(top_finding, [])
    found_symptoms = {s.lower() for s in nlp.symptoms}

    if not expected_symptoms:
        return 0.7, []  # No known mapping — neutral

    matches = sum(
        1 for es in expected_symptoms
        if any(es in fs or fs in es for fs in found_symptoms)
    )
    agreement = matches / len(expected_symptoms) if expected_symptoms else 0.5

    if agreement < 0.2 and nlp.symptoms:
        flags.append(
            f"Low symptom-finding agreement: {top_finding} expected "
            f"{expected_symptoms[:3]} but got {list(found_symptoms)[:3]}"
        )

    # Check if negated symptoms contradict vision
    negated = {s.lower() for s in nlp.negated_symptoms}
    for es in expected_symptoms:
        if es in negated:
            flags.append(f"Contradiction: '{es}' negated but vision shows {top_finding}")

    return float(agreement), flags


def check_rag_coverage(
    vision: VisionFindings,
    rag: RAGFindings,
) -> tuple[float, list[str]]:
    """
    Checks if retrieved documents cover the top vision finding.
    """
    flags = []
    if not vision or not rag or not rag.retrieved_docs:
        return 0.3, ["No RAG documents retrieved"]

    top_finding = vision.top_finding.lower()
    coverage_count = sum(
        1 for doc in rag.retrieved_docs
        if top_finding in doc.content.lower() or top_finding in doc.title.lower()
    )

    coverage_score = min(1.0, coverage_count / 2.0)  # 2+ covering docs = full score

    if coverage_count == 0:
        flags.append(f"No retrieved documents mention top finding: {top_finding}")
    elif rag.evidence_level == "low":
        flags.append("Evidence quality is low — consider additional verification")

    return coverage_score, flags


def llm_hallucination_check(
    vision: VisionFindings,
    nlp: NLPFindings,
    rag: RAGFindings,
) -> tuple[float, str]:
    """
    Uses Claude to critically evaluate the combined findings.
    Returns (llm_confidence 0-1, critique_text).
    """
    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=settings.GROQ_API_KEY,
            temperature=0.1,
            max_tokens=400,
        )

        vision_summary = (
            f"Top finding: {vision.top_finding} ({vision.confidence:.1%}). "
            f"Other findings: {list(vision.pathologies.keys())[:3]}"
            if vision else "No vision findings"
        )
        nlp_summary = (
            f"Symptoms: {', '.join(nlp.symptoms[:5])}. "
            f"Severity: {nlp.severity_score:.2f}"
            if nlp else "No NLP findings"
        )
        rag_summary = (
            f"Top document: {rag.retrieved_docs[0].title if rag.retrieved_docs else 'none'}. "
            f"Evidence: {rag.evidence_level}"
            if rag else "No RAG findings"
        )

        prompt = f"""You are a medical AI quality reviewer.

Vision findings: {vision_summary}
NLP findings: {nlp_summary}
RAG retrieval: {rag_summary}

Evaluate:
1. Are the vision and symptom findings clinically consistent?
2. Is the retrieved evidence relevant?
3. Are there any contradictions or hallucinations?

Respond with:
CONFIDENCE: <0.0-1.0>
CRITIQUE: <2-3 sentences>"""

        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content

        # Parse confidence
        confidence = 0.6  # default
        for line in content.split("\n"):
            if line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass

        critique = content.split("CRITIQUE:")[-1].strip() if "CRITIQUE:" in content else content[:200]
        return confidence, critique

    except Exception as e:
        logger.error(f"[CriticAgent] LLM check failed: {e}")
        return 0.6, "LLM evaluation unavailable"


# ── Main agent function ───────────────────────────────────────────────────────

def run_critic_agent(state: ClinicalState) -> ClinicalState:
    """
    LangGraph node: Critic Agent.
    Evaluates the quality of all prior agent outputs.
    Sets quality_gate_passed and reflection_needed in CriticEvaluation.
    """
    start = time.time()
    tracker = MLflowTracker()
    logger.info("[CriticAgent] Starting evaluation")

    vision = state.get("vision_findings")
    nlp = state.get("nlp_findings")
    rag = state.get("rag_findings")

    prior_eval = state.get("critic_evaluation")
    reflection_count = prior_eval.reflection_count + 1 if prior_eval else 0

    all_flags = []

    # 1. Vision–NLP agreement
    agreement_score, agreement_flags = check_vision_nlp_agreement(vision, nlp)
    all_flags.extend(agreement_flags)

    # 2. RAG coverage
    rag_coverage, rag_flags = check_rag_coverage(vision, rag)
    all_flags.extend(rag_flags)

    # 3. LLM hallucination check (only on first pass or if there were flags)
    if reflection_count == 0 or all_flags:
        llm_confidence, critique = llm_hallucination_check(vision, nlp, rag)
    else:
        llm_confidence, critique = 0.8, "Passed LLM check in prior iteration"

    # 4. Vision confidence
    vision_conf = vision.confidence if vision else 0.0

    # 5. Composite confidence
    weights = {"vision": 0.35, "agreement": 0.25, "rag": 0.20, "llm": 0.20}
    overall = (
        weights["vision"] * vision_conf
        + weights["agreement"] * agreement_score
        + weights["rag"] * rag_coverage
        + weights["llm"] * llm_confidence
    )

    # 6. Quality gate
    quality_passed = (
        overall >= CONFIDENCE_THRESHOLD
        and len([f for f in all_flags if "contradiction" in f.lower()]) == 0
    )
    reflection_needed = not quality_passed and reflection_count < MAX_REFLECTIONS

    evaluation = CriticEvaluation(
        overall_confidence=overall,
        vision_nlp_agreement=agreement_score,
        hallucination_flags=all_flags,
        missing_evidence_flags=rag_flags,
        reflection_needed=reflection_needed,
        reflection_count=reflection_count,
        quality_gate_passed=quality_passed or reflection_count >= MAX_REFLECTIONS,
        critique_text=critique,
    )

    elapsed = (time.time() - start) * 1000

    tracker.log_metrics(state["mlflow_run_id"], {
        "critic_overall_confidence": overall,
        "critic_agreement_score": agreement_score,
        "critic_rag_coverage": rag_coverage,
        "critic_llm_confidence": llm_confidence,
        "critic_flags_count": len(all_flags),
        "critic_reflection_count": reflection_count,
        "critic_quality_passed": int(quality_passed),
        "critic_eval_ms": elapsed,
    })

    status_msg = "✅ Quality gate PASSED" if quality_passed else f"🔄 Reflection needed (pass {reflection_count}/{MAX_REFLECTIONS})"
    logger.info(
        f"[CriticAgent] confidence={overall:.2f}, {status_msg} in {elapsed:.0f}ms"
    )

    return {
        **state,
        "critic_evaluation": evaluation,
        "status": "critic_review",
        "messages": state["messages"] + [{
            "role": "assistant",
            "agent": "critic_agent",
            "content": (
                f"Quality evaluation complete. Confidence: {overall:.1%}. "
                f"{status_msg}. Flags: {len(all_flags)}."
            ),
        }],
    }
