"""
agents/orchestrator/synthesizer.py

Synthesizer node for HealthGuard AI.
Uses Gemini 2.0 Flash (FREE — Google AI Studio) to generate a structured
clinical report from the combined Vision + NLP + RAG + Critic findings.
Outputs a validated ClinicalReport dataclass.
"""

from __future__ import annotations
import time
import uuid
import logging
import json
from datetime import datetime

from langchain_groq import ChatGroq
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from core.state.clinical_state import ClinicalState, ClinicalReport
from monitoring.mlflow_tracking.tracker import MLflowTracker
from config.settings import settings

logger = logging.getLogger(__name__)


# ── Pydantic schema for structured LLM output ────────────────────────────────

class ClinicalReportSchema(BaseModel):
    patient_summary: str = Field(description="2-3 sentence patient overview")
    primary_diagnosis: str = Field(description="Most likely diagnosis")
    differential_diagnoses: list[str] = Field(description="2-4 alternative diagnoses")
    supporting_evidence: list[str] = Field(description="Evidence supporting primary diagnosis")
    recommended_actions: list[str] = Field(description="Clinical recommendations")
    urgency_level: str = Field(description="routine | urgent | emergency")
    confidence_explanation: str = Field(description="Why this confidence level was assigned")


SYSTEM_PROMPT = """You are HealthGuard AI, an advanced clinical decision support system.
You synthesise findings from medical imaging analysis, symptom extraction, and clinical literature retrieval.

CRITICAL RULES:
1. Base ALL conclusions strictly on the provided findings — do not invent information
2. Explicitly state uncertainty when evidence is limited
3. Always recommend professional medical consultation
4. Flag emergencies immediately and prominently
5. Use precise clinical language
6. If findings are contradictory, acknowledge the contradiction rather than ignoring it

Output ONLY valid JSON matching the specified schema. No prose outside the JSON."""


def build_synthesis_prompt(state: ClinicalState) -> str:
    """Constructs the full synthesis prompt from all agent findings."""
    vision = state.get("vision_findings")
    nlp = state.get("nlp_findings")
    rag = state.get("rag_findings")
    critic = state.get("critic_evaluation")

    vision_section = "VISION FINDINGS (chest X-ray analysis):\n"
    if vision:
        pathologies_str = "\n".join(
            f"  - {k}: {v:.1%}" for k, v in sorted(
                vision.pathologies.items(), key=lambda x: x[1], reverse=True
            )[:6]
        )
        vision_section += (
            f"  Top finding: {vision.top_finding} (confidence: {vision.confidence:.1%})\n"
            f"  All pathologies:\n{pathologies_str}\n"
            f"  Image quality: {vision.image_quality_score:.1%}\n"
        )
    else:
        vision_section += "  No vision findings available\n"

    nlp_section = "NLP FINDINGS (symptom extraction):\n"
    if nlp:
        nlp_section += (
            f"  Symptoms: {', '.join(nlp.symptoms[:8]) or 'none identified'}\n"
            f"  Negated symptoms: {', '.join(nlp.negated_symptoms[:4]) or 'none'}\n"
            f"  Medications: {', '.join(nlp.medications[:5]) or 'none'}\n"
            f"  Severity score: {nlp.severity_score:.2f}/1.0\n"
            f"  Urgency flag: {'⚠️ YES' if nlp.urgency_flag else 'No'}\n"
            f"  Patient age: {state.get('patient_age', 'unknown')}, "
            f"Sex: {state.get('patient_sex', 'unknown')}\n"
        )
    else:
        nlp_section += "  No NLP findings available\n"

    rag_section = "CLINICAL EVIDENCE (retrieved literature):\n"
    if rag and rag.retrieved_docs:
        for i, doc in enumerate(rag.retrieved_docs[:3], 1):
            rag_section += (
                f"  [{i}] {doc.title} (source: {doc.source}, "
                f"relevance: {doc.relevance_score:.2f})\n"
                f"      {doc.content[:300]}...\n\n"
            )
        rag_section += f"  Evidence level: {rag.evidence_level}\n"
    else:
        rag_section += "  No clinical evidence retrieved\n"

    critic_section = "QUALITY ASSESSMENT:\n"
    if critic:
        critic_section += (
            f"  Overall confidence: {critic.overall_confidence:.1%}\n"
            f"  Vision-symptom agreement: {critic.vision_nlp_agreement:.1%}\n"
            f"  Quality flags: {'; '.join(critic.hallucination_flags) if critic.hallucination_flags else 'none'}\n"
            f"  Reflection loops completed: {critic.reflection_count}\n"
            f"  Critic assessment: {critic.critique_text[:200]}\n"
        )

    schema_instruction = """
OUTPUT JSON SCHEMA (respond with ONLY this JSON, no other text):
{
  "patient_summary": "string",
  "primary_diagnosis": "string",
  "differential_diagnoses": ["string", "string"],
  "supporting_evidence": ["string", "string"],
  "recommended_actions": ["string", "string"],
  "urgency_level": "routine|urgent|emergency",
  "confidence_explanation": "string"
}"""

    return "\n\n".join([
        vision_section,
        nlp_section,
        rag_section,
        critic_section,
        schema_instruction,
    ])


def run_synthesizer(state: ClinicalState) -> ClinicalState:
    """
    LangGraph node: Synthesizer.
    Generates the final structured clinical report using Gemini 2.0 Flash (free).
    """
    start = time.time()
    tracker = MLflowTracker()
    logger.info("[Synthesizer] Generating clinical report")

    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=settings.GROQ_API_KEY,
            temperature=0.15,
            max_tokens=1500,
        )

        prompt = build_synthesis_prompt(state)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = llm.invoke(messages)
        raw_content = response.content

        # Parse JSON response
        try:
            # Strip markdown code fences if present
            clean = raw_content.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            report_data = json.loads(clean)
        except json.JSONDecodeError as e:
            logger.error(f"[Synthesizer] JSON parse error: {e}. Raw: {raw_content[:200]}")
            # Fallback report
            report_data = {
                "patient_summary": "Analysis completed with parsing issues.",
                "primary_diagnosis": state["vision_findings"].top_finding
                    if state.get("vision_findings") else "Undetermined",
                "differential_diagnoses": [],
                "supporting_evidence": ["Vision analysis completed"],
                "recommended_actions": ["Consult a licensed radiologist"],
                "urgency_level": "urgent" if state.get("nlp_findings") and
                    state["nlp_findings"].urgency_flag else "routine",
                "confidence_explanation": "Parsing error — manual review recommended",
            }

        elapsed = (time.time() - start) * 1000
        critic = state.get("critic_evaluation")

        report = ClinicalReport(
            patient_summary=report_data.get("patient_summary", ""),
            primary_diagnosis=report_data.get("primary_diagnosis", ""),
            differential_diagnoses=report_data.get("differential_diagnoses", []),
            supporting_evidence=report_data.get("supporting_evidence", []),
            recommended_actions=report_data.get("recommended_actions", []),
            urgency_level=report_data.get("urgency_level", "routine"),
            confidence_score=critic.overall_confidence if critic else 0.6,
            report_id=str(uuid.uuid4())[:8].upper(),
            generated_at=datetime.utcnow().isoformat() + "Z",
            llm_model="gemini-2.0-flash",
            total_pipeline_time_ms=(time.time() - state.get("pipeline_start_time", time.time())) * 1000,
        )

        # Log synthesis metrics to MLflow
        tracker.log_metrics(state["mlflow_run_id"], {
            "synthesis_time_ms": elapsed,
            "report_confidence": report.confidence_score,
            "differentials_count": len(report.differential_diagnoses),
            "recommendations_count": len(report.recommended_actions),
        })
        tracker.log_param(state["mlflow_run_id"], "urgency_level", report.urgency_level)
        tracker.log_param(state["mlflow_run_id"], "primary_diagnosis", report.primary_diagnosis)
        tracker.end_run(state["mlflow_run_id"])

        logger.info(
            f"[Synthesizer] Report {report.report_id} generated. "
            f"Diagnosis: {report.primary_diagnosis}, "
            f"Urgency: {report.urgency_level}, "
            f"Confidence: {report.confidence_score:.1%} in {elapsed:.0f}ms"
        )

        return {
            **state,
            "final_report": report,
            "status": "complete",
            "messages": state["messages"] + [{
                "role": "assistant",
                "agent": "synthesizer",
                "content": (
                    f"Clinical report {report.report_id} generated. "
                    f"Primary: {report.primary_diagnosis}. "
                    f"Urgency: {report.urgency_level}."
                ),
            }],
        }

    except Exception as e:
        logger.error(f"[Synthesizer] Error: {e}", exc_info=True)
        return {
            **state,
            "error_log": state["error_log"] + [f"Synthesizer: {str(e)}"],
            "status": "failed",
        }
