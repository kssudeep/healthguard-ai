from __future__ import annotations
import time, uuid, logging, json
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from core.state.clinical_state import ClinicalState, ClinicalReport
from monitoring.mlflow_tracking.tracker import MLflowTracker
from config.settings import settings

logger = logging.getLogger(__name__)

class ClinicalReportSchema(BaseModel):
    patient_summary: str = Field(description="2-3 sentence patient overview")
    primary_diagnosis: str = Field(description="Most likely diagnosis")
    differential_diagnoses: list[str] = Field(description="2-4 alternative diagnoses")
    supporting_evidence: list[str] = Field(description="Evidence supporting primary diagnosis")
    recommended_actions: list[str] = Field(description="Clinical recommendations")
    urgency_level: str = Field(description="routine | urgent | emergency")
    confidence_explanation: str = Field(description="Why this confidence level was assigned")

SYSTEM_PROMPT = """You are HealthGuard AI, an advanced clinical decision support system.
Output ONLY valid JSON matching the specified schema. No prose outside the JSON."""

def build_synthesis_prompt(state: ClinicalState) -> str:
    vision = state.get("vision_findings")
    nlp = state.get("nlp_findings")
    rag = state.get("rag_findings")
    critic = state.get("critic_evaluation")

    vision_section = "VISION FINDINGS:\n"
    if vision:
        pathologies_str = "\n".join(
            f"  - {k}: {v:.1%}" for k, v in sorted(
                vision.pathologies.items(), key=lambda x: x[1], reverse=True)[:6])
        vision_section += (
            f"  Top finding: {vision.top_finding} (confidence: {vision.confidence:.1%})\n"
            f"  Pathologies:\n{pathologies_str}\n"
            f"  Image quality: {vision.image_quality_score:.1%}\n")
    else:
        vision_section += "  No vision findings available\n"

    nlp_section = "NLP FINDINGS:\n"
    if nlp:
        nlp_section += (
            f"  Symptoms: {', '.join(nlp.symptoms[:8]) or 'none identified'}\n"
            f"  Severity score: {nlp.severity_score:.2f}/1.0\n"
            f"  Urgency flag: {'YES' if nlp.urgency_flag else 'No'}\n"
            f"  Patient age: {state.get('patient_age', 'unknown')}, Sex: {state.get('patient_sex', 'unknown')}\n")
    else:
        nlp_section += "  No NLP findings available\n"

    rag_section = "CLINICAL EVIDENCE:\n"
    if rag and rag.retrieved_docs:
        for i, doc in enumerate(rag.retrieved_docs[:3], 1):
            rag_section += f"  [{i}] {doc.title}: {doc.content[:200]}...\n"
    else:
        rag_section += "  No clinical evidence retrieved\n"

    critic_section = "QUALITY ASSESSMENT:\n"
    if critic:
        critic_section += (
            f"  Overall confidence: {critic.overall_confidence:.1%}\n"
            f"  Flags: {'; '.join(critic.hallucination_flags) if critic.hallucination_flags else 'none'}\n")

    schema_instruction = """OUTPUT JSON:
{
  "patient_summary": "string",
  "primary_diagnosis": "string",
  "differential_diagnoses": ["string"],
  "supporting_evidence": ["string"],
  "recommended_actions": ["string"],
  "urgency_level": "routine|urgent|emergency",
  "confidence_explanation": "string"
}"""

    return "\n\n".join([vision_section, nlp_section, rag_section, critic_section, schema_instruction])


def run_synthesizer(state: ClinicalState) -> ClinicalState:
    start = time.time()
    tracker = MLflowTracker()
    logger.info("[Synthesizer] Generating clinical report")

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.15,
            max_output_tokens=1500,
        )

        prompt = build_synthesis_prompt(state)
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        raw_content = response.content

        try:
            if isinstance(raw_content, str):
                clean = raw_content.strip()
            elif isinstance(raw_content, list):
                clean = "".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in raw_content
                ).strip()
            else:
                clean = str(raw_content).strip()

            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            report_data = json.loads(clean)

        except json.JSONDecodeError as e:
            logger.error(f"[Synthesizer] JSON parse error: {e}. Raw: {str(raw_content)[:200]}")
            report_data = {
                "patient_summary": "Analysis completed with parsing issues.",
                "primary_diagnosis": state["vision_findings"].top_finding if state.get("vision_findings") else "Undetermined",
                "differential_diagnoses": [],
                "supporting_evidence": ["Vision analysis completed"],
                "recommended_actions": ["Consult a licensed radiologist"],
                "urgency_level": "urgent" if state.get("nlp_findings") and state["nlp_findings"].urgency_flag else "routine",
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
            llm_model="gemini-3.6-flash",
            total_pipeline_time_ms=(time.time() - state.get("pipeline_start_time", time.time())) * 1000,
        )

        tracker.log_metrics(state["mlflow_run_id"], {
            "synthesis_time_ms": elapsed,
            "report_confidence": report.confidence_score,
            "differentials_count": len(report.differential_diagnoses),
            "recommendations_count": len(report.recommended_actions),
        })
        tracker.log_param(state["mlflow_run_id"], "urgency_level", report.urgency_level)
        tracker.log_param(state["mlflow_run_id"], "primary_diagnosis", report.primary_diagnosis)
        tracker.end_run(state["mlflow_run_id"])

        logger.info(f"[Synthesizer] Report {report.report_id}: {report.primary_diagnosis}, {report.urgency_level}")

        return {
            **state,
            "final_report": report,
            "status": "complete",
            "messages": state["messages"] + [{
                "role": "assistant",
                "agent": "synthesizer",
                "content": f"Report {report.report_id}: {report.primary_diagnosis}. Urgency: {report.urgency_level}.",
            }],
        }

    except Exception as e:
        logger.error(f"[Synthesizer] Error: {e}", exc_info=True)
        return {
            **state,
            "error_log": state["error_log"] + [f"Synthesizer: {str(e)}"],
            "status": "failed",
        }
