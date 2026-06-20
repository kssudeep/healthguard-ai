from pydantic import BaseModel
from typing import Optional


class AnalysisRequest(BaseModel):
    symptoms: str
    patient_age: Optional[int] = None
    patient_sex: Optional[str] = "unknown"


class AnalysisResponse(BaseModel):
    job_id: str
    status: str
    message: str
    estimated_time_seconds: int = 25


class ReportStatus(BaseModel):
    job_id: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None


class ClinicalReportOut(BaseModel):
    report_id: str
    patient_summary: str
    primary_diagnosis: str
    differential_diagnoses: list[str]
    supporting_evidence: list[str]
    recommended_actions: list[str]
    urgency_level: str
    confidence_score: float
    disclaimer: str
    generated_at: str
    total_pipeline_time_ms: float
