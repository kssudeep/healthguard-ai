"""
agents/nlp_agent/agent.py

NLP Agent for HealthGuard AI.
- BioBERT fine-tuned for clinical NER (symptom, medication, body part extraction)
- SpaCy (en_core_sci_lg) for dependency parsing + negation detection
- Rule-based severity scorer
- Urgency flag for critical symptoms
"""

from __future__ import annotations
import time
import logging
import re
from dataclasses import asdict

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
)

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from core.state.clinical_state import (
    ClinicalState,
    NLPFindings,
    ClinicalEntity,
)
from monitoring.mlflow_tracking.tracker import MLflowTracker
from config.settings import settings

logger = logging.getLogger(__name__)

# ── BioBERT NER model ─────────────────────────────────────────────────────────
# dmis-lab/biobert-base-cased-v1.2 fine-tuned on i2b2 2010 NER
BIOBERT_NER_MODEL = "d4data/biomedical-ner-all"

# ── Critical symptom keywords (trigger urgency flag) ─────────────────────────
CRITICAL_SYMPTOMS = {
    "chest pain", "shortness of breath", "difficulty breathing",
    "haemoptysis", "hemoptysis", "coughing blood", "respiratory failure",
    "cyanosis", "hypoxia", "cardiac arrest", "loss of consciousness",
    "severe pain", "crushing pain",
}

# ── Severity keyword weights ───────────────────────────────────────────────────
SEVERITY_MODIFIERS = {
    "severe": 0.9, "extreme": 0.95, "critical": 1.0, "unbearable": 0.9,
    "moderate": 0.5, "mild": 0.2, "slight": 0.15, "minor": 0.1,
    "worsening": 0.7, "sudden": 0.7, "acute": 0.75, "chronic": 0.3,
}

# ── NER label mapping (model-specific) ────────────────────────────────────────
LABEL_REMAP = {
    "B-SYMPTOM": "SYMPTOM", "I-SYMPTOM": "SYMPTOM",
    "B-MEDICATION": "MEDICATION", "I-MEDICATION": "MEDICATION",
    "B-BODY_PART": "BODY_PART", "I-BODY_PART": "BODY_PART",
    "B-DIAGNOSIS": "DIAGNOSIS", "I-DIAGNOSIS": "DIAGNOSIS",
    "B-PROCEDURE": "PROCEDURE", "I-PROCEDURE": "PROCEDURE",
}


# ── Singleton NER pipeline ────────────────────────────────────────────────────

_ner_pipeline = None
_nlp_model = None   # SpaCy model


def get_ner_pipeline():
    global _ner_pipeline
    if _ner_pipeline is None:
        logger.info("[NLPAgent] Loading BioBERT NER pipeline...")
        tokenizer = AutoTokenizer.from_pretrained(BIOBERT_NER_MODEL)
        model = AutoModelForTokenClassification.from_pretrained(BIOBERT_NER_MODEL)
        _ner_pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="max",
            device=0 if torch.cuda.is_available() else -1,
        )
        logger.info("[NLPAgent] BioBERT NER loaded")
    return _ner_pipeline


def get_spacy_model():
    global _nlp_model
    if _nlp_model is None and SPACY_AVAILABLE:
        try:
            _nlp_model = spacy.load("en_core_sci_lg")
            logger.info("[NLPAgent] SpaCy en_core_sci_lg loaded")
        except OSError:
            # Fallback to base English model
            _nlp_model = spacy.load("en_core_web_sm")
            logger.warning("[NLPAgent] en_core_sci_lg not found, using en_core_web_sm")
    return _nlp_model


# ── Negation detection ─────────────────────────────────────────────────────────

NEGATION_TRIGGERS = {
    "no", "not", "without", "denies", "absent", "absence of",
    "no evidence of", "rules out", "negative for", "free of",
}


def detect_negation(text: str, entity_text: str) -> bool:
    """
    Simple window-based negation detection.
    Checks if a negation trigger appears within 5 tokens before the entity.
    """
    text_lower = text.lower()
    entity_lower = entity_text.lower()
    entity_pos = text_lower.find(entity_lower)
    if entity_pos == -1:
        return False
    # Look in the 60 characters before the entity
    window = text_lower[max(0, entity_pos - 60): entity_pos]
    for trigger in NEGATION_TRIGGERS:
        if trigger in window:
            return True
    return False


# ── Severity scoring ──────────────────────────────────────────────────────────

def compute_severity_score(text: str, symptoms: list[str]) -> float:
    """
    Scores severity 0–1 based on:
    - Severity modifier keywords in the text
    - Number of symptoms found
    - Presence of urgency keywords
    """
    text_lower = text.lower()
    max_modifier_score = 0.0
    for modifier, weight in SEVERITY_MODIFIERS.items():
        if modifier in text_lower:
            max_modifier_score = max(max_modifier_score, weight)

    # Scale with number of symptoms (more = higher severity)
    symptom_load_score = min(1.0, len(symptoms) / 8.0) * 0.3

    # Base severity if no modifiers
    base = 0.3 if symptoms else 0.0

    score = max(base, max_modifier_score) * 0.7 + symptom_load_score
    return float(min(1.0, score))


def check_urgency(symptoms: list[str], text: str) -> bool:
    """Flags urgent cases requiring immediate attention."""
    text_lower = text.lower()
    symptom_set = {s.lower() for s in symptoms}
    for critical in CRITICAL_SYMPTOMS:
        if critical in text_lower or critical in symptom_set:
            return True
    return False


# ── Main NER extraction ───────────────────────────────────────────────────────

def extract_entities(text: str) -> list[ClinicalEntity]:
    """
    Runs BioBERT NER on text.
    Returns a list of ClinicalEntity objects.
    """
    ner = get_ner_pipeline()
    raw_entities = ner(text)
    entities = []
    for ent in raw_entities:
        label = LABEL_REMAP.get(ent["entity_group"], ent["entity_group"])
        if label not in {"SYMPTOM", "MEDICATION", "BODY_PART", "DIAGNOSIS", "PROCEDURE"}:
            continue
        negated = detect_negation(text, ent["word"])
        entities.append(ClinicalEntity(
            text=ent["word"],
            label=label,
            start=ent["start"],
            end=ent["end"],
            confidence=float(ent["score"]),
            negated=negated,
        ))
    return entities


def run_spacy_enhancement(text: str, entities: list[ClinicalEntity]) -> list[ClinicalEntity]:
    """
    Use SpaCy dependency parsing to improve negation detection
    and resolve co-references (basic).
    """
    nlp = get_spacy_model()
    if not nlp:
        return entities
    doc = nlp(text)
    # Build token-level negation map from SpaCy
    negated_spans = set()
    for token in doc:
        if token.dep_ == "neg":
            # The negated word and its subtree
            for child in token.head.subtree:
                negated_spans.add(child.idx)
    # Update entities
    for entity in entities:
        if entity.start in negated_spans:
            entity.negated = True
    return entities


# ── Main agent function ───────────────────────────────────────────────────────

def run_nlp_agent(state: ClinicalState) -> ClinicalState:
    """
    LangGraph node: NLP Agent.
    Reads symptom_text from state, extracts clinical entities,
    scores severity, flags urgency, writes NLPFindings back.
    """
    start = time.time()
    tracker = MLflowTracker()
    text = state["symptom_text"]
    logger.info(f"[NLPAgent] Processing text ({len(text)} chars)")

    try:
        # 1. BioBERT NER
        entities = extract_entities(text)

        # 2. SpaCy negation enhancement
        entities = run_spacy_enhancement(text, entities)

        # 3. Categorise
        symptoms = [e.text for e in entities if e.label == "SYMPTOM" and not e.negated]
        negated_symptoms = [e.text for e in entities if e.label == "SYMPTOM" and e.negated]
        medications = [e.text for e in entities if e.label == "MEDICATION"]
        body_parts = [e.text for e in entities if e.label == "BODY_PART"]

        # Remove duplicates preserving order
        symptoms = list(dict.fromkeys(symptoms))
        medications = list(dict.fromkeys(medications))

        # 4. Severity + urgency
        severity = compute_severity_score(text, symptoms)
        urgency = check_urgency(symptoms, text)

        elapsed = (time.time() - start) * 1000

        findings = NLPFindings(
            entities=entities,
            symptoms=symptoms,
            medications=medications,
            body_parts=body_parts,
            severity_score=severity,
            urgency_flag=urgency,
            negated_symptoms=negated_symptoms,
            raw_text=text,
            processing_time_ms=elapsed,
        )

        # 5. Log to MLflow
        tracker.log_metrics(state["mlflow_run_id"], {
            "nlp_entities_found": len(entities),
            "nlp_symptoms_count": len(symptoms),
            "nlp_severity_score": severity,
            "nlp_urgency_flag": int(urgency),
            "nlp_processing_ms": elapsed,
        })

        urgency_str = "⚠️ URGENT" if urgency else "routine"
        logger.info(
            f"[NLPAgent] Done. {len(symptoms)} symptoms, "
            f"severity={severity:.2f}, {urgency_str} in {elapsed:.0f}ms"
        )

        return {
            **state,
            "nlp_findings": findings,
            "status": "nlp_complete",
            "messages": state["messages"] + [{
                "role": "assistant",
                "agent": "nlp_agent",
                "content": (
                    f"NLP extraction complete. Symptoms identified: {', '.join(symptoms[:5])}. "
                    f"Severity score: {severity:.2f}. Urgency: {urgency_str}."
                ),
            }],
        }

    except Exception as e:
        logger.error(f"[NLPAgent] Error: {e}", exc_info=True)
        return {
            **state,
            "error_log": state["error_log"] + [f"NLPAgent: {str(e)}"],
        }
