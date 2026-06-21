"""
tests/unit/test_agents.py

Unit tests for all HealthGuard AI agents.
Uses mocking to avoid requiring actual model weights or API keys.
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
from PIL import Image
import io

# ── Vision Agent Tests ────────────────────────────────────────────────────────

class TestVisionAgent:
    def test_assess_image_quality_high(self):
        from agents.vision_agent.agent import assess_image_quality
        # Create a well-exposed grayscale image
        arr = np.random.randint(80, 180, (512, 512, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        score = assess_image_quality(img)
        assert 0.0 <= score <= 1.0

    def test_assess_image_quality_low_contrast(self):
        from agents.vision_agent.agent import assess_image_quality
        # All-white image = zero contrast
        arr = np.full((256, 256, 3), 255, dtype=np.uint8)
        img = Image.fromarray(arr)
        score = assess_image_quality(img)
        assert score < 0.5

    def test_pathology_classes_count(self):
        from agents.vision_agent.agent import PATHOLOGY_CLASSES
        assert len(PATHOLOGY_CLASSES) == 14

    @patch("agents.vision_agent.agent.get_model")
    def test_run_vision_agent_success(self, mock_get_model):
        """Tests the full agent node with mocked model."""
        from agents.vision_agent.agent import run_vision_agent, PATHOLOGY_CLASSES

        # Mock model returning probabilities
        mock_model = MagicMock()
        mock_preds = torch.tensor([0.8, 0.1, 0.3, 0.05] + [0.02] * 10)
        mock_model.return_value = mock_preds.unsqueeze(0)
        mock_device = torch.device("cpu")
        mock_get_model.return_value = (mock_model, mock_device)

        # Create test image file
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f.name)
            image_path = f.name

        state = {
            "session_id": "test01",
            "image_path": image_path,
            "symptom_text": "test symptoms",
            "status": "pending",
            "error_log": [],
            "messages": [],
            "mlflow_run_id": "mock_run",
            "pipeline_start_time": 0.0,
        }

        with patch("agents.vision_agent.agent.MLflowTracker") as mock_tracker:
            mock_tracker.return_value.log_metrics = MagicMock()
            result = run_vision_agent(state)

        os.unlink(image_path)
        assert result["vision_findings"] is not None
        assert result["vision_findings"].top_finding in PATHOLOGY_CLASSES
        assert 0.0 <= result["vision_findings"].confidence <= 1.0
        assert result["status"] == "vision_complete"


# ── NLP Agent Tests ───────────────────────────────────────────────────────────

class TestNLPAgent:
    def test_detect_negation_positive(self):
        from agents.nlp_agent.agent import detect_negation
        text = "Patient denies any chest pain or shortness of breath"
        assert detect_negation(text, "chest pain") is True

    def test_detect_negation_negative(self):
        from agents.nlp_agent.agent import detect_negation
        text = "Patient presents with severe chest pain and fever"
        assert detect_negation(text, "chest pain") is False

    def test_compute_severity_severe(self):
        from agents.nlp_agent.agent import compute_severity_score
        text = "Patient has severe crushing chest pain with extreme shortness of breath"
        symptoms = ["chest pain", "shortness of breath", "dyspnea"]
        score = compute_severity_score(text, symptoms)
        assert score >= 0.6

    def test_compute_severity_mild(self):
        from agents.nlp_agent.agent import compute_severity_score
        text = "Patient has mild cough"
        symptoms = ["cough"]
        score = compute_severity_score(text, symptoms)
        assert score < 0.5

    def test_check_urgency_critical(self):
        from agents.nlp_agent.agent import check_urgency
        symptoms = ["chest pain", "shortness of breath"]
        assert check_urgency(symptoms, "severe chest pain and difficulty breathing") is True

    def test_check_urgency_routine(self):
        from agents.nlp_agent.agent import check_urgency
        symptoms = ["mild cough", "fatigue"]
        assert check_urgency(symptoms, "mild cough for 2 weeks") is False


# ── RAG Agent Tests ────────────────────────────────────────────────────────────

class TestRAGAgent:
    def test_reciprocal_rank_fusion(self):
        from agents.rag_agent.agent import reciprocal_rank_fusion
        from langchain_core.documents import Document

        dense = [Document(page_content=f"doc {i}") for i in range(5)]
        sparse = [Document(page_content=f"doc {i}") for i in [0, 2, 4, 6, 8]]
        fused = reciprocal_rank_fusion(dense, sparse)

        assert len(fused) > 0
        # Scores should be in descending order
        scores = [s for _, s in fused]
        assert scores == sorted(scores, reverse=True)

    def test_build_enriched_query_with_findings(self):
        from agents.rag_agent.agent import build_enriched_query
        from core.state.clinical_state import VisionFindings, NLPFindings

        vision = VisionFindings(
            top_finding="Pneumonia",
            confidence=0.85,
            pathologies={"Pneumonia": 0.85, "Consolidation": 0.4},
        )
        nlp = NLPFindings(symptoms=["fever", "cough", "dyspnea"])

        state = {
            "vision_findings": vision,
            "nlp_findings": nlp,
            "symptom_text": "fever and cough",
            "_rag_query_override": None,
        }
        query = build_enriched_query(state)
        assert "Pneumonia" in query
        assert "fever" in query


# ── Critic Agent Tests ─────────────────────────────────────────────────────────

class TestCriticAgent:
    def test_vision_nlp_agreement_high(self):
        from agents.critic_agent.agent import check_vision_nlp_agreement
        from core.state.clinical_state import VisionFindings, NLPFindings

        vision = VisionFindings(top_finding="Pneumonia", confidence=0.85)
        nlp = NLPFindings(symptoms=["fever", "cough", "shortness of breath"])
        score, flags = check_vision_nlp_agreement(vision, nlp)
        assert score > 0.3  # Some matching expected

    def test_vision_nlp_agreement_contradiction(self):
        from agents.critic_agent.agent import check_vision_nlp_agreement
        from core.state.clinical_state import VisionFindings, NLPFindings

        vision = VisionFindings(top_finding="Pneumonia", confidence=0.85)
        nlp = NLPFindings(
            symptoms=["headache", "nausea"],
            negated_symptoms=["fever", "cough"],
        )
        score, flags = check_vision_nlp_agreement(vision, nlp)
        # Negated expected symptoms should flag contradiction
        assert len(flags) > 0

    def test_rag_coverage_no_docs(self):
        from agents.critic_agent.agent import check_rag_coverage
        from core.state.clinical_state import VisionFindings, RAGFindings

        vision = VisionFindings(top_finding="Pneumonia", confidence=0.8)
        rag = RAGFindings(retrieved_docs=[])
        score, flags = check_rag_coverage(vision, rag)
        assert score < 0.5
        assert len(flags) > 0


# ── State Tests ───────────────────────────────────────────────────────────────

class TestClinicalState:
    def test_vision_findings_default(self):
        from core.state.clinical_state import VisionFindings
        vf = VisionFindings()
        assert vf.pathologies == {}
        assert vf.confidence == 0.0
        assert vf.image_quality_score == 1.0

    def test_clinical_entity_negation(self):
        from core.state.clinical_state import ClinicalEntity
        ent = ClinicalEntity(
            text="chest pain", label="SYMPTOM",
            start=10, end=20, confidence=0.9, negated=True
        )
        assert ent.negated is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
