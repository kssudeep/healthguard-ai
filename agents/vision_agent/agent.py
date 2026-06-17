"""
agents/vision_agent/agent.py

Vision Agent for HealthGuard AI.
- Loads DenseNet-121 fine-tuned on NIH ChestX-ray14 (14 pathologies)
- Supports DICOM (.dcm) and standard image formats (PNG, JPEG)
- Generates Grad-CAM heatmaps for explainability
- Assesses image quality before inference
- Logs all metrics to MLflow
"""

from __future__ import annotations
import time
import logging
import os
from pathlib import Path
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

# Grad-CAM via torchcam
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask

# DICOM support
try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False

from core.state.clinical_state import ClinicalState, VisionFindings
from monitoring.mlflow_tracking.tracker import MLflowTracker
from config.settings import settings

logger = logging.getLogger(__name__)

# ── 14 NIH ChestX-ray14 pathology classes ────────────────────────────────────
PATHOLOGY_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]

# ── Image preprocessing ────────────────────────────────────────────────────────
TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ChestXRayModel(nn.Module):
    """
    DenseNet-121 modified for multi-label classification (14 pathologies).
    Based on the CheXNet architecture (Rajpurkar et al., 2017).
    Final sigmoid instead of softmax — each pathology is independent.
    """

    def __init__(self, num_classes: int = 14, pretrained: bool = True):
        super().__init__()
        densenet = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # Replace classifier head
        in_features = densenet.classifier.in_features
        densenet.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
            nn.Sigmoid(),
        )
        self.model = densenet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_features_layer(self):
        """Returns the final conv layer for GradCAM."""
        return self.model.features.denseblock4


def load_model(weights_path: str = None) -> ChestXRayModel:
    """
    Loads model weights. Falls back to ImageNet-pretrained DenseNet
    if no fine-tuned weights are available (for development).
    """
    model = ChestXRayModel(num_classes=14, pretrained=True)
    if weights_path and Path(weights_path).exists():
        checkpoint = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"[VisionAgent] Loaded fine-tuned weights from {weights_path}")
    else:
        logger.warning("[VisionAgent] No weights found — using ImageNet pretrained (demo mode)")
    model.eval()
    return model


# Singleton model (loaded once at import time)
_model: ChestXRayModel = None
_device: torch.device = None


def get_model() -> tuple[ChestXRayModel, torch.device]:
    global _model, _device
    if _model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = load_model(settings.VISION_MODEL_WEIGHTS_PATH).to(_device)
        logger.info(f"[VisionAgent] Model loaded on {_device}")
    return _model, _device


# ── DICOM / image loading ─────────────────────────────────────────────────────

def load_image(image_path: str) -> tuple[Image.Image, dict]:
    """
    Loads image from path. Handles:
    - DICOM (.dcm): extracts pixel array + metadata
    - PNG / JPEG: standard PIL loading
    Returns (PIL Image in RGB, metadata dict).
    """
    path = Path(image_path)
    metadata = {"format": path.suffix.lower()}

    if path.suffix.lower() == ".dcm":
        if not DICOM_AVAILABLE:
            raise ImportError("pydicom required for DICOM support: pip install pydicom")
        ds = pydicom.dcmread(str(path))
        pixel_array = ds.pixel_array.astype(np.float32)
        # Normalize to 0-255
        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-8)
        pixel_array = (pixel_array * 255).astype(np.uint8)
        img = Image.fromarray(pixel_array).convert("RGB")
        metadata.update({
            "PatientID": str(getattr(ds, "PatientID", "unknown")),
            "StudyDate": str(getattr(ds, "StudyDate", "unknown")),
            "Modality": str(getattr(ds, "Modality", "unknown")),
            "ViewPosition": str(getattr(ds, "ViewPosition", "unknown")),
        })
    else:
        img = Image.open(path).convert("RGB")

    return img, metadata


def assess_image_quality(img: Image.Image) -> float:
    """
    Heuristic image quality score (0–1).
    Checks: resolution, contrast, brightness distribution.
    """
    arr = np.array(img.convert("L"), dtype=np.float32)
    # Contrast (std of pixel values)
    contrast = arr.std() / 128.0
    # Brightness (avoid over/under exposed)
    mean_brightness = arr.mean() / 255.0
    brightness_score = 1.0 - abs(mean_brightness - 0.5) * 2
    # Resolution
    w, h = img.size
    resolution_score = min(1.0, (w * h) / (512 * 512))
    quality = (contrast * 0.4 + brightness_score * 0.4 + resolution_score * 0.2)
    return float(np.clip(quality, 0.0, 1.0))


# ── Grad-CAM generation ───────────────────────────────────────────────────────

def generate_gradcam(
    model: ChestXRayModel,
    tensor: torch.Tensor,
    target_class_idx: int,
    output_path: str,
    original_img: Image.Image,
) -> str:
    """
    Generates and saves a Grad-CAM heatmap overlay.
    Returns the saved path.
    """
    try:
        cam_extractor = GradCAM(model, target_layer=model.get_features_layer())
        with torch.enable_grad():
            out = model(tensor.unsqueeze(0))
            activation_map = cam_extractor(target_class_idx, out)

        # Overlay on original image
        result = overlay_mask(
            original_img,
            Image.fromarray(activation_map[0].squeeze().numpy()),
            alpha=0.5,
        )
        result.save(output_path)
        logger.info(f"[VisionAgent] GradCAM saved to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"[VisionAgent] GradCAM failed: {e}")
        return ""


# ── Main agent function ───────────────────────────────────────────────────────

def run_vision_agent(state: ClinicalState) -> ClinicalState:
    """
    LangGraph node: Vision Agent.
    Reads image_path from state, runs inference, writes VisionFindings back.
    """
    start = time.time()
    tracker = MLflowTracker()
    logger.info(f"[VisionAgent] Processing image: {state['image_path']}")

    try:
        # 1. Load image
        img, dicom_meta = load_image(state["image_path"])

        # 2. Quality check
        quality_score = assess_image_quality(img)
        if quality_score < 0.2:
            logger.warning(f"[VisionAgent] Low quality image: {quality_score:.2f}")

        # 3. Preprocess
        model, device = get_model()
        tensor = TRANSFORM(img).to(device)

        # 4. Inference
        with torch.no_grad():
            predictions = model(tensor.unsqueeze(0)).squeeze().cpu().numpy()

        # 5. Build pathology scores dict
        pathology_scores = {
            cls: float(score)
            for cls, score in zip(PATHOLOGY_CLASSES, predictions)
        }
        # Filter to clinically significant findings (>0.1)
        significant = {k: v for k, v in pathology_scores.items() if v > 0.1}
        top_finding = max(pathology_scores, key=pathology_scores.get)
        top_confidence = pathology_scores[top_finding]

        # 6. Generate Grad-CAM for top finding
        top_idx = PATHOLOGY_CLASSES.index(top_finding)
        gradcam_path = os.path.join(
            settings.GRADCAM_OUTPUT_DIR,
            f"{state['session_id']}_gradcam.png",
        )
        os.makedirs(settings.GRADCAM_OUTPUT_DIR, exist_ok=True)
        gradcam_path = generate_gradcam(
            model, tensor, top_idx, gradcam_path, img
        )

        elapsed = (time.time() - start) * 1000

        # 7. Build findings object
        findings = VisionFindings(
            pathologies=significant,
            top_finding=top_finding,
            confidence=top_confidence,
            gradcam_heatmap_path=gradcam_path,
            image_quality_score=quality_score,
            dicom_metadata=dicom_meta,
            model_version="densenet121-chestxray14-v2",
            inference_time_ms=elapsed,
        )

        # 8. Log to MLflow
        tracker.log_metrics(state["mlflow_run_id"], {
            "vision_top_confidence": top_confidence,
            "vision_quality_score": quality_score,
            "vision_inference_ms": elapsed,
            "vision_significant_findings": len(significant),
        })

        logger.info(
            f"[VisionAgent] Done. Top finding: {top_finding} "
            f"({top_confidence:.2%}) in {elapsed:.0f}ms"
        )

        return {
            **state,
            "vision_findings": findings,
            "status": "vision_complete",
            "messages": state["messages"] + [{
                "role": "assistant",
                "agent": "vision_agent",
                "content": (
                    f"Vision analysis complete. Primary finding: {top_finding} "
                    f"(confidence: {top_confidence:.1%}). "
                    f"Additional findings: {', '.join(list(significant.keys())[:3])}."
                ),
            }],
        }

    except Exception as e:
        logger.error(f"[VisionAgent] Error: {e}", exc_info=True)
        return {
            **state,
            "error_log": state["error_log"] + [f"VisionAgent: {str(e)}"],
            "status": "failed",
        }
