"""
ui/app.py

Streamlit dashboard for HealthGuard AI.
- Image upload (X-ray / DICOM)
- Symptom text input
- Live pipeline status tracker
- Structured report display
- Grad-CAM heatmap viewer
- MLflow metrics panel
"""

import streamlit as st
import requests
import time
import json
from pathlib import Path
from PIL import Image
import io

API_BASE = "http://api:8000"

st.set_page_config(
    page_title="HealthGuard AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏥 HealthGuard AI")
    st.markdown("**Multimodal Clinical Intelligence Platform**")
    st.divider()
    st.markdown("### About")
    st.markdown(
        "This system combines:\n"
        "- 🔬 **DenseNet-121** chest X-ray analysis\n"
        "- 🧠 **BioBERT NER** symptom extraction\n"
        "- 📚 **Hybrid RAG** clinical knowledge\n"
        "- 🤖 **LangGraph** 5-agent orchestration\n"
        "- ✅ **Critic agent** quality control"
    )
    st.divider()
    st.warning(
        "⚠️ **Research Use Only**\n\n"
        "This tool is for educational and research purposes. "
        "Always consult a licensed medical professional."
    )

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("🏥 HealthGuard AI — Clinical Decision Support")
st.caption("Upload a chest X-ray and describe symptoms for AI-assisted analysis")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📷 Chest X-Ray Upload")
    uploaded_file = st.file_uploader(
        "Upload chest X-ray",
        type=["jpg", "jpeg", "png", "dcm"],
        help="Supports JPEG, PNG, and DICOM formats",
    )
    if uploaded_file:
        if uploaded_file.name.endswith(".dcm"):
            st.info("DICOM file detected — will be processed with pydicom")
        else:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded X-ray", use_column_width=True)

with col2:
    st.subheader("📝 Patient Information")
    symptoms = st.text_area(
        "Describe symptoms",
        height=150,
        placeholder=(
            "e.g. Patient presents with 3-day history of productive cough, "
            "fever of 38.5°C, shortness of breath on exertion, and right-sided chest pain..."
        ),
    )
    col_age, col_sex = st.columns(2)
    with col_age:
        patient_age = st.number_input("Patient Age", min_value=0, max_value=120, value=45)
    with col_sex:
        patient_sex = st.selectbox("Sex", ["unknown", "M", "F"])

# ── Submit ────────────────────────────────────────────────────────────────────
st.divider()
submit = st.button("🚀 Run Clinical Analysis", type="primary", use_container_width=True)

if submit:
    if not uploaded_file:
        st.error("Please upload a chest X-ray image")
    elif not symptoms.strip():
        st.error("Please describe patient symptoms")
    else:
        # Submit to API
        with st.spinner("Submitting to analysis pipeline..."):
            try:
                uploaded_file.seek(0)
                response = requests.post(
                    f"{API_BASE}/api/v1/analyse",
                    files={"image": (uploaded_file.name, uploaded_file, "image/jpeg")},
                    data={
                        "symptoms": symptoms,
                        "patient_age": patient_age,
                        "patient_sex": patient_sex,
                    },
                    timeout=120,
                )
                job_data = response.json()
                job_id = job_data["job_id"]
                st.session_state["job_id"] = job_id
                st.success(f"✅ Job submitted: **{job_id}**")
            except Exception as e:
                st.error(f"API error: {e}")
                st.stop()

        # Poll for results with progress
        st.subheader("🔄 Pipeline Progress")
        progress_bar = st.progress(0)
        status_placeholder = st.empty()

        pipeline_stages = [
            "👁️ Vision Agent — Analyzing X-ray...",
            "🧠 NLP Agent — Extracting symptoms...",
            "📚 RAG Agent — Retrieving clinical evidence...",
            "✅ Critic Agent — Quality evaluation...",
            "📋 Synthesizer — Generating report...",
        ]

        stage = 0
        for attempt in range(60):  # 60 second timeout
            time.sleep(1)
            try:
                result_resp = requests.get(
                    f"{API_BASE}/api/v1/results/{job_id}", timeout=30
                )
                result_data = result_resp.json()
            except Exception:
                continue

            status = result_data.get("status")
            progress = min(0.95, attempt / 30)
            progress_bar.progress(progress)

            if stage < len(pipeline_stages):
                status_placeholder.info(pipeline_stages[min(stage, len(pipeline_stages)-1)])
                stage = int(progress * len(pipeline_stages))

            if status == "complete":
                progress_bar.progress(1.0)
                status_placeholder.success("✅ Analysis complete!")
                st.session_state["result"] = result_data["result"]
                break
            elif status == "failed":
                st.error(f"Pipeline failed: {result_data.get('error')}")
                st.stop()
        else:
            st.warning("Analysis timed out — try again or check the API logs")

# ── Display Results ───────────────────────────────────────────────────────────
if "result" in st.session_state:
    result = st.session_state["result"]
    report = result.get("report", {})

    st.divider()
    st.header("📋 Clinical Analysis Report")

    # Urgency banner
    urgency = report.get("urgency_level", "routine")
    urgency_colors = {"routine": "🟢", "urgent": "🟡", "emergency": "🔴"}
    st.markdown(
        f"**Urgency:** {urgency_colors.get(urgency, '⚪')} {urgency.upper()}"
    )

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Confidence", f"{report.get('confidence_score', 0):.1%}")
    m2.metric("Report ID", report.get("report_id", "N/A"))
    m3.metric("Reflection Loops", result.get("reflection_loops", 0))
    m4.metric("Pipeline Time", f"{result.get('total_time_ms', 0)/1000:.1f}s")

    # Main report content
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.subheader("🎯 Primary Diagnosis")
        st.info(report.get("primary_diagnosis", "N/A"))

        st.subheader("🔀 Differential Diagnoses")
        for d in report.get("differential_diagnoses", []):
            st.markdown(f"- {d}")

        st.subheader("📄 Patient Summary")
        st.write(report.get("patient_summary", ""))

    with col_r2:
        st.subheader("🔬 Supporting Evidence")
        for e in report.get("supporting_evidence", []):
            st.markdown(f"- {e}")

        st.subheader("💊 Recommended Actions")
        for r_item in report.get("recommended_actions", []):
            st.markdown(f"- {r_item}")

    # Grad-CAM
    if st.session_state.get("job_id"):
        st.subheader("🔥 Grad-CAM Heatmap (Explainability)")
        try:
            session_id = result.get("session_id", "")
            gcam_resp = requests.get(
                f"{API_BASE}/api/v1/gradcam/{session_id}", timeout=30
            )
            if gcam_resp.status_code == 200:
                gcam_img = Image.open(io.BytesIO(gcam_resp.content))
                st.image(gcam_img, caption="Regions influencing AI diagnosis", width=400)
            else:
                st.info("GradCAM not available for this session")
        except Exception:
            st.info("GradCAM visualization unavailable")

    # Raw JSON expander
    with st.expander("🔧 Raw JSON Response"):
        st.json(result)

    # Disclaimer
    st.divider()
    st.caption(
        "⚠️ " + report.get(
            "disclaimer",
            "This report is AI-generated for research purposes only. "
            "Always consult a licensed medical professional.",
        )
    )
