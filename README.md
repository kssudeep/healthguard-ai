# HealthGuard AI — Multimodal Clinical Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-green)
![Azure](https://img.shields.io/badge/Azure-Deployed-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracked-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

> A production-grade, multi-agent AI system for automated clinical decision support. Combines computer vision (chest X-ray analysis), NLP (symptom extraction), RAG (clinical knowledge retrieval), and LLM-based synthesis — orchestrated via LangGraph with a supervisor-reflection pattern, deployed on Azure with full MLflow experiment tracking and LangSmith observability.

---

## Architecture Overview

```
User Input (Image + Symptom Text)
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│              SUPERVISOR AGENT (LangGraph)                    │
│   Routes → Retries → Resolves Conflicts → Quality Gates      │
└──────────┬──────────────┬──────────────────┬────────────────┘
           │              │                  │
    ┌──────▼──────┐ ┌──────▼──────┐ ┌────────▼────────┐
    │ VISION      │ │ NLP/NER     │ │  RAG RETRIEVAL  │
    │ AGENT       │ │ AGENT       │ │  AGENT          │
    │ DenseNet121 │ │ BioBERT NER │ │  FAISS + BM25   │
    │ + GradCAM   │ │ + SpaCy     │ │  Hybrid Search  │
    │ + DICOM     │ │ + Sentiment │ │  PubMed + Guide │
    └──────┬──────┘ └──────┬──────┘ └────────┬────────┘
           └──────────────┬┘                  │
                          ▼                   │
              ┌───────────────────────────────▼──────┐
              │         CRITIC AGENT                  │
              │  Confidence Scoring + Hallucination   │
              │  Detection + Reflection Loop          │
              └───────────────────┬──────────────────┘
                                  │
                          ┌───────▼───────┐
                          │  SYNTHESIZER  │
                          │  Claude / GPT │
                          │  Structured   │
                          │  Report Gen.  │
                          └───────┬───────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │  FastAPI REST Backend        │
                    │  + Streamlit Dashboard       │
                    │  + MLflow Tracking           │
                    │  + LangSmith Observability   │
                    └─────────────────────────────┘
```

## Key Features

- **5-agent LangGraph system** with supervisor-reflection pattern and circuit breakers
- **DenseNet-121** fine-tuned on NIH ChestX-ray14 (14 pathologies) with Grad-CAM explainability
- **BioBERT NER** for clinical entity extraction (symptoms, medications, body parts)
- **Hybrid RAG** (FAISS dense + BM25 sparse) over PubMed abstracts + clinical guidelines
- **Critic agent** with hallucination detection and confidence-gated reflection loops
- **DICOM support** for real medical imaging files
- **MLflow** experiment tracking with model registry
- **LangSmith** full trace observability for every agent node
- **Azure Container Apps** deployment with GitHub Actions CI/CD
- **Redis** for agent state caching and conversation memory
- **Prometheus + Grafana** for production monitoring

## Tech Stack

| Layer | Technology |
|---|---|
| Vision | PyTorch, DenseNet-121, OpenCV, pydicom, torchcam (Grad-CAM) |
| NLP | BioBERT, SpaCy (en_core_sci_lg), Hugging Face Transformers |
| RAG | LangChain, FAISS, BM25Retriever, sentence-transformers |
| Orchestration | LangGraph 0.2, LangSmith (optional) |
| LLM — Synthesizer | **Gemini 2.0 Flash** (FREE — aistudio.google.com) |
| LLM — Critic Agent | **Groq Llama 3.3 70B** (FREE — console.groq.com) |
| Embeddings | sentence-transformers/all-MiniLM (FREE — Hugging Face) |
| Backend | FastAPI, Pydantic v2, Redis, Celery |
| Frontend | Streamlit |
| Tracking | MLflow, Prometheus, Grafana |
| Infra | Docker, Azure Container Apps, GitHub Actions |

## Project Structure

```
healthguard_ai/
├── agents/
│   ├── orchestrator/       # LangGraph supervisor + graph definition
│   ├── vision_agent/       # DenseNet + GradCAM + DICOM
│   ├── nlp_agent/          # BioBERT NER + symptom parser
│   ├── rag_agent/          # Hybrid retriever + reranker
│   └── critic_agent/       # Confidence scoring + reflection
├── core/
│   ├── state/              # LangGraph shared state schema
│   ├── memory/             # Redis conversation memory
│   └── tools/              # LangChain tool wrappers
├── ml/
│   ├── vision/             # Model training + evaluation
│   ├── nlp/                # NER fine-tuning
│   └── rag/                # Vector store + indexing
├── api/
│   ├── routes/             # FastAPI routers
│   ├── middleware/          # Auth, rate limit, CORS
│   └── schemas/            # Pydantic request/response models
├── monitoring/
│   ├── mlflow_tracking/    # Experiment logging
│   └── langsmith/          # Trace configuration
├── ui/                     # Streamlit dashboard
├── tests/                  # Unit + integration + agent tests
├── config/                 # Settings, env management
├── scripts/                # Data download, DB seeding
├── .github/workflows/      # CI/CD pipelines
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Quickstart

```bash
git clone https://github.com/kssudeep/healthguard-ai
cd healthguard_ai
cp .env.example .env          # Fill in API keys
docker-compose up --build     # Spins up all services
# Visit http://localhost:8501  (Streamlit UI)
# Visit http://localhost:8000/docs  (FastAPI Swagger)
# Visit http://localhost:5000  (MLflow UI)
```

## Dataset Setup

```bash
python scripts/download_data.py   # Downloads NIH ChestX-ray14 subset
python scripts/build_vector_db.py # Indexes PubMed + clinical guidelines
python scripts/seed_redis.py      # Seeds conversation memory store
```

## Training

```bash
python ml/vision/train.py --epochs 30 --batch_size 32 --model densenet121
python ml/nlp/finetune_ner.py --model biobert --dataset i2b2
```

## Author
Sudeep K S | MS Applied AI, Northeastern University
