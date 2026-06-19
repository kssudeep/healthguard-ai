"""
agents/orchestrator/graph.py

LangGraph supervisor graph for HealthGuard AI.
Implements the supervisor-reflection pattern:
  - Supervisor routes to Vision, NLP, RAG agents (parallel where possible)
  - Critic agent evaluates combined findings
  - Reflection loop retries weak agents if confidence < threshold
  - Circuit breaker prevents infinite loops (max 3 reflections)
  - Synthesizer produces final structured clinical report
"""

from __future__ import annotations
import time
import uuid
import logging
from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from core.state.clinical_state import (
    ClinicalState,
    WorkflowStatus,
)
from agents.vision_agent.agent import run_vision_agent
from agents.nlp_agent.agent import run_nlp_agent
from agents.rag_agent.agent import run_rag_agent
from agents.critic_agent.agent import run_critic_agent
from agents.orchestrator.synthesizer import run_synthesizer
from monitoring.mlflow_tracking.tracker import MLflowTracker
from config.settings import settings

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
MAX_REFLECTION_LOOPS = 3
CONFIDENCE_THRESHOLD = 0.65


# ── Supervisor node ──────────────────────────────────────────────────────────

def supervisor_node(state: ClinicalState) -> ClinicalState:
    """
    Entry point. Validates inputs, initialises tracking,
    and sets the routing plan for parallel execution.
    """
    logger.info(f"[Supervisor] Starting session {state['session_id']}")

    if not state.get("image_path") or not state.get("symptom_text"):
        return {
            **state,
            "status": "failed",
            "error_log": ["Supervisor: missing image_path or symptom_text"],
        }

    tracker = MLflowTracker()
    run_id = tracker.start_run(
        run_name=f"healthguard-{state['session_id']}",
        tags={
            "session_id": state["session_id"],
            "patient_age": str(state.get("patient_age", "unknown")),
        },
    )

    return {
        **state,
        "status": "pending",
        "current_agent": "supervisor",
        "next_agent": "vision_agent",
        "iteration_count": 0,
        "pipeline_start_time": time.time(),
        "mlflow_run_id": run_id,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are HealthGuard AI, a clinical decision support system. "
                    "Analyse the provided chest X-ray and symptom description. "
                    "Be evidence-based, precise, and flag uncertainty explicitly."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Patient symptoms: {state['symptom_text']}. "
                    f"Image submitted for analysis."
                ),
            },
        ],
    }


# ── Parallel fan-out node ─────────────────────────────────────────────────────

def parallel_analysis_node(state: ClinicalState) -> ClinicalState:
    """
    Runs Vision + NLP agents.
    LangGraph handles true parallelism via Send() API in the graph edges.
    This node is a coordinator for the fan-in step.
    """
    logger.info("[Supervisor] Fan-out to Vision + NLP agents")
    return {**state, "status": "pending", "current_agent": "supervisor"}


# ── RAG routing node ──────────────────────────────────────────────────────────

def rag_routing_node(state: ClinicalState) -> ClinicalState:
    """
    After Vision + NLP complete, build an enriched RAG query
    combining image findings and NLP symptoms, then route to RAG agent.
    """
    vision = state.get("vision_findings")
    nlp = state.get("nlp_findings")

    query_parts = []
    if vision:
        top = vision.top_finding
        query_parts.append(f"chest X-ray showing {top}")
    if nlp:
        symptoms = ", ".join(nlp.symptoms[:5])
        query_parts.append(f"patient symptoms: {symptoms}")

    enriched_query = ". ".join(query_parts)
    logger.info(f"[Supervisor] RAG query: {enriched_query}")

    return {
        **state,
        "current_agent": "rag_agent",
        "rag_findings": None,   # reset for fresh retrieval
        "_rag_query_override": enriched_query,
    }


# ── Critic routing ────────────────────────────────────────────────────────────

def route_after_critic(state: ClinicalState) -> Literal[
    "synthesizer", "vision_agent", "nlp_agent", "rag_agent", "fail"
]:
    """
    Conditional edge after Critic Agent.
    - If quality gate passed → synthesize
    - If reflection needed and under limit → re-run weakest agent
    - If reflection limit exceeded → synthesize anyway (with low-confidence flag)
    - If critical failure → fail
    """
    critic = state.get("critic_evaluation")
    if not critic:
        return "fail"

    if critic.quality_gate_passed:
        logger.info("[Supervisor] Quality gate passed → Synthesizer")
        return "synthesizer"

    if critic.reflection_count >= MAX_REFLECTION_LOOPS:
        logger.warning("[Supervisor] Max reflections reached → forcing synthesis")
        return "synthesizer"

    # Determine weakest link for targeted reflection
    vision_conf = state["vision_findings"].confidence if state.get("vision_findings") else 0
    rag_docs = len(state["rag_findings"].retrieved_docs) if state.get("rag_findings") else 0

    if vision_conf < 0.5:
        logger.info("[Supervisor] Reflecting on Vision Agent")
        return "vision_agent"
    elif rag_docs < 2:
        logger.info("[Supervisor] Reflecting on RAG Agent")
        return "rag_agent"
    else:
        logger.info("[Supervisor] Reflecting on NLP Agent")
        return "nlp_agent"


def route_after_supervisor(state: ClinicalState) -> Literal["parallel", "fail"]:
    """Initial routing after supervisor validation."""
    if state["status"] == "failed":
        return "fail"
    return "parallel"


# ── Build the LangGraph ───────────────────────────────────────────────────────

def build_graph(redis_url: str = None) -> StateGraph:
    """
    Constructs and compiles the full LangGraph multi-agent workflow.

    Graph topology:
        supervisor → [vision_agent ‖ nlp_agent] → rag_routing → rag_agent
                   → critic_agent → (synthesizer | reflection) → END
    """
    builder = StateGraph(ClinicalState)

    # Add nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("parallel_analysis", parallel_analysis_node)
    builder.add_node("vision_agent", run_vision_agent)
    builder.add_node("nlp_agent", run_nlp_agent)
    builder.add_node("rag_routing", rag_routing_node)
    builder.add_node("rag_agent", run_rag_agent)
    builder.add_node("critic_agent", run_critic_agent)
    builder.add_node("synthesizer", run_synthesizer)

    # Entry point
    builder.set_entry_point("supervisor")

    # Supervisor → parallel or fail
    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"parallel": "parallel_analysis", "fail": END},
    )

    # Sequential execution: vision -> nlp -> rag
    builder.add_edge("parallel_analysis", "vision_agent")
    builder.add_edge("vision_agent", "nlp_agent")
    builder.add_edge("nlp_agent", "rag_routing")

    # RAG routing → RAG agent
    builder.add_edge("rag_routing", "rag_agent")

    # RAG → Critic
    builder.add_edge("rag_agent", "critic_agent")

    # Critic → conditional routing (synthesize or reflect)
    builder.add_conditional_edges(
        "critic_agent",
        route_after_critic,
        {
            "synthesizer": "synthesizer",
            "vision_agent": "vision_agent",
            "nlp_agent": "nlp_agent",
            "rag_agent": "rag_agent",
            "fail": END,
        },
    )

    # Synthesizer → END
    builder.add_edge("synthesizer", END)

    # Compile with Redis checkpointer for state persistence
    checkpointer = None
    if True:
        checkpointer = MemorySaver()

    return builder.compile(checkpointer=checkpointer)


# ── Public runner ─────────────────────────────────────────────────────────────

def run_pipeline(
    image_path: str,
    symptom_text: str,
    patient_age: int = None,
    patient_sex: str = "unknown",
) -> dict:
    """
    Public entry point. Initialises state and runs the full pipeline.
    Returns the final ClinicalReport as a dict.
    """
    session_id = str(uuid.uuid4())[:8]

    initial_state: ClinicalState = {
        "session_id": session_id,
        "image_path": image_path,
        "symptom_text": symptom_text,
        "patient_age": patient_age,
        "patient_sex": patient_sex,
        "status": "pending",
        "current_agent": "supervisor",
        "next_agent": "supervisor",
        "error_log": [],
        "iteration_count": 0,
        "vision_findings": None,
        "nlp_findings": None,
        "rag_findings": None,
        "critic_evaluation": None,
        "final_report": None,
        "messages": [],
        "pipeline_start_time": time.time(),
        "mlflow_run_id": "",
        "langsmith_trace_id": "",
    }

    graph = build_graph(redis_url=settings.REDIS_URL)
    config = {"configurable": {"thread_id": session_id}}

    logger.info(f"[Pipeline] Starting session {session_id}")
    final_state = graph.invoke(initial_state, config=config)

    report = final_state.get("final_report")
    if not report:
        return {"error": "Pipeline failed", "error_log": final_state.get("error_log", [])}

    return {
        "session_id": session_id,
        "report": report.__dict__,
        "vision_confidence": final_state["vision_findings"].confidence
        if final_state.get("vision_findings") else 0,
        "reflection_loops": final_state["critic_evaluation"].reflection_count
        if final_state.get("critic_evaluation") else 0,
        "total_time_ms": (time.time() - initial_state["pipeline_start_time"]) * 1000,
    }
