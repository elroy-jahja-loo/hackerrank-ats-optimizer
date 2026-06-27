import os
import sys
import tempfile
from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parent
HIRING_AGENT_DIR = APP_DIR / "hiring-agent"
sys.path.insert(0, str(HIRING_AGENT_DIR))

from evaluator import ResumeEvaluator
from models import ModelProvider
from pdf import PDFHandler
from prompt import MODEL_PARAMETERS
from transform import convert_json_resume_to_text


CATEGORY_MAXES = {
    "open_source": 35,
    "self_projects": 30,
    "production": 25,
    "technical_skills": 10,
}

PROVIDER_PRESETS = {
    "Gemini (Google AI Studio free tier)": {
        "provider": ModelProvider.GEMINI.value,
        "models": ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-flash"],
        "base_url": "",
        "needs_key": True,
        "help": "Often the easiest free hosted option. Create a key in Google AI Studio and paste it here.",
    },
    "Groq (free hosted Llama/Gemma)": {
        "provider": ModelProvider.OPENAI.value,
        "models": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "gemma2-9b-it"],
        "base_url": "https://api.groq.com/openai/v1",
        "needs_key": True,
        "help": "Free hosted OpenAI-compatible API for Llama/Gemma models. Paste your Groq API key.",
    },
    "Ollama (local free)": {
        "provider": ModelProvider.OLLAMA.value,
        "models": ["gemma3:4b", "qwen3:4b", "mistral:7b", "gemma3:1b"],
        "base_url": "",
        "needs_key": False,
        "help": "Runs on your own computer with Ollama. Not available on Streamlit Community Cloud unless you provide a reachable Ollama host.",
    },
    "OpenAI": {
        "provider": ModelProvider.OPENAI.value,
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"],
        "base_url": "",
        "needs_key": True,
        "help": "Paid OpenAI API key.",
    },
    "DeepSeek": {
        "provider": ModelProvider.OPENAI.value,
        "models": ["deepseek-v4-flash", "deepseek-v4-pro", "deepseek-chat"],
        "base_url": "https://api.deepseek.com",
        "needs_key": True,
        "help": "OpenAI-compatible DeepSeek endpoint. Prefer deepseek-v4-flash or deepseek-v4-pro; deepseek-chat is deprecated by DeepSeek on 2026-07-24.",
    },
    "Anthropic": {
        "provider": ModelProvider.ANTHROPIC.value,
        "models": ["claude-haiku-4-5-20251001", "claude-sonnet-4-6"],
        "base_url": "",
        "needs_key": True,
        "help": "Paid Anthropic API key.",
    },
    "Custom OpenAI-compatible": {
        "provider": ModelProvider.OPENAI.value,
        "models": ["custom-model"],
        "base_url": "",
        "needs_key": True,
        "help": "Use any OpenAI-compatible chat/completions API by providing model, base URL, and API key.",
    },
}


def grade_label(score: float) -> str:
    if score >= 80:
        return "Excellent"
    if score >= 60:
        return "Strong"
    if score >= 40:
        return "Average"
    return "Needs Work"


def score_resume(uploaded_file, provider_name, model_name, api_key, base_url):
    model_params = MODEL_PARAMETERS.get(model_name, {"temperature": 0.1, "top_p": 0.9})

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        pdf_handler = PDFHandler(
            model_name=model_name,
            model_params=model_params,
            provider_name=provider_name,
            api_key=api_key or None,
            base_url=base_url or None,
        )
        resume_data = pdf_handler.extract_json_from_pdf(tmp_path)
        if not resume_data:
            raise ValueError("Failed to parse resume. Make sure it is a readable PDF and the selected model can return JSON.")

        evaluator = ResumeEvaluator(
            model_name=model_name,
            model_params=model_params,
            provider_name=provider_name,
            api_key=api_key or None,
            base_url=base_url or None,
        )
        resume_text = convert_json_resume_to_text(resume_data)
        evaluation = evaluator.evaluate_resume(resume_text)

        total_score = 0
        scores_out = {}
        for category, max_value in CATEGORY_MAXES.items():
            category_score = getattr(evaluation.scores, category)
            capped = min(category_score.score, max_value)
            total_score += capped
            scores_out[category] = {
                "score": capped,
                "raw_score": category_score.score,
                "max": category_score.max,
                "evidence": category_score.evidence,
            }

        total_score += evaluation.bonus_points.total
        total_score -= evaluation.deductions.total
        total_score = max(-20, min(120, total_score))

        return resume_data, evaluation, scores_out, round(total_score, 1)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


st.set_page_config(page_title="HackerRank ATS Optimizer", page_icon="📄", layout="wide")

st.title("HackerRank ATS Optimizer")
st.caption("Upload a PDF resume, choose an LLM provider, and get an explainable HackerRank-style ATS score.")

with st.sidebar:
    st.header("LLM Provider")
    preset_name = st.selectbox("Provider", list(PROVIDER_PRESETS.keys()))
    preset = PROVIDER_PRESETS[preset_name]
    st.info(preset["help"])

    if preset_name == "Custom OpenAI-compatible":
        model_name = st.text_input("Model", value="")
    else:
        model_name = st.selectbox("Model", preset["models"])

    base_url = preset["base_url"]
    if preset_name in {"Custom OpenAI-compatible", "Ollama (local free)"}:
        base_url = st.text_input(
            "Base URL / host",
            value=base_url,
            placeholder="https://api.example.com/v1 or http://localhost:11434",
        )
    elif base_url:
        st.text_input("Base URL", value=base_url, disabled=True)

    api_key = ""
    if preset["needs_key"]:
        api_key = st.text_input("API key", type="password", help="Stored only in this browser session. It is not committed to the repo.")

    st.divider()
    st.write("Free options:")
    st.write("- Gemini free tier: hosted, needs your free Google AI Studio key.")
    st.write("- Groq free tier: hosted Llama/Gemma, needs your Groq key.")
    st.write("- Ollama: free local models, works when running locally.")

uploaded_file = st.file_uploader("Upload your resume PDF", type=["pdf"])

if uploaded_file:
    st.write(f"Selected: `{uploaded_file.name}`")

can_run = bool(uploaded_file and model_name and (api_key or not preset["needs_key"]))
if st.button("Analyze Resume", type="primary", disabled=not can_run):
    with st.spinner("Parsing resume and scoring. This can take 30-90 seconds depending on the model."):
        try:
            resume_data, evaluation, scores_out, total_score = score_resume(
                uploaded_file,
                preset["provider"],
                model_name.strip(),
                api_key.strip(),
                base_url.strip(),
            )
            st.session_state["result"] = (resume_data, evaluation, scores_out, total_score)
        except Exception as exc:
            st.error(str(exc))

if "result" in st.session_state:
    resume_data, evaluation, scores_out, total_score = st.session_state["result"]

    left, right = st.columns([1, 2])
    with left:
        st.metric("Overall Score", f"{total_score}/120", grade_label(total_score))
        st.subheader("Bonus")
        st.write(f"+{evaluation.bonus_points.total}")
        st.caption(evaluation.bonus_points.breakdown)
        st.subheader("Deductions")
        st.write(f"-{evaluation.deductions.total}")
        st.caption(evaluation.deductions.reasons)

    with right:
        st.subheader("Category Breakdown")
        for category, data in scores_out.items():
            label = category.replace("_", " ").title()
            st.write(f"**{label}: {data['score']}/{data['max']}**")
            st.progress(max(0.0, min(1.0, data["score"] / data["max"])))
            st.caption(data["evidence"])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Key Strengths")
        for item in evaluation.key_strengths:
            st.write(f"- {item}")

    with col2:
        st.subheader("Areas To Improve")
        for item in evaluation.areas_for_improvement:
            st.write(f"- {item}")

    with st.expander("Parsed Resume JSON"):
        st.json(resume_data.model_dump())
