# HackerRank ATS Optimizer

An open-source, AI-powered web app that parses, scores, and annotates a software engineering resume against the same style of rubric HackerRank uses for intern screening, giving you a transparent, actionable ATS-style score before you apply.

---

## What It Does

Upload a PDF resume in the Flask app or Streamlit app and get back:

- **Overall score out of 120** with a letter-grade tier (Excellent / Strong / Average / Needs Improvement)
- **Four category scores** matching HackerRank's actual hiring criteria
- **Color-coded resume viewer** where every section is highlighted by the category it contributes to, with the AI's evidence shown inline
- **Key strengths** and **areas to improve** written in plain English
- **Bonus points and deductions** breakdown (GSoC, startup experience, missing links, tutorial-only projects, etc.)

The codebase contains functional parsing logic in `hiring-agent/pdf.py`, `hiring-agent/pymupdf_rag.py`, `hiring-agent/llm_utils.py`, and `hiring-agent/github.py`. It combines deterministic PDF text extraction, LLM-powered NLP section extraction, Pydantic schema validation, JSON response cleanup, and regex-based GitHub identity parsing.

---

## Scoring Rubric

| Category | Max Points | What It Measures |
|---|---|---|
| Open Source | 35 | Contributions to external repos, GSoC, community involvement |
| Self Projects | 30 | Complexity, real-world impact, live demos, GitHub links |
| Production Experience | 25 | Internships, full-time roles, startup early-engineer credit |
| Technical Skills | 10 | Breadth of languages, frameworks, and demonstrated depth |
| Bonus Points | +20 | GSoC (+5), Girl Script SoC (+3), founder/co-founder (+5), LinkedIn (+1), portfolio (+2), blogs (+3) |
| Deductions | -20 | Tutorial-only projects, missing links, broken URLs, generic project names |

Total range: -20 to 120.

---

## How It Works

```
PDF Upload
    |
    v
PyMuPDF -> Markdown text
    |
    v
Claude Haiku (6 parallel calls)
    |-- basics extraction
    |-- work experience extraction
    |-- education extraction
    |-- skills extraction
    |-- projects extraction
    `-- awards extraction
    |
    v
JSONResume structured object
    |
    v
Claude Haiku -> evaluation call
(scored against HackerRank rubric via Jinja prompt templates)
    |
    v
EvaluationData: scores + evidence + strengths + improvements
    |
    v
Flask/Streamlit UI -> score cards + parsed resume JSON
```

The grading logic is based on [interviewstreet/hiring-agent](https://github.com/interviewstreet/hiring-agent), HackerRank's open-source hiring pipeline. This project adds Flask and Streamlit UIs on top, extends the provider system to support Claude, OpenAI, OpenAI-compatible endpoints, Gemini, Groq, DeepSeek, and Ollama, and fixes template path resolution so it runs from any working directory.

---

## Algorithmic Methodology

This project performs resume analysis as a deterministic pipeline with constrained LLM/NLP steps. The application never sends the raw PDF directly to the scoring prompt. It first parses the resume into a structured JSON Resume object, normalizes that object, then evaluates only the structured content and optional GitHub evidence.

### 1. PDF ingestion and text normalization

- `app.py` accepts only PDF uploads through the `/analyze` Flask endpoint and stores each upload in a temporary file.
- `hiring-agent/pdf.py` opens the PDF with PyMuPDF and passes every page to `to_markdown` in `hiring-agent/pymupdf_rag.py`.
- `pymupdf_rag.py` reconstructs text in reading order, converts headings to Markdown based on font-size heuristics, preserves links and tables where possible, and emits a Markdown-like text stream for downstream parsing.
- This stage is deterministic: the same readable PDF produces the same Markdown text before model calls.

### 2. Section-level NLP parsing

- `PDFHandler.extract_json_from_pdf` calls `_extract_all_sections_separately` after text extraction.
- The resume is split conceptually into six target sections: `basics`, `work`, `education`, `skills`, `projects`, and `awards`.
- For each section, `pdf.py` renders a dedicated Jinja prompt from `hiring-agent/prompts/templates/*.jinja` and sends the full normalized resume text plus section-specific instructions to the configured LLM provider.
- Supported NLP backends are selected through `hiring-agent/prompt.py` and initialized by `hiring-agent/llm_utils.py`: Anthropic, OpenAI, OpenAI-compatible endpoints via `OPENAI_BASE_URL`, Google Gemini, and local Ollama models.
- Each model call requests structured JSON using the matching Pydantic section schema, such as `BasicsSection`, `WorkSection`, `SkillsSection`, or `ProjectsSection` from `hiring-agent/models.py`.

### 3. JSON repair, validation, and normalization

- LLM responses are passed through `extract_json_from_response` in `hiring-agent/llm_utils.py`, which removes Markdown code fences and provider reasoning tags before JSON decoding.
- `pdf.py` slices the response to the outermost JSON object and parses it with `json.loads`.
- `hiring-agent/transform.py` normalizes loose provider output into JSON Resume-compatible keys and list shapes.
- The final object is validated by the `JSONResume` Pydantic model in `hiring-agent/models.py`. Invalid or empty parses fail safely instead of being scored as real data.

### 4. Regex-based GitHub identity parsing and enrichment

- `hiring-agent/github.py` contains explicit regex parsing logic in `extract_github_username`.
- It recognizes GitHub profiles in forms such as `https://github.com/user`, `github.com/user`, `@user`, or a bare username using these patterns: `https?://github\.com/([^/]+)`, `github\.com/([^/]+)`, `@([^/]+)`, and `^([a-zA-Z0-9-]+)$`.
- When a GitHub profile is present, the app can fetch profile, repository, contributor, and language metadata through the GitHub API.
- Repository signals are classified into self-project and open-source evidence, then provided as additional context for scoring.

### 5. Resume-to-text conversion for scoring

- `hiring-agent/transform.py` converts the validated `JSONResume` object into a compact evaluator text format.
- This keeps the scoring prompt focused on normalized evidence rather than noisy PDF formatting.
- The optional GitHub and blog data are appended as separate evidence blocks when available.

### 6. HackerRank-style scoring algorithm

- `hiring-agent/evaluator.py` renders `resume_evaluation_criteria.jinja`, which encodes the scoring methodology.
- The evaluator scores four required categories: Open Source out of 35, Self Projects out of 30, Production Experience out of 25, and Technical Skills out of 10.
- It applies bonus points up to 20 for signals such as Google Summer of Code, Girl Script Summer of Code, founder or early-stage startup experience, portfolio links, LinkedIn, and technical blogs.
- It applies deductions for weak signals such as tutorial-only projects, missing project links, broken links, generic project names, and lack of true open-source contributions.
- Fairness constraints in the prompt explicitly exclude name, gender, school, GPA, geography, and unrelated demographic information from scoring.
- `app.py` and `streamlit_app.py` cap category scores and the final score to the documented range of `-20` to `120` before returning the result.

### 7. Explainability output

- The evaluator must return strict JSON matching `EvaluationData` in `models.py`.
- Every category includes a numeric score, max score, and evidence string.
- The frontend renders the structured resume, category evidence, key strengths, areas for improvement, bonus points, deductions, and color-coded highlights so users can see why each score was assigned.

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/elroy-jahja-loo/hackerrank-ats-optimizer
cd hackerrank-ats-optimizer
pip install -r requirements.txt
```

### 2. Configure your LLM provider

Create `hiring-agent/.env` and pick one provider:

**Anthropic Claude (recommended, about $0.001 per resume)**
```env
LLM_PROVIDER=anthropic
DEFAULT_MODEL=claude-haiku-4-5-20251001
ANTHROPIC_API_KEY=sk-ant-...
```

**OpenAI**
```env
LLM_PROVIDER=openai
DEFAULT_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-proj-...
```

**OpenAI-compatible endpoint**
```env
LLM_PROVIDER=openai
DEFAULT_MODEL=deepseek-v4-flash
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_API_KEY=sk-...
```

**Google Gemini**
```env
LLM_PROVIDER=gemini
DEFAULT_MODEL=gemini-2.0-flash
GEMINI_API_KEY=AIzaSy...
```

**Ollama (local, free)**
```env
LLM_PROVIDER=ollama
DEFAULT_MODEL=gemma3:4b
```
Requires [Ollama](https://ollama.com) installed and `ollama pull gemma3:4b`.

### 3. Run

```bash
python app.py
```

Open [http://localhost:5050](http://localhost:5050), drag in a PDF, and wait ~30 seconds.

---

## Project Structure

```
.
|-- app.py                        # Flask server + /analyze endpoint
|-- streamlit_app.py              # Streamlit app with runtime provider/key selection
|-- requirements.txt              # Streamlit Cloud/root install dependencies
|-- templates/
|   `-- index.html                # Single-page UI (Tailwind CSS, vanilla JS)
`-- hiring-agent/                 # HackerRank's evaluation engine
    |-- pdf.py                    # PDF -> JSONResume via LLM section extraction
    |-- evaluator.py              # JSONResume -> EvaluationData scoring
    |-- models.py                 # Pydantic models + LLM provider classes
    |-- llm_utils.py              # Provider initialization
    |-- prompt.py                 # Model registry + API key loading
    |-- transform.py              # Data normalization utilities
    `-- prompts/templates/        # Jinja templates for each resume section
        |-- resume_evaluation_criteria.jinja   # The core scoring rubric
        |-- basics.jinja
        |-- work.jinja
        |-- education.jinja
        |-- skills.jinja
        |-- projects.jinja
        `-- awards.jinja
```

---

## LLM Provider Support

| Provider | Models | Notes |
|---|---|---|
| Anthropic | `claude-haiku-4-5-20251001`, `claude-sonnet-4-6` | Recommended, uses prefill trick for reliable JSON |
| OpenAI | `gpt-4o-mini`, `gpt-4o`, `gpt-4.1-mini`, `gpt-4.1` | Uses structured output JSON schema |
| OpenAI-compatible | DeepSeek, Groq, or any compatible chat API | Uses runtime base URL for compatible chat APIs |
| DeepSeek | `deepseek-v4-flash`, `deepseek-v4-pro`, `deepseek-chat` | `deepseek-chat` is deprecated by DeepSeek on 2026-07-24; prefer V4 models |
| Groq | `llama-3.1-8b-instant`, `llama-3.3-70b-versatile`, `gemma2-9b-it` | Free hosted Llama/Gemma option with a Groq key |
| Gemini | `gemini-2.0-flash`, `gemini-2.5-flash`, `gemini-2.5-pro` | Uses the official `google-genai` SDK |
| Ollama | `gemma3:4b`, `qwen3:4b`, `mistral:7b`, others | Fully local, no API key needed |

---

## Streamlit Deployment

The Streamlit app is the recommended public deployment because it does not require committing any API keys. Users choose a provider and paste their own key in the sidebar at runtime.

### Run Streamlit Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Deploy On Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to Streamlit Community Cloud and create a new app from the repo.
3. Set the main file path to `streamlit_app.py`.
4. Do not add your DeepSeek/OpenAI/Gemini/Anthropic key to the repository.
5. Keep `runtime.txt` in the repo so Streamlit uses Python 3.11 instead of Python 3.14.
6. Share the app URL. Each user enters their own provider, model, and API key in the app sidebar.

If dependency installation fails with `pydantic-core`, `pyarrow`, `PyO3`, or `CPython 3.14` build errors, Streamlit is using too-new a Python runtime. Confirm `runtime.txt` contains `python-3.11`, push it, then reboot/redeploy the app from Streamlit Cloud.

### Free Or Low-Cost Provider Choices

| Option | Hosted on Streamlit Cloud? | Cost model | Notes |
|---|---|---|---|
| Gemini free tier | Yes | Free quota with user's Google AI Studio key | Best hosted free default for most users |
| Groq Llama/Gemma | Yes | Free quota with user's Groq key | OpenAI-compatible, fast, no local server needed |
| Ollama | Local only by default | Free local compute | Works with `streamlit run streamlit_app.py` on a user's machine after `ollama pull gemma3:4b` |
| DeepSeek | Yes | User pays/uses their own DeepSeek quota | OpenAI-compatible endpoint |
| OpenAI/Anthropic | Yes | User pays with their own key | Good JSON reliability |

Streamlit Community Cloud cannot reliably host a local Llama/Ollama server inside the app process. For a no-cost public deployment, use Gemini free tier or Groq's free hosted Llama/Gemma options and let users provide their own free API keys.

---

## Open Source License

This project is released under the MIT License. See `LICENSE`.

---

## Credits

- Scoring rubric and evaluation pipeline: [interviewstreet/hiring-agent](https://github.com/interviewstreet/hiring-agent) by HackerRank
- PDF extraction: [PyMuPDF](https://pymupdf.readthedocs.io) + [pymupdf4llm](https://github.com/pymupdf/RAG)
- Web app fork, provider extensions, README methodology, and open-source packaging: Elroy Loo
