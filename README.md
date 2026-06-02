# agentic-content-pipeline

A multi-agent content publishing pipeline built on **LangGraph**. It researches a topic against
real web sources, writes an article, fact-checks every claim against those sources, scores SEO,
and loops back through an editor agent until quality thresholds pass — then a human approves and
it exports a Medium-ready article.

> **Status:** under active construction (built milestone-by-milestone). This README is a skeleton
> and is fully fleshed out in the final milestone (architecture diagram, screenshots, design
> decisions).

## Architecture (at a glance)

```
research → writer → fact_check → seo → [ROUTER] ─▶ editor ─(re-validate)─┐
                                          │                              │
                                          └────────────────▶ publisher (human approval gate)
```

Six agents share one typed state object. A conditional **router** after the SEO agent decides
whether to loop back to the **editor** (low fact-check or SEO score) or proceed to the
**publisher**. A loop guard caps revisions so the graph always terminates.

## Quick start (Windows / PowerShell)

```powershell
# 1. Create and activate a virtual environment (Python 3.11+)
py -3.12 -m venv .venv          # or: python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure keys
copy .env.example .env          # then edit .env and add OPENAI_API_KEY + TAVILY_API_KEY

# 4. Run (Streamlit demo — added in a later milestone)
streamlit run app.py
```

No keys handy? Run everything offline with deterministic fakes:

```powershell
$env:PIPELINE_MOCK = "1"
pytest
```

## Configuration

All thresholds, model names, and the loop guard live in [`pipeline/config.py`](pipeline/config.py)
and can be overridden via environment variables (see [`.env.example`](.env.example)).

