# EvalForge

**EvalForge** is a backend service and dashboard for generating and running evaluation benchmarks for RAG systems and AI-agent workflows.

Instead of manually writing evaluation questions one by one, EvalForge lets users upload source documents, policy files, CSVs, PDFs, and optional tool schemas. It automatically ingests the files, chunks the documents, extracts rules, generates benchmark cases, validates citations, exports datasets, runs evaluations, stores results, and compares runs for regressions.

---

## Why EvalForge Exists

Many RAG systems and AI agents are tested with only a few manual questions before deployment. That is risky because the system may:

- answer with unsupported claims,
- cite the wrong source,
- fail to ask clarifying questions,
- refuse when it should answer,
- answer when it should refuse,
- call the wrong tool,
- miss required tool arguments,
- regress after a prompt, model, or retriever update.

EvalForge solves this by creating structured, source-grounded evaluation datasets from the same files used by the RAG system or AI agent.

---

## Core Workflow

User uploads files:

```text
refund_policy.md
shipping_policy.md
warranty_policy.md
support_tool_schema.json
Customer_support_data.csv
policy_manual.pdf
```

EvalForge automatically runs:

```text
file upload
  → document loading
  → chunking
  → rule extraction
  → benchmark generation
  → citation validation
  → optional evaluation run
  → SQLite persistence
  → JSON / JSONL / CSV exports
  → run comparison
```

---

## Main Features

- Upload `.md`, `.txt`, `.json`, `.csv`, and `.pdf` files.
- Generate grounded QA test cases.
- Generate ambiguity / clarification test cases.
- Generate adversarial / policy-bypass test cases.
- Generate tool-use correctness test cases from JSON tool schemas.
- Validate schema correctness and citation coverage.
- Export benchmark datasets as JSON, JSONL, and CSV.
- Run evaluations against demo target systems.
- Run evaluations against real HTTP RAG/agent endpoints.
- Store pipeline runs in SQLite.
- View persisted run history.
- Compare two runs to detect regressions.
- Use a browser dashboard or FastAPI Swagger docs.
- Run with Docker Compose.

---

## Dashboard

Start the backend, then open:

```text
http://127.0.0.1:8000/dashboard
```

The dashboard supports:

- file upload,
- full pipeline execution,
- quality metrics,
- evaluation metrics,
- generated case preview,
- persisted run table,
- export links,
- run comparison.

---

## Architecture

```text
EvalForge
│
├── FastAPI backend
│   ├── /pipeline/run
│   ├── /pipeline/runs
│   ├── /pipeline/runs/{run_id}/cases
│   ├── /pipeline/runs/{run_id}/export/json
│   ├── /pipeline/runs/{run_id}/export/jsonl
│   ├── /pipeline/runs/{run_id}/export/csv
│   ├── /pipeline/runs/{run_id}/quality-report
│   ├── /pipeline/runs/{run_id}/compare/{baseline_run_id}
│   └── /eval-runs/http-target
│
├── Dashboard frontend
│   └── /dashboard
│
├── Streamlit MVP
│   └── app.py
│
├── Core source modules
│   ├── document_loader.py
│   ├── chunker.py
│   ├── rule_extractor.py
│   ├── generators/
│   ├── validator.py
│   ├── exporter.py
│   └── eval_runner.py
│
├── SQLite persistence
│   └── data/evalforge.db
│
└── Docker Compose
```

---

## Supported Input Files

EvalForge supports:

```text
.md
.txt
.json
.csv
.pdf
```

Typical inputs:

```text
policy manuals
refund policies
shipping policies
warranty policies
employee handbooks
FAQ files
terms of service
customer support CSVs
tool schema JSON files
API/workflow specs
```

For large CSV files, EvalForge limits ingestion by default so the pipeline remains fast and reviewable.

---

## Generated Test Types

| Test Type | Purpose |
|---|---|
| Grounded QA | Checks whether the system answers using source documents. |
| Ambiguity | Checks whether the system asks clarification instead of guessing. |
| Adversarial | Checks whether the system resists policy-bypass or prompt-injection style requests. |
| Tool-use correctness | Checks whether an agent selects the right tool and handles missing arguments. |

---

## Example Generated Case

```json
{
  "test_id": "refund_policy_0001_grounded_001",
  "test_type": "grounded_policy_qa",
  "risk_level": "medium",
  "user_query": "My package arrived more than 7 days late. Can I get a shipping refund?",
  "expected_behavior": "Answer using the late-delivery refund policy and cite the source.",
  "expected_answer_outline": [
    "Customer may be eligible for a shipping-fee refund.",
    "Delivery must be delayed by more than 7 days.",
    "Customer must not already have received compensation."
  ],
  "required_citations": [
    {
      "chunk_id": "refund_policy_0001",
      "source": "refund_policy.md",
      "required_evidence": "Late delivery refund eligibility."
    }
  ],
  "review_status": "pending_review"
}
```

---

## Local Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the backend:

```bash
uvicorn backend.app.main:app --reload --port 8000
```

Open the dashboard:

```text
http://127.0.0.1:8000/dashboard
```

Open Swagger docs:

```text
http://127.0.0.1:8000/docs
```

Run the Streamlit MVP:

```bash
streamlit run app.py
```

Open Streamlit:

```text
http://127.0.0.1:8501
```

---

## Docker Setup

Build and start everything:

```bash
docker compose up --build
```

Then open:

```text
Backend dashboard: http://127.0.0.1:8000/dashboard
Swagger docs:       http://127.0.0.1:8000/docs
Streamlit app:      http://127.0.0.1:8501
```

Stop containers:

```bash
docker compose down
```

---

## One-Shot Pipeline API

The main endpoint is:

```text
POST /pipeline/run
```

Example:

```bash
curl -X POST "http://127.0.0.1:8000/pipeline/run" \
  -F "files=@sample_docs/refund_policy.md" \
  -F "files=@sample_docs/shipping_policy.md" \
  -F "files=@sample_docs/warranty_policy.md" \
  -F "files=@sample_docs/support_tool_schema.json" \
  -F "project_id=support_demo" \
  -F "dataset_version=v0.1.0" \
  -F "max_cases_per_type=5" \
  -F "citation_support_threshold=0.35" \
  -F "run_eval=true" \
  -F "target_system=demo_target_system" \
  -F "pass_threshold=0.70"
```

Example response:

```json
{
  "project_id": "support_demo",
  "dataset_version": "v0.1.0",
  "pipeline_run_id": "pipeline_abc123",
  "document_count": 4,
  "chunk_count": 11,
  "rule_count": 16,
  "case_count": 20,
  "quality_summary": {
    "total_cases": 20,
    "valid_cases": 20,
    "validity_rate": 1.0,
    "citation_coverage": 1.0
  },
  "eval_summary": {
    "target_system": "demo_target_system",
    "pass_rate": 1.0,
    "average_score": 0.86
  }
}
```

---

## Persisted Runs

List saved runs:

```bash
curl "http://127.0.0.1:8000/pipeline/runs"
```

Get one run:

```bash
curl "http://127.0.0.1:8000/pipeline/runs/YOUR_RUN_ID"
```

Get generated cases:

```bash
curl "http://127.0.0.1:8000/pipeline/runs/YOUR_RUN_ID/cases"
```

Export JSON:

```bash
curl "http://127.0.0.1:8000/pipeline/runs/YOUR_RUN_ID/export/json"
```

Export JSONL:

```bash
curl "http://127.0.0.1:8000/pipeline/runs/YOUR_RUN_ID/export/jsonl"
```

Export CSV:

```bash
curl "http://127.0.0.1:8000/pipeline/runs/YOUR_RUN_ID/export/csv"
```

Quality report:

```bash
curl "http://127.0.0.1:8000/pipeline/runs/YOUR_RUN_ID/quality-report"
```

---

## Run Comparison

Compare a current run against a baseline run:

```bash
curl "http://127.0.0.1:8000/pipeline/runs/CURRENT_RUN_ID/compare/BASELINE_RUN_ID"
```

EvalForge returns metric deltas and flags regressions.

Example:

```json
{
  "current_run_id": "pipeline_good",
  "baseline_run_id": "pipeline_bad",
  "regression_detected": false,
  "metric_comparison": {
    "average_score": {
      "current": 0.86,
      "baseline": 0.31,
      "delta": 0.55
    }
  }
}
```

---

## HTTP Target Evaluation

EvalForge can evaluate a real RAG or AI-agent HTTP endpoint.

Start the fake target server:

```bash
uvicorn scripts.fake_rag_target:app --reload --port 9000
```

Generate cases first using `/pipeline/run`, then run:

```bash
curl -X POST "http://127.0.0.1:8000/eval-runs/http-target" \
  -H "Content-Type: application/json" \
  -d '{
    "target": {
      "target_url": "http://127.0.0.1:9000/answer",
      "method": "POST",
      "request_field": "question",
      "response_answer_field": "answer",
      "response_citations_field": "citations",
      "response_tool_calls_field": "tool_calls",
      "timeout_seconds": 30
    },
    "pass_threshold": 0.70
  }'
```

---

## Running Tests

Run all tests:

```bash
python -m pytest -q
```

Run selected backend tests:

```bash
python -m pytest tests/test_pipeline_api.py -q
python -m pytest tests/test_persistence.py -q
python -m pytest tests/test_persisted_exports.py -q
python -m pytest tests/test_run_comparison.py -q
python -m pytest tests/test_dashboard.py -q
```

---

## Sample Documents

The repository includes sample support-policy files:

```text
sample_docs/refund_policy.md
sample_docs/shipping_policy.md
sample_docs/warranty_policy.md
sample_docs/support_tool_schema.json
```

These are useful for quickly testing the full EvalForge pipeline.

---

## Current Limitations

- Generation is mostly deterministic/template-based in the current version.
- Human review exists through review status and case metadata, but a full edit/approve/reject workflow can be expanded.
- Current automated grading is a hybrid deterministic scoring demo, not a calibrated LLM-as-judge system.
- Large CSV files are sampled by default to keep generation fast and reviewable.
- High-stakes domains such as healthcare, legal, and finance require expert review before using generated benchmarks.

---

## Future Improvements

- Add more test types: citation alignment, boundary-condition, refusal-specific, and multi-hop tests.
- Add a richer human review queue with edit/approve/reject actions in the dashboard.
- Add embedding-based citation verification.
- Add LLM-based case generation with strict Pydantic validation.
- Add PostgreSQL support.
- Add GitHub Actions CI regression gate.
- Add React frontend.
- Add failure clustering and difficulty scoring.

---

## Resume Summary

Built **EvalForge**, an AI evaluation-infrastructure platform that converts documents, policies, PDFs, CSVs, and tool schemas into source-grounded benchmark datasets for RAG systems and AI-agent workflows. Implemented document ingestion, chunking, rule extraction, multi-type test generation, citation validation, evaluation scoring, HTTP target evaluation, SQLite persistence, run comparison, dataset exports, FastAPI APIs, dashboard UI, Docker Compose, and pytest coverage.
