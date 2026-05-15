from __future__ import annotations

import tempfile
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

from src.chunker import chunk_documents
from src.document_loader import load_documents, load_documents_from_directory
from src.exporter import export_dataset_bundle
from src.generators.adversarial import generate_adversarial_cases
from src.generators.ambiguity import generate_ambiguity_cases
from src.generators.grounded_qa import generate_grounded_qa_cases
from src.generators.tool_use import generate_tool_use_cases
from src.rule_extractor import extract_rules_from_chunks
from src.schemas import Chunk, Document, EvalCase, ReviewStatus
from src.validator import (
    attach_validation_errors,
    summarize_validation_results,
    validate_cases,
)
from src.eval_runner import (
    demo_target_system,
    eval_results_to_dicts,
    intentionally_bad_target_system,
    run_eval_dataset,
)


st.set_page_config(
    page_title="EvalForge",
    page_icon="🧪",
    layout="wide",
)


def initialize_session_state() -> None:
    """
    Initialize all Streamlit session variables used by the app.
    """

    defaults = {
    "documents": [],
    "chunks": [],
    "rules": [],
    "cases": [],
    "validation_results": {},
    "quality_summary": {},
    "export_paths": {},
    "tool_schema_text": "",
    "eval_results": [],
    "eval_summary": None,
}

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def save_uploaded_files(uploaded_files) -> List[Path]:
    """
    Save Streamlit uploaded files into a temporary directory.

    The document loader expects file paths, so uploaded files need to be written
    to disk first.
    """

    temp_dir = Path(tempfile.mkdtemp(prefix="evalforge_uploads_"))
    saved_paths: List[Path] = []

    for uploaded_file in uploaded_files:
        file_path = temp_dir / uploaded_file.name
        file_path.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(file_path)

    return saved_paths


def load_sample_documents() -> Tuple[List[Document], str]:
    """
    Load sample docs from sample_docs/.

    Returns:
        documents, tool_schema_text
    """

    sample_dir = Path("sample_docs")

    if not sample_dir.exists():
        raise FileNotFoundError("sample_docs directory does not exist.")

    documents = load_documents_from_directory(sample_dir)

    tool_schema_path = sample_dir / "support_tool_schema.json"
    tool_schema_text = ""

    if tool_schema_path.exists():
        tool_schema_text = tool_schema_path.read_text(encoding="utf-8")

    return documents, tool_schema_text


def extract_tool_schema_text(documents: List[Document]) -> str:
    """
    Find tool schema text from uploaded JSON documents if available.
    """

    for document in documents:
        if "tool" in document.filename.lower() or "schema" in document.filename.lower():
            return document.text

    return ""


def generate_all_cases(
    rules,
    tool_schema_text: str,
    project_id: str,
    dataset_version: str,
    max_cases_per_type: int,
) -> List[EvalCase]:
    """
    Generate all Stage 1 case types.
    """

    cases: List[EvalCase] = []

    cases.extend(
        generate_grounded_qa_cases(
            rules=rules,
            project_id=project_id,
            dataset_version=dataset_version,
            max_cases=max_cases_per_type,
        )
    )

    cases.extend(
        generate_ambiguity_cases(
            rules=rules,
            project_id=project_id,
            dataset_version=dataset_version,
            max_cases=max_cases_per_type,
        )
    )

    cases.extend(
        generate_adversarial_cases(
            rules=rules,
            project_id=project_id,
            dataset_version=dataset_version,
            max_cases=max_cases_per_type,
        )
    )

    if tool_schema_text.strip():
        cases.extend(
            generate_tool_use_cases(
                rules=rules,
                tool_schema_text=tool_schema_text,
                project_id=project_id,
                dataset_version=dataset_version,
                max_cases=max_cases_per_type,
            )
        )

    return cases


def update_case_review_status(test_id: str, new_status: ReviewStatus) -> None:
    """
    Update one case's review status in session state.
    """

    updated_cases: List[EvalCase] = []

    for case in st.session_state.cases:
        if case.test_id == test_id:
            case_data = case.model_dump()
            case_data["review_status"] = new_status
            updated_cases.append(EvalCase(**case_data))
        else:
            updated_cases.append(case)

    st.session_state.cases = updated_cases


def cases_to_dataframe(cases: List[EvalCase]) -> pd.DataFrame:
    """
    Convert cases to a simple display table.
    """

    rows = []

    for case in cases:
        rows.append(
            {
                "test_id": case.test_id,
                "test_type": case.test_type.value,
                "risk_level": case.risk_level.value,
                "review_status": case.review_status.value,
                "query": case.user_query,
                "citations": ", ".join(
                    citation.chunk_id for citation in case.required_citations
                ),
                "errors": len(case.validation_errors),
            }
        )

    return pd.DataFrame(rows)

def eval_results_to_dataframe(eval_results) -> pd.DataFrame:
    """
    Convert eval results to a display table for Streamlit.
    """

    rows = []

    for result in eval_results:
        rows.append(
            {
                "test_id": result.test_id,
                "test_type": result.test_type.value,
                "passed": result.passed,
                "overall_score": result.grade.overall_score,
                "faithfulness": result.grade.faithfulness,
                "answer_relevance": result.grade.answer_relevance,
                "citation_accuracy": result.grade.citation_accuracy,
                "policy_correctness": result.grade.policy_correctness,
                "refusal_correctness": result.grade.refusal_correctness,
                "clarification_correctness": result.grade.clarification_correctness,
                "tool_call_correctness": result.grade.tool_call_correctness,
                "safety": result.grade.safety,
                "latency_ms": result.target_response.latency_ms,
                "answer_preview": result.target_response.answer[:180],
            }
        )

    return pd.DataFrame(rows)


def render_header() -> None:
    st.title("🧪 EvalForge")
    st.caption(
        "Autonomous benchmark generator for RAG systems and AI-agent workflows. "
        "Stage 1 MVP: documents → chunks → rules → eval cases → validation → review → export."
    )


def render_sidebar() -> Dict[str, object]:
    st.sidebar.header("Generation Settings")

    project_id = st.sidebar.text_input("Project ID", value="support_demo")
    dataset_version = st.sidebar.text_input("Dataset Version", value="v0.1.0")

    max_chars = st.sidebar.slider(
        "Chunk max characters",
        min_value=300,
        max_value=2000,
        value=700,
        step=100,
    )

    overlap_chars = st.sidebar.slider(
        "Chunk overlap characters",
        min_value=0,
        max_value=300,
        value=100,
        step=25,
    )

    max_cases_per_type = st.sidebar.slider(
        "Max cases per test type",
        min_value=1,
        max_value=30,
        value=8,
        step=1,
    )

    citation_threshold = st.sidebar.slider(
        "Citation support threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05,
    )

    st.sidebar.divider()

    st.sidebar.markdown("### Current Session")
    st.sidebar.write(f"Documents: **{len(st.session_state.documents)}**")
    st.sidebar.write(f"Chunks: **{len(st.session_state.chunks)}**")
    st.sidebar.write(f"Rules: **{len(st.session_state.rules)}**")
    st.sidebar.write(f"Cases: **{len(st.session_state.cases)}**")

    return {
        "project_id": project_id,
        "dataset_version": dataset_version,
        "max_chars": max_chars,
        "overlap_chars": overlap_chars,
        "max_cases_per_type": max_cases_per_type,
        "citation_threshold": citation_threshold,
    }


def render_ingestion_tab(settings: Dict[str, object]) -> None:
    st.subheader("1. Source Ingestion")

    st.write(
        "Upload Markdown, TXT, or JSON or CSV or pdf files. For the sample support-agent demo, "
        "you can also load the built-in sample docs."
    )

    uploaded_files = st.file_uploader(
        "Upload source documents",
        type=["md", "txt", "json", "csv", "pdf"],
        accept_multiple_files=True,
    )

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("Load Uploaded Documents", use_container_width=True):
            if not uploaded_files:
                st.warning("Upload at least one document first.")
            else:
                saved_paths = save_uploaded_files(uploaded_files)
                documents = load_documents(saved_paths)

                st.session_state.documents = documents
                st.session_state.tool_schema_text = extract_tool_schema_text(documents)

                st.success(f"Loaded {len(documents)} uploaded document(s).")

    with col_b:
        if st.button("Load Sample Support Docs", use_container_width=True):
            try:
                documents, tool_schema_text = load_sample_documents()
                st.session_state.documents = documents
                st.session_state.tool_schema_text = tool_schema_text
                st.success(f"Loaded {len(documents)} sample document(s).")
            except Exception as exc:
                st.error(f"Failed to load sample docs: {exc}")

    if st.session_state.documents:
        st.markdown("### Loaded Documents")

        doc_rows = []

        for document in st.session_state.documents:
            doc_rows.append(
                {
                    "source_id": document.source_id,
                    "filename": document.filename,
                    "type": document.source_type.value,
                    "chars": document.metadata.get("char_count"),
                    "checksum": str(document.metadata.get("checksum_sha256", ""))[:12],
                }
            )

        st.dataframe(pd.DataFrame(doc_rows), use_container_width=True)

        with st.expander("Preview loaded document text"):
            for document in st.session_state.documents:
                st.markdown(f"#### {document.filename}")
                st.code(document.text[:2000])


def render_processing_tab(settings: Dict[str, object]) -> None:
    st.subheader("2. Chunking and Rule Extraction")

    if not st.session_state.documents:
        st.info("Load documents first.")
        return

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("Create Chunks", use_container_width=True):
            chunks = chunk_documents(
                st.session_state.documents,
                max_chars=int(settings["max_chars"]),
                overlap_chars=int(settings["overlap_chars"]),
            )
            st.session_state.chunks = chunks
            st.success(f"Created {len(chunks)} chunk(s).")

    with col_b:
        if st.button("Extract Rules", use_container_width=True):
            if not st.session_state.chunks:
                st.warning("Create chunks before extracting rules.")
            else:
                rules = extract_rules_from_chunks(st.session_state.chunks)
                st.session_state.rules = rules
                st.success(f"Extracted {len(rules)} rule(s).")

    if st.session_state.chunks:
        st.markdown("### Source Chunks")

        chunk_rows = []
        for chunk in st.session_state.chunks:
            chunk_rows.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "section": chunk.section,
                    "chars": len(chunk.text),
                }
            )

        st.dataframe(pd.DataFrame(chunk_rows), use_container_width=True)

        with st.expander("Inspect chunk text"):
            for chunk in st.session_state.chunks:
                st.markdown(f"#### {chunk.chunk_id} — {chunk.section or 'No Section'}")
                st.caption(chunk.source)
                st.write(chunk.text)

    if st.session_state.rules:
        st.markdown("### Extracted Rules")

        rule_rows = []

        for rule in st.session_state.rules:
            rule_rows.append(
                {
                    "rule_id": rule.rule_id,
                    "chunk_id": rule.source_chunk_id,
                    "section": rule.section,
                    "type": rule.rule_type.value,
                    "risk": rule.risk_level.value,
                    "condition": rule.condition,
                    "expected_action": rule.expected_action,
                }
            )

        st.dataframe(pd.DataFrame(rule_rows), use_container_width=True)

        with st.expander("Inspect rule text"):
            for rule in st.session_state.rules:
                st.markdown(f"#### {rule.rule_id}")
                st.caption(f"{rule.rule_type.value} · {rule.risk_level.value}")
                st.write(rule.rule_text)


def render_generation_tab(settings: Dict[str, object]) -> None:
    st.subheader("3. Generate Evaluation Cases")

    if not st.session_state.rules:
        st.info("Extract rules first.")
        return

    st.write(
        "Generate Stage 1 test cases: grounded QA, ambiguity, adversarial, and tool-use correctness."
    )

    if st.session_state.tool_schema_text:
        st.success("Tool schema detected. Tool-use cases will be generated.")
    else:
        st.warning("No tool schema detected. Tool-use cases will be skipped.")

    if st.button("Generate Benchmark Cases", type="primary", use_container_width=True):
        cases = generate_all_cases(
            rules=st.session_state.rules,
            tool_schema_text=st.session_state.tool_schema_text,
            project_id=str(settings["project_id"]),
            dataset_version=str(settings["dataset_version"]),
            max_cases_per_type=int(settings["max_cases_per_type"]),
        )

        validation_results = validate_cases(
            cases=cases,
            chunks=st.session_state.chunks,
            citation_support_threshold=float(settings["citation_threshold"]),
        )

        cases_with_errors = attach_validation_errors(cases, validation_results)
        quality_summary = summarize_validation_results(cases_with_errors, validation_results)

        st.session_state.cases = cases_with_errors
        st.session_state.validation_results = validation_results
        st.session_state.quality_summary = quality_summary

        st.success(f"Generated {len(cases_with_errors)} evaluation case(s).")

    if st.session_state.cases:
        st.markdown("### Generated Case Overview")
        st.dataframe(cases_to_dataframe(st.session_state.cases), use_container_width=True)


def render_validation_tab() -> None:
    st.subheader("4. Validation and Quality Report")

    if not st.session_state.cases:
        st.info("Generate cases first.")
        return

    summary = st.session_state.quality_summary

    if not summary:
        st.warning("No quality summary available.")
        return

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Cases", summary.get("total_cases", 0))
    col2.metric("Valid Cases", summary.get("valid_cases", 0))
    col3.metric("Validity Rate", summary.get("validity_rate", 0))
    col4.metric("Citation Coverage", summary.get("citation_coverage", 0))

    col5, col6 = st.columns(2)

    with col5:
        st.markdown("### Test Type Distribution")
        test_type_distribution = summary.get("test_type_distribution", {})
        if test_type_distribution:
            st.bar_chart(pd.Series(test_type_distribution))

    with col6:
        st.markdown("### Risk Distribution")
        risk_distribution = summary.get("risk_distribution", {})
        if risk_distribution:
            st.bar_chart(pd.Series(risk_distribution))

    st.markdown("### Validation Details")

    validation_rows = []

    for case in st.session_state.cases:
        result = st.session_state.validation_results.get(case.test_id)
        validation_rows.append(
            {
                "test_id": case.test_id,
                "valid": result.is_valid if result else None,
                "errors": "; ".join(result.errors) if result else "",
                "warnings": "; ".join(result.warnings) if result else "",
            }
        )

    st.dataframe(pd.DataFrame(validation_rows), use_container_width=True)


def render_review_tab() -> None:
    st.subheader("5. Human Review Queue")

    if not st.session_state.cases:
        st.info("Generate cases first.")
        return

    filter_status = st.selectbox(
        "Filter by review status",
        options=[
            "all",
            ReviewStatus.PENDING.value,
            ReviewStatus.APPROVED.value,
            ReviewStatus.REJECTED.value,
            ReviewStatus.NEEDS_FIX.value,
        ],
    )

    filtered_cases = st.session_state.cases

    if filter_status != "all":
        filtered_cases = [
            case for case in filtered_cases if case.review_status.value == filter_status
        ]

    st.write(f"Showing **{len(filtered_cases)}** case(s).")

    for case in filtered_cases:
        with st.expander(
            f"{case.test_id} · {case.test_type.value} · {case.risk_level.value} · {case.review_status.value}",
            expanded=False,
        ):
            st.markdown("#### User Query")
            st.write(case.user_query)

            st.markdown("#### Expected Behavior")
            st.write(case.expected_behavior)

            st.markdown("#### Expected Answer Outline")
            for item in case.expected_answer_outline:
                st.markdown(f"- {item}")

            st.markdown("#### Required Citations")
            for citation in case.required_citations:
                st.markdown(
                    f"- `{citation.chunk_id}` from **{citation.source}** — {citation.required_evidence}"
                )

            st.markdown("#### Disallowed Behaviors")
            for behavior in case.disallowed_behaviors:
                st.markdown(f"- {behavior}")

            if case.tool_expectation:
                st.markdown("#### Tool Expectation")
                st.json(case.tool_expectation.model_dump(mode="json"))

            if case.validation_errors:
                st.error("Validation Errors")
                for error in case.validation_errors:
                    st.markdown(f"- {error}")

            st.markdown("#### Tags")
            st.write(", ".join(case.tags))

            col_a, col_b, col_c = st.columns(3)

            with col_a:
                if st.button(
                    "Approve",
                    key=f"approve_{case.test_id}",
                    use_container_width=True,
                ):
                    update_case_review_status(case.test_id, ReviewStatus.APPROVED)
                    st.rerun()

            with col_b:
                if st.button(
                    "Needs Fix",
                    key=f"needs_fix_{case.test_id}",
                    use_container_width=True,
                ):
                    update_case_review_status(case.test_id, ReviewStatus.NEEDS_FIX)
                    st.rerun()

            with col_c:
                if st.button(
                    "Reject",
                    key=f"reject_{case.test_id}",
                    use_container_width=True,
                ):
                    update_case_review_status(case.test_id, ReviewStatus.REJECTED)
                    st.rerun()


def render_export_tab() -> None:
    st.subheader("6. Export Dataset")

    if not st.session_state.cases:
        st.info("Generate and review cases first.")
        return

    dataset_name = st.text_input("Dataset export name", value="support_eval_stage1")

    approved_only = st.checkbox("Export approved cases only", value=False)

    if st.button("Export Dataset Bundle", type="primary", use_container_width=True):
        paths = export_dataset_bundle(
            cases=st.session_state.cases,
            output_dir="outputs",
            dataset_name=dataset_name,
            quality_summary=st.session_state.quality_summary,
            approved_only=approved_only,
        )

        st.session_state.export_paths = paths
        st.success("Export complete.")

    if st.session_state.export_paths:
        st.markdown("### Exported Files")

        for name, path in st.session_state.export_paths.items():
            st.write(f"**{name}**: `{path}`")

        st.markdown("### Download Files")

        for name, path in st.session_state.export_paths.items():
            path = Path(path)

            if path.exists():
                st.download_button(
                    label=f"Download {name}",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime="application/octet-stream",
                    use_container_width=True,
                )

def render_eval_runner_tab() -> None:
    st.subheader("7. Evaluation Runner")

    if not st.session_state.cases:
        st.info("Generate cases first.")
        return

    st.write(
        "Run the generated benchmark against a demo target system. "
        "The good demo target follows the case structure, while the bad demo target ignores policies and citations."
    )

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        target_choice = st.selectbox(
            "Target system",
            options=[
                "demo_target_system",
                "intentionally_bad_target_system",
            ],
        )

    with col_b:
        pass_threshold = st.slider(
            "Pass threshold",
            min_value=0.10,
            max_value=0.95,
            value=0.70,
            step=0.05,
        )

    with col_c:
        eval_approved_only = st.checkbox(
            "Evaluate approved cases only",
            value=False,
        )

    eval_cases = st.session_state.cases

    if eval_approved_only:
        eval_cases = [
            case for case in eval_cases if case.review_status == ReviewStatus.APPROVED
        ]

    st.write(f"Cases selected for evaluation: **{len(eval_cases)}**")

    if not eval_cases:
        st.warning("No cases match the selected evaluation filter.")
        return

    if st.button("Run Evaluation", type="primary", use_container_width=True):
        if target_choice == "demo_target_system":
            target_fn = demo_target_system
        else:
            target_fn = intentionally_bad_target_system

        results, summary = run_eval_dataset(
            cases=eval_cases,
            target_system=target_fn,
            target_system_name=target_choice,
            dataset_version=str(eval_cases[0].dataset_version if eval_cases else "v0.1.0"),
            pass_threshold=float(pass_threshold),
        )

        st.session_state.eval_results = results
        st.session_state.eval_summary = summary

        st.success(f"Evaluation complete. Ran {summary.total_cases} case(s).")

    if st.session_state.eval_summary is not None:
        summary = st.session_state.eval_summary

        st.markdown("### Evaluation Summary")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Cases", summary.total_cases)
        col2.metric("Passed", summary.passed_cases)
        col3.metric("Pass Rate", summary.pass_rate)
        col4.metric("Average Score", summary.average_score)

        st.markdown("### Score by Test Type")

        if summary.score_by_test_type:
            st.bar_chart(pd.Series(summary.score_by_test_type))
        else:
            st.info("No test-type score breakdown available.")

        st.markdown("### Evaluation Results")

        if st.session_state.eval_results:
            df = eval_results_to_dataframe(st.session_state.eval_results)
            st.dataframe(df, use_container_width=True)

            with st.expander("Inspect detailed responses"):
                for result in st.session_state.eval_results:
                    st.markdown(f"#### {result.test_id}")
                    st.caption(
                        f"{result.test_type.value} · passed={result.passed} · score={result.grade.overall_score}"
                    )

                    st.markdown("**User query**")
                    st.write(result.user_query)

                    st.markdown("**Target answer**")
                    st.write(result.target_response.answer)

                    st.markdown("**Citations returned**")
                    st.write(result.target_response.citations)

                    if result.target_response.tool_calls:
                        st.markdown("**Tool calls**")
                        st.json(result.target_response.tool_calls)

                    st.markdown("**Grade**")
                    st.json(result.grade.model_dump(mode="json"))

        st.markdown("### Download Evaluation Results")

        eval_payload = {
            "summary": summary.model_dump(mode="json"),
            "results": eval_results_to_dicts(st.session_state.eval_results),
        }

        st.download_button(
            label="Download eval_run_results.json",
            data=json.dumps(eval_payload, indent=2).encode("utf-8"),
            file_name="eval_run_results.json",
            mime="application/json",
            use_container_width=True,
        )


def main() -> None:
    initialize_session_state()
    render_header()

    settings = render_sidebar()

    tabs = st.tabs(
    [
        "1. Ingest",
        "2. Process",
        "3. Generate",
        "4. Validate",
        "5. Review",
        "6. Export",
        "7. Eval Runner",
    ]
)

    with tabs[0]:
        render_ingestion_tab(settings)

    with tabs[1]:
        render_processing_tab(settings)

    with tabs[2]:
        render_generation_tab(settings)

    with tabs[3]:
        render_validation_tab()

    with tabs[4]:
        render_review_tab()

    with tabs[5]:
        render_export_tab()
    
    with tabs[6]:
        render_eval_runner_tab()


if __name__ == "__main__":
    main()