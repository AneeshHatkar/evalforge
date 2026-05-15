from pathlib import Path

from src.chunker import chunk_documents
from src.document_loader import load_documents_from_directory
from src.eval_runner import demo_target_system, intentionally_bad_target_system, run_eval_dataset
from src.exporter import export_dataset_bundle
from src.generators.adversarial import generate_adversarial_cases
from src.generators.ambiguity import generate_ambiguity_cases
from src.generators.grounded_qa import generate_grounded_qa_cases
from src.generators.tool_use import generate_tool_use_cases
from src.rule_extractor import extract_rules_from_chunks
from src.validator import (
    split_valid_invalid_cases,
    summarize_validation_results,
    validate_cases,
)


def test_stage1_full_pipeline_from_sample_docs_to_eval_run(tmp_path: Path):
    """
    End-to-end Stage 1 pipeline test.

    This proves EvalForge can:
    1. Load sample support docs
    2. Chunk documents
    3. Extract rules
    4. Generate multiple case types
    5. Validate generated cases
    6. Export dataset bundle
    7. Run eval against good and bad demo targets
    """

    sample_dir = Path("sample_docs")
    tool_schema_path = sample_dir / "support_tool_schema.json"

    assert sample_dir.exists(), "sample_docs directory is missing"
    assert tool_schema_path.exists(), "support_tool_schema.json is missing"

    documents = load_documents_from_directory(sample_dir)

    assert len(documents) >= 4

    chunks = chunk_documents(
        documents=documents,
        max_chars=700,
        overlap_chars=100,
    )

    assert len(chunks) > 0
    assert all(chunk.chunk_id for chunk in chunks)
    assert all(chunk.source for chunk in chunks)

    rules = extract_rules_from_chunks(chunks)

    assert len(rules) > 0
    assert all(rule.source_chunk_id for rule in rules)

    tool_schema_text = tool_schema_path.read_text(encoding="utf-8")

    cases = []
    cases.extend(generate_grounded_qa_cases(rules, max_cases=5))
    cases.extend(generate_ambiguity_cases(rules, max_cases=5))
    cases.extend(generate_adversarial_cases(rules, max_cases=5))
    cases.extend(
        generate_tool_use_cases(
            rules=rules,
            tool_schema_text=tool_schema_text,
            max_cases=5,
        )
    )

    assert len(cases) > 0

    generated_types = {case.test_type.value for case in cases}

    assert "grounded_policy_qa" in generated_types
    assert "ambiguity" in generated_types
    assert "adversarial" in generated_types
    assert "tool_use_correctness" in generated_types

    validation_results = validate_cases(cases, chunks)
    summary = summarize_validation_results(cases, validation_results)

    assert summary["total_cases"] == len(cases)
    assert summary["citation_coverage"] == 1.0

    valid_cases, invalid_cases = split_valid_invalid_cases(cases, validation_results)

    assert len(valid_cases) > 0

    # We do not require zero invalid cases because deterministic citation scoring
    # can be conservative, but most generated cases should be valid.
    assert len(valid_cases) >= len(cases) * 0.70

    export_paths = export_dataset_bundle(
        cases=cases,
        output_dir=tmp_path,
        dataset_name="integration_eval_dataset",
        quality_summary=summary,
    )

    assert export_paths["json"].exists()
    assert export_paths["jsonl"].exists()
    assert export_paths["csv"].exists()
    assert export_paths["quality_report"].exists()

    good_results, good_summary = run_eval_dataset(
        valid_cases,
        target_system=demo_target_system,
        target_system_name="demo_target_system",
        dataset_version="v0.1.0",
    )

    bad_results, bad_summary = run_eval_dataset(
        valid_cases,
        target_system=intentionally_bad_target_system,
        target_system_name="intentionally_bad_target_system",
        dataset_version="v0.1.0",
    )

    assert len(good_results) == len(valid_cases)
    assert len(bad_results) == len(valid_cases)

    assert good_summary.total_cases == len(valid_cases)
    assert bad_summary.total_cases == len(valid_cases)

    assert good_summary.average_score > bad_summary.average_score