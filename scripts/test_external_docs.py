from pathlib import Path
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.chunker import chunk_documents
from src.document_loader import load_documents_from_directory
from src.eval_runner import demo_target_system, intentionally_bad_target_system, run_eval_dataset
from src.exporter import export_dataset_bundle
from src.generators.adversarial import generate_adversarial_cases
from src.generators.ambiguity import generate_ambiguity_cases
from src.generators.grounded_qa import generate_grounded_qa_cases
from src.generators.tool_use import generate_tool_use_cases
from src.rule_extractor import extract_rules_from_chunks
from src.validator import summarize_validation_results, validate_cases


DOC_DIR = Path("external_test_docs")
OUTPUT_DIR = Path("outputs/external_test")


def main() -> None:
    documents = load_documents_from_directory(DOC_DIR)
    chunks = chunk_documents(documents, max_chars=700, overlap_chars=100)
    rules = extract_rules_from_chunks(chunks)

    tool_schema_text = ""
    for doc in documents:
        if "schema" in doc.filename.lower() or "tool" in doc.filename.lower():
            tool_schema_text = doc.text
            break

    cases = []
    cases.extend(generate_grounded_qa_cases(rules, max_cases=10))
    cases.extend(generate_ambiguity_cases(rules, max_cases=10))
    cases.extend(generate_adversarial_cases(rules, max_cases=10))

    if tool_schema_text:
        cases.extend(
            generate_tool_use_cases(
                rules=rules,
                tool_schema_text=tool_schema_text,
                max_cases=10,
            )
        )

    validation_results = validate_cases(cases, chunks)
    summary = summarize_validation_results(cases, validation_results)

    export_paths = export_dataset_bundle(
        cases=cases,
        output_dir=OUTPUT_DIR,
        dataset_name="external_policy_eval",
        quality_summary=summary,
    )

    good_results, good_summary = run_eval_dataset(
        cases,
        target_system=demo_target_system,
        target_system_name="demo_target_system",
    )

    bad_results, bad_summary = run_eval_dataset(
        cases,
        target_system=intentionally_bad_target_system,
        target_system_name="intentionally_bad_target_system",
    )

    print("=" * 80)
    print("External Document EvalForge Test")
    print("=" * 80)
    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created:   {len(chunks)}")
    print(f"Rules extracted:  {len(rules)}")
    print(f"Cases generated:  {len(cases)}")
    print(f"Validity rate:    {summary['validity_rate']}")
    print(f"Citation coverage:{summary['citation_coverage']}")
    print()
    print("Generated test type distribution:")
    for test_type, count in summary["test_type_distribution"].items():
        print(f"  - {test_type}: {count}")
    print()
    print("Good target average score:", good_summary.average_score)
    print("Bad target average score: ", bad_summary.average_score)
    print()
    print("Exported files:")
    for name, path in export_paths.items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()