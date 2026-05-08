"""
run_all.py — Run all Day 22 lab steps sequentially.

Usage:
    python run_all.py            # run all steps (1 → 4)
    python run_all.py --step 3   # run only step 3
    python run_all.py --step 1 2 # run steps 1 and 2
"""

import sys
import time
import argparse
import traceback
from pathlib import Path

# Ensure the project root is on the Python path
sys.path.insert(0, str(Path(__file__).parent))


def run_step(step_number: int) -> bool:
    """
    Import and run the main() function of the requested step module.

    Returns True if the step completed without raising an exception.
    """
    step_modules = {
        1: ("01_langsmith_rag_pipeline",  "Step 1 — LangSmith RAG Pipeline"),
        2: ("02_prompt_hub_ab_routing",   "Step 2 — Prompt Hub & A/B Routing"),
        3: ("03_ragas_evaluation",        "Step 3 — RAGAS Evaluation"),
        4: ("04_guardrails_validator",    "Step 4 — Guardrails Validators"),
    }

    if step_number not in step_modules:
        print(f"❌ Unknown step: {step_number}. Valid steps: 1–4")
        return False

    module_name, description = step_modules[step_number]

    print("\n" + "█" * 62)
    print(f"  RUNNING {description}")
    print("█" * 62 + "\n")

    t_start = time.time()
    try:
        module = __import__(module_name)
        module.main()
        elapsed = time.time() - t_start
        print(f"\n✅ {description} completed in {elapsed:.1f}s")
        return True
    except Exception as exc:
        elapsed = time.time() - t_start
        print(f"\n❌ {description} FAILED after {elapsed:.1f}s")
        print(f"   Error: {exc}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Day 22 Lab — run all steps or a specific step"
    )
    parser.add_argument(
        "--step",
        type=int,
        nargs="+",
        choices=[1, 2, 3, 4],
        help="Step number(s) to run (default: all steps 1–4)",
    )
    args = parser.parse_args()

    steps_to_run = args.step if args.step else [1, 2, 3, 4]

    print("=" * 62)
    print("  Day 22 Lab — LangSmith + Prompt Versioning")
    print(f"  Running steps: {steps_to_run}")
    print("=" * 62)

    results: dict[int, bool] = {}
    overall_start = time.time()

    for step in steps_to_run:
        results[step] = run_step(step)

    # Final summary
    total_time = time.time() - overall_start
    print("\n" + "=" * 62)
    print("  SUMMARY")
    print("=" * 62)
    for step, ok in results.items():
        icon = "✅" if ok else "❌"
        print(f"  {icon}  Step {step}")
    print(f"\n  Total time: {total_time:.1f}s")

    failed = [s for s, ok in results.items() if not ok]
    if failed:
        print(f"\n⚠️  Steps {failed} failed. Check errors above.")
        sys.exit(1)
    else:
        print("\n🎉 All steps completed successfully!")


if __name__ == "__main__":
    main()