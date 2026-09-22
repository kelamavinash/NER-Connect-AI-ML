"""Command-line entrypoint for a reviewed disruption dataset."""

import argparse
import json
from pathlib import Path

from training.ml_v2.benchmark import run_disruption_benchmark
from training.ml_v2.config import ExperimentConfig
from training.ml_v2.schema import load_training_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark disruption_next_6h on real NER labels")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--runs-root", default=Path("runs"), type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--allow-missing-optional-models", action="store_true")
    args = parser.parse_args()
    config = ExperimentConfig.from_json(args.config)
    table = load_training_table(args.data, args.manifest)
    result = run_disruption_benchmark(
        table,
        args.runs_root,
        config,
        run_id=args.run_id,
        require_all=not args.allow_missing_optional_models,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
