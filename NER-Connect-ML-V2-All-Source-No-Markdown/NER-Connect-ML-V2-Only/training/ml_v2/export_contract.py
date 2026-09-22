"""Export the canonical V2 feature registry for data stewards."""

import argparse
import json
from pathlib import Path

from app.ml_v2.contract import contract_as_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(contract_as_dict(), stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
