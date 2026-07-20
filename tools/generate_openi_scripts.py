from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openi_baselines.registry import iter_tasks


def _script(model: str, dataset: str, pred_len: int | None) -> str:
    arguments = f"--model {model} --dataset {dataset}"
    if pred_len is not None:
        arguments += f" --pred-len {pred_len}"
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"\n\n'
        f'exec python -u "$REPO_ROOT/train_openi.py" {arguments} "$@"\n'
    )


def expected_scripts() -> dict[Path, str]:
    expected: dict[Path, str] = {}
    for task in iter_tasks():
        base = Path(task.model) / task.dataset
        expected[base / "all.sh"] = _script(task.model, task.dataset, None)
        for pred_len in task.horizons:
            expected[base / f"pred_{pred_len}.sh"] = _script(
                task.model, task.dataset, pred_len
            )
    return expected


def generate_scripts(output_root: str | Path) -> None:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    expected = expected_scripts()

    for stale in root.rglob("*.sh"):
        stale.unlink()
    for directory in sorted(root.rglob("*"), key=lambda path: len(path.parts), reverse=True):
        if directory.is_dir():
            try:
                directory.rmdir()
            except OSError:
                pass

    for relative, content in sorted(expected.items(), key=lambda item: str(item[0])):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)


def check_scripts(output_root: str | Path) -> list[str]:
    root = Path(output_root)
    expected = expected_scripts()
    problems: list[str] = []
    actual_files = {path.relative_to(root) for path in root.rglob("*.sh")}
    expected_files = set(expected)
    for missing in sorted(expected_files - actual_files, key=str):
        problems.append(f"missing: {missing}")
    for extra in sorted(actual_files - expected_files, key=str):
        problems.append(f"unexpected: {extra}")
    for relative in sorted(expected_files & actual_files, key=str):
        if (root / relative).read_text(encoding="utf-8") != expected[relative]:
            problems.append(f"content differs: {relative}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate OpenI baseline launch scripts")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "scripts" / "openi",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    if args.check:
        problems = check_scripts(args.output_root)
        if problems:
            print("\n".join(problems), file=sys.stderr)
            return 1
        print(f"OpenI launch scripts are current ({len(expected_scripts())} files)")
        return 0

    generate_scripts(args.output_root)
    print(f"Generated {len(expected_scripts())} scripts under {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
