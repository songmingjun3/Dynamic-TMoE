from pathlib import Path

from tools.generate_openi_scripts import expected_scripts, generate_scripts


def test_generator_creates_all_and_per_horizon_scripts(tmp_path):
    generate_scripts(tmp_path)
    base = tmp_path / "PatchTST" / "ETTh1"

    assert (base / "all.sh").is_file()
    assert (base / "pred_96.sh").is_file()
    assert "--pred-len 96" in (base / "pred_96.sh").read_text(encoding="utf-8")
    assert "--pred-len" not in (base / "all.sh").read_text(encoding="utf-8")


def test_generator_emits_expected_script_count(tmp_path):
    generate_scripts(tmp_path)

    assert len(list(tmp_path.rglob("*.sh"))) == 450
    assert len(expected_scripts()) == 450


def test_generator_includes_dynamic_tmoe_and_timemixer_ili(tmp_path):
    generate_scripts(tmp_path)

    assert (tmp_path / "TimeMixer" / "Exchange" / "all.sh").is_file()
    assert (
        tmp_path / "TimeMixer" / "Exchange" / "pred_720.sh"
    ).is_file()
    assert (tmp_path / "TimeMixer" / "ILI" / "pred_60.sh").is_file()
    assert (tmp_path / "TimeMixer" / "Weather" / "pred_720.sh").is_file()
    assert (tmp_path / "Dynamic_TMoE" / "ETTh1" / "pred_336.sh").is_file()


def test_generated_script_resolves_repo_and_delegates_to_shared_cli(tmp_path):
    generate_scripts(tmp_path)
    script = (tmp_path / "DLinear" / "ETTh1" / "pred_336.sh").read_text(
        encoding="utf-8"
    )

    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert 'REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"' in script
    assert '"$REPO_ROOT/train_openi.py"' in script
    assert "--model DLinear --dataset ETTh1 --pred-len 336" in script


def test_regeneration_removes_only_stale_shell_scripts(tmp_path):
    stale = tmp_path / "stale.sh"
    keep = tmp_path / "notes.txt"
    stale.write_text("stale", encoding="utf-8")
    keep.write_text("keep", encoding="utf-8")

    generate_scripts(tmp_path)

    assert not stale.exists()
    assert keep.read_text(encoding="utf-8") == "keep"
