from pathlib import Path

import pytest

from openi_baselines.platform import prepare_platform


def test_local_context_uses_explicit_roots_without_c2net(tmp_path):
    code = tmp_path / "code"
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"

    context = prepare_platform(
        local=True,
        code_root=code,
        dataset_root=dataset,
        output_root=output,
    )

    assert context.code_path == code
    assert context.dataset_path == dataset
    assert context.output_path == output
    assert output.is_dir()
    context.upload_output()


def test_local_context_requires_all_roots(tmp_path):
    with pytest.raises(ValueError, match="code_root.*dataset_root.*output_root"):
        prepare_platform(local=True, code_root=tmp_path)


def test_openi_context_adapts_c2net_object(monkeypatch, tmp_path):
    calls = []

    class RawContext:
        code_path = str(tmp_path / "code")
        dataset_path = str(tmp_path / "dataset")
        output_path = str(tmp_path / "output")

    def fake_loader():
        return (lambda: RawContext()), (lambda: calls.append("uploaded"))

    monkeypatch.setattr("openi_baselines.platform._load_c2net", fake_loader)

    context = prepare_platform(local=False)
    context.upload_output()

    assert context.code_path == Path(RawContext.code_path)
    assert calls == ["uploaded"]
