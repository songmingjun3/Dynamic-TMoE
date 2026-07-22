import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FEDFORMER_ROOT = REPO_ROOT / "baselines" / "FEDformer"


def test_fedformer_entrypoint_resolves_its_local_packages(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    monkeypatch.syspath_prepend(str(FEDFORMER_ROOT))

    import importlib.util

    expected = {
        "exp": FEDFORMER_ROOT / "exp" / "__init__.py",
        "models": FEDFORMER_ROOT / "models" / "__init__.py",
        "layers": FEDFORMER_ROOT / "layers" / "__init__.py",
        "data_provider": FEDFORMER_ROOT / "data_provider" / "__init__.py",
        "utils": FEDFORMER_ROOT / "utils" / "__init__.py",
    }
    for package, expected_origin in expected.items():
        sys.modules.pop(package, None)
        spec = importlib.util.find_spec(package)
        assert spec is not None
        assert Path(spec.origin).resolve() == expected_origin.resolve()
