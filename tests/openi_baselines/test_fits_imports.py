import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FITS_ROOT = REPO_ROOT / "baselines" / "FITS"


def test_fits_entrypoint_resolves_its_local_packages(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    monkeypatch.syspath_prepend(str(FITS_ROOT))

    import importlib.util

    expected = {
        "exp": FITS_ROOT / "exp" / "__init__.py",
        "models": FITS_ROOT / "models" / "__init__.py",
        "layers": FITS_ROOT / "layers" / "__init__.py",
        "data_provider": FITS_ROOT / "data_provider" / "__init__.py",
        "utils": FITS_ROOT / "utils" / "__init__.py",
    }
    for package, expected_origin in expected.items():
        sys.modules.pop(package, None)
        spec = importlib.util.find_spec(package)
        assert spec is not None
        assert Path(spec.origin).resolve() == expected_origin.resolve()
