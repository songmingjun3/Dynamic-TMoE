from types import SimpleNamespace

from openi_baselines.native import loader_kwargs


def test_loader_kwargs_preserve_baseline_defaults(monkeypatch):
    for name in (
        "OPENI_PIN_MEMORY",
        "OPENI_PERSISTENT_WORKERS",
        "OPENI_PREFETCH_FACTOR",
    ):
        monkeypatch.delenv(name, raising=False)

    kwargs = loader_kwargs(
        SimpleNamespace(num_workers=2),
        default_pin_memory=True,
        default_persistent_workers=True,
        default_prefetch_factor=2,
    )

    assert kwargs == {
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2,
    }


def test_loader_kwargs_apply_environment_and_handle_zero_workers(monkeypatch):
    monkeypatch.setenv("OPENI_PIN_MEMORY", "1")
    monkeypatch.setenv("OPENI_PERSISTENT_WORKERS", "1")
    monkeypatch.setenv("OPENI_PREFETCH_FACTOR", "4")

    kwargs = loader_kwargs(SimpleNamespace(num_workers=0))

    assert kwargs == {
        "pin_memory": True,
        "persistent_workers": False,
    }
