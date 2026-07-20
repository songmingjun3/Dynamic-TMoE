from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


def _noop_upload() -> None:
    return None


@dataclass
class PlatformContext:
    code_path: Path
    dataset_path: Path
    output_path: Path
    _uploader: Callable[[], None] = field(repr=False, default=_noop_upload)

    def upload_output(self) -> None:
        self._uploader()


def _load_c2net():
    from c2net.context import prepare, upload_output

    return prepare, upload_output


def prepare_platform(
    *,
    local: bool,
    code_root: str | Path | None = None,
    dataset_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> PlatformContext:
    if local:
        if code_root is None or dataset_root is None or output_root is None:
            raise ValueError(
                "Local mode requires code_root, dataset_root, and output_root"
            )
        context = PlatformContext(
            code_path=Path(code_root).expanduser().resolve(),
            dataset_path=Path(dataset_root).expanduser().resolve(),
            output_path=Path(output_root).expanduser().resolve(),
        )
    else:
        prepare, upload_output = _load_c2net()
        raw_context = prepare()
        context = PlatformContext(
            code_path=Path(raw_context.code_path).expanduser().resolve(),
            dataset_path=Path(raw_context.dataset_path).expanduser().resolve(),
            output_path=Path(raw_context.output_path).expanduser().resolve(),
            _uploader=upload_output,
        )

    context.output_path.mkdir(parents=True, exist_ok=True)
    return context
