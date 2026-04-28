# SPDX-License-Identifier: GPL-3.0-or-later
"""Smoke test that the Gradio Blocks app actually boots and serves.

We launch the UI on an ephemeral port in the same process, fetch
``/config``, assert the Blocks schema is present, then shut down. This
catches regressions like the pydantic / fastapi / gradio incompatibility
that previously broke ``demo.launch()``.
"""

from __future__ import annotations

import socket
import time
import urllib.request
from pathlib import Path

import pytest


def _free_port() -> int:
    """Return a free TCP port (bound briefly then closed)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_gradio_launch_serves_config(tmp_path: Path) -> None:
    """Launching Blocks binds a port and serves a valid ``/config`` endpoint."""
    import os

    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

    from spar_measure.ui import PathManager, build_blocks

    port = _free_port()
    path_mgt = PathManager(out_dir=str(tmp_path))
    demo, _ = build_blocks(path_mgt)
    demo.queue()
    demo.launch(
        server_port=port,
        prevent_thread_lock=True,
        quiet=True,
    )
    try:
        # Wait briefly for uvicorn to bind.
        for _ in range(20):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/config", timeout=1) as r:
                    body = r.read().decode()
                break
            except Exception:
                time.sleep(0.2)
        else:
            pytest.fail("Gradio /config endpoint never responded")

        # Gradio 6 /config returns JSON with "version" present
        assert '"version"' in body
    finally:
        demo.close()


def test_run_gui_whitelists_sample_data_dir(tmp_path: Path) -> None:
    """``run_gui`` must inject the bundled sample_data dir into ``allowed_paths``.

    Reproduces the Colab failure where clicking "Load Example Dataset and
    Scales" raised ``gradio.exceptions.InvalidPathError`` because the
    sample CSV/NPY live inside site-packages and Gradio 6 only serves
    files under cwd, /tmp, or ``allowed_paths``. The fix is in
    ``run_gui`` and this guards against regressions.
    """
    import spar_measure.ui as ui_mod
    from spar_measure.ui import PathManager, run_gui

    captured: dict = {}

    class _Stub:
        def queue(self) -> None:
            return None

        def launch(self, **kwargs) -> None:
            captured.update(kwargs)

    real_build = ui_mod.build_blocks
    ui_mod.build_blocks = lambda pm: (_Stub(), None)  # type: ignore[assignment]
    try:
        run_gui(out_dir=str(tmp_path), mode="local")
    finally:
        ui_mod.build_blocks = real_build

    allowed = [str(p) for p in (captured.get("allowed_paths") or [])]
    sample_dir = str(PathManager(out_dir=str(tmp_path)).sample_data_dir)
    assert sample_dir in allowed, (
        f"run_gui did not whitelist {sample_dir!r}; allowed_paths={allowed!r}"
    )


def test_measure_docs_returns_error_textbox_when_scales_missing(tmp_path: Path) -> None:
    """``measure_docs`` must catch errors so the spinner doesn't get stuck.

    Reproduces the workshop bug where clicking *Measure Documents* before
    saving scales raised ``ValueError: scale_embeddings is empty`` from
    ``project_documents``. Without a try/except in the UI handler, Gradio
    leaves every output bound to the click in its in-flight (spinning)
    state forever. The fix returns an inline error textbox and clears the
    download File component instead.
    """
    from spar_measure.state import MeasurementState
    from spar_measure.ui import Measurement, PathManager

    m = Measurement(PathManager(out_dir=str(tmp_path)))
    state = MeasurementState()
    # No scale_embeddings, no embeddings — the original code would raise.
    textbox, file_update = m.measure_docs(
        single_subspace="No", whitening="No", measurement_state=state
    )
    assert getattr(textbox, "value", "").lower().startswith("measurement failed"), (
        f"expected an error textbox, got {textbox!r}"
    )
    assert getattr(file_update, "value", "missing") is None


def test_run_gui_preserves_user_allowed_paths(tmp_path: Path) -> None:
    """User-supplied ``allowed_paths`` must be preserved alongside sample_data."""
    import spar_measure.ui as ui_mod
    from spar_measure.ui import PathManager, run_gui

    captured: dict = {}

    class _Stub:
        def queue(self) -> None:
            return None

        def launch(self, **kwargs) -> None:
            captured.update(kwargs)

    real_build = ui_mod.build_blocks
    ui_mod.build_blocks = lambda pm: (_Stub(), None)  # type: ignore[assignment]
    user_extra = str(tmp_path / "extra")
    try:
        run_gui(out_dir=str(tmp_path), mode="local", allowed_paths=[user_extra])
    finally:
        ui_mod.build_blocks = real_build

    allowed = [str(p) for p in (captured.get("allowed_paths") or [])]
    sample_dir = str(PathManager(out_dir=str(tmp_path)).sample_data_dir)
    assert sample_dir in allowed
    assert user_extra in allowed
