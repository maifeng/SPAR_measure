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
