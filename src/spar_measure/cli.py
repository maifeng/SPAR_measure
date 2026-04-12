# SPDX-License-Identifier: GPL-3.0-or-later
"""Fire-based command-line entry point for SPAR.

Two subcommands:

- ``gui``: launch the Gradio UI (equivalent to the legacy
  ``python -m spar_measure.gui`` invocation).
- ``measure``: headless scoring from a CSV plus a JSON scale spec.

Usage:

    python -m spar_measure gui
    python -m spar_measure gui --mode=public --username=foo --password=bar
    python -m spar_measure measure --docs=corpus.csv --scales=scales.json --out=scored.csv
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import fire
import pandas as pd

from . import ui
from .api import score
from .state import DEFAULT_EMBED_BATCH_SIZE, DEFAULT_OPENAI_EMBEDDING_MODEL, DEFAULT_SBERT_MODEL


def _setup_logging() -> None:
    """Configure a sensible default logger for the CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def gui(
    out_dir: str = "measure_output/",
    mode: str = "local",
    username: str | None = None,
    password: str | None = None,
    server_port: int | None = None,
    server_name: str | None = None,
    share: bool = False,
    inbrowser: bool = False,
    quiet: bool = False,
) -> None:
    """Launch the SPAR Gradio UI.

    Args:
        out_dir: Directory where outputs are written.
        mode: ``"local"`` or ``"public"``.
        username: Basic-auth username (public mode).
        password: Basic-auth password (public mode).
        server_port: Port to bind (default 7860).
        server_name: Host to bind.
        share: Emit a gradio share link.
        inbrowser: Open the UI in a browser on launch.
        quiet: Suppress verbose launch banner.
    """
    _setup_logging()
    extra: dict[str, object] = {}
    if server_port is not None:
        extra["server_port"] = server_port
    if server_name is not None:
        extra["server_name"] = server_name
    if share:
        extra["share"] = True
    if inbrowser:
        extra["inbrowser"] = True
    if quiet:
        extra["quiet"] = True
    ui.run_gui(
        out_dir=out_dir,
        mode=mode,
        username=username,
        password=password,
        **extra,
    )


def measure(
    docs: str,
    scales: str,
    out: str | None = None,
    text_col: str = "text",
    id_col: str | None = None,
    embedding_model: str | None = None,
    openai_api_key: str | None = None,
    openai_embedding_model: str = DEFAULT_OPENAI_EMBEDDING_MODEL,
    whiten: bool = False,
    single_subspace: bool = False,
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
) -> None:
    """Score a CSV of documents against a JSON scale spec, headless.

    Args:
        docs: Path to the input CSV.
        scales: Path to a JSON file defining scales. See :func:`spar_measure.score`.
        out: Path to write the scored CSV. Defaults to ``<docs>.scored.csv``.
        text_col: Text column name.
        id_col: ID column name. If ``None``, a ``doc_id`` is added.
        embedding_model: SBERT or OpenAI model name (see :func:`spar_measure.score`).
        openai_api_key: OpenAI API key (switches backend if supplied).
        openai_embedding_model: OpenAI model name.
        whiten: Apply ZCA whitening.
        single_subspace: Use joint-subspace projection.
        batch_size: Embedding batch size.
    """
    _setup_logging()
    docs_df = pd.read_csv(docs)
    with open(scales) as f:
        scales_spec = json.load(f)
    result = score(
        docs_df,
        scales_spec,
        embedding_model=embedding_model,
        openai_api_key=openai_api_key,
        openai_embedding_model=openai_embedding_model,
        whiten=whiten,
        single_subspace=single_subspace,
        text_col=text_col,
        id_col=id_col,
        batch_size=batch_size,
    )
    out_path = Path(out) if out else Path(docs).with_suffix(".scored.csv")
    result.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


def main() -> None:
    """Fire entry point for ``python -m spar_measure``."""
    fire.Fire({"gui": gui, "measure": measure})


if __name__ == "__main__":
    main()
