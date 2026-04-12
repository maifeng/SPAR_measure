# SPDX-License-Identifier: GPL-3.0-or-later
"""Gradio UI: Blocks layout, callbacks, and :func:`run_gui` entrypoint.

This module is a thin adapter over :mod:`spar_measure.core`. All real work
(embedding, projection, scoring) lives there; the ``Measurement`` class here
exists to mediate Gradio's dict-of-updates return convention.
"""

from __future__ import annotations

import json
import logging
import os
from operator import itemgetter
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd
import torch

from . import core, util_funcs
from .io import load_embeddings, read_csv_smart
from .state import (
    DEFAULT_EMBED_BATCH_SIZE,
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    DEFAULT_SBERT_MODEL,
    MAX_DIMENSIONS,
    OPENAI_EMBEDDING_MODELS,
    MeasurementState,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logger = logging.getLogger(__name__)


class CVFDemo:
    """Competing Values Framework example dimensions and scales.

    Used by the "Load Example Dataset and Scales" button in the UI. The
    dimensions and two example scales are the same ones used in the paper.

    Attributes:
        dim_names: Four CVF dimension names.
        dim_seeds: Four seed sentences, one per dimension.
        scales: Two scale definitions combining the dimensions.
    """

    dim_names: list[str] = ["Create", "Collaborate", "Control", "Compete"]
    dim_seeds: list[str] = [
        "We should adapt and innovate.",
        "We should empathize and collaborate.",
        "We should control and stabilize.",
        "We should respond swiftly and serve customers.",
    ]
    scales: dict[str, dict[str, list[str]]] = {
        "External-Internal": {
            "Positive": ["Create", "Compete"],
            "Negative": ["Control", "Collaborate"],
        },
        "Flexible-Stable": {
            "Positive": ["Collaborate", "Create"],
            "Negative": ["Control", "Compete"],
        },
    }


class PathManager:
    """Resolves input/output directories for a SPAR session.

    Attributes:
        out_dir: Directory where measurement CSVs, JSON exports, and cached
            embeddings are written.
        root_dir: Directory of the installed ``spar_measure`` package.
        sample_data_dir: ``<root_dir>/sample_data``, source of the bundled
            ``sample_text.csv`` and ``sample_emb.npy``.
    """

    def __init__(self, out_dir: str | os.PathLike[str]) -> None:
        self.out_dir: str = str(out_dir)
        self.root_dir: str = os.path.dirname(os.path.abspath(__file__))
        self.sample_data_dir: Path = Path(self.root_dir, "sample_data")
        logger.debug("sample_data_dir=%s", self.sample_data_dir)


class Measurement:
    """Gradio callback adapter that mediates a :class:`MeasurementState`.

    Each method here corresponds to one Gradio callback and returns the
    Gradio update objects the UI expects. The actual numeric work is
    delegated to :mod:`spar_measure.core`.

    Attributes:
        path_mgt: Resolved paths for this session.
    """

    def __init__(self, path_mgt: PathManager) -> None:
        self.path_mgt = path_mgt

    # --- Tab 1: upload + embed -----------------------------------------------

    def read_csv_cols(
        self, file_obj: Any, measurement_state: MeasurementState
    ) -> tuple[Any, Any]:
        """Load a CSV from the file uploader and populate the column dropdowns.

        Uses UTF-8 with a latin-1 fallback (fixes H-02 per 05_SPAR_code_review.md).
        """
        if file_obj is not None:
            measurement_state["input_df"] = read_csv_smart(file_obj.name)
            measurement_state["col_names"] = measurement_state[
                "input_df"
            ].columns.tolist()
            logger.info("Loaded CSV columns: %s", measurement_state["col_names"])
        return (
            gr.Dropdown(choices=measurement_state["col_names"], interactive=True),
            gr.Dropdown(choices=measurement_state["col_names"], interactive=True),
        )

    def read_input_embedding(
        self, file_obj: Any, measurement_state: MeasurementState
    ) -> tuple[Any, Any]:
        """Load a ``.npy`` matrix of precomputed embeddings and validate its shape.

        Validates 2D, float dtype, no NaN/Inf (fixes M-11).
        """
        try:
            measurement_state["embeddings"] = load_embeddings(file_obj.name)
        except ValueError as e:
            return gr.Textbox(value=f"Error: {e}"), gr.Button(visible=False)
        if measurement_state["embeddings"].shape[0] != len(
            measurement_state["input_df"]
        ):
            msg = (
                "Error: The number of rows in the input data and the number of rows in "
                f"the embedding matrix do not match. Embedding shape "
                f"{measurement_state['embeddings'].shape}, docs {len(measurement_state['input_df'])}."
            )
        else:
            msg = (
                "Precomputed embedding uploaded. \nThe shape of the embeddings is "
                f"{measurement_state['embeddings'].shape}. Proceed to the next tab to "
                "define the dimensions."
            )
        return gr.Textbox(value=msg), gr.Button(visible=False)

    def set_doc_col(
        self,
        doc_col_name: str,
        doc_id_col_name: str,
        measurement_state: MeasurementState,
    ) -> Any:
        """Set text and ID column names in state after validating they exist."""
        logger.info("Setting doc_col_name to %s", doc_col_name)
        if measurement_state["input_df"].columns.tolist() is not None:
            if doc_col_name in measurement_state["input_df"].columns.tolist():
                measurement_state["doc_col_name"] = doc_col_name
        logger.info("Setting doc_id_col_name to %s", doc_id_col_name)
        if measurement_state["input_df"].columns.tolist() is not None:
            if doc_id_col_name in measurement_state["input_df"].columns.tolist():
                measurement_state["doc_id_col_name"] = doc_id_col_name
        return gr.Textbox(value="Column names set.")

    def set_sbert(self, model_name: str, measurement_state: MeasurementState) -> None:
        """Load a Sentence-BERT model into state."""
        bundle = core.load_sbert(model_name)
        measurement_state["model_name"] = bundle.name
        measurement_state["tokenizer"] = bundle.tokenizer
        measurement_state["model"] = bundle.model

    def set_openai_api_key(
        self, api_key: str, measurement_state: MeasurementState
    ) -> Any:
        """Set the OpenAI API key in state; fall back to the environment variable."""
        logger.info("Setting OpenAI API key")
        if api_key:
            measurement_state["openai_api_key"] = api_key
            return None
        env_key = os.environ.get("OPENAI_API_KEY", "")
        if env_key:
            measurement_state["openai_api_key"] = env_key
            return None
        return gr.Textbox(
            value=(
                "Invalid API key. Enter a key in the textbox or set the "
                "OPENAI_API_KEY environment variable."
            ),
            visible=True,
        )

    def toggle_embedding_model_visibility(
        self, embedding_model_dropdown: str
    ) -> tuple[Any, Any, Any]:
        """Show/hide the Sentence-BERT or OpenAI text boxes per the dropdown."""
        if embedding_model_dropdown.startswith("Sentence Transformers"):
            return (
                gr.Textbox(visible=True),
                gr.Textbox(visible=False),
                gr.Button(value="Set Embedding Model"),
            )
        if embedding_model_dropdown.startswith("OpenAI"):
            return (
                gr.Textbox(visible=False),
                gr.Textbox(visible=True),
                gr.Button(value="Set Embedding Model"),
            )
        return gr.Textbox(visible=False), gr.Textbox(visible=False), gr.Button()

    def set_embedding_model(
        self,
        embedding_model_dropdown: str,
        sbert_model_textbox: str,
        openai_api_key: str,
        measurement_state: MeasurementState,
    ) -> Any:
        """Initialize the embedding backend (Sentence-BERT or OpenAI) in state."""
        logger.info("Setting embedding model to %s", embedding_model_dropdown)
        if embedding_model_dropdown.startswith("Sentence Transformers"):
            self.set_sbert(sbert_model_textbox, measurement_state)
            measurement_state["use_openai"] = False
        if embedding_model_dropdown.startswith("OpenAI"):
            # The dropdown label carries the selected OpenAI model name.
            for m in OPENAI_EMBEDDING_MODELS:
                if m in embedding_model_dropdown:
                    measurement_state["openai_embedding_model"] = m
                    break
            self.set_openai_api_key(openai_api_key, measurement_state)
            measurement_state["use_openai"] = True
        return gr.Button(value="Embedding model set.")

    def reset_set_emb_btn(self) -> Any:
        """Reset the Set Embedding Model button label."""
        return gr.Button(value="Set Embedding Model")

    def set_embedding_option(self, embedding_option: str) -> tuple[Any, Any]:
        """Show/hide the upload vs. embed-in-app controls per radio choice."""
        if embedding_option == "Upload Precomputed Embeddings":
            return gr.File(visible=True), gr.Button(visible=False)
        if embedding_option == "Embed Documents":
            return gr.File(visible=False), gr.Button(visible=True)
        return gr.File(visible=False), gr.Button(visible=False)

    @classmethod
    def embed_texts(
        cls,
        docs: list[str] | pd.Series,
        progress: Any,
        measurement_state: MeasurementState,
    ) -> np.ndarray:
        """Embed ``docs`` with the backend configured in state.

        Delegates to :func:`spar_measure.core.embed_texts`. The ``progress``
        argument is ``gr.Progress`` inside the UI and is used to drive the
        progress bar. Fixes C-02 and C-03: OpenAI batch failures now raise
        instead of corrupting the embedding matrix.
        """
        use_openai = bool(measurement_state["use_openai"])
        batch_size = int(measurement_state.get("batch_size", DEFAULT_EMBED_BATCH_SIZE))

        def _tqdm_range(seq: Any) -> Any:
            # gr.Progress.tqdm accepts any iterable.
            return progress.tqdm(seq, unit="batch")

        if use_openai:
            return core.embed_with_openai(
                [str(x) for x in docs],
                api_key=measurement_state["openai_api_key"],
                model=measurement_state.get(
                    "openai_embedding_model", DEFAULT_OPENAI_EMBEDDING_MODEL
                ),
                batch_size=batch_size,
                progress_iter=_tqdm_range,
            )
        bundle = core.SbertBundle(
            tokenizer=measurement_state["tokenizer"],
            model=measurement_state["model"],
            name=measurement_state.get("model_name", DEFAULT_SBERT_MODEL),
        )
        return core.embed_with_sbert(
            [str(x) for x in docs],
            bundle=bundle,
            batch_size=batch_size,
            progress_iter=_tqdm_range,
        )

    def embed_df(
        self, measurement_state: MeasurementState, progress: Any = gr.Progress()
    ) -> tuple[Any, Any, str | None]:
        """Embed the selected text column and save the result as ``embeddings.npy``."""
        try:
            measurement_state["embeddings"] = self.embed_texts(
                measurement_state["input_df"][measurement_state["doc_col_name"]],
                progress=progress,
                measurement_state=measurement_state,
            )
            out_path = Path(self.path_mgt.out_dir, "embeddings.npy")
            np.save(out_path, measurement_state["embeddings"])
            return (
                gr.Textbox(
                    value=(
                        "Embedding completed. \nThe shape of the embeddings is "
                        f"{measurement_state['embeddings'].shape}. You can download and "
                        "save the embeddings below. Proceed to the next tab to define "
                        "the dimensions."
                    )
                ),
                gr.File(visible=True),
                str(out_path),
            )
        except Exception as e:
            logger.exception("Embedding failed")
            return (
                gr.Textbox(
                    value=(
                        "Embedding failed. Make sure you clicked Confirm Column "
                        f"Selections and Set Embedding Model. \nError: {e}"
                    )
                ),
                gr.File(visible=False),
                None,
            )

    # --- Tab 2/3: dimension + scale rows --------------------------------------

    def toggle_row_vis(
        self, n_rows: int, measurement_state: MeasurementState
    ) -> list[Any]:
        """Show/hide dimension rows on Tab 2 per the active slider value."""
        measurement_state["n_dims"] = n_rows
        return [gr.Row(visible=i < n_rows) for i in range(MAX_DIMENSIONS)]

    def toggle_row_vis_scales(
        self, n_rows: int, measurement_state: MeasurementState
    ) -> list[Any]:
        """Show/hide scale rows on Tab 3 per the active slider value."""
        measurement_state["n_scales"] = n_rows
        return [gr.Row(visible=i < n_rows) for i in range(MAX_DIMENSIONS)]

    def semantic_search(
        self,
        query: str,
        n_results: int,
        measurement_state: MeasurementState,
        progress: Any = gr.Progress(),
    ) -> Any:
        """Return the top ``n_results`` corpus matches for ``query``."""
        queries = [q for q in query.split("\n") if q.strip()]
        if not queries:
            return gr.Textbox(value="No query provided.")
        try:
            query_embedding = self.embed_texts(
                queries, progress=progress, measurement_state=measurement_state
            )
            mean_vect = query_embedding.mean(axis=0)
            hits = util_funcs.semantic_search(
                mean_vect,
                measurement_state["embeddings"],
                score_function=util_funcs.dot_score,
                top_k=n_results,
            )
            hit_ids = [h["corpus_id"] for h in hits[0]]
            hit_scores = [h["score"] for h in hits[0]]
            examples = list(
                itemgetter(*hit_ids)(
                    measurement_state["input_df"][measurement_state["doc_col_name"]]
                )
            )
            example_doc_ids = list(
                itemgetter(*hit_ids)(
                    measurement_state["input_df"][measurement_state["doc_id_col_name"]]
                )
            )
            assert len(hit_ids) == len(hit_scores) == len(examples)
            parts: list[str] = []
            for i in range(len(hit_ids)):
                parts.append(f"Document ID: {example_doc_ids[i]}")
                parts.append(f"Score: {round(hit_scores[i], 3)}")
                parts.append("------------------------")
                parts.append(str(examples[i]))
                parts.append("------------------------")
            return gr.Textbox(value="\n".join(parts) + "\n")
        except Exception as e:
            logger.exception("Semantic search failed")
            return gr.Textbox(
                value=f"Error. Make sure the query is not empty. \nError: {e}"
            )

    def save_dims(
        self,
        measurement_state: MeasurementState,
        progress: Any = gr.Progress(),
        *dims_boxes: str,
    ) -> list[Any]:
        """Embed each dimension's queries and persist ``dimensions_queries.json``."""
        try:
            half = len(dims_boxes) // 2
            all_dim_names = dims_boxes[:half]
            all_dim_queries = dims_boxes[half:]
            measurement_state["dim_embeddings"] = {}
            measurement_state["dim_queries"] = {}
            for i in range(measurement_state["n_dims"]):
                dim_name = (
                    all_dim_names[i].strip()
                    if all_dim_names[i].strip()
                    else f"Dimension_{i + 1}"
                )
                queries = [q for q in all_dim_queries[i].split("\n") if q.strip()]
                dim_embedding = self.embed_texts(
                    queries, progress=progress, measurement_state=measurement_state
                )
                measurement_state["dim_embeddings"][dim_name] = dim_embedding.mean(axis=0)
                measurement_state["dim_queries"][dim_name] = queries
            json_path = Path(self.path_mgt.out_dir, "dimensions_queries.json")
            with open(json_path, "w") as f:
                json.dump(measurement_state["dim_queries"], f)
            return (
                [gr.Dropdown(choices=list(measurement_state["dim_queries"].keys()))]
                * 20
                + [
                    gr.Textbox(
                        visible=True,
                        value=(
                            f"Dimensions saved: {list(measurement_state['dim_queries'].keys())}. "
                            "You can download the json file below to keep a record of "
                            "the final queries. Proceed to the next tab to define scales."
                        ),
                    )
                ]
                + [gr.File(visible=True)]
                + [str(json_path)]
                + [gr.Markdown(visible=False)]
            )
        except Exception as e:
            logger.exception("save_dims failed")
            return (
                [gr.Dropdown(choices=list(measurement_state["dim_queries"].keys()))]
                * 20
                + [
                    gr.Textbox(
                        visible=True,
                        value=(
                            "Please check that you have filled in all dimension names "
                            "and queries. Move the 'Number of dimensions' slider to "
                            f"add or remove dimensions. \nError: {e}."
                        ),
                    )
                ]
                + [gr.File(visible=False)]
                + [None]
                + [gr.Markdown(visible=True)]
            )

    def save_scales(
        self, measurement_state: MeasurementState, *scale_boxes: Any
    ) -> list[Any]:
        """Combine dimension embeddings into scales and persist their definitions."""
        try:
            third = len(scale_boxes) // 3
            all_scale_names = scale_boxes[:third]
            all_pos_scales = scale_boxes[third : 2 * third]
            all_neg_scales = scale_boxes[2 * third :]
            measurement_state["scale_embeddings"] = {}
            measurement_state["scale_definitions"] = {}
            for i in range(measurement_state["n_scales"]):
                scale_name = (
                    all_scale_names[i].strip()
                    if all_scale_names[i].strip()
                    else f"Scale_{i + 1}"
                )
                measurement_state["scale_definitions"][scale_name] = {
                    "Positive": list(all_pos_scales[i]),
                    "Negative": list(all_neg_scales[i]),
                }
                pos = [
                    measurement_state["dim_embeddings"][d] for d in all_pos_scales[i]
                ]
                neg = [
                    measurement_state["dim_embeddings"][d] for d in all_neg_scales[i]
                ]
                measurement_state["scale_embeddings"][scale_name] = core.combine_scale(
                    pos, neg
                )
            json_path = Path(self.path_mgt.out_dir, "scale_definitions.json")
            with open(json_path, "w") as f:
                json.dump(measurement_state["scale_definitions"], f)
            return (
                [
                    gr.Textbox(
                        visible=True,
                        value=(
                            f"Scales saved: {list(measurement_state['scale_definitions'].keys())}. "
                            "You can download the json file below. Proceed to the next "
                            "tab to measure using semantic projection."
                        ),
                    )
                ]
                + [gr.File(visible=True)]
                + [str(json_path)]
                + [gr.Markdown(visible=False)]
            )
        except Exception as e:
            logger.exception("save_scales failed")
            return (
                [
                    gr.Textbox(
                        visible=True,
                        value=(
                            "Error: Please check your scale definitions. Make sure you "
                            "clicked Embed Queries and Save Dimensions in Tab 2, and "
                            "there are no empty scales or dimensions. Move the 'Number "
                            f"of scales' slider to add or remove scales. \nError: {e}"
                        ),
                    )
                ]
                + [gr.File(visible=False)]
                + [None]
                + [gr.Markdown(visible=True)]
            )

    # --- Tab 4: project + score ----------------------------------------------

    def measure_docs(
        self,
        single_subspace: str,
        whitening: str,
        measurement_state: MeasurementState,
    ) -> tuple[Any, Any, str]:
        """Score all documents against the defined scales and save as CSV.

        Fixes C-04: dropped the dead relative-path ``mkdir("measure_output")``
        and the naked ``Path(...)`` expression on line 796 of the old file.

        Fixes M-05: uses :func:`spar_measure.core.project_documents` which
        switched from ``numpy.linalg.inv`` to ``numpy.linalg.pinv``.
        """
        logger.info(
            "measure_docs: single_subspace=%s whitening=%s",
            single_subspace,
            whitening,
        )
        scores = core.project_documents(
            measurement_state["embeddings"],
            measurement_state["scale_embeddings"],
            single_subspace=(single_subspace == "Yes"),
            whiten=(whitening == "Yes"),
        )
        scores = scores.round(4)
        id_col = measurement_state["doc_id_col_name"]
        scores.insert(0, id_col, measurement_state["input_df"][id_col].to_numpy())
        Path(self.path_mgt.out_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(self.path_mgt.out_dir, "measurement_output.csv")
        scores.to_csv(out_path, index=False)
        return (
            gr.Textbox(
                visible=True,
                value="Measurement completed. Download the results below.",
            ),
            gr.File(visible=True),
            str(out_path),
        )

    def load_example_dataset(self, measurement_state: MeasurementState) -> list[Any]:
        """Load the bundled 2000-post sample corpus and populate the CVF example."""
        measurement_state["input_df"] = pd.read_csv(
            Path(self.path_mgt.sample_data_dir, "sample_text.csv")
        )
        measurement_state["embeddings"] = np.load(
            Path(self.path_mgt.sample_data_dir, "sample_emb.npy")
        )
        self.set_sbert(DEFAULT_SBERT_MODEL, measurement_state)
        measurement_state["doc_col_name"] = "text"
        measurement_state["doc_id_col_name"] = "doc_id"
        measurement_state["use_openai"] = False

        dim_name_updates: list[Any] = []
        dim_seed_updates: list[Any] = []
        for i in range(MAX_DIMENSIONS):
            if i < len(CVFDemo.dim_names):
                dim_name_updates.append(gr.Textbox(value=CVFDemo.dim_names[i]))
                dim_seed_updates.append(gr.Textbox(value=CVFDemo.dim_seeds[i]))
            else:
                dim_name_updates.append(gr.Textbox(value=""))
                dim_seed_updates.append(gr.Textbox(value=""))

        scale_name_updates: list[Any] = []
        scale_pos_updates: list[Any] = []
        scale_neg_updates: list[Any] = []
        demo_scale_names = list(CVFDemo.scales.keys())
        for i in range(MAX_DIMENSIONS):
            if i < len(demo_scale_names):
                scale_name_updates.append(gr.Textbox(value=demo_scale_names[i]))
                scale_pos_updates.append(
                    gr.Textbox(value=CVFDemo.scales[demo_scale_names[i]]["Positive"])
                )
                scale_neg_updates.append(
                    gr.Textbox(value=CVFDemo.scales[demo_scale_names[i]]["Negative"])
                )
            else:
                scale_name_updates.append(None)
                scale_pos_updates.append(None)
                scale_neg_updates.append(None)

        return (
            [
                gr.Row(visible=True),
                str(Path(self.path_mgt.sample_data_dir, "sample_text.csv")),
                str(Path(self.path_mgt.sample_data_dir, "sample_emb.npy")),
                gr.Radio(value="Embed Documents"),
                gr.Textbox(
                    value=(
                        "Embedding completed. The shape of the embeddings is (2000, 384). "
                        "You can download and save the embeddings below. Proceed to the "
                        "next tab to define the dimensions."
                    )
                ),
                gr.Dropdown(value="text"),
                gr.Dropdown(value="doc_id"),
                gr.Dropdown(value="Sentence Transformers (Local)"),
                gr.Textbox(value=DEFAULT_SBERT_MODEL),
            ]
            + dim_name_updates
            + dim_seed_updates
            + scale_name_updates
            + scale_pos_updates
            + scale_neg_updates
            + [gr.Slider(value=4), gr.Slider(value=2)]
        )


# --- Blocks layout ---------------------------------------------------------


def build_blocks(path_mgt: PathManager) -> tuple[gr.Blocks, Measurement]:
    """Build the Gradio Blocks layout for the SPAR UI.

    Args:
        path_mgt: Resolved paths for this session.

    Returns:
        ``(demo, measurement)`` tuple. The caller owns launching ``demo``.
    """
    torch.set_grad_enabled(False)

    with gr.Blocks(title="SPAR") as demo:
        m = Measurement(path_mgt=path_mgt)
        state = gr.State(MeasurementState())
        gr.Markdown(
            "### SPAR: Semantic Projection with Active Retrieval (Research Preview)"
        )

        all_dim_name_boxes: list[Any] = []
        all_search_query_boxes: list[Any] = []
        all_search_btns: list[Any] = []
        all_search_results: list[Any] = []
        all_rows_dims: list[Any] = []
        all_rows_scale: list[Any] = []
        all_scale_name_boxes: list[Any] = []
        all_scale_pos_selector: list[Any] = []
        all_scale_neg_selector: list[Any] = []

        # Header ------------------------------------------------------------
        with gr.Row():
            with gr.Column(scale=1):
                example_btn = gr.Button(value="Load Example Dataset and Scales")
            with gr.Column(scale=8):
                gr.Markdown(
                    value=(
                        "* SPAR is a Python package and web interface for measuring "
                        "short text documents using semantic projection.\n"
                        "* Reference: __Bei Yan, Feng Mai, Chaojiang Wu, Rui Chen, "
                        "Xiaolin Li (2024). A Computational Framework for Understanding "
                        "Firm Communication During Disasters. Information Systems "
                        "Research 35(2):590-608. https://doi.org/10.1287/isre.2022.0128__\n"
                        "* Source: [GitHub](https://github.com/maifeng/SPAR_measure) "
                        "(GPL-3.0)."
                    ),
                    label="",
                )
        example_row = gr.Row(visible=False, variant="panel")
        with example_row:
            with gr.Column(scale=2):
                gr.Markdown(
                    visible=True,
                    label="",
                    value=(
                        "__Example dataset and embeddings loaded.__ You may download "
                        "both on the right. Do not change Tab 1; proceed to Tab 2. To "
                        "use your own dataset, refresh the page and upload a CSV."
                    ),
                )
            with gr.Column(scale=1):
                example_file_download = gr.File(
                    visible=True, label="Download Example File"
                )
                example_emb_download = gr.File(
                    visible=True, label="Download Example Embeddings"
                )

        # Tab 1 -------------------------------------------------------------
        with gr.Tab("1. Upload File and Embed"):
            gr.Markdown(
                value=(
                    "Upload a **CSV file** that contains at least 2 columns: (1) the "
                    "documents to be measured, and (2) the document IDs. Alternatively "
                    "click Load Example Dataset and Scales to explore with a 2000-row "
                    "sample corpus and pre-defined CVF dimensions and scales."
                ),
                label="",
            )
            with gr.Row(variant="panel"):
                input_file = gr.File(
                    file_count="single", file_types=[".csv"], label="Input CSV File"
                )
                doc_col_selector = gr.Dropdown(
                    choices="",
                    label="Select Text Column",
                    interactive=False,
                    allow_custom_value=True,
                )
                doc_id_col_selector = gr.Dropdown(
                    choices="",
                    label="Select Document ID Column",
                    interactive=False,
                    allow_custom_value=True,
                )
                doc_id_col_btn = gr.Button(value="Confirm Column Selections")
                doc_id_col_btn.click(
                    fn=m.set_doc_col,
                    inputs=[doc_col_selector, doc_id_col_selector, state],
                    outputs=doc_id_col_btn,
                )

            gr.Markdown(
                value=(
                    "Select an embedding method. You can use Sentence Transformers "
                    "locally (default: all-MiniLM-L6-v2) or [any SBERT model]"
                    "(https://www.sbert.net/docs/pretrained_models.html). Alternatively, "
                    "use the OpenAI Embeddings API; default model: "
                    f"`{DEFAULT_OPENAI_EMBEDDING_MODEL}`. Get an API key [here]"
                    "(https://platform.openai.com/account/api-keys)."
                ),
            )

            openai_choices = [
                f"OpenAI {m} (API Key Required)" for m in OPENAI_EMBEDDING_MODELS
            ]
            with gr.Row(variant="panel"):
                embedding_model_dropdown = gr.Dropdown(
                    choices=["Sentence Transformers (Local)"] + openai_choices,
                    label="Select an embedding method:",
                    interactive=True,
                    value="Sentence Transformers (Local)",
                    allow_custom_value=True,
                    multiselect=False,
                )
                openai_api_key = gr.Textbox(
                    placeholder="",
                    label="OpenAI API Key (leave blank to use OPENAI_API_KEY env var)",
                    value="",
                    visible=False,
                    interactive=True,
                    type="password",
                )
                sbert_model_textbox = gr.Textbox(
                    label="Sentence Transformers Model Name",
                    value=DEFAULT_SBERT_MODEL,
                    interactive=True,
                    visible=True,
                )
                set_emb_btn = gr.Button(value="Set Embedding Model", visible=True)
                set_emb_btn.click(
                    fn=m.set_embedding_model,
                    inputs=[
                        embedding_model_dropdown,
                        sbert_model_textbox,
                        openai_api_key,
                        state,
                    ],
                    outputs=set_emb_btn,
                )
            gr.Markdown(
                value=(
                    "Click Embed Documents. Alternatively, upload a numpy array "
                    "(.npy or .npz) with precomputed embeddings. Shape must be "
                    "(n_docs, embedding_dim) and must have been produced by the "
                    "same embedding model selected above."
                )
            )
            with gr.Row(variant="panel"):
                upload_emb_choice = gr.Radio(
                    choices=["Embed Documents", "Upload Precomputed Embeddings"],
                    label="Embedding Options",
                    visible=True,
                    interactive=True,
                )
                embed_btn = gr.Button("Embed Documents", visible=False)
                input_embedding_uploader = gr.File(
                    file_count="single",
                    file_types=[".npz", ".npy"],
                    label="Precomputed Embeddings",
                    visible=False,
                )
                emb_result_txtbox = gr.Textbox(value="", label="Embedding Progress")
                emb_results_file = gr.File(visible=False)

            upload_emb_choice.change(
                fn=m.set_embedding_option,
                inputs=upload_emb_choice,
                outputs=[input_embedding_uploader, embed_btn],
                api_visibility="private",
            )
            input_file.change(
                fn=m.read_csv_cols,
                inputs=[input_file, state],
                outputs=[doc_col_selector, doc_id_col_selector],
                api_visibility="private",
            )
            input_embedding_uploader.change(
                fn=m.read_input_embedding,
                inputs=[input_embedding_uploader, state],
                outputs=[emb_result_txtbox, embed_btn],
                api_visibility="private",
            )
            embedding_model_dropdown.change(
                fn=m.toggle_embedding_model_visibility,
                inputs=[embedding_model_dropdown],
                outputs=[sbert_model_textbox, openai_api_key, set_emb_btn],
                api_visibility="private",
            )
            sbert_model_textbox.change(
                fn=m.reset_set_emb_btn, outputs=[set_emb_btn]
            )
            openai_api_key.change(
                fn=m.set_openai_api_key,
                inputs=[openai_api_key, state],
                outputs=[emb_result_txtbox],
                api_visibility="private",
            )
            embed_btn.click(
                fn=m.embed_df,
                inputs=state,
                outputs=[emb_result_txtbox, emb_results_file, emb_results_file],
                api_visibility="private",
            )

        # Tab 2 -------------------------------------------------------------
        with gr.Tab("2. Define Dimensions and Semantic Search"):
            gr.Markdown(
                value=(
                    "Move the sliders to set the number of dimensions and the "
                    "number of results per search. Do not leave any dimension empty."
                ),
                label="",
            )
            n_dim_slider = gr.Slider(
                1,
                MAX_DIMENSIONS,
                step=1,
                value=4,
                interactive=True,
                label="Number of dimensions",
            )
            n_results_slider = gr.Slider(
                10,
                200,
                step=5,
                value=10,
                interactive=True,
                label="Number of results in search",
            )
            gr.Markdown(
                value=(
                    "Enter dimension names and seed queries; click 'Search Dimension' "
                    "to find similar documents in the corpus. Copy relevant documents "
                    "back into the query box to iterate. Multiple queries per dimension "
                    "are fine, one per line."
                ),
                label="",
            )
            for i in range(MAX_DIMENSIONS):
                with gr.Row(variant="panel") as a_row:
                    all_rows_dims.append(a_row)
                    a_row.visible = i < 4
                    with gr.Column(scale=4, min_width=400):
                        if i < 4:
                            all_dim_name_boxes.append(
                                gr.Textbox(
                                    lines=1,
                                    max_lines=1,
                                    interactive=True,
                                    placeholder="e.g. " + CVFDemo.dim_names[i],
                                    value="",
                                    label=f"Dimension {i + 1} Name (Optional)",
                                    visible=True,
                                )
                            )
                            all_search_query_boxes.append(
                                gr.Textbox(
                                    lines=5,
                                    interactive=True,
                                    label=f"Query (Seed) Sentences for Dimension {i + 1}. One per line. (Required)",
                                    value="",
                                    placeholder="e.g. " + CVFDemo.dim_seeds[i],
                                    visible=True,
                                )
                            )
                        else:
                            all_dim_name_boxes.append(
                                gr.Textbox(
                                    lines=1,
                                    max_lines=1,
                                    placeholder=None,
                                    interactive=True,
                                    label=f"Dimension {i + 1} Name (Optional)",
                                    value="",
                                    visible=True,
                                )
                            )
                            all_search_query_boxes.append(
                                gr.Textbox(
                                    lines=5,
                                    interactive=True,
                                    label=f"Query (Seed) Sentences for Dimension {i + 1}. One per line. (Required)",
                                    value="",
                                    placeholder=f"Dimension {i + 1} seed sentences, one per line",
                                    visible=True,
                                )
                            )
                    with gr.Column(scale=1):
                        all_search_btns.append(
                            gr.Button(f"Search Dimension {i + 1}", visible=True)
                        )
                    with gr.Column(scale=4, min_width=400):
                        all_search_results.append(
                            gr.Textbox(
                                value="",
                                label=(
                                    f"Search Results for Dimension {i + 1}. Copy "
                                    "relevant sentences into the query box."
                                ),
                                visible=True,
                            )
                        )

            n_dim_slider.change(
                fn=m.toggle_row_vis,
                inputs=[n_dim_slider, state],
                outputs=all_rows_dims,
                api_visibility="private",
            )
            for box in all_search_query_boxes:
                box.change(
                    fn=m.toggle_row_vis,
                    inputs=[n_dim_slider, state],
                    outputs=[],
                    api_visibility="private",
                )
            for dim_i, btn in enumerate(all_search_btns):
                btn.click(
                    fn=m.semantic_search,
                    inputs=[all_search_query_boxes[dim_i], n_results_slider, state],
                    outputs=all_search_results[dim_i],
                    api_visibility="private",
                )
            gr.Markdown(
                value=(
                    "After defining dimensions with final context-specific queries, "
                    "click 'Embed Queries and Save Dimensions'. Each dimension must "
                    "contain at least one final query."
                ),
                label="",
            )
            save_dim_button = gr.Button("Embed Queries and Save Dimensions")
            dim_define_results = gr.Textbox(visible=False, label="")
            dimension_def_file_download = gr.File(visible=False)

        # Tab 3 -------------------------------------------------------------
        with gr.Tab("3. Define Scales"):
            tab2_warn = gr.Markdown(
                value="<span style='color:red'>Embed Queries and Save Dimensions in Tab 2 first.</span>",
                visible=True,
            )
            gr.Markdown(value="Move the slider to set the number of scales.", label="")
            n_scale_slider = gr.Slider(
                1,
                MAX_DIMENSIONS,
                step=1,
                value=2,
                interactive=True,
                label="Number of scales",
            )
            gr.Markdown(
                value=(
                    "Enter scale names and select the relevant dimensions; click "
                    "Save Scales to compute the scale embeddings. Scales are linear "
                    "combinations of dimensions, e.g. Safety = Safe (+) - Danger (-), "
                    "Wellness = Physical + Mental - Illness - Stress. Each scale "
                    "needs at least one positive or negative dimension."
                ),
                label="",
            )
            for i in range(MAX_DIMENSIONS):
                with gr.Row(variant="panel") as a_row:
                    all_rows_scale.append(a_row)
                    if i >= 2:
                        a_row.visible = False
                        with gr.Column(scale=2, min_width=400):
                            all_scale_name_boxes.append(
                                gr.Textbox(
                                    lines=1,
                                    max_lines=1,
                                    interactive=True,
                                    placeholder=None,
                                    value="",
                                    label=f"Scale {i + 1} Name",
                                    visible=True,
                                )
                            )
                    else:
                        a_row.visible = True
                        with gr.Column(scale=2, min_width=400):
                            all_scale_name_boxes.append(
                                gr.Textbox(
                                    lines=1,
                                    max_lines=1,
                                    interactive=True,
                                    placeholder="e.g. "
                                    + list(CVFDemo.scales.keys())[i],
                                    value="",
                                    label=f"Scale {i + 1} Name",
                                    visible=True,
                                )
                            )
                    with gr.Column(scale=4):
                        all_scale_pos_selector.append(
                            gr.Dropdown(
                                interactive=True,
                                multiselect=True,
                                label=f"Positive Dimensions for Scale {i + 1} (Required)",
                                value=None,
                                visible=True,
                                allow_custom_value=True,
                            )
                        )
                    with gr.Column(scale=4):
                        all_scale_neg_selector.append(
                            gr.Dropdown(
                                interactive=True,
                                multiselect=True,
                                label=f"Negative Dimensions for Scale {i + 1} (Optional)",
                                value=None,
                                visible=True,
                                allow_custom_value=True,
                            )
                        )

            n_scale_slider.change(
                fn=m.toggle_row_vis_scales,
                inputs=[n_scale_slider, state],
                outputs=all_rows_scale,
                api_visibility="private",
            )
            for box in all_scale_name_boxes + all_scale_pos_selector + all_scale_neg_selector:
                box.change(
                    fn=m.toggle_row_vis_scales,
                    inputs=[n_scale_slider, state],
                    outputs=[],
                    api_visibility="private",
                )
            save_scale_button = gr.Button("Save Scales")
            scale_define_results = gr.Textbox(visible=False, label="")
            scale_def_file_download = gr.File(visible=False)

            save_dim_button.click(
                fn=m.save_dims,
                inputs=[state] + all_dim_name_boxes + all_search_query_boxes,
                outputs=all_scale_pos_selector
                + all_scale_neg_selector
                + [dim_define_results]
                + [dimension_def_file_download] * 2
                + [tab2_warn],
            )

        # Tab 4 -------------------------------------------------------------
        with gr.Tab("4. Measurement"):
            tab3_warn = gr.Markdown(
                value="<span style='color:red'>Save Scales in Tab 3 first.</span>",
                visible=True,
            )
            save_scale_button.click(
                fn=m.save_scales,
                inputs=[state]
                + all_scale_name_boxes
                + all_scale_pos_selector
                + all_scale_neg_selector,
                outputs=[scale_define_results]
                + [scale_def_file_download] * 2
                + [tab3_warn],
            )
            gr.Markdown(
                value=(
                    "Click Measure Documents Using Semantic Projection to score each "
                    "document. The output CSV contains the document ID plus one column "
                    "per scale."
                ),
                label="",
            )
            with gr.Row(variant="panel"):
                single_subspace_radio_btn = gr.Radio(
                    choices=["Yes", "No"],
                    value="No",
                    label=(
                        "Single subspace: 'Yes' if all k scales span a single k-d "
                        "subspace (recommended when scales are semantically similar). "
                        "'No' to treat each scale as independent."
                    ),
                    visible=True,
                )
                whitening_radio_btn = gr.Radio(
                    choices=["Yes", "No"],
                    value="Yes",
                    label=(
                        "Whitening Output: 'Yes' to decorrelate the scores "
                        "(recommended if scales are theoretically orthogonal)."
                    ),
                    visible=True,
                )
            measure_button = gr.Button("Measure Documents Using Semantic Projection")
            measure_result = gr.Textbox(visible=False, label="")
            measure_results_file = gr.File(visible=False)
            measure_button.click(
                fn=m.measure_docs,
                inputs=[
                    single_subspace_radio_btn,
                    whitening_radio_btn,
                    state,
                ],
                outputs=[measure_result, measure_results_file, measure_results_file],
            )

        example_btn.click(
            fn=m.load_example_dataset,
            inputs=state,
            outputs=[
                example_row,
                example_file_download,
                example_emb_download,
                upload_emb_choice,
                emb_result_txtbox,
                doc_col_selector,
                doc_id_col_selector,
                embedding_model_dropdown,
                sbert_model_textbox,
            ]
            + all_dim_name_boxes
            + all_search_query_boxes
            + all_scale_name_boxes
            + all_scale_pos_selector
            + all_scale_neg_selector
            + [n_dim_slider, n_scale_slider],
        )
    return demo, m


# --- Launch helper ---------------------------------------------------------


_LAUNCH_ALLOWED: frozenset[str] = frozenset(
    {
        "share",
        "auth",
        "server_name",
        "server_port",
        "favicon_path",
        "ssl_keyfile",
        "ssl_certfile",
        "ssl_verify",
        "inbrowser",
        "quiet",
        "debug",
        "height",
        "width",
        "root_path",
        "max_threads",
        "show_error",
        "allowed_paths",
        "blocked_paths",
    }
)
"""Whitelisted keyword arguments forwarded to :meth:`gradio.Blocks.launch`.

Blocks unknown kwargs (e.g. ``--help`` forwarded by fire). See H-05.
"""


def run_gui(
    out_dir: str = "measure_output/",
    mode: str = "local",
    username: str | None = None,
    password: str | None = None,
    **kwargs: Any,
) -> None:
    """Launch the SPAR Gradio UI.

    Args:
        out_dir: Directory where outputs are written; created if missing.
        mode: ``"local"`` (default, localhost only) or ``"public"`` (share link
            with optional basic auth).
        username: Basic-auth username, required in public mode for auth.
        password: Basic-auth password.
        **kwargs: Extra arguments forwarded to :meth:`gradio.Blocks.launch`.
            Only keys in :data:`_LAUNCH_ALLOWED` are forwarded; others are
            logged and dropped (fixes H-05).
    """
    ignored = sorted(set(kwargs) - _LAUNCH_ALLOWED)
    if ignored:
        logger.warning(
            "Ignoring unknown CLI flags not accepted by gradio.launch: %s", ignored
        )
    kwargs = {k: v for k, v in kwargs.items() if k in _LAUNCH_ALLOWED}

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path_mgt = PathManager(out_dir=out_dir)

    demo, _ = build_blocks(path_mgt)
    demo.queue()

    if username is None or password is None:
        auth = None
    else:
        auth = (username, password)

    favicon = path_mgt.sample_data_dir / "favicon.png"
    if mode == "public":
        demo.launch(
            share=True,
            auth=auth,
            server_name="0.0.0.0",
            favicon_path=favicon,
            **kwargs,
        )
    elif mode == "local":
        demo.launch(favicon_path=favicon, **kwargs)
    else:
        raise ValueError(f"Invalid mode {mode!r}; must be 'local' or 'public'.")


__all__ = ["Measurement", "PathManager", "CVFDemo", "build_blocks", "run_gui"]
