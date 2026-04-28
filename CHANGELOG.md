# Changelog

All notable changes to `spar-measure` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [0.3.6] - 2026-04-28

### Added
- `CHANGELOG.md` (this file), covering v0.3.0 to v0.3.6.
- ChromaStore usage section in `README.MD` with three-step persist/load/score workflow.
- Table-of-contents entry for the ChromaStore section.

### Changed
- `resources/pypi_intro.MD` fully rewritten for the post-refactor API: correct GUI
  launch command (`python -m spar_measure gui`), headless `score()` quickstart,
  ChromaStore snippet, Colab badges, citation.
- `README.MD` line 246: dead `Manual.MD` link replaced with documentation of the
  `spar` and `spar-measure` console entry points.
- `README.MD` line 89: dead `Manual.MD` link sentence removed.

### Fixed
- BibTeX `author` field in `README.MD` citation: `Chen, Rong` corrected to `Chen, Rui`.

### Removed
- `resources/Manual.MD` (20-byte stub, "Under construction.") deleted via `git rm`.

---

## [0.3.5] - 2026-04-28

### Added
- Headless Colab notebook (`resources/example_colab_headless.ipynb`) for running
  `score()` on the bundled 2,000-document corpus without a GUI or API key
  (commit `8ae5dc0`).
- Canonical `scales.json` hand-off: `io.py` gains `gui_state_to_scales_spec()` and
  `write_scales_spec()`; the GUI now writes one JSON file consumed by `score()`
  directly (commit `a1fb043`).
- 103 passing tests (up from 25 at v0.3.0a1).

### Changed
- God-module split: 1460-line `gui.py` replaced by six focused modules
  (`state`, `io`, `core`, `api`, `ui`, `cli`) (commit `9020bd6`).
- Numerical stability: `project_documents` uses `numpy.linalg.pinv` instead of
  `inv` for the joint-subspace projection path.
- README rewritten around the semantic-projection mental model with a new
  projection diagram and GUI hand-off snippet.

### Fixed
- Correct coauthor name: `Rong Chen` changed to `Rui Chen` across all files
  (commit `9a28622`).

---

## [0.3.4] - 2026-04-28

### Fixed
- `measure_docs` callback now wrapped in `try/except`; previously an empty
  scale list raised `ValueError` and left all Gradio outputs in a permanent
  spinning state (commit `b5c4ad4`).
- Removed two dead `box.change(..., outputs=[])` bindings that fired on every
  keystroke and flooded the warning log.

---

## [0.3.3] - 2026-04-28

### Fixed
- Same release series as 0.3.4; intermediate patch (commit `b5c4ad4`).

---

## [0.3.2] - 2026-04-27

### Fixed
- `sample_data` directory added to `run_gui` allowed paths so the bundled
  example dataset loads without a Gradio security error (commit `b600f39`).

---

## [0.3.1] - 2026-04-27

### Added
- Fast-fail on bad OpenAI API key in `embed_with_openai`: raises `AuthenticationError`
  immediately rather than surfacing it mid-batch (commit `8f4570e`).

### Changed
- Gradio 6 migration (commits `60b19f5` to `20e1c93`):
  - Dropped removed `show_api` launch kwarg (commit `8e64372`).
  - `api_name=False` renamed to `api_visibility="private"` (commit `34cb9af`).
  - Fixed `gr.Dropdown` `choices` and `value` strictness for Gradio 5+/6
    (commit `20e1c93`).
  - Bumped `requirements.txt` to `gradio>=6.0.0,<7`; removed `pydantic` and
    `fastapi` version caps (commit `60b19f5`).

---

## [0.3.0] - 2026-04-15

### Added
- `ChromaStore` class in `src/spar_measure/vector_store.py`: persistent
  ChromaDB-backed vector store for large corpora; `[vector]` optional extra
  added to `pyproject.toml` (commit `64f222e`).
- `spar` and `spar-measure` console entry points registered in `pyproject.toml`.
- `[dev]` optional extra for `pytest` and `gradio_client`.
- 61 new tests for scale validation, extended core/io/state paths, batch size
  handling, and `ChromaStore`.

### Changed
- Package status promoted from Alpha to Beta in PyPI classifiers.
- Minimum Python version raised to 3.10.

### Fixed
- Author name `Rong Chen` changed to `Rui Chen` in `pypi_intro.MD` and other
  files (completed in v0.3.5 commit `9a28622`; partially addressed here).

---

[0.3.6]: https://github.com/maifeng/SPAR_measure/compare/v0.3.5...v0.3.6
[0.3.5]: https://github.com/maifeng/SPAR_measure/compare/v0.3.4...v0.3.5
[0.3.4]: https://github.com/maifeng/SPAR_measure/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/maifeng/SPAR_measure/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/maifeng/SPAR_measure/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/maifeng/SPAR_measure/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/maifeng/SPAR_measure/compare/v0.2.0...v0.3.0
