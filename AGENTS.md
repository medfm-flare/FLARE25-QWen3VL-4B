# Repository Guidelines

## Project Structure & Module Organization
`Qwen3-VL/qwen-vl-finetune/` hosts the upstream finetuning stack—avoid reshuffling its packages so training scripts continue to import cleanly. `data_conversion/` contains the conversion and validation scripts that read from `organized_dataset/` and write JSON into `converted_data/`; keep that trio in sync with the folder schema shown in `README.md`. Root `pyproject.toml` and `uv.lock` lock the runtime, while `main.py` is only a stub wrapper.

## Build, Test, and Development Commands
Install dependencies with `uv sync`. Generate starter conversions via `uv run python data_conversion/convert_flare_to_qwen3vl.py --datasets starter --output_dir converted_data`. Validate outputs using `uv run python data_conversion/validate_conversion.py --converted_dir converted_data`. Switch into `Qwen3-VL/qwen-vl-finetune` to launch finetuning with `bash scripts/sft_qwen3_4b_flare.sh` after updating dataset lists. `uv run python data_conversion/dataset_configs.py` prints modality metadata when sanity-checking coverage.

## Coding Style & Naming Conventions
Python modules use 4-space indentation, type hints, and pathlib-based path handling; keep docstrings descriptive and sentence-cased. Prefer structured logging (`logging.getLogger`) to bare prints, and reuse the small helper patterns found in `convert_flare_to_qwen3vl.py`. Dataset registry keys follow `flare_<dataset>%<sample_rate>`—maintain that casing when introducing new splits. Match upstream Qwen3 naming (camelCase arguments, terse comments) when touching files under `Qwen3-VL/`.

## Testing Guidelines
Run `uv run python Qwen3-VL/qwen-vl-finetune/scripts/test_data_loading.py` before every PR; it checks annotation JSONs, opens sample images, and smoke-tests the processor. Start with the starter datasets, then scale to all 19 once the lightweight pass is clean. Use `uv run python data_conversion/validate_conversion.py --files <json>` for focused checks after parser edits. When DeepSpeed configs change, execute a short ZeRO-2 training dry run to ensure launches still succeed.

## Commit & Pull Request Guidelines
Commit subjects follow the repository’s short, imperative style (`Add Apache 2.0 license`, `clean the readme`); include dataset or script names when relevant. Squash unrelated tweaks into separate commits. Pull requests should outline dataset coverage, note any required checkpoints, and attach key logs or WandB run URLs. Link to challenge issues or Hugging Face discussions whenever mirroring upstream fixes.

## Data & Security Notes
Do not commit raw datasets, converted JSON, or credential files; verify `.gitignore` catches new artifacts. Keep absolute paths, API keys, and WANDB tokens in local configs rather than hardcoding them. Scrub PHI-bearing filenames and private URLs before sharing logs in issues or PRs.
