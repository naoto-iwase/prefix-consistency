# generation

Generates initial answers, regenerates from truncated CoT, and annotates each answer with an extracted answer, logprob statistics, and model-reported confidences.

## Pipeline overview

```
generate.py → regenerate.py → enrich.py → convert_to_jsonl.py → analysis/
  Phase 1        Phase 2       Phase 3         Phase 4           Phase 5
  (GPU)          (GPU)       (GPU or not)      (no GPU)          (no GPU)
```

Phase 2 has two variants:
- `regenerate.py`: Fixed-percentage or fixed-token truncation of the CoT, then regenerate.
- `regenerate_from_markers.py`: Truncate at transition markers (Hammoud et al., 2025), then regenerate.

Phase 3 enrichment is handled by `enrich.py`, which runs five steps sequentially:
1. Extracted answer (regex, no GPU)
2. Logprob confidence / DeepConf + Response Probability (from .npy, no GPU)
3. Verbal 0-100 confidence / CISC (LLM query, GPU needed)
4. Binary confidence / CISC (LLM query, GPU needed) -- writes both Verbal Binary and P(True) headers from a single API call
5. Answer map + canonical answer (LLM judge for judge datasets, otherwise no GPU)

Use `--skip-extraction`, `--skip-confidence`, `--skip-verbal-0-100`, `--skip-binary`, `--skip-answer-map` to skip individual steps. Use `--regen-verbal-0-100` or `--regen-binary` to also run those confidence types on regen files.

Shared library code lives in `core/`. The shell script `run_regen.sh` orchestrates the full pipeline.

<details>
<summary>Full directory structure</summary>

```
generation/
├── core/                              # Shared library modules (imported, not run directly)
│   ├── config.py                      # Model/dataset configurations and constants
│   ├── io.py                          # File I/O (read_*/write_* for answers, headers, logprobs, metadata)
│   ├── text.py                        # Prompt rendering, CoT parsing, naming helpers
│   ├── api.py                         # OpenAI API call wrapper
│   ├── answer_extraction.py           # Answer extraction from \boxed{} and normalization
│   ├── answer_map.py                  # Answer normalization map construction
│   ├── llm_judge.py                   # LLM-based answer equivalence judge
│   ├── logprob_confidence.py          # DeepConf logprob confidence (Fu et al., ICLR 2026)
│   ├── transition_markers.py          # Transition marker detection (Hammoud et al., 2025)
│   └── verbalized_confidence.py       # Verbalized confidence + P(True) (Taubenfeld+ 2025, Kadavath+ 2022)
│
│   # --- Main pipeline (run in order) ---
├── generate.py                        # Phase 1:  Initial answer generation (GPU)
├── regenerate.py                      # Phase 2a: Truncate CoT + regenerate (GPU)
├── regenerate_from_markers.py         # Phase 2b: Marker-based truncation + regenerate (GPU)
├── enrich.py                          # Phase 3:  Confidence + verbalized + answer map
│
│   # --- Conversion (no GPU needed) ---
├── convert_to_jsonl.py                # Phase 4:  Pure file conversion -> JSONL
│                                      #   Reads: .txt headers, answer_map.json
│                                      #   All data pre-computed by enrich.py
│
│   # --- Shell scripts ---
├── run_regen.sh                       # Pipeline orchestrator (Phase 1-5)
├── start_vllm_server.sh               # Start/stop vLLM server
└── start_sglang_server.sh             # Start/stop SGLang server (alternative backend)
```

</details>

## Setup

The pipeline expects model weights at `<parent>/models/` and datasets at `<parent>/datasets/`, where `<parent>` is the parent of this repository:

```
<parent>/                  # auto-detected as workspace root
├── prefix-consistency/    # this repo
├── models/                # model weights
└── datasets/              # evaluation datasets
```

Override with `MODELS_DIR` / `DATASETS_DIR` env vars or by symlinking.

Download:

```bash
uv run --script utils/download_datasets.py                              # all datasets
uv run --script utils/download_datasets.py -- aime2025 frontierscience  # subset
uv run --script utils/download_models.py                                # all models
uv run --script utils/download_models.py -- gpt-oss-20b gpt-oss-120b    # subset
```

vLLM and torch are installed in a dedicated venv (CUDA-version-specific, managed separately from the project's `uv sync`; see the [vLLM installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/)):

```bash
uv venv .venv-vllm --python 3.12 --seed --managed-python
uv pip install --python .venv-vllm/bin/python vllm --torch-backend=auto
```

If multiple machines with different CUDA versions share the same storage, create the venv on machine-local storage to avoid conflicts.

## Quick start

```bash
cd <repo-root>
TAG=gpt-oss-20b_aime2025
mkdir -p logs

# 1. Start vLLM server (boots in background; --venv points to vllm-specific venv)
nohup bash generation/start_vllm_server.sh gpt-oss-20b --dp --venv .venv-vllm \
  > "logs/${TAG}_vllm.log" 2>&1 &

# 2. Run pipeline (this can start immediately because run_regen.sh waits for vLLM internally)
nohup bash generation/run_regen.sh \
  --tag "$TAG" \
  --dataset aime2025 --model gpt-oss-20b \
  --remove-pcts 50 --regen-count 3 \
  --n-init-gens 128 \
  > "logs/${TAG}_regen.log" 2>&1 &

# 3. Monitor
tail -f "logs/${TAG}_regen.log"
```

```bash
bash generation/run_regen.sh --help
bash generation/start_vllm_server.sh --help
bash generation/start_sglang_server.sh --help
```

<details>
<summary>Using SGLang instead of vLLM</summary>

SGLang is provided as an alternative backend. It requires a separate venv because sglang and vllm have conflicting dependencies.

**One-time setup:**

```bash
uv venv .venv-sglang --python 3.12
uv pip install --python .venv-sglang/bin/python sglang
pip install ninja  # required for JIT kernel compilation
uv pip install --python .venv-sglang/bin/python "nvidia-cudnn-cu12==9.16.0.29"  # fix PyTorch 2.9 + cuDNN compat
```

**Usage:**

```bash
nohup bash generation/start_sglang_server.sh gpt-oss-20b --dp \
  --venv .venv-sglang \
  > "logs/${TAG}_sglang.log" 2>&1 &
```

`run_regen.sh` is backend-agnostic (OpenAI-compatible API), so no changes needed on the pipeline side.

</details>

### Process management

<details>
<summary>Commands for listing, killing, and cancelling pipelines</summary>

```bash
# List running pipelines
ps aux | grep run_regen | grep -v grep

# Kill parent + children (prevents orphan processes)
pkill -P <PID> && kill <PID>
```

Killing only the parent `run_regen.sh` leaves child Python processes running
as orphans. Always use `pkill -P` to kill children first.

</details>

### Experiment templates

<details>
<summary>New experiment</summary>

```bash
TAG=gpt-oss-20b_hmmt

nohup bash generation/start_vllm_server.sh gpt-oss-20b \
  --venv .venv-vllm --max-model-len=40960 --dp \
  > "logs/${TAG}_vllm.log" 2>&1 &

nohup bash generation/run_regen.sh \
  --tag "$TAG" \
  --dataset hmmt --model gpt-oss-20b \
  --remove-pcts 50 --regen-count 3 \
  --n-init-gens 128 \
  --parallel 16 \
  > "logs/${TAG}_regen.log" 2>&1 &
```

</details>

<details>
<summary>Marker-based regeneration (BLA)</summary>

Marker-based regen is integrated into the pipeline via `--marker-regen` on `run_regen.sh`.
Per-problem token budget early stopping is enabled by default.
Use `--no-marker-budget-stop` to disable it.

```bash
# Integrated: marker regen as part of run_regen.sh
bash generation/run_regen.sh \
  --tag "$TAG" \
  --dataset aime2025 --model gpt-oss-20b \
  --remove-pcts 50 --regen-count 3 \
  --n-init-gens 128 \
  --marker-regen --marker-regen-count 1

# Standalone: regenerate_from_markers.py directly
uv run python generation/regenerate_from_markers.py \
  --dataset aime2025 --model-name gpt-oss-20b \
  --init-dir "$INIT_DIR" \
  --regen-count 1 \
  --out-dir "$MARKER_DIR" \
  --save-logprobs \
  --parallel 128 \
  --budget-stop
```

</details>

<details>
<summary>Resume</summary>

Specify the old TAG. Existing files are auto-skipped.

```bash
OLD_TAG=gpt-oss-20b_aime2025

nohup bash generation/run_regen.sh \
  --tag "$OLD_TAG" \
  --dataset aime2025 --model gpt-oss-20b \
  --remove-pcts 50 --regen-count 3 \
  --n-init-gens 128 \
  --parallel 16 \
  > "logs/${OLD_TAG}_resume.log" 2>&1 &
```

</details>


## Output structure

Removal tag encodes the truncation setting: `rm{N}pct_{scope}` or `rm{N}tok_{scope}`.
Examples: `rm50pct_full`, `rm1000tok_cot`.

```
generation/${TAG}_data/
├── init/                       # generate.py output
│   ├── *.txt                   # answer files
│   ├── *.txt.npy               # top-k logprobs sidecar
│   ├── *.txt.tok_logprobs.npy  # per-token generated logprobs
│   └── answer_map.json         # answer normalization (enrich.py)
├── regen_${RTAG}/              # regenerate.py output (e.g. regen_rm50pct_full/)
└── regen_from_markers/         # regenerate_from_markers.py output

analysis/${TAG}_jsonl/
├── analysis_${DATASET}_${MODEL}_init.jsonl
├── analysis_${DATASET}_${MODEL}_${RTAG}_x${REGEN_COUNT}.jsonl
└── analysis_${DATASET}_${MODEL}_markers_x${REGEN_COUNT}.jsonl
```

Downstream outputs are documented in [analysis/README.md](../analysis/README.md).

### File formats

<details>
<summary>.txt (answer files)</summary>

`generate.py` and `regenerate.py` produce a `.txt` file per answer. Enrichment scripts add header fields after generation.

```
Dataset: aime2025
Problem Number: 0
Answer Index: 3
Gold Answer: 42
Generated Tokens: 1500
CoT Tokens: 1200
Final Tokens: 300
Extracted Answer: 735
Mean Conf: 0.847321
Bottom10 Conf: 0.623145
Tail Conf: 0.791234
First Token Conf: 0.912345
Block Min Conf: 0.567890
Response Probability: 0.00234567
Verbal 0-100 Confidence: 85
Verbal 0-100 Actual Tokens: 30
Verbal 0-100 Min Tokens: 12
Verbal Binary Confidence: 1
Binary Query Actual Tokens: 18
Binary Query Min Tokens: 6
P(True): 0.923456
Canonical Answer: 735
Prompt: <rendered prompt text>
==================================================
<model output (CoT + final answer)>
```

Header fields added at different stages:
- Generation (`generate.py` / `regenerate.py`):
  `Generated Tokens`, `CoT Tokens`, `Final Tokens`
- Enrichment step 1 (extracted answer):
  `Extracted Answer` (set to `PARSE_FAILED` when extraction fails)
- Enrichment step 2 (logprob confidence + response probability):
  `Mean Conf`, `Bottom10 Conf`, `Tail Conf`, `First Token Conf`, `Block Min Conf`, `Response Probability`
- Enrichment step 3 (verbal 0-100 confidence):
  `Verbal 0-100 Confidence`, `Verbal 0-100 Actual Tokens`, `Verbal 0-100 Min Tokens`
- Enrichment step 4 (binary confidence, single API call):
  `Verbal Binary Confidence`, `P(True)`, `Binary Query Actual Tokens`, `Binary Query Min Tokens`
- Enrichment step 5 (answer map):
  `Canonical Answer`

`*_Actual_Tokens` is the generated-token count of the confidence query
itself; `*_Min_Tokens` is the prompt-truncation lower bound used by the
analysis side to enforce cost-fair comparison.

Additional headers for regen files:
- `regenerate.py`: `Init Tokens`, `Kept Tokens`, `Cut Tokens`
- `regenerate_from_markers.py`: `Marker Index`, `Marker Position`, `Init Tokens`, `Kept Tokens`, `Cut Tokens`

Filename conventions:

```
{dataset}_{model}_prob{P}_answer{A}.txt                        # init
{dataset}_{model}_prob{P}_answer{A}_regen{R}.txt               # fixed-% regen
{dataset}_{model}_prob{P}_answer{A}_marker{B}_regen{R}.txt     # marker regen
```

</details>

<details>
<summary>.txt.npy and .txt.tok_logprobs.npy (logprob sidecars)</summary>

Saved alongside `.txt` files by `generate.py` / `regenerate.py` with `--save-logprobs`.

- `.txt.npy`: `float32` array of shape `(T, K)` where T = generated tokens, K = top-k (default 20). Each row holds top-k log-probabilities sorted descending. Used by `enrich.py` step 2 to compute DeepConf metrics.

- `.txt.tok_logprobs.npy`: `float64` 1D array of length T. The log-probability of each actually generated token (regardless of sampling temperature).

</details>

<details>
<summary>answer_map.json</summary>

Generated by `enrich.py` step 5, saved in the init directory. Maps each unique extracted answer to its canonical (normalized) form, and includes normalized gold answers. For judge datasets (FrontierScience, HMMT, etc.), equivalent answers are clustered via LLM judge.

```jsonc
{
  "answers": {
    "\\dfrac{1}{2}": "\\frac{1}{2}",   // raw -> canonical
    "0.5": "\\frac{1}{2}",
    "735": "735"
  },
  "golds": {
    "0": "\\frac{1}{2}",               // problem_num -> normalized gold
    "1": "735"
  }
}
```

</details>

<details>
<summary>JSONL (convert_to_jsonl.py output)</summary>

Each line is a JSON object representing one problem. Confidence fields are auto-included when the corresponding .txt headers are present.

```jsonc
{
  "dataset": "aime2025",
  "model_name": "gpt-oss-20b",
  "problem_num": 0,
  "gold_answer": "42",

  // Each element: [answer, generated_tokens, cot_tokens, final_tokens]
  // cot_tokens/final_tokens are null if not recorded in the file header.
  "all_answers": [["42", 1500, 1200, 300], ...],

  // --- Regen answers (optional, requires --regen-dir) ---
  // regen_answers[i][j] = [answer, tokens, cot_tokens, final_tokens,
  //                         init_tokens, kept_tokens, cut_tokens]
  //   i = init answer index, j = regen iteration index
  "regen_answers": [[["42", 800, 600, 200, 1500, 750, 750], ...], ...],

  // --- Marker regen answers (optional, requires --marker-regen-dir) ---
  // marker_regen_answers[i][j] = [answer, tokens, cot_tokens, final_tokens,
  //                                marker_idx, regen_idx, marker_position,
  //                                init_tokens, kept_tokens, cut_tokens]
  "marker_regen_answers": [[["42", 800, 600, 200, 3, 0, 1024, 8192, 1024, 7168], ...], ...],

  // --- DeepConf logprob confidence (auto-included if headers present) ---
  "confidences": [{"mean_conf": 0.85, "bottom10_conf": 0.6, ...}, ...],
  "regen_confidences": [[{"mean_conf": 0.9, ...}, ...], ...],
  "marker_regen_confidences": [[{"mean_conf": 0.88, ...}, ...], ...],

  // --- Response Probability: exp(mean(token_logprobs)) ---
  "response_probabilities": [0.0023, 0.0018, ...],
  "regen_response_probabilities": [[0.0021, ...], ...],
  "marker_regen_response_probabilities": [[0.0019, ...], ...],

  // --- Verbal 0-100: self-rated confidence on a 0-100 scale ---
  "verbal_0_100_confidences": [85, 72, null, ...],
  "verbal_0_100_actual_tokens": [30, 28, ...],   // tokens generated by the 0-100 query
  "verbal_0_100_min_tokens":    [12, 12, ...],   // prompt-truncation lower bound
  "regen_verbal_0_100_confidences": [[90, 85, ...], ...],
  "marker_regen_verbal_0_100_confidences": [[80, 75, ...], ...],

  // --- Verbal Binary: text-parsed 0/1 confidence ---
  "verbal_binary_confidences": [1.0, 0.0, 1.0, ...],
  "binary_query_actual_tokens": [18, 16, ...],   // shared by Verbal Binary and P(True)
  "binary_query_min_tokens":    [ 6,  6, ...],
  "regen_verbal_binary_confidences": [[1.0, 0.0, ...], ...],
  "marker_regen_verbal_binary_confidences": [[1.0, 1.0, ...], ...],

  // --- P(True): logprob-extracted P("1") from the final answer token ---
  // Reuses binary_query_*_tokens above (same API call as Verbal Binary).
  "p_true_confidences": [0.92, 0.15, 0.87, ...],
  "regen_p_true_confidences": [[0.88, 0.12, ...], ...],
  "marker_regen_p_true_confidences": [[0.91, 0.20, ...], ...],

  // --- Metadata ---
  "reasoning_effort": "high",           // optional
  "no_think": true                      // optional
}
```

DeepConf logprob confidence dict keys: `mean_conf`, `bottom10_conf`, `tail_conf`, `first_token_conf`, `block_min_conf`.

</details>
