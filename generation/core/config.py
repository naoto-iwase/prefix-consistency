"""
Dataset and model configuration.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

# =====================================================================
# Model and dataset directories
# =====================================================================
# Auto-detected from repo location (repo's parent directory).
# Override with MODELS_DIR / DATASETS_DIR env vars to change locations.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PARENT = _REPO_ROOT.parent
MODELS_DIR = Path(os.environ.get("MODELS_DIR", str(_DEFAULT_PARENT / "models")))
DATASETS_DIR = Path(os.environ.get("DATASETS_DIR", str(_DEFAULT_PARENT / "datasets")))

# =====================================================================
# Constants
# =====================================================================
BASE_URL = "http://localhost:8100/v1"
MIN_FILE_SIZE_BYTES = 2048  # Skip files >= 2KB (resumability)
TOP_LOGPROBS = int(os.environ.get("TOP_LOGPROBS", "20"))  # vLLM max allowed value

# =====================================================================
# Dataset configs
# =====================================================================
_DATASET_CONFIGS = {
    # https://huggingface.co/datasets/MathArena/aime_2025
    "aime2025": {
        "data_file": str(DATASETS_DIR / "AIME2025" / "aime2025.jsonl"),
        "system_prompt": "You are a helpful assistant specialized in solving mathematical problems.",
        "prompt_suffix": "\n Please reason step by step, and put your final answer within \\boxed{}.",
        "complete_max_tokens": 100_000,
        "gold_key": "answer",
        "question_key": "problem",
        "llm_judge": False,
    },
    # https://huggingface.co/datasets/openai/frontierscience (olympiad subset)
    "frontierscience_olympiad": {
        "data_file": str(DATASETS_DIR / "FrontierScience" / "frontierscience_olympiad.jsonl"),
        "system_prompt": "You are an expert scientist solving olympiad-level problems in physics, chemistry, and biology.",
        "prompt_suffix": "\n Please reason step by step, and put your final answer within \\boxed{}.",
        "complete_max_tokens": 100_000,
        "gold_key": "answer",
        "question_key": "problem",
        "llm_judge": True,
    },
    # https://huggingface.co/datasets/MathArena/hmmt_feb_2026
    "hmmt": {
        "data_file": str(DATASETS_DIR / "HMMT" / "hmmt.jsonl"),
        "system_prompt": "You are a helpful assistant specialized in solving mathematical problems.",
        "prompt_suffix": "\n Please reason step by step, and put your final answer within \\boxed{}.",
        "complete_max_tokens": 100_000,
        "gold_key": "answer",
        "question_key": "problem",
        "llm_judge": True,
    },
    # https://huggingface.co/datasets/MathArena/brumo_2025
    "brumo": {
        "data_file": str(DATASETS_DIR / "BRUMO" / "brumo.jsonl"),
        "system_prompt": "You are a helpful assistant specialized in solving mathematical problems.",
        "prompt_suffix": "\n Please reason step by step, and put your final answer within \\boxed{}.",
        "complete_max_tokens": 100_000,
        "gold_key": "answer",
        "question_key": "problem",
        "llm_judge": True,
    },
}

DATASET_NAMES = list(_DATASET_CONFIGS.keys())


def get_dataset_config(dataset: str) -> Dict:
    """Return the dataset config dict for the given dataset name."""
    return _DATASET_CONFIGS[dataset]


# =====================================================================
# Model configs
# =====================================================================
_MODEL_TYPE_CONFIGS = {
    # GPT-OSS (custom CoT delimiters, reasoning_effort modes)
    "gpt-oss": {
        "cot_prefix": "<|channel|>analysis<|message|>",
        "cot_suffix": "<|end|>",
        "final_prefix": "<|start|>assistant<|channel|>final<|message|>",
        "max_context_length": 131072,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 40,
        "template_kwargs": {},
        "no_think_overrides": {},
    },
    # Nemotron-Nano-9B-v2 (Hybrid Mamba-2/Transformer, requires trust_remote_code)
    "nemotron-nano-v2": {
        "cot_prefix": "<think>",
        "cot_suffix": "</think>",
        "final_prefix": None,
        "max_context_length": 131072,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "template_kwargs": {},
        "no_think_overrides": {},
    },
    # Nemotron-3-Nano-30B (Hybrid Mamba-2/MoE/GQA, requires trust_remote_code)
    "nemotron-3-nano-30b": {
        "cot_prefix": "<think>",
        "cot_suffix": "</think>",
        "final_prefix": None,
        "max_context_length": 131072,  # native 256K, capped to 128K
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 20,
        "template_kwargs": {},
        "no_think_overrides": {},
    },
    # Ministral reasoning ([THINK]/[/THINK] delimiters)
    "ministral-reasoning": {
        "cot_prefix": "[THINK]",
        "cot_suffix": "[/THINK]",
        "final_prefix": None,
        "max_context_length": 131072,  # native 256K, capped to 128K
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
        "template_kwargs": {},
        "no_think_overrides": {},  # no-think not supported
    },
}

_MODEL_NAME_TO_TYPE = {
    "gpt-oss-20b": "gpt-oss",
    "gpt-oss-120b": "gpt-oss",
    "Nemotron-Nano-9B-v2": "nemotron-nano-v2",
    "Nemotron-3-Nano-30B-A3B": "nemotron-3-nano-30b",
    "Ministral-3-14B-Reasoning-2512": "ministral-reasoning",
}

_MODEL_CONFIGS = {
    name: {**_MODEL_TYPE_CONFIGS[model_type], "default_model_path": str(MODELS_DIR / name)}
    for name, model_type in _MODEL_NAME_TO_TYPE.items()
}

MODEL_NAMES = list(_MODEL_CONFIGS.keys())


# =====================================================================
# Model config builder
# =====================================================================

# Keys that go into api_params (used by call_completions / call_chat_completions)
_SAMPLING_KEYS = {"temperature", "top_p", "top_k", "presence_penalty"}


def build_model_config(
    model_name: str,
    no_think: bool = False,
    reasoning_effort: Optional[str] = None,
) -> Tuple[Dict, Dict]:
    """Build model config with runtime overrides applied.

    Returns (model_info, api_params). Does not mutate _MODEL_CONFIGS.

    model_info: model metadata (paths, CoT delimiters, template_kwargs, max_context_length).
    api_params: pre-built kwargs for API calls. Spread directly into
        client.completions.create / client.chat.completions.create via **api_params.
    """
    base = _MODEL_CONFIGS[model_name]

    # Resolve sampling overrides
    sampling = {k: base[k] for k in _SAMPLING_KEYS if k in base}
    if no_think:
        for k, v in base["no_think_overrides"].items():
            sampling[k] = v

    # Build api_params (ready to unpack into API call kwargs)
    api_params = {
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "extra_body": {"skip_special_tokens": False, "top_k": sampling["top_k"]},
    }
    if "presence_penalty" in sampling:
        api_params["presence_penalty"] = sampling["presence_penalty"]

    # Build model_info (everything else)
    template_kwargs = dict(base["template_kwargs"])
    if no_think:
        template_kwargs["enable_thinking"] = False
    if reasoning_effort is not None:
        template_kwargs["reasoning_effort"] = reasoning_effort

    model_info = {
        "default_model_path": base["default_model_path"],
        "max_context_length": base["max_context_length"],
        "cot_prefix": None if no_think else base["cot_prefix"],
        "cot_suffix": None if no_think else base["cot_suffix"],
        "final_prefix": base["final_prefix"],
        "template_kwargs": template_kwargs,
    }

    return model_info, api_params
