from pathlib import Path
from typing import Dict, Any, Optional, cast

import torch
from torch.utils.data import Dataset


class PreTokenizedRetrieverDataset(Dataset[Dict[str, Dict[str, torch.Tensor]]]):
    def __init__(self, split_cache: Dict[str, torch.Tensor]):
        self.q_input_ids = split_cache["q_input_ids"]
        self.q_attention_mask = split_cache["q_attention_mask"]
        self.p_input_ids = split_cache["p_input_ids"]
        self.p_attention_mask = split_cache["p_attention_mask"]
        self.n_input_ids = split_cache["n_input_ids"]
        self.n_attention_mask = split_cache["n_attention_mask"]

    def __len__(self) -> int:
        return self.q_input_ids.size(0)

    def __getitem__(self, idx: int) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "q_tok": {
                "input_ids": self.q_input_ids[idx],
                "attention_mask": self.q_attention_mask[idx],
            },
            "p_tok": {
                "input_ids": self.p_input_ids[idx],
                "attention_mask": self.p_attention_mask[idx],
            },
            "n_tok": {
                "input_ids": self.n_input_ids[idx],
                "attention_mask": self.n_attention_mask[idx],
            },
        }


def load_pretokenized_cache(
    cache_path: str,
    map_location: str = "cpu",
    allow_unsafe_fallback: bool = True,
) -> Dict[str, Any]:
    path = Path(cache_path)

    # PyTorch >=2.6 defaults to weights_only=True. Try safe mode first.
    try:
        cache = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # Older PyTorch versions may not support the weights_only argument.
        cache = torch.load(path, map_location=map_location)
    except Exception as safe_err:
        if not allow_unsafe_fallback:
            raise RuntimeError(
                "Safe pretokenized cache load failed and unsafe fallback is disabled. "
                f"cache_path={path}. Original error: {safe_err}"
            ) from safe_err

        # Backward-compatible fallback for trusted local cache files.
        cache = torch.load(path, map_location=map_location, weights_only=False)

    if not isinstance(cache, dict):
        raise TypeError(f"Expected pretokenized cache to be a dict, got {type(cache)}")

    return cast(Dict[str, Any], cache)


def validate_cache_meta(
    cache: Dict[str, Any],
    model_name: Optional[str] = None,
    max_q_len: Optional[int] = None,
    max_c_len: Optional[int] = None,
) -> None:
    meta = cache.get("meta", {})
    if model_name is not None and meta.get("model_name") != model_name:
        raise ValueError(f"model_name mismatch: expected {model_name}, got {meta.get('model_name')}")
    if max_q_len is not None and meta.get("max_q_len") != max_q_len:
        raise ValueError(f"max_q_len mismatch: expected {max_q_len}, got {meta.get('max_q_len')}")
    if max_c_len is not None and meta.get("max_c_len") != max_c_len:
        raise ValueError(f"max_c_len mismatch: expected {max_c_len}, got {meta.get('max_c_len')}")
