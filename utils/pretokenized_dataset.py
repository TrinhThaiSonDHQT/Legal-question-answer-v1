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

    def _read_header_bytes(p: Path, n: int = 256) -> bytes:
        with p.open("rb") as f:
            return f.read(n)

    def _is_lfs_pointer(header: bytes) -> bool:
        return header.startswith(b"version https://git-lfs.github.com/spec/v1")

    def _looks_like_text(header: bytes) -> bool:
        if not header:
            return False
        try:
            decoded = header.decode("utf-8")
        except UnicodeDecodeError:
            return False
        # Heuristic: plain text headers usually indicate wrong/corrupted cache format.
        return all((ch.isprintable() or ch in "\r\n\t") for ch in decoded)

    header = _read_header_bytes(path)
    if _is_lfs_pointer(header):
        raise RuntimeError(
            "Pretokenized cache is a Git LFS pointer, not the real .pt binary: "
            f"{path}. Run `git lfs pull` then retry."
        )

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
        try:
            cache = torch.load(path, map_location=map_location, weights_only=False)
        except Exception as unsafe_err:
            if _looks_like_text(header):
                preview = header[:80].decode("utf-8", errors="replace").replace("\n", "\\n")
                raise RuntimeError(
                    "Pretokenized cache is not a valid torch binary (text-like header). "
                    f"cache_path={path}, header_preview='{preview}'. "
                    "If this file comes from Git LFS, run `git lfs pull`."
                ) from unsafe_err

            raise RuntimeError(
                "Pretokenized cache load failed in both safe and unsafe modes. "
                f"cache_path={path}. Safe error: {safe_err}. Unsafe error: {unsafe_err}"
            ) from unsafe_err

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
