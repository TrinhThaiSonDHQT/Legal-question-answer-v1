import json
from pathlib import Path


def get_jsonl_data(file_path):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    data = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                # Common on cloud notebooks: the file is a Git LFS pointer instead of real dataset content.
                if line.startswith("version https://git-lfs.github.com/spec/v1"):
                    raise ValueError(
                        f"{path} appears to be a Git LFS pointer, not real JSONL data. "
                        "Download/attach the actual dataset file in Kaggle and point to that path."
                    ) from e

                preview = line[:120]
                raise ValueError(
                    f"Invalid JSON at {path}:{line_no}. "
                    f"Preview: {preview!r}. Original error: {e}"
                ) from e

    if not data:
        raise ValueError(f"No valid JSON records found in file: {path}")

    return data
