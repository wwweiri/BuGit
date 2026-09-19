from __future__ import annotations

from pathlib import Path

from huggingface_hub import get_token, hf_hub_download

from bugit.paths import RAW_DATA_DIR

HF_REPO_ID = "Weiri/ApacheJIT"
HF_REPO_TYPE = "dataset"

DEFAULT_FILES: tuple[str, ...] = (
    "apachejit_train.csv",
    "apachejit_test_large.csv",
)


def download_apachejit_file(
    filename: str,
    dest_dir: str | Path = RAW_DATA_DIR,
    repo_id: str = HF_REPO_ID,
    token: str | None = None,
) -> Path:
    token = token or get_token()
    if token is None:
        raise RuntimeError(
            "Could not find the token: not in the argument, not in HF_TOKEN, and not in the cache "
            "`hf auth login`. Check `hf auth whoami` to make sure you're actually logged in."
        )

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    local_path = hf_hub_download(
        repo_id=repo_id,
        repo_type=HF_REPO_TYPE,
        filename=filename,
        local_dir=dest_dir,
        token=token,
    )
    return Path(local_path)


def download_apachejit(
    filenames: tuple[str, ...] = DEFAULT_FILES,
    dest_dir: str | Path = RAW_DATA_DIR,
    repo_id: str = HF_REPO_ID,
    token: str | None = None,
) -> list[Path]:
    return [
        download_apachejit_file(name, dest_dir=dest_dir, repo_id=repo_id, token=token)
        for name in filenames
    ]


if __name__ == "__main__":
    if HF_REPO_ID.startswith("CHANGE_ME"):
        raise SystemExit(
            "Replace HF_REPO_ID in download.py with the actual repo_id "
            "of your private dataset before running the script."
        )

    paths = download_apachejit()
    for p in paths:
        print(f"Downloaded: {p}")