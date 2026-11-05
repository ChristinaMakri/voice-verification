"""Normalise a dataset into one table: one row per recording.

Columns: utt_id, path, speaker, gender, age, duration_s, sentence_id, dataset.
Everything downstream works on this table and never on dataset-specific files.
"""

from pathlib import Path

import pandas as pd

COLUMNS = ["utt_id", "path", "speaker", "gender", "age", "duration_s", "sentence_id", "dataset"]

_GENDER = {
    "male": "male",
    "male_masculine": "male",
    "female": "female",
    "female_feminine": "female",
}


def _read_tsv(path: Path) -> pd.DataFrame:
    # quoting=3 (QUOTE_NONE): Common Voice sentences contain stray quote characters.
    return pd.read_csv(path, sep="\t", quoting=3, low_memory=False, dtype=str)


def load_commonvoice(root: Path, split_file: str = "validated.tsv") -> pd.DataFrame:
    """``root`` is the locale folder, e.g. ``cv-corpus-27.0-.../el``."""
    root = Path(root)
    clips = _read_tsv(root / split_file)
    durations = _read_tsv(root / "clip_durations.tsv")
    durations.columns = ["path", "ms"]

    df = clips.merge(durations, on="path", how="left")
    missing = df["ms"].isna().sum()
    if missing:
        raise ValueError(f"{missing} clips have no entry in clip_durations.tsv")

    out = pd.DataFrame(
        {
            "utt_id": df["path"].str.replace(r"\.mp3$", "", regex=True),
            "path": [str(root / "clips" / p) for p in df["path"]],
            "speaker": df["client_id"],
            "gender": df["gender"].map(_GENDER),  # anything else -> NaN (undeclared)
            "age": df["age"],
            "duration_s": df["ms"].astype(float) / 1000.0,
            "sentence_id": df.get("sentence_id"),
            "dataset": f"commonvoice:{root.name}",
        }
    )
    # One declared gender per speaker (Common Voice repeats it per clip; a few
    # speakers change it between clips, so take the most frequent value).
    out["gender"] = out.groupby("speaker")["gender"].transform(_mode_or_nan)
    out["age"] = out.groupby("speaker")["age"].transform(_mode_or_nan)
    if out["utt_id"].duplicated().any():
        raise ValueError("duplicate clip paths in manifest")
    return out[COLUMNS].reset_index(drop=True)


def _mode_or_nan(s: pd.Series):
    m = s.dropna().mode()
    return m.iat[0] if len(m) else float("nan")
