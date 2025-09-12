
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

from pattern_similarity import (
    PreprocessConfig,
    RepresentationConfig,
    attach_pattern_similarity,
)

# ---- User-provided helper (unchanged): JSON I/O ----
def load_json(path: str):
    p = Path(path)
    if not p.exists():
        return []
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []

def write_json(path: str, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---- Data loader for local files ----
def local_timeseries_loader(meta_doc: Dict[str, Any], info: Dict[str, Any]) -> pd.DataFrame:
    """
    Load a time series DataFrame from local files based on meta_doc/info.
    Customize this mapping as needed.
    Assumptions:
      - One CSV per tag set; file path pattern can be customized.
      - The CSV has either a datetime index or a 'time' column.
    Example file naming:
      ./data/{table_name}_{serial_number}.csv
    """
    table = meta_doc.get("table_name") or meta_doc.get("measurement") or "unknown"
    tags = info.get("filtered_tag_set", {})
    # Choose a key from tags to identify the file (customize as needed)
    tag_key, tag_val = next(iter(tags.items())) if tags else ("", "")
    # Build a path like: ./data/airQualityMeasurement_kw_n_OC3KL2300002.csv
    fname = f"{table}_{tag_val}.csv" if tag_val else f"{table}.csv"
    path = Path("./data") / fname
    if not path.exists():
        # Try a generic fallback
        path = Path("./data") / f"{table}.csv"
        if not path.exists():
            return pd.DataFrame()  # fail gracefully

    df = pd.read_csv(path)
    # Index normalization is handled in pattern_similarity.to_datetime_index
    return df


def set_pattern_similarity():
    """
    Orchestrate the end-to-end run:
      1) Load anchor and (optional) comparison meta JSONs
      2) Compute pattern similarities (preprocess -> representation -> similarity)
      3) Save updated anchor JSON with results written under statistical_info[*].pattern_similarity
    Customize file paths and configs below.
    """
    # -------- File paths (EDIT THESE) --------
    anchor_meta_path = Path("./meta/anchor.json")
    compare_meta_paths = [Path("./meta/compare_a.json"), Path("./meta/compare_b.json")]  # can be empty for self-compare
    out_path = Path("./meta/anchor_with_similarity.json")

    # -------- Load JSONs --------
    anchor_meta = load_json(str(anchor_meta_path))
    compare_meta_list: List[Dict[str, Any]] = []
    for p in compare_meta_paths:
        if p.exists():
            compare_meta_list.append(load_json(str(p)))

    # If no compare metas provided, default to self-compare mode
    if not compare_meta_list:
        compare_meta_list = [anchor_meta]

    # -------- Configs --------
    pre_cfg = PreprocessConfig(
        impute_method="time",
        impute_limit=None,
        scaling={"flag": True, "method": "standard"},  # minmax/standard/maxabs/robust
    )
    rep_cfg = RepresentationConfig(
        method="pca",            # pca | tsne | autoencoder | lstm | tcn | transformer
        params={"n_components": 10},
        summarize="mean",        # mean | median
    )
    similarity = "cosine"        # cosine | euclidean

    # -------- Run --------
    updated_anchor = attach_pattern_similarity(
        anchor_meta=anchor_meta,
        compared_meta=compare_meta_list,
        loader=local_timeseries_loader,
        pre_cfg=pre_cfg,
        rep_cfg=rep_cfg,
        sim=similarity,
    )

    # -------- Save --------
    write_json(str(out_path), updated_anchor)


if __name__ == "__main__":
    set_pattern_similarity()
