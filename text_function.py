
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

# === Representation methods (do not modify, just import & call) ===
try:
    from representation_methods import (
        pca_representation,
        tsne_representation,
        autoencoder_representation,
        lstm_representation,
        tcn_representation,
        transformer_representation,
    )
except Exception as e:  # graceful fallback for environments where the module isn't present
    pca_representation = tsne_representation = autoencoder_representation = None
    lstm_representation = tcn_representation = transformer_representation = None


# ============================
# Utilities
# ============================

def make_entry(doc: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a meta+info pair into a lightweight identifier entry.
    - doc: dataset-level metadata (contains bucket_name or bucket, table_name or measurement)
    - info: one element from statistical_info list (contains filtered_tag_set)
    Returns {"bucket", "measurement", "tags"}.
    """
    bucket = doc.get("bucket_name") or doc.get("bucket") or ""
    measurement = doc.get("table_name") or doc.get("measurement") or ""
    tags = info.get("filtered_tag_set") or {}
    return {"bucket": bucket, "measurement": measurement, "tags": dict(tags)}


def to_datetime_index(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Ensure DataFrame index is a proper DateTimeIndex (sorted, unique). If a "time" column
    exists and index is not datetime-like, use it as the index.
    """
    if not inplace:
        df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        # Try "time" column first, then the first column
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
            df = df.set_index("time")
        else:
            # Try to coerce the index itself
            idx = pd.to_datetime(df.index, errors="coerce", utc=True)
            df.index = idx
    # Drop rows where index is NaT and sort
    df = df[~df.index.isna()].sort_index()
    # Deduplicate by taking the last occurrence
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]
    return df


def infer_freq_from_index(idx: pd.DatetimeIndex) -> Optional[pd.Timedelta]:
    """
    Infer a representative sampling interval from a DateTimeIndex.
    Use median of diffs for robustness.
    """
    if len(idx) < 2:
        return None
    diffs = np.diff(idx.view("int64"))  # nanoseconds
    if len(diffs) == 0:
        return None
    # Convert to Timedelta (nanoseconds)
    td = pd.to_timedelta(np.median(diffs), unit="ns")
    if pd.isna(td) or td <= pd.Timedelta(0):
        return None
    return td


def choose_common_frequency(freq_a: Optional[pd.Timedelta],
                            freq_b: Optional[pd.Timedelta]) -> Optional[pd.Timedelta]:
    """
    Pick the coarser (longer) frequency between two. If one is missing, return the other.
    If both missing, return None.
    """
    if freq_a is None and freq_b is None:
        return None
    if freq_a is None:
        return freq_b
    if freq_b is None:
        return freq_a
    return max(freq_a, freq_b)  # longer interval


def make_static_frequency_data(data: pd.DataFrame, freq: str | pd.Timedelta) -> pd.DataFrame:
    """
    Resample a DataFrame to a static frequency using mean aggregation.
    - Convert all columns to numeric (coerce errors to NaN)
    - Resample to freq and take mean
    - Enforce the frequency using asfreq
    NOTE: Mirrors user's provided behavior.
    """
    out = data.copy()
    out = out.sort_index()
    out = out.apply(pd.to_numeric, errors="coerce")
    if isinstance(freq, pd.Timedelta):
        # Convert Timedelta to pandas offset string if possible
        # fallback: use freq directly in resample
        resampled = out.resample(freq).mean()
        return resampled.asfreq(freq=freq)
    else:
        resampled = out.resample(freq).mean()
        return resampled.asfreq(freq=freq)


def simple_impute(data: pd.DataFrame, method: str = "time", limit: Optional[int] = None) -> pd.DataFrame:
    """
    Simple interpolation-based imputation.
    - method: pandas interpolate method (e.g., 'time', 'linear', 'nearest', ...)
    - limit: max consecutive NaNs to fill
    Direction is both sides by default (as per user's snippet).
    """
    return data.interpolate(method=method, limit=limit, limit_direction="both")


def scale_dataframe(data: pd.DataFrame, scaling_param: Dict[str, Any]) -> pd.DataFrame:
    """
    Scale numeric columns by the chosen scaler.
    scaling_param example: {"flag": True, "method": "standard"}.
    Supported methods: minmax, standard, maxabs, robust.
    """
    flag = scaling_param.get("flag", False)
    if not flag:
        return data

    method = scaling_param.get("method", "standard").lower()
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
    scalers = {
        "minmax": MinMaxScaler,
        "standard": StandardScaler,
        "maxabs": MaxAbsScaler,
        "robust": RobustScaler,
    }
    if method not in scalers:
        raise ValueError(f"Unsupported scaling method: {method}")

    scaler = scalers[method]()
    numeric = data.select_dtypes(include=[np.number])
    scaled = pd.DataFrame(
        scaler.fit_transform(numeric.values),
        columns=list(numeric.columns),
        index=data.index,
    )
    # Preserve non-numeric columns if any (rare for time series)
    result = data.copy()
    result[scaled.columns] = scaled
    return result


def run_representation(data: pd.DataFrame,
                       method: str = "pca",
                       rep_params: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Run a representation method (imported from representation_methods.py) on a processed DataFrame.
    Returns a 2D numpy array of embeddings (time steps x embedding_dim).
    """
    rep_params = rep_params or {}
    # Ensure numeric-only and drop all-NaN columns
    X = data.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    if X.empty:
        raise ValueError("No numeric columns available for representation.")

    method = method.lower()
    if method == "pca" and pca_representation is not None:
        return pca_representation(X, **{k: v for k, v in rep_params.items() if k in ("n_components",)})
    if method == "tsne" and tsne_representation is not None:
        return tsne_representation(X, **{k: v for k, v in rep_params.items() if k in ("n_components",)})
    if method == "autoencoder" and autoencoder_representation is not None:
        return autoencoder_representation(X, **{k: v for k, v in rep_params.items()
                                                if k in ("encoding_dim", "epochs", "batch_size")})
    if method == "lstm" and lstm_representation is not None:
        return lstm_representation(X, **{k: v for k, v in rep_params.items()
                                         if k in ("hidden_dim", "output_dim", "num_layers", "epochs", "batch_size")})
    if method == "tcn" and tcn_representation is not None:
        return tcn_representation(X, **{k: v for k, v in rep_params.items()
                                        if k in ("num_filters", "kernel_size", "output_dim", "epochs", "batch_size")})
    if method == "transformer" and transformer_representation is not None:
        return transformer_representation(X, **{k: v for k, v in rep_params.items()
                                                if k in ("embed_dim", "num_heads", "num_layers", "output_dim",
                                                         "epochs", "batch_size")})

    raise ValueError(f"Unsupported or unavailable representation method: {method}")


def summarize_embedding(embeddings: np.ndarray, mode: str = "mean") -> np.ndarray:
    """
    Reduce a (T x D) embedding sequence to a single vector.
    Supported: 'mean' (default), 'median'.
    """
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be 2D (time x dim).")
    if mode == "mean":
        return embeddings.mean(axis=0)
    elif mode == "median":
        return np.median(embeddings, axis=0)
    else:
        raise ValueError(f"Unsupported summarize mode: {mode}")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D vectors."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


@dataclass
class PreprocessConfig:
    impute_method: str = "time"   # pandas interpolate method
    impute_limit: Optional[int] = None
    scaling: Dict[str, Any] = None  # e.g., {"flag": True, "method": "standard"}

    def __post_init__(self):
        if self.scaling is None:
            self.scaling = {"flag": True, "method": "standard"}


@dataclass
class RepresentationConfig:
    method: str = "pca"  # pca | tsne | autoencoder | lstm | tcn | transformer
    params: Dict[str, Any] = None   # method-specific parameters
    summarize: str = "mean"         # how to reduce T x D to 1 x D

    def __post_init__(self):
        if self.params is None:
            if self.method == "pca":
                self.params = {"n_components": 10}
            elif self.method == "tsne":
                self.params = {"n_components": 2}
            elif self.method == "autoencoder":
                self.params = {"encoding_dim": 10, "epochs": 50, "batch_size": 32}
            else:
                self.params = {}  # sensible default


def preprocess_to_common_freq(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    desired_freq: Optional[str | pd.Timedelta] = None,
    cfg: Optional[PreprocessConfig] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[str | pd.Timedelta]]:
    """
    Given two time-indexed DataFrames, resample both to a common (coarser) frequency,
    then impute and scale them consistently.
    - If desired_freq is None: infer each freq & pick the longer (coarser) one.
    """
    cfg = cfg or PreprocessConfig()

    ref_df = to_datetime_index(ref_df)
    tgt_df = to_datetime_index(tgt_df)

    if desired_freq is None:
        f_ref = infer_freq_from_index(ref_df.index)
        f_tgt = infer_freq_from_index(tgt_df.index)
        common = choose_common_frequency(f_ref, f_tgt)
    else:
        common = desired_freq

    # If still None (e.g., too few points), skip resampling
    if common is not None:
        ref_df = make_static_frequency_data(ref_df, common)
        tgt_df = make_static_frequency_data(tgt_df, common)

    # Align columns by intersection to ensure same dimensionality
    common_cols = sorted(set(ref_df.columns).intersection(set(tgt_df.columns)))
    if not common_cols:
        raise ValueError("No overlapping numeric columns between the two datasets.")
    ref_df = ref_df[common_cols]
    tgt_df = tgt_df[common_cols]

    # Impute
    ref_df = simple_impute(ref_df, method=cfg.impute_method, limit=cfg.impute_limit)
    tgt_df = simple_impute(tgt_df, method=cfg.impute_method, limit=cfg.impute_limit)

    # Scale
    ref_df = scale_dataframe(ref_df, cfg.scaling)
    tgt_df = scale_dataframe(tgt_df, cfg.scaling)

    return ref_df, tgt_df, common


def compute_similarity_for_pair(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    pre_cfg: Optional[PreprocessConfig] = None,
    rep_cfg: Optional[RepresentationConfig] = None,
    sim: str = "cosine"
) -> float:
    """
    Full pipeline for two datasets:
    - resample to common freq (coarser)
    - impute, scale
    - representation (same method & dimension)
    - summarize to a single vector and compute similarity
    """
    rep_cfg = rep_cfg or RepresentationConfig()
    ref_p, tgt_p, _ = preprocess_to_common_freq(ref_df, tgt_df, cfg=(pre_cfg or PreprocessConfig()))

    ref_emb = run_representation(ref_p, rep_cfg.method, rep_cfg.params)
    tgt_emb = run_representation(tgt_p, rep_cfg.method, rep_cfg.params)

    ref_vec = summarize_embedding(ref_emb, rep_cfg.summarize)
    tgt_vec = summarize_embedding(tgt_emb, rep_cfg.summarize)

    if sim == "cosine":
        return cosine_similarity(ref_vec, tgt_vec)
    elif sim == "euclidean":
        return float(np.linalg.norm(ref_vec - tgt_vec))
    else:
        raise ValueError(f"Unsupported similarity: {sim}")


def attach_pattern_similarity(
    anchor_meta: Dict[str, Any],
    compared_meta: List[Dict[str, Any]],
    loader,  # callable(meta, info) -> pd.DataFrame
    pre_cfg: Optional[PreprocessConfig] = None,
    rep_cfg: Optional[RepresentationConfig] = None,
    sim: str = "cosine",
) -> Dict[str, Any]:
    """
    For a single anchor document (with possibly multiple statistical_info items),
    compute similarity against each info in each compared document.
    The `loader` should take (meta_doc, info_dict) and return the corresponding time-series DataFrame.
    Results are stored under info["pattern_similarity"] in the anchor_meta.
    """
    pre_cfg = pre_cfg or PreprocessConfig()
    rep_cfg = rep_cfg or RepresentationConfig()

    anchor_infos = anchor_meta.get("statistical_info", [])
    if not isinstance(anchor_infos, list):
        return anchor_meta

    for a_info in anchor_infos:
        # load anchor data
        ref_df = loader(anchor_meta, a_info)
        if ref_df is None or ref_df.empty:
            # still create an empty result for clarity
            a_info.setdefault("pattern_similarity", [])
            a_info["pattern_similarity"] = [{
                "method": rep_cfg.method,
                "summarize": rep_cfg.summarize,
                "similarity": sim,
                "targets": []
            }]
            continue

        targets: List[Dict[str, Any]] = []

        # compare against every info in every compared_meta
        for cmp_doc in compared_meta:
            cmp_infos = cmp_doc.get("statistical_info", [])
            for c_info in cmp_infos:
                tgt_df = loader(cmp_doc, c_info)
                if tgt_df is None or tgt_df.empty:
                    continue
                try:
                    score = compute_similarity_for_pair(ref_df, tgt_df, pre_cfg, rep_cfg, sim=sim)
                except Exception as e:
                    # On failure, record NaN score with reason
                    score = float("nan")

                targets.append({
                    "entry": make_entry(cmp_doc, c_info),
                    "score": score,
                })

        # sort descending for cosine (higher is more similar), ascending for euclidean
        if sim == "cosine":
            targets.sort(key=lambda x: (np.nan_to_num(x["score"], nan=-1.0)), reverse=True)
        elif sim == "euclidean":
            targets.sort(key=lambda x: (np.nan_to_num(x["score"], nan=np.inf)))

        a_info.setdefault("pattern_similarity", [])
        a_info["pattern_similarity"] = [{
            "method": rep_cfg.method,
            "summarize": rep_cfg.summarize,
            "similarity": sim,
            "targets": targets
        }]

    return anchor_meta
