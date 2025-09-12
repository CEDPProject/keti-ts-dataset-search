from itertools import product
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from tslearn.metrics import dtw
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from fastdtw import fastdtw as _fastdtw
import numpy as np
import pandas as pd
from itertools import product
from datetime import timedelta

import representation_methods


def compute_similarity_metrics(embeddings1, embeddings2, selected_method, method_max=None):
    result = {}
    if selected_method == 'cosine':
        a_mean = np.mean(embeddings1, axis=0)
        b_mean = np.mean(embeddings2, axis=0)
        cosine = cosine_similarity(a_mean.reshape(1, -1), b_mean.reshape(1, -1))[0][0]
        result = {"raw": cosine, "normalized": (cosine + 1) / 2 } 
        
    elif selected_method == 'euclidean':
        a_mean = np.mean(embeddings1, axis=0)
        b_mean = np.mean(embeddings2, axis=0)
        eucl = euclidean(a_mean, b_mean)
        result = { "raw": eucl, "normalized": 1 - (eucl / method_max) } 
        
    elif selected_method == 'dtw':
        dtw_val = dtw(embeddings1, embeddings2)
        result = {"raw": dtw_val, "normalized": 1 - (dtw_val / method_max) } 

    elif selected_method == 'fast_dtw':
        fast_dtw_val, _ = _fastdtw(embeddings1, embeddings2, dist=euclidean)
        result = {"raw": fast_dtw_val, "normalized": 1 - (fast_dtw_val / method_max) } 

    return result


def make_static_frequency_data(data, freq):
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
        resampled = out.resample(freq).mean()
        return resampled.asfreq(freq=freq)
    else:
        resampled = out.resample(freq).mean()
        return resampled.asfreq(freq=freq)


def simple_impute(data, param):
    """ Get imputed data from simple other methods
    """
    if param['flag'] == True:
        method = param['method']
        limit = param['limit']
        result = data.interpolate(method=method, limit=limit, limit_direction="both")
    else:
        result = data.copy()
    return result


def data_scaling(data, scaling_param):
    if scaling_param['flag']==True:
        method = scaling_param['method']
        scalers = {
            "minmax": MinMaxScaler,
            "standard": StandardScaler,
            "maxabs": MaxAbsScaler,
            "robust": RobustScaler }
        scaler = scalers.get(method.lower())()
        data = pd.DataFrame(scaler.fit_transform(data), columns=list(data.columns), index = data.index)
    else:
        data = data.copy()

    return data

def check_all_nan_col(data):
    nan_cols = []
    if data.isnull().values.any():
        nan_cols = data.columns[data.isnull().all()].tolist()
        if nan_cols:
            data = data.drop(columns=nan_cols)
    return data


def get_freq(data):
    if len(data)> 3:
        # Simply compare 2 intervals from 3 data points.
        # And get estimated frequency.
        
        inferred_freq1 = (data.index[1]-data.index[0])
        inferred_freq2 = (data.index[2]-data.index[1])
        
        if inferred_freq1 == inferred_freq2:
            estimated_freq = inferred_freq1
        else:
            inferred_freq1 = (data.index[-1]-data.index[-2])
            inferred_freq2 = (data.index[-2]-data.index[-3])
            if inferred_freq1 == inferred_freq2:
                estimated_freq = inferred_freq1
            else :
                estimated_freq = None
    else:
        estimated_freq = None
    
    if not estimated_freq:
        try:
            estimated_freq = (data.index[1]-data.index[0])
        except :
            print("예외가 발생했습니다. data : ", data)

    time_freq = timedelta(seconds=pd.Timedelta(estimated_freq).total_seconds())
    
    return time_freq

def remove_duplication(data):
    """ Get Clean Data without redundency using all data preprocessing functions.

    Args:
        data (DataFrame): input data
        
    Returns:
        DataFrame: result, output data

    Example:

        >>> output = ExcludeRedundancy().get_result(data)
    """

    data = data.loc[:, ~data.columns.duplicated()]
    # duplicated Index Drop
    data = data.sort_index()
    result = data[~data.index.duplicated(keep='first')]
    
    return result


def set_data_preprocessing(data, param):
    
    freq_data = make_static_frequency_data(data, param['frequency'])
    remove_data = remove_duplication(freq_data)
    impute_data = simple_impute(remove_data, param['impute_param'])
    clean_data = check_all_nan_col(impute_data)
    scaled_data = data_scaling(clean_data, param['scaling_param'])
    
    return scaled_data
    
    
def _choose_freq(user_freq, f1: timedelta, f2: timedelta):
    return user_freq if user_freq else max(f1, f2)

    
def get_data_similarity(data_list, param):
    
    sim_method   = param.get("similarity_method", "cosine")
    scaling_param = param.get("scaling_param", {"flag": True, "method": "minmax"})
    impute_param  = param.get("impute", {"flag": False, "method": "linear", "limit": None})
    method_max    = param.get("method_max", None)

    ae_param     = param.get("representation_param", {}).get("autoencoder_param", {})
    encoding_dim = int(ae_param.get("encoding_dim", 10))
    epochs       = int(ae_param.get("epochs", 50))
    batch_size   = int(ae_param.get("batch_size", 32))


    pair_cache = {}
    results  = []
    for ref in data_list:
        ref_file = ref["file_name"]
        ref_df_raw: pd.DataFrame = ref["df"]
        ref_freq = get_freq(ref_df_raw)

        similarity_result_list = []

        for tgt in data_list:
            tgt_file = tgt["file_name"]
            if tgt_file == ref_file:
                continue  
            
            pair_key = tuple(sorted((ref_file, tgt_file)))
            if pair_key in pair_cache:
                score = pair_cache[pair_key]
                similarity_result_list.append({
                    "score": score,
                    "target": {"file_name": tgt_file}
                })
                continue

            tgt_df_raw: pd.DataFrame = tgt["df"]
            tgt_freq = get_freq(tgt_df_raw)

            common_freq = _choose_freq(param.get("frequency"), ref_freq, tgt_freq)
            preprocessing_param = {"frequency": common_freq, "scaling_param": scaling_param, 'impute_param':impute_param}

            data1 = set_data_preprocessing(ref_df_raw, preprocessing_param)
            data2 = set_data_preprocessing(tgt_df_raw, preprocessing_param)


            if data1.empty or data2.empty:
                score = float("nan")
                pair_cache[pair_key] = score
                similarity_result_list.append({
                    "score": score,
                    "target": {"file_name": tgt_file}
                })
                continue

            try:
                emb1 = representation_methods.autoencoder_representation(
                    data1, encoding_dim, epochs, batch_size
                )
                emb2 = representation_methods.autoencoder_representation(
                    data2, encoding_dim, epochs, batch_size
                )
            except Exception:
                score = float("nan")
                pair_cache[pair_key] = score
                similarity_result_list.append({
                    "score": score,
                    "target": {"file_name": tgt_file}
                })
                continue

            score_info = compute_similarity_metrics(emb1, emb2, sim_method, method_max)
            norm = score_info["normalized"]
            score = round(float(norm), 4) if np.isfinite(norm) else float("nan")

            pair_cache[pair_key] = score

            similarity_result_list.append({
                "score": score,
                "target": {"file_name": tgt_file}
            })

        filtered_info = build_filtered_info(ref_df_raw, ref_freq)

        similarity_meta = {"method": sim_method, "result": similarity_result_list}
        meta_obj = {
            "domain": ref["domain"],
            "file_name": ref_file,
            "statistical_info": [
                {
                    "filtered_tag_set": {},
                    "filtered_info": filtered_info,
                    "similarity": [similarity_meta],
                }
            ],
        }
        results.append(meta_obj)

    return results
    
    
def build_filtered_info(df, freq):
    """
    JSON의 filtered_info 필드 구성:
      - numberOfColumns: 수치 컬럼 수
      - pointCount: 행 수
      - frequency: 문자열 (e.g., '0 days 00:01:00')
      - startTime, endTime: ISO8601
      - columns: 수치 컬럼 리스트
    """
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    point_count = int(df.shape[0])
    start_time = df.index[0].isoformat() if point_count > 0 else ""
    end_time = df.index[-1].isoformat() if point_count > 0 else ""
    freq_str = str(pd.Timedelta(freq)) if freq else ""

    return {
        "numberOfColumns": int(len(numeric_cols)),
        "pointCount": point_count,
        "frequency": freq_str,
        "startTime": start_time,
        "endTime": end_time,
        "columns": numeric_cols,
    }