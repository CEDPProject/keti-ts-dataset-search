from itertools import product
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from tslearn.metrics import dtw
from fastdtw import fastdtw
import numpy as np
import pandas as pd
from itertools import product

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
        fast_dtw_val, _ = fastdtw(embeddings1, embeddings2, dist=euclidean)
        result = {"raw": fast_dtw_val, "normalized": 1 - (fast_dtw_val / method_max) } 

    return result


def set_all_entries(db_client):
    all_entries = []

    for db_name in db_client.get_db_list():
        for ms_name in db_client.measurement_list(db_name):
            tag_keys = db_client.get_tag_list(db_name, ms_name)
            tag_value_map = {} 
            for tag_key in tag_keys:
                value_dict = db_client.get_tag_value(db_name, ms_name, tag_key)
                # value_dict: {"FRM_ID": ["F00013", "F00014"]}
                if value_dict and tag_key in value_dict:
                    tag_value_map[tag_key] = value_dict[tag_key]
            # 각 key별 모든 조합 생성
            keys = list(tag_value_map.keys())
            value_lists = list(tag_value_map.values())
            for values in product(*value_lists):
                tag_combo = dict(zip(keys, values))
                all_entries.append({
                    "bucket": db_name,
                    "measurement": ms_name,
                    "tags": tag_combo
                })
    return all_entries


def set_parameter(data_freq1, data_freq2):
    if data_freq1 > data_freq2:
        selected_freq = data_freq1
    else:
        selected_freq = data_freq2
    
    pipeline_parameter = [
        ['data_refinement', 
         {'remove_duplication': {'flag': True}, 
          'static_frequency': {'flag': True, 'frequency': selected_freq}}],
        ['data_imputation',
         {'flag': True,
          'imputation_method': [{'min': 0,'max': 20000,'method': 'linear','parameter': {}}],
          'total_non_NaN_ratio': 1}],
        ['data_scaling', {'flag': True, 'method': 'minmax'}] ]
    
    return pipeline_parameter


def save_meta_to_mongo(mongo_client, selected_entry, similarity_results):
    db_name, other_bk_name = selected_entry['bucket'].split('_', 1)
    collection_name, division = other_bk_name.rsplit('_', 1)

    search = {
        'table_name': selected_entry['measurement'],
        "statistical_info": {
            "$elemMatch": {
                **{f"filtered_tag_set.{k}": v for k, v in selected_entry['tags'].items()}
            }
        }
    }

    find_doc = mongo_client.get_document_by_json(db_name, collection_name, search)[0]

    for info in find_doc.get("statistical_info", []):
        if info.get("filtered_tag_set") == selected_entry["tags"]:
            if "similarity" not in info:
                info["similarity"] = []
            info["similarity"].append(similarity_results)

    update_data = {"statistical_info": find_doc["statistical_info"]}
    mongo_client.update_one_document(db_name, collection_name, search, update_data)
    
    
def check_all_nan_col(data):
    nan_cols = []
    if data.isnull().values.any():
        nan_cols = data.columns[data.isnull().all()].tolist()
        if nan_cols:
            data = data.drop(columns=nan_cols)
    return data





