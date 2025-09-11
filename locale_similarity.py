from itertools import product
from geopy.distance import geodesic

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

def get_search_meta(mongo_client, selected_meta):
    db_name, other_bk_name = selected_meta['bucket'].split('_', 1)
    collection_name, division = other_bk_name.rsplit('_', 1)

    # MongoDB 검색 조건
    search = {
        'table_name': selected_meta['measurement'],
        "statistical_info": {
            "$elemMatch": {
                **{f"filtered_tag_set.{k}": v for k, v in selected_meta['tags'].items()}
            }
        }
    }

    find_doc = mongo_client.get_document_by_json(db_name, collection_name, search)[0]

    for info in find_doc.get("statistical_info", []):
        if info.get("filtered_tag_set") == selected_meta["tags"]:
            find_doc["statistical_info"] = [info]  # <- 해당 info 하나만 리스트로 다시 덮어쓰기
            
            return find_doc


def search_nearby_by_meta(ref_latlng, target_latlng, n_km):
    """
    기준 위치(ref_latlng)로부터 target 위치(target_latlng)가 n_km 이내에 있는지 판단

    Args:
        ref_latlng (list or tuple): [위도, 경도]
        target_latlng (list or tuple): [위도, 경도]
        n_km (float): 거리 기준 (단위: km)

    Returns:
        bool: n_km 이내에 있으면 True, 아니면 False
    """
    distance = geodesic(ref_latlng, target_latlng).km
    return distance <= n_km


def save_nearby_meta(mongo_client, selected_doc, nearby_data):
    db_name = selected_doc["domain"]
    collection_name = selected_doc["subDomain"]  # collection 이름으로 추정됨

    # tags 기준 필터 만들기
    selected_tags = selected_doc["statistical_info"][0]["filtered_tag_set"]

    search = { "table_name": selected_doc["table_name"]}
    update_data = {"statistical_info.$[elem].location.nearby": nearby_data}
    array_filters = [{f"elem.filtered_tag_set.{k}": v for k, v in selected_tags.items()}]
    
    mongo_client.update_one_document(db_name, collection_name, search, update_data, array_filters)