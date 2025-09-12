from geopy.distance import geodesic

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


def get_nearby_meta(n_km, data_meta, data_meta2 = None):
    """
    - data_meta: 기준 문서(dict)
    - data_meta2: 타겟 문서(dict) 또는 None(내부 비교)
    - 각 문서의 statistical_info[*].location(lat/lng)로 거리 판단.
    - 반경 n_km 이내면 기준 info.location.nearby에 아래 형태로 기록:
        [{"distance": n_km, "unit": "km", "target": [ {"bucket", "measurement", "tags"}, ... ]}]
    - 결과는 data_meta를 in-place로 갱신하여 반환.
    """
    if not isinstance(data_meta, dict):
        raise TypeError("data_meta는 dict여야 합니다.")
    if data_meta2 is not None and not isinstance(data_meta2, dict):
        raise TypeError("data_meta2는 dict 또는 None이어야 합니다.")

    ref_infos = data_meta.get("statistical_info", [])
    tgt_doc = data_meta if data_meta2 is None else data_meta2
    tgt_infos= tgt_doc.get("statistical_info", [])

    if not isinstance(ref_infos, list) or not isinstance(tgt_infos, list):
        return data_meta 

    for i, ref_info in enumerate(ref_infos):
        ref_loc = ref_info.get("location")
        if not isinstance(ref_loc, dict) or "lat" not in ref_loc or "lng" not in ref_loc:
            continue

        ref_latlng = (ref_loc["lat"], ref_loc["lng"])
        nearby_list = []
        ref_entry = make_entry(data_meta, ref_info)

        for j, tgt_info in enumerate(tgt_infos):
            if data_meta2 is None and i == j:
                continue

            tgt_loc = tgt_info.get("location")
            if not isinstance(tgt_loc, dict) or "lat" not in tgt_loc or "lng" not in tgt_loc:
                continue

            if ref_entry == make_entry(tgt_doc, tgt_info):
                continue

            if _same_coords(ref_loc, tgt_loc, decimals=6):
                continue

            tgt_latlng = (tgt_loc["lat"], tgt_loc["lng"])
            if search_nearby_by_meta(ref_latlng, tgt_latlng, n_km):
                nearby_list.append(make_entry(tgt_doc, tgt_info))

        ref_info.setdefault("location", {})
        ref_info["location"]["nearby"] = [{ "distance": n_km, "unit": "km","target": nearby_list}]

    return data_meta

def make_entry(doc, info):
    """
    doc에서 bucket/measurement 키 이름 차이를 흡수하고,
    info의 filtered_tag_set을 담아 표준 엔트리로 반환.
    """
    bucket = doc.get("bucket_name") or doc.get("bucket") or ""
    measurement = doc.get("table_name") or doc.get("measurement") or ""
    tags = info.get("filtered_tag_set") or {}
    return {"bucket": bucket, "measurement": measurement, "tags": dict(tags)}

def _same_coords(loc_a, loc_b, decimals=6):
    return ( round(loc_a.get("lat", 0), decimals) == round(loc_b.get("lat", 0), decimals) and
            round(loc_a.get("lng", 0), decimals) == round(loc_b.get("lng", 0), decimals) )

