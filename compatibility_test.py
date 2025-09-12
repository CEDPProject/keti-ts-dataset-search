import locale_similarity, pattern_similarity, use_json
from pathlib import Path
import pandas as pd


if __name__ == "__main__":

      # 지역 유사도 탐색
      dataset_path = Path('./test_dataset/for_locale/dataset.json')
      update_locale_dataset_path = Path('./test_dataset/for_locale/update_locale_dataset.json')
      dataset_meta = use_json.load_json(dataset_path)

      n_km = 3
      update_dataset_meta = locale_similarity.get_nearby_meta(n_km, dataset_meta)
      use_json.write_json(update_locale_dataset_path, update_dataset_meta)
      

      # 패턴 유사도 탐색
      data_dir = Path("./test_dataset/for_pattern")
      out_path = Path("./test_dataset/for_pattern/pattern_similarity_result.json")

      data_list= []
      for csv_path in sorted(data_dir.glob("*.csv")):
            df = pd.read_csv(csv_path, parse_dates=["time"], index_col="time")
            df = df.sort_index()
            data_list.append({
            "file_name": csv_path.name,
            "df": df,
            "domain": "air"})
            
      param = {
        "frequency": '',
        "scaling_param": {"flag": True, "method": 'minmax'},
        "similarity_method":"cosine",
        "impute": {"flag": True, "method": "linear", "limit": None},
        "representation_param":{
            'autoencoder_param':
                  {"encoding_dim":10, "epochs":50, "batch_size":32}
            }
      }
      
      similarity_meta = pattern_similarity.get_data_similarity(data_list, param)
      use_json.write_json(out_path, similarity_meta)