# KETI Time Series Dataset Search

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![Dependencies](https://img.shields.io/badge/Dependencies-PyTorch%20%7C%20Scikit--learn%20%7C%20Pandas-green)](requirements.txt)

A comprehensive Python library for **time series dataset similarity search and discovery** based on geographical proximity and temporal pattern analysis. This library enables intelligent dataset recommendation by combining location-based filtering with advanced representation learning techniques.

## 🎯 **Overview**

KETI Time Series Dataset Search provides two complementary approaches for discovering similar datasets:

1. **🌍 Geographical Similarity**: Find datasets from sensors located within specified radius
2. **📈 Pattern Similarity**: Discover datasets with similar temporal patterns using advanced machine learning representations

## ✨ **Key Features**

### 🌍 **Location-Based Search**
- **Geospatial Filtering**: Find datasets within N kilometers radius
- **GPS Coordinate Support**: Latitude/longitude based proximity search
- **Flexible Distance Metrics**: Configurable search radius

### 📊 **Pattern-Based Search**
- **Multiple Representation Methods**: 
  - **Classical**: PCA, t-SNE
  - **Deep Learning**: Autoencoder, LSTM, TCN, Transformer
- **Similarity Metrics**: Cosine similarity, Euclidean distance, DTW (Dynamic Time Warping)
- **Preprocessing Pipeline**: Automatic resampling, imputation, scaling
- **Robust Handling**: Missing data, irregular sampling rates

### 🔧 **Data Processing**
- **Automatic Frequency Detection**: Infer sampling rates from time series
- **Multi-format Support**: CSV, JSON metadata
- **Preprocessing Automation**: Scaling, imputation, frequency alignment

## 🚀 **Quick Start**

### Installation

```bash
git clone https://github.com/your-repo/keti-ts-dataset-search.git
cd keti-ts-dataset-search
pip install -r requirements.txt
```

### Basic Usage

#### 1. Geographical Similarity Search

```python
import locale_similarity
from pathlib import Path
import use_json

# Load dataset metadata
dataset_path = Path('./test_dataset/for_locale/dataset.json')
dataset_meta = use_json.load_json(dataset_path)

# Find datasets within 3km radius
n_km = 3
updated_meta = locale_similarity.get_nearby_meta(n_km, dataset_meta)

# Save results
output_path = Path('./test_dataset/for_locale/nearby_datasets.json')
use_json.write_json(output_path, updated_meta)
```

#### 2. Pattern Similarity Search

```python
import pattern_similarity
import pandas as pd
from pathlib import Path

# Prepare dataset list
data_list = []
data_dir = Path("./test_dataset/for_pattern")

for csv_path in data_dir.glob("*.csv"):
    df = pd.read_csv(csv_path, parse_dates=["time"], index_col="time")
    data_list.append({
        "file_name": csv_path.name,
        "df": df.sort_index(),
        "domain": "environmental"
    })

# Configure similarity parameters
param = {
    "frequency": '',  # Auto-detect
    "scaling_param": {"flag": True, "method": 'minmax'},
    "similarity_method": "cosine",
    "impute": {"flag": True, "method": "linear", "limit": None},
    "representation_param": {
        'autoencoder_param': {
            "encoding_dim": 10, 
            "epochs": 50, 
            "batch_size": 32
        }
    }
}

# Compute similarities
similarity_results = pattern_similarity.get_data_similarity(data_list, param)

# Save results
output_path = Path("./results/pattern_similarity.json")
use_json.write_json(output_path, similarity_results)
```

#### 3. Advanced Text-Based API

```python
import text_function as tf
import pandas as pd

# Load two time series datasets
df1 = pd.read_csv('dataset1.csv', parse_dates=['time'], index_col='time')
df2 = pd.read_csv('dataset2.csv', parse_dates=['time'], index_col='time')

# Configure preprocessing and representation
pre_cfg = tf.PreprocessConfig(
    impute_method="linear",
    scaling={"flag": True, "method": "standard"}
)

rep_cfg = tf.RepresentationConfig(
    method="autoencoder",
    params={"encoding_dim": 16, "epochs": 100},
    summarize="mean"
)

# Compute similarity
similarity_score = tf.compute_similarity_for_pair(
    df1, df2, 
    pre_cfg=pre_cfg, 
    rep_cfg=rep_cfg, 
    sim="cosine"
)

print(f"Similarity Score: {similarity_score:.4f}")
```

## 📚 **Core Modules**

### 🌍 `locale_similarity.py`
Geographical proximity-based dataset discovery.

**Key Functions:**
- `search_nearby_by_meta(ref_latlng, target_latlng, n_km)`: Check if two locations are within distance
- `get_nearby_meta(n_km, data_meta)`: Find all nearby datasets within radius

### 📈 `pattern_similarity.py`
Temporal pattern-based similarity computation.

**Key Functions:**
- `get_data_similarity(data_list, param)`: Compute pairwise similarities for dataset list
- `compute_similarity_metrics(emb1, emb2, method)`: Calculate similarity between embeddings
- `set_data_preprocessing(data, param)`: Preprocessing pipeline

### 🧠 `representation_methods.py`
Advanced representation learning techniques.

**Available Methods:**
- `pca_representation()`: Principal Component Analysis
- `tsne_representation()`: t-Distributed Stochastic Neighbor Embedding
- `autoencoder_representation()`: Deep autoencoder embeddings
- `lstm_representation()`: LSTM-based sequential embeddings
- `tcn_representation()`: Temporal Convolutional Networks
- `transformer_representation()`: Attention-based embeddings

### 🔧 `text_function.py`
High-level API with comprehensive preprocessing and configuration.

**Key Classes:**
- `PreprocessConfig`: Preprocessing configuration
- `RepresentationConfig`: Representation method configuration

**Key Functions:**
- `compute_similarity_for_pair()`: End-to-end similarity computation
- `attach_pattern_similarity()`: Batch similarity computation with metadata

## ⚙️ **Configuration Options**

### Preprocessing Parameters
```python
preprocessing_config = {
    "frequency": "10T",  # Resample to 10-minute intervals
    "scaling_param": {
        "flag": True,
        "method": "minmax"  # Options: minmax, standard, robust, maxabs
    },
    "impute": {
        "flag": True,
        "method": "linear",  # Options: linear, time, nearest, spline
        "limit": None  # Max consecutive NaN to fill
    }
}
```

### Representation Parameters
```python
representation_config = {
    # Autoencoder
    "autoencoder_param": {
        "encoding_dim": 10,
        "epochs": 50,
        "batch_size": 32
    },
    
    # LSTM
    "lstm_param": {
        "hidden_dim": 64,
        "output_dim": 32,
        "num_layers": 2,
        "epochs": 50
    },
    
    # Transformer
    "transformer_param": {
        "embed_dim": 64,
        "num_heads": 4,
        "num_layers": 2,
        "output_dim": 32
    }
}
```

### Similarity Methods
- **`cosine`**: Cosine similarity (range: -1 to 1, higher = more similar)
- **`euclidean`**: Euclidean distance (lower = more similar)
- **`dtw`**: Dynamic Time Warping distance
- **`fast_dtw`**: Fast DTW approximation

## 📁 **Input Data Format**

### CSV Time Series Format
```csv
time,temperature,humidity,pressure
2023-01-01 00:00:00,22.5,65.2,1013.2
2023-01-01 00:10:00,22.3,65.8,1013.1
2023-01-01 00:20:00,22.1,66.1,1012.9
```

### JSON Metadata Format
```json
{
  "bucket_name": "environmental_data",
  "table_name": "sensor_readings", 
  "statistical_info": [
    {
      "filtered_tag_set": {"location": "building_a", "sensor_type": "temp"},
      "location": {
        "lat": 37.5665,
        "lng": 126.9780
      },
      "filtered_info": {
        "numberOfColumns": 3,
        "pointCount": 1440,
        "frequency": "0 days 00:10:00",
        "startTime": "2023-01-01T00:00:00Z",
        "endTime": "2023-01-01T23:50:00Z",
        "columns": ["temperature", "humidity", "pressure"]
      }
    }
  ]
}
```

## 🔍 **Output Format**

### Similarity Results
```json
{
  "domain": "environmental",
  "file_name": "sensor_A.csv",
  "statistical_info": [
    {
      "similarity": [
        {
          "method": "cosine",
          "result": [
            {
              "score": 0.8945,
              "target": {"file_name": "sensor_B.csv"}
            },
            {
              "score": 0.7234,
              "target": {"file_name": "sensor_C.csv"}
            }
          ]
        }
      ]
    }
  ]
}
```

### Location-Based Results
```json
{
  "statistical_info": [
    {
      "location": {
        "lat": 37.5665,
        "lng": 126.9780,
        "nearby": [
          {
            "distance": 3,
            "unit": "km", 
            "target": [
              {
                "bucket": "environmental_data",
                "measurement": "weather_station",
                "tags": {"location": "building_b"}
              }
            ]
          }
        ]
      }
    }
  ]
}
```

## 🧪 **Testing**

Run the compatibility test to verify functionality:

```bash
python compatibility_test.py
```

This will:
1. ✅ Test geographical similarity search (3km radius)
2. ✅ Test pattern similarity with autoencoder representations
3. 📁 Generate results in `test_dataset/` directories

## 📦 **Dependencies**

**Core Requirements:**
- `pandas >= 1.3.0`: Data manipulation and analysis
- `numpy >= 1.21.0`: Numerical computations
- `scikit-learn >= 1.0.0`: Machine learning utilities

**Deep Learning:**
- `torch >= 1.9.0`: PyTorch for neural networks
- `tslearn >= 0.5.0`: Time series analysis

**Geospatial:**
- `geopy >= 2.2.0`: Geographic calculations

**Optional:**
- `fastdtw >= 0.3.4`: Fast Dynamic Time Warping
- `scipy >= 1.7.0`: Scientific computing

## 🛣️ **Roadmap**

- [ ] **Multi-modal Similarity**: Combine location + pattern scores
- [ ] **Streaming Support**: Real-time dataset discovery
- [ ] **Advanced Metrics**: Earth Mover's Distance, Wasserstein
- [ ] **Visualization**: Interactive similarity maps and plots
- [ ] **API Server**: REST API for web integration
- [ ] **Caching**: Persistent similarity computation results

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏢 **Acknowledgments**

- **KETI (Korea Electronics Technology Institute)**: Research support and domain expertise
- **Open Source Community**: Libraries and tools that make this project possible
- **Research Partners**: Time series analysis and representation learning insights

---

**For questions, issues, or contributions, please visit our [GitHub Issues](https://github.com/your-repo/keti-ts-dataset-search/issues) page.**