import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

class Autoencoder(nn.Module):
    """
    Description: Autoencoder 모델 정의

    Args:
        input_dim (int): 입력 데이터 차원
        encoding_dim (int): 인코딩 차원 (임베딩 차원)

    Methods:
        forward(x): 데이터를 입력받아 인코딩과 디코딩 결과를 반환
    """
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def autoencoder_representation(data, encoding_dim=10, epochs=50, batch_size=32):
    """
        Description: Autoencoder를 사용하여 임베딩을 생성하는 함수

    Args:
        data (DataFrame): 입력 데이터
        encoding_dim (int): 인코딩 차원 (임베딩 차원)
        epochs (int): 학습 에폭 수
        batch_size (int): 배치 크기

    Returns:
        numpy array: Autoencoder로 학습된 임베딩
    """
    input_dim = data.shape[1]
    model = Autoencoder(input_dim, encoding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    data_tensor = torch.tensor(data.values, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(data_tensor, data_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for batch_data, _ in dataloader:
            optimizer.zero_grad()
            encoded, decoded = model(batch_data)
            loss = criterion(decoded, batch_data)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    with torch.no_grad():
        embeddings, _ = model(data_tensor)
    return embeddings.numpy()

def pca_representation(data, n_components=10):
    """
    Description: PCA를 사용하여 데이터를 압축하고 임베딩을 생성하는 함수

    Args:
        data (DataFrame): 입력 데이터
        n_components (int): PCA로 압축할 차원 수

    Returns:
        numpy array: PCA로 변환된 임베딩
    """
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)

def tsne_representation(data, n_components=2):
    """
    Description: t-SNE를 사용하여 고차원 데이터를 시각화 가능한 차원으로 변환하는 함수

    Args:
        data (DataFrame): 입력 데이터
        n_components (int): 출력 차원 수 (기본값: 2)

    Returns:
        numpy array: t-SNE로 변환된 임베딩
    """
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=30)
    return tsne.fit_transform(data)

class LSTMRepresentation(nn.Module):
    """
    Description: LSTM 기반 Representation Learning 모델 정의

    Args:
        input_dim (int): 입력 데이터 차원
        hidden_dim (int): LSTM 히든 차원
        output_dim (int): 출력 임베딩 차원
        num_layers (int): LSTM 레이어 수

    Methods:
        forward(x): LSTM을 통해 입력 데이터를 처리하고 임베딩을 반환
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMRepresentation, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x: (Batch Size, Seq Length, Input Dim)
        """
        _, (hidden, _) = self.lstm(x)  # hidden: (num_layers, Batch Size, Hidden Dim)
        return self.fc(hidden[-1])    # Use last layer's hidden state

def lstm_representation(data, hidden_dim=64, output_dim=32, num_layers=2, epochs=50, batch_size=32):
    """
    Description: LSTM 모델을 사용하여 시계열 데이터를 임베딩으로 변환하는 함수

    Args:
        data (DataFrame): 입력 데이터
        hidden_dim (int): LSTM 히든 차원
        output_dim (int): 출력 임베딩 차원
        num_layers (int): LSTM 레이어 수
        epochs (int): 학습 에폭 수
        batch_size (int): 배치 크기

    Returns:
        numpy array: LSTM으로 학습된 임베딩
    """
    input_dim = data.shape[1]
    model = LSTMRepresentation(input_dim, hidden_dim, output_dim, num_layers)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # 데이터 변환: (Batch Size, Seq Length, Input Dim)
    data_tensor = torch.tensor(data.values, dtype=torch.float32).unsqueeze(1)  # Add Seq Length dimension
    dataset = torch.utils.data.TensorDataset(data_tensor, data_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_data, _ in dataloader:
            optimizer.zero_grad()
            embeddings = model(batch_data)
            
            # Target 생성: Batch Data를 Output Dim으로 매핑
            target = batch_data.mean(dim=1)[:, :output_dim]  # Match Output Dim
            
            loss = criterion(embeddings, target)  # Match Loss Dimensions
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")

    # Embeddings 추출
    model.eval()
    with torch.no_grad():
        embeddings = model(data_tensor)
    return embeddings.numpy()

class TCNRepresentation(nn.Module):
    """
    Description: TCN (Temporal Convolutional Network) 모델 정의

    Args:
        input_dim (int): 입력 데이터 차원
        num_filters (int): 컨볼루션 필터 개수
        kernel_size (int): 커널 크기
        output_dim (int): 출력 임베딩 차원

    Methods:
        forward(x): 입력 데이터를 처리하여 임베딩을 반환
    """
    def __init__(self, input_dim, num_filters, kernel_size, output_dim):
        super(TCNRepresentation, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, num_filters, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size, padding=kernel_size // 2)
        self.fc = nn.Linear(num_filters, output_dim)

    def forward(self, x):
        """
        x: (Batch Size, Channels, Sequence Length)
        """
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.mean(x, dim=2)  # Global Average Pooling over Sequence Length
        return self.fc(x)

def tcn_representation(data, num_filters=64, kernel_size=3, output_dim=32, epochs=50, batch_size=32):
    """
    Description: TCN 모델을 사용하여 데이터를 임베딩으로 변환하는 함수

    Args:
        data (DataFrame): 입력 데이터
        num_filters (int): 필터의 개수
        kernel_size (int): 컨볼루션 커널의 크기
        output_dim (int): 출력 임베딩 차원
        epochs (int): 학습 에폭 수
        batch_size (int): 배치 크기

    Returns:
        numpy array: TCN으로 학습된 임베딩
    """
    input_dim = 1  # TCN은 단일 채널로 처리
    model = TCNRepresentation(input_dim=input_dim, num_filters=num_filters, kernel_size=kernel_size, output_dim=output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # 데이터 변환: (batch_size, channels=1, seq_len)
    data_tensor = torch.tensor(data.values, dtype=torch.float32).unsqueeze(1)
    dataset = torch.utils.data.TensorDataset(data_tensor, data_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_data, _ in dataloader:
            optimizer.zero_grad()
            embeddings = model(batch_data)
            loss = criterion(embeddings, batch_data.mean(dim=2))  # Match mean of sequence to embeddings
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")

    # Embeddings 추출
    model.eval()
    with torch.no_grad():
        embeddings = model(data_tensor)
    return embeddings.numpy()

class TimeSeriesTransformer(nn.Module):
    """
    Description: Transformer 기반 Representation Learning 모델 정의

    Args:
        input_dim (int): 입력 데이터 차원
        embed_dim (int): 입력 데이터의 임베딩 차원
        num_heads (int): Multi-Head Attention의 헤드 수
        num_layers (int): Transformer 레이어 수
        output_dim (int): 출력 임베딩 차원

    Methods:
        forward(x): Transformer를 통해 입력 데이터를 처리하고 임베딩을 반환
    """
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, output_dim):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.transformer = nn.Transformer(embed_dim, num_heads, num_layers, batch_first=True)
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x, x)
        return self.fc(x.mean(dim=1))

def transformer_representation(data, embed_dim=64, num_heads=4, num_layers=2, output_dim=32, epochs=50, batch_size=32):
    """
    Description: Transformer 모델을 사용하여 시계열 데이터를 임베딩으로 변환하는 함수

    Args:
        data (DataFrame): 입력 데이터
        embed_dim (int): 입력 데이터의 임베딩 차원
        num_heads (int): Multi-Head Attention의 헤드 수
        num_layers (int): Transformer 레이어 수
        output_dim (int): 출력 임베딩 차원
        epochs (int): 학습 에폭 수
        batch_size (int): 배치 크기

    Returns:
        numpy array: Transformer로 학습된 임베딩
    """
    input_dim = data.shape[1]
    model = TimeSeriesTransformer(input_dim=input_dim, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, output_dim=output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    data_tensor = torch.tensor(data.values, dtype=torch.float32).unsqueeze(1)  # Add sequence length dimension
    dataset = torch.utils.data.TensorDataset(data_tensor, data_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for batch_data, _ in dataloader:
            optimizer.zero_grad()
            embeddings = model(batch_data)
            loss = criterion(embeddings, torch.zeros_like(embeddings))  # Dummy loss
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    with torch.no_grad():
        embeddings = model(data_tensor)
    return embeddings.numpy()