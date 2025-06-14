import multiprocessing
import os
import time
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch.utils.data import TensorDataset, Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from tqdm import tqdm
import matplotlib.pyplot as plt

# 時間をHH:MM:SSまたはMM:SS形式でフォーマットするヘルパー関数
def format_time(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"
    
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """VAE損失関数（再構成誤差 + KLダイバージェンス）"""
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss / x.size(0), recon_loss, kl_loss / x.size(0)

class MicroenvironmentVAE(nn.Module):
    """微小環境データ用の変分オートエンコーダー"""
    
    def __init__(self, input_dim=30000, dim_1 = 1024, dim_2 = 256, latent_dim=64):
        super(MicroenvironmentVAE, self).__init__()

        # エンコーダー
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, dim_1),
            nn.BatchNorm1d(dim_1),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(dim_1, dim_2),
            nn.BatchNorm1d(dim_2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # 潜在変数の平均と分散
        self.mu_layer = nn.Linear(dim_2, latent_dim)
        self.logvar_layer = nn.Linear(dim_2, latent_dim)
        
        # デコーダー
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, dim_2),
            nn.BatchNorm1d(dim_2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(dim_2, dim_1),
            nn.BatchNorm1d(dim_1),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(dim_1, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        """再パラメータ化トリック"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # エンコード
        encoded = self.encoder(x)
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)
        
        # 再パラメータ化
        z = self.reparameterize(mu, logvar)
        
        # デコード
        decoded = self.decoder(z)
        
        return decoded, mu, logvar


class SpatialMicroenvironmentAnalyzer:
    """空間転写データの微小環境解析クラス"""
    
    def __init__(self, coords, expression_data, k_neighbors=30, device = torch.device('cpu')):
        """
        Parameters:
        coords: numpy array of shape (n_cells, 2) - XY座標
        expression_data: numpy array of shape (n_cells, n_genes) - 発現量データ
        k_neighbors: int - 近傍細胞数
        """
        self.coords = coords
        self.expression_data = expression_data
        self.k_neighbors = k_neighbors
        self.n_cells, self.n_genes = expression_data.shape
        self.device = device
        print(f"細胞数: {self.n_cells:,}")
        print(f"遺伝子数: {self.n_genes:,}")
        
    def build_microenvironment_data(self):
        """各細胞の微小環境データを構築"""
        print("k-NN検索を実行中...")
        
        # k-NNモデル構築
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors, 
                               algorithm='ball_tree').fit(self.coords)
        
        # 各細胞の近傍細胞を取得
        distances, indices = nbrs.kneighbors(self.coords)
        
        # 微小環境データの初期化
        # microenv_data = np.zeros((self.n_cells, self.k_neighbors * self.n_genes))
        microenv_data = np.zeros((self.n_cells, self.n_genes))
        
        print("微小環境データを構築中...")
        for i in tqdm(range(self.n_cells)):
            # 近傍細胞のインデックス
            neighbor_indices = indices[i, 0:]  # 最初は自分自身、除外しない
            #neighbor_indices = indices[i, 1:]  # 最初は自分自身なので除外
            
            # 近傍細胞の発現データを取得
            neighbor_expr = self.expression_data[neighbor_indices]
            
            # # 各遺伝子ごとに降順ソート
            # sorted_expr = np.sort(neighbor_expr, axis=0)[::-1]  # 降順
            
            # # 平坦化してmicroenv_dataに格納
            # microenv_data[i] = sorted_expr.flatten()
            microenv_data[i] = np.mean(neighbor_expr, axis=0)

        microenv_data = microenv_data / microenv_data.max()
        
        self.microenv_data = microenv_data
        print(f"微小環境データ形状: {microenv_data.shape}")
        
        return indices, microenv_data
    
    def train_vae(self, dim_1 = 1024, dim_2 = 256, latent_dim=32, epochs=100, batch_size=256, lr=1e-3, weight_decay=1e-4, beta=1.0):
        """VAEの訓練"""
        print("VAEの訓練を開始...")
        
        # データローダー準備
        cpu_count = multiprocessing.cpu_count()
        num_workers = min(cpu_count - 1, cpu_count // os.cpu_count() if os.cpu_count() else 1)
        dataset = TensorDataset(torch.FloatTensor(self.microenv_data))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, multiprocessing_context="fork")
        
        # モデル初期化
        # input_dim = self.k_neighbors * self.n_genes
        input_dim = self.n_genes
        self.vae = MicroenvironmentVAE(dim_1 = dim_1, dim_2 = dim_2, input_dim=input_dim, latent_dim=latent_dim).to(self.device)
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr, weight_decay=weight_decay)
        
        # 訓練ループ
        losses = []
        self.vae.train()
        start_time = time.time()
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_Rec_loss = 0
            epoch_KL_loss = 0
            for batch_data, in dataloader:
                batch_data = batch_data.to(self.device)
                beta_current = min(1.0, epoch / 50.0)
                optimizer.zero_grad()
                recon_batch, mu, logvar = self.vae(batch_data)
                loss, recon_loss, kl_loss = vae_loss(recon_batch, batch_data, mu, logvar, beta=beta_current)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_Rec_loss += recon_loss.item()
                epoch_KL_loss += kl_loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            avg_Rec_loss = epoch_Rec_loss / len(dataloader)
            avg_KL_loss = epoch_KL_loss / len(dataloader)
            losses.append(avg_loss)
            if (epoch + 1) % 2 == 0:
                current_time = time.time()
                elapsed_time_sec = current_time - start_time
                epochs_completed = epoch + 1
                if epochs_completed > 0:
                    avg_time_per_epoch = elapsed_time_sec / epochs_completed
                    remaining_epochs = epochs - epochs_completed
                    estimated_remaining_time_sec = avg_time_per_epoch * remaining_epochs
                else:
                    avg_time_per_epoch = 0
                    estimated_remaining_time_sec = 0 
                elapsed_time_formatted = format_time(elapsed_time_sec)
                estimated_remaining_time_formatted = format_time(estimated_remaining_time_sec)
                output = f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f} (Rec: {avg_Rec_loss:.4f}, KL: {avg_KL_loss:.4f}) [Elapsed: {elapsed_time_formatted}, Est. Remaining: {estimated_remaining_time_formatted}]   "
                print(f"\r{output}", end="", flush=True)
        self.losses = losses
        return self.vae
    
    def extract_latent_features(self):
        """潜在特徴量（mu）を抽出"""
        print("潜在特徴量を抽出中...")
        
        # データの抽出
        if isinstance(self.microenv_data, pd.DataFrame):
            microenv_data_np = self.microenv_data.values
        else:
            microenv_data_np = self.microenv_data
        
        self.vae.eval() # モデルを評価モードに設定
        latent_features = [] # 抽出された潜在特徴量を格納するリスト

        # バッチ処理で潜在特徴量を抽出
        batch_size = 1024 # バッチサイズを定義
        
        # microenv_data_np をバッチごとに処理
        for i in range(0, len(microenv_data_np), batch_size):
            # 現在のバッチをNumPy配列として取得
            batch_np = microenv_data_np[i:i+batch_size]
            
            # バッチをPyTorchのFloatTensorに変換し、デバイスに送る
            batch_tensor = torch.FloatTensor(batch_np).to(self.device)
            
            with torch.no_grad(): # 勾配計算を無効化（推論モード）
                encoded = self.vae.encoder(batch_tensor) # エンコーダーでエンコード
                mu = self.vae.mu_layer(encoded) # muレイヤーから潜在特徴量muを抽出
                latent_features.append(mu.cpu().numpy()) # NumPy配列に変換してリストに追加

        # 全バッチの潜在特徴量を垂直方向に結合
        self.latent_features = np.vstack(latent_features)
        
        print(f"潜在特徴量形状: {self.latent_features.shape}")
        return self.latent_features
    
    def reconstruct_from_latent(self):
        """潜在特徴量から再構成データを得る"""
        print("潜在特徴量から再構成データを生成中...")
    
        # モデルを評価モードに
        self.vae.eval()
    
        # 潜在特徴量をTensorに変換
        latent_tensor = torch.FloatTensor(self.latent_features).to(self.device)
    
        # デコーダーを通して再構成
        with torch.no_grad():
            recon_x = self.vae.decoder(latent_tensor)
    
        # NumPy配列に変換して返す
        self.reconstructed_data = recon_x.cpu().numpy()
        print(f"再構成データの形状: {self.reconstructed_data.shape}")
        return self.reconstructed_data
    
    def perform_umap_clustering(self, n_neighbors=15, min_dist=0.5, n_components=2, 
                               cell_type_data=None, seed=42):
        """UMAP次元削減とLeidenクラスタリング"""
        print("AnnDataオブジェクトを作成中...")
        
        # AnnDataオブジェクト作成
        adata = sc.AnnData(X=self.latent_features)
        
        # データの前処理（欠損値補完）
        print("データの前処理中...")
        if hasattr(adata.X, "toarray"):
            X_dense = adata.X.toarray()
        else:
            X_dense = adata.X
        
        imputer = SimpleImputer(strategy='constant', fill_value=0)
        adata.X = imputer.fit_transform(X_dense)
        adata.raw = None
     
        # 近傍グラフ構築
        print("近傍グラフ構築中...")
        sc.pp.neighbors(adata, random_state=seed, n_neighbors=n_neighbors, use_rep='X')
        
        # UMAP
        print("UMAP次元削減実行中...")
        sc.tl.umap(adata, min_dist=min_dist, random_state=seed)
        
        # Leidenクラスタリング
        print("Leidenクラスタリング実行中...")
        sc.tl.leiden(adata, resolution=0.3, key_added='leiden', random_state=seed)
        
        # Cell type情報を追加（提供されている場合）
        if cell_type_data is not None:
            adata.obs['cell_type'] = cell_type_data.tolist()
        
        self.clusters = adata.obs['leiden'].values.astype(int)
        self.adata = adata
        self.umap_embedding = adata.obsm['X_umap']
        
        print(f"クラスター数: {len(np.unique(self.clusters))}")
        
        return self.umap_embedding, self.clusters
    
    def visualize_results(self, figsize=(15, 12)):
        # 基本的な可視化
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # 1. 訓練ロス
        axes[0, 0].plot(self.losses)
        axes[0, 0].set_title('VAE Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        
        # 2. 空間分布
        scatter = axes[0, 1].scatter(self.coords[:, 1], self.coords[:, 0], 
                                   c=self.clusters, cmap='tab20', s=0.5)
        axes[0, 1].set_title('Spatial Distribution (Colored by Cluster)')
        axes[0, 1].set_xlabel('Array Column')
        axes[0, 1].set_ylabel('Array Row')
        plt.colorbar(scatter, ax=axes[0, 1])
        
        # 3. UMAP embedding
        scatter2 = axes[0, 2].scatter(self.umap_embedding[:, 0], 
                                    self.umap_embedding[:, 1],
                                    c=self.clusters, cmap='tab20', s=0.5)
        axes[0, 2].set_title('UMAP Embedding (Colored by Cluster)')
        axes[0, 2].set_xlabel('UMAP 1')
        axes[0, 2].set_ylabel('UMAP 2')
        plt.colorbar(scatter2, ax=axes[0, 2])
        
        # 4. クラスター分布
        cluster_counts = np.bincount(self.clusters)
        axes[1, 0].bar(range(len(cluster_counts)), cluster_counts)
        axes[1, 0].set_title('Cluster Size Distribution')
        axes[1, 0].set_xlabel('Cluster ID')
        axes[1, 0].set_ylabel('Number of Cells')
        
        # 5. 潜在特徴量の分布（最初の2次元）
        axes[1, 1].scatter(self.latent_features[:, 0], 
                          self.latent_features[:, 1],
                          c=self.clusters, cmap='tab20', s=0.5, alpha=0.6)
        axes[1, 1].set_title('Latent Features (Dim 0 vs 1)')
        axes[1, 1].set_xlabel('Latent Feature Dimension 0')
        axes[1, 1].set_ylabel('Latent Feature Dimension 1')
        
        # 6. 潜在特徴量のヒートマップ（各クラスターの平均）
        cluster_means = []
        for cluster_id in np.unique(self.clusters):
            mask = self.clusters == cluster_id
            cluster_mean = self.latent_features[mask].mean(axis=0)
            cluster_means.append(cluster_mean)
        
        cluster_means = np.array(cluster_means)
        im = axes[1, 2].imshow(cluster_means, aspect='auto', cmap='viridis')
        axes[1, 2].set_title('Cluster Mean Latent Features')
        axes[1, 2].set_xlabel('Latent Dimension')
        axes[1, 2].set_ylabel('Cluster ID')
        plt.colorbar(im, ax=axes[1, 2])
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def visualize_scanpy_results(self):
        """Scanpy形式での詳細可視化"""
        if not hasattr(self, 'adata'):
            print("先にperform_umap_clustering()を実行してください")
            return
        
        # 基本的なUMAP可視化
        print("基本的なUMAP可視化:")
        if 'cell_type' in self.adata.obs.columns:
            sc.pl.umap(self.adata, color=['cell_type', 'leiden'], wspace=0.4)
        else:
            sc.pl.umap(self.adata, color='leiden')
