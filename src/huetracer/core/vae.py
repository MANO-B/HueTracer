import multiprocessing
import os
import time
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch.utils.data import TensorDataset, Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import warnings
import torch.multiprocessing as mp
from .reproducibility import set_global_seed, get_seed_from_env

# Helper function to format time in HH:MM:SS or MM:SS format
def format_time(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"
    
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """VAE loss function (reconstruction error + KL divergence)"""
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss / x.size(0), recon_loss, kl_loss / x.size(0)

class MicroenvironmentVAE(nn.Module):
    """Variational Autoencoder for microenvironment data"""
    
    def __init__(self, input_dim=30000, dim_1 = 1024, dim_2 = 256, latent_dim=64):
        super(MicroenvironmentVAE, self).__init__()

        # Encoder
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
        
        # Mean and variance of latent variables
        self.mu_layer = nn.Linear(dim_2, latent_dim)
        self.logvar_layer = nn.Linear(dim_2, latent_dim)
        
        # Decoder
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
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        decoded = self.decoder(z)
        
        return decoded, mu, logvar


# -----------------------------
# Utilities
# -----------------------------
def pick_torch_device(prefer_gpu=True):
    """
    Returns torch.device("cuda"|"mps"|"cpu") with a friendly print.
    """
    if prefer_gpu and torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"✅ Using CUDA: {torch.cuda.get_device_name(0)}")
        return dev
    if prefer_gpu and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("✅ Using Apple MPS (Metal)")
        return dev
    dev = torch.device("cpu")
    print("✅ Using CPU")
    return dev


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _has_rapids():
    try:
        import cupy  # noqa
        import cuml  # noqa
        return True
    except Exception:
        return False


def _has_cugraph():
    try:
        import cugraph  # noqa
        import cudf     # noqa
        return True
    except Exception:
        return False


# -----------------------------
# You must provide these
#   - MicroenvironmentVAE
#   - vae_loss
# -----------------------------
# from your_module import MicroenvironmentVAE, vae_loss


class SpatialMicroenvironmentAnalyzer:
    """Spatial transcriptomics microenvironment analysis with CPU/CUDA/MPS support."""

    def __init__(self, coords, expression_data, k_neighbors=30, device=None, prefer_gpu=True, seed=None, deterministic_torch: bool = True):
        """
        coords: np.ndarray (n_cells, 2)
        expression_data: np.ndarray (n_cells, n_genes)  (dense recommended)
        deterministic_torch: bool
            If True, requests deterministic PyTorch algorithms (cuDNN/cuBLAS).
            Set to False to allow non-deterministic paths, which can be faster.
            Has no effect on CPU; on MPS only warn_only is used.
        """
        self.coords = np.asarray(coords)
        self.expression_data = np.asarray(expression_data)
        self.k_neighbors = int(k_neighbors)
        self.seed = get_seed_from_env(default=42) if seed is None else int(seed)
        self.deterministic_torch = bool(deterministic_torch)

        # Keep global RNGs deterministic for this analysis instance.
        set_global_seed(self.seed, deterministic_torch=self.deterministic_torch)

        self.n_cells, self.n_genes = self.expression_data.shape
        self.device = device if device is not None else pick_torch_device(prefer_gpu=prefer_gpu)

        print(f"Number of cells: {self.n_cells:,}")
        print(f"Number of genes: {self.n_genes:,}")

        # flags
        self.rapids_available = _has_rapids()
        self.cugraph_available = _has_cugraph()

        if str(self.device) == "mps":
            # MPS exists only for torch ops; rapids won't exist on mac typically.
            pass

    # -------------------------
    # 1) Microenvironment construction (kNN)
    # -------------------------
    def build_microenvironment_data(
        self,
        agg="mean",             # "mean"|"sum"
        normalize="max",        # "max"|None
        use_gpu_knn_if_available=True,
        include_self=True,
        dtype=np.float32,
    ):
        """
        Builds microenv features per cell by aggregating neighbor expressions.
        Returns (indices, microenv_data)
        """
        print("Executing k-NN search...")

        # Decide backend
        use_gpu_knn = (
            use_gpu_knn_if_available
            and self.rapids_available
            and (str(self.device).startswith("cuda"))  # RAPIDS is CUDA-only
        )

        indices = None

        if use_gpu_knn:
            try:
                import cupy as cp
                from cuml.neighbors import NearestNeighbors as cuNN

                coords_gpu = cp.asarray(self.coords, dtype=cp.float32)
                knn = cuNN(n_neighbors=self.k_neighbors, algorithm="brute")
                knn.fit(coords_gpu)
                dists_gpu, idx_gpu = knn.kneighbors(coords_gpu)
                indices = cp.asnumpy(idx_gpu)

                print("✅ kNN on GPU (cuML)")
            except Exception as e:
                print(f"⚠️ GPU kNN failed -> fallback to sklearn. Reason: {e}")
                use_gpu_knn = False

        if not use_gpu_knn:
            nbrs = NearestNeighbors(
                n_neighbors=self.k_neighbors,
                algorithm="ball_tree" if self.coords.shape[1] <= 10 else "auto",
            ).fit(self.coords)
            _, indices = nbrs.kneighbors(self.coords)
            print("✅ kNN on CPU (sklearn)")

        # Aggregate neighbor expression
        microenv = np.zeros((self.n_cells, self.n_genes), dtype=dtype)

        start_col = 0 if include_self else 1
        print("Constructing microenvironment data...")
        for i in tqdm(range(self.n_cells)):
            neigh = indices[i, start_col:]
            Xn = self.expression_data[neigh]
            if agg == "sum":
                microenv[i] = Xn.sum(axis=0)
            else:
                microenv[i] = Xn.mean(axis=0)

        if normalize == "max":
            mx = float(microenv.max()) if microenv.size else 1.0
            microenv = microenv / (mx + 1e-8)

        self.microenv_data = microenv
        self.neighbor_indices = indices
        print(f"Microenvironment data shape: {microenv.shape}")
        return indices, microenv

    # -------------------------
    # 2) Train VAE (Torch: CUDA/MPS/CPU)
    # -------------------------
    def train_vae(
        self,
        dim_1=1024,
        dim_2=256,
        latent_dim=32,
        epochs=100,
        batch_size=256,
        lr=1e-3,
        weight_decay=1e-4,
        beta_warmup_epochs=50,
        shuffle=True,
        num_workers=None,
        pin_memory=None,
    ):
        """
        Train VAE on microenv_data.
        Notes:
          - On MPS, use num_workers=0 for stability.
          - On CUDA, pin_memory True can speed up.
        """
        if not hasattr(self, "microenv_data"):
            raise RuntimeError("Run build_microenvironment_data() first.")

        # DataLoader knobs (safe defaults)
        if num_workers is None:
            if str(self.device) == "cuda" or str(self.device) == "mps":
                num_workers = 0
            else:
                num_workers = max(0, min(os.cpu_count() - 1 if os.cpu_count() else 0, 8))

        if pin_memory is None:
            pin_memory = str(self.device).startswith("cuda")

        dataset = TensorDataset(torch.tensor(self.microenv_data, dtype=torch.float32))
        # 例：自動決定（必要なら引数で上書き）
        if num_workers is None:
            # Linuxなら 2〜8 くらいが現実的。巨大配列なので増やしすぎ注意
            num_workers = max(0, min(4, (os.cpu_count() or 4) - 2))
        
        # pin_memory は CUDA のときだけ意味あり
        if pin_memory is None:
            pin_memory = (self.device.type == "cuda")
        
        # multiprocessing_context は num_workers>0 のときだけ渡す
        loader_kwargs = dict(
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        )
        
        if num_workers > 0:
            ctx = mp.get_context("spawn")  # mac / notebook 安全寄り
            loader_kwargs["multiprocessing_context"] = ctx
            loader_kwargs["prefetch_factor"] = 2
        else:
            # ★ここが今回のポイント：num_workers=0 では prefetch_factor を渡さない
            loader_kwargs["prefetch_factor"] = None  # もしくは key自体を入れない
        dataloader = DataLoader(dataset, **{k:v for k,v in loader_kwargs.items() if v is not None})

        # Build model
        input_dim = self.n_genes
        self.vae = MicroenvironmentVAE(
            dim_1=dim_1,
            dim_2=dim_2,
            input_dim=input_dim,
            latent_dim=latent_dim,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr, weight_decay=weight_decay)

        self.vae.train()
        losses = []
        t0 = time.time()
        early_stop = True
        patience = 20
        min_delta = 1e-4
        monitor = "loss"   # or "rec"
        
        best = float("inf")
        bad_epochs = 0
        best_state = None
        best_epoch = 0

        print("Starting VAE training...")
        for epoch in range(epochs):
            ep_loss = 0.0
            ep_rec = 0.0
            ep_kl = 0.0

            beta = min(1.0, (epoch + 1) / max(1, beta_warmup_epochs))

            for (x,) in dataloader:
                x = x.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                recon, mu, logvar = self.vae(x)
                loss, rec_loss, kl_loss = vae_loss(recon, x, mu, logvar, beta=beta)
                loss.backward()
                optimizer.step()

                ep_loss += float(loss.item())
                ep_rec += float(rec_loss.item())
                ep_kl += float(kl_loss.item())

            ep_loss /= max(1, len(dataloader))
            ep_rec /= max(1, len(dataloader))
            ep_kl /= max(1, len(dataloader))
            losses.append(ep_loss)
            metric = ep_loss if monitor == "loss" else ep_rec  # あなたの変数名に合わせて

            if metric < best - min_delta:
                best = metric
                bad_epochs = 0
                best_epoch = epoch + 1
                # ベストモデルを保存（メモリOKならこれが確実）
                best_state = {k: v.detach().cpu().clone() for k, v in self.vae.state_dict().items()}
            else:
                bad_epochs += 1
            
            if early_stop and bad_epochs >= patience:
                print(f"\n🛑 Early stopping at epoch {epoch+1} (best {monitor}={best:.6g} @ epoch {best_epoch})")
                break

            if (epoch + 1) % 2 == 0:
                elapsed = time.time() - t0
                avg_ep = elapsed / (epoch + 1)
                remain = avg_ep * (epochs - (epoch + 1))
                msg = (
                    f"Epoch {epoch+1}/{epochs}  "
                    f"Loss:{ep_loss:.4g} (Rec:{ep_rec:.4g}, KL:{ep_kl:.4g}, beta:{beta:.2g})  "
                    f"[Elapsed:{elapsed:.1f} sec > Remain:{remain:.1f} sec]"
                )
                
                # 1行を上書き表示（末尾の空白は「前の長い行の残り」を消すため）
                print("\r" + msg + " " * 30, end="", flush=True)
                
                # 最後だけ改行（終わったらプロンプトを綺麗に戻す）
                if epoch + 1 == epochs:
                    print()

        if best_state is not None:
            self.vae.load_state_dict(best_state)
        self.losses = losses
        return self.vae

    # -------------------------
    # 3) Extract latent (mu)
    # -------------------------
    def extract_latent_features(self, batch_size=1024):
        if not hasattr(self, "vae"):
            raise RuntimeError("Run train_vae() first.")
        if not hasattr(self, "microenv_data"):
            raise RuntimeError("Run build_microenvironment_data() first.")

        self.vae.eval()
        latents = []

        X = self.microenv_data
        for i in range(0, X.shape[0], batch_size):
            xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                encoded = self.vae.encoder(xb)
                mu = self.vae.mu_layer(encoded)
            latents.append(mu.detach().cpu().numpy())

        self.latent_features = np.vstack(latents)
        print(f"Latent features shape: {self.latent_features.shape}")

    # -------------------------
    # 4) UMAP + Leiden
    #   - GPU if RAPIDS+cugraph available (CUDA only)
    #   - else CPU via scanpy
    # -------------------------
    def perform_umap_clustering(
        self,
        n_neighbors=15,
        min_dist=0.5,
        n_components=2,
        resolution=0.3,
        seed=None,
        cell_type_data=None,
        use_gpu_if_available=True,
        clustering_backend="auto",  # "auto"|"cpu"|"gpu"
        missing_vertex_policy="neighbor",  # "neighbor"|"noise"|"singleton"
    ):
        """Run UMAP + Leiden clustering with optional backend/policy controls.

        Parameters
        ----------
        clustering_backend
            "auto": prefer GPU when available and allowed.
            "cpu": force Scanpy CPU path.
            "gpu": force RAPIDS path (raises if unavailable).
        missing_vertex_policy
            Policy for vertices that do not receive a partition label on GPU path:
            - "neighbor": copy first available neighbor label (default, stable cluster counts)
            - "noise": keep as -1
            - "singleton": assign unique label per missing vertex (legacy behavior)
        """
        seed = self.seed if seed is None else int(seed)
        set_global_seed(seed, deterministic_torch=self.deterministic_torch)

        if not hasattr(self, "latent_features"):
            raise RuntimeError("Run extract_latent_features() first.")

        X = self.latent_features

        # Fill missing (rare)
        imputer = SimpleImputer(strategy="constant", fill_value=0)
        X = imputer.fit_transform(X)

        backend = str(clustering_backend).lower()
        if backend not in {"auto", "cpu", "gpu"}:
            raise ValueError("clustering_backend must be one of: auto, cpu, gpu")

        missing_policy = str(missing_vertex_policy).lower()
        if missing_policy not in {"neighbor", "noise", "singleton"}:
            raise ValueError("missing_vertex_policy must be one of: neighbor, noise, singleton")

        gpu_ready = (
            self.rapids_available
            and self.cugraph_available
            and str(self.device).startswith("cuda")
        )

        if backend == "cpu":
            use_gpu = False
        elif backend == "gpu":
            if not gpu_ready:
                raise RuntimeError("GPU clustering_backend requested but RAPIDS/cugraph CUDA path is unavailable.")
            use_gpu = True
        else:  # auto
            use_gpu = use_gpu_if_available and gpu_ready

        self.last_clustering_backend = "gpu" if use_gpu else "cpu"

        if use_gpu:
            try:
                import cupy as cp
                import cudf
                import cugraph
                from cuml.manifold import UMAP as cuUMAP
                from cuml.neighbors import NearestNeighbors as cuNN

                Xg = cp.asarray(X, dtype=cp.float32)

                # ---- kNN (GPU) ----
                knn = cuNN(n_neighbors=n_neighbors, algorithm="brute")
                knn.fit(Xg)
                dists, idx = knn.kneighbors(Xg)
                
                n = Xg.shape[0]
                src = cp.repeat(cp.arange(n, dtype=cp.int32), n_neighbors)
                dst = idx.reshape(-1).astype(cp.int32)
                
                # self-edge 除去（任意。入れても良いが普通は不要）
                mask = dst != src
                src = src[mask]; dst = dst[mask]
                
                # edges（重み無し）
                edges = cudf.DataFrame({"src": src, "dst": dst})
                
                # 重複除去（kNNの性質上ほぼ無いが念のため）
                edges = edges.drop_duplicates()
                # Keep deterministic edge order before graph construction.
                edges = edges.sort_values(by=["src", "dst"])
                print("edges:", len(edges))
                print("unique vertices in edges:", int(cudf.concat([edges["src"], edges["dst"]]).nunique()))
                G = cugraph.Graph(directed=False)
                G.from_cudf_edgelist(edges, source="src", destination="dst", renumber=False)
                # ---- UMAP (GPU) ----
                umap = cuUMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=seed)
                emb = umap.fit_transform(Xg)
                
                print("✅ UMAP+Leiden on GPU (RAPIDS)")
                m = len(edges)
                uniq_v = int(cudf.concat([edges["src"], edges["dst"]]).nunique())
                print("edges:", m)
                print("unique vertices in edges:", uniq_v)
                print("n vertices:", int(n))
                print("avg out-degree ~", m / n)        # directed view
                print("avg degree (undirected approx) ~", 2*m / n)
                # ---- ここが重要：全頂点に揃える ----
                parts, _ = cugraph.leiden(G, resolution=resolution, random_state=seed)
                parts = parts.sort_values("vertex")
                
                clusters = np.full(n, -1, dtype=np.int32)
                v = parts["vertex"].to_numpy()
                p = parts["partition"].to_numpy()
                clusters[v] = p
                
                missing = np.sum(clusters < 0)
                if missing > 0:
                    print(f"⚠️ leiden returned partitions for {n-missing}/{n} vertices. Applying missing policy: {missing_policy}.")
                    if missing_policy == "neighbor":
                        idx_np = cp.asnumpy(idx)
                        unresolved = clusters < 0
                        for i in np.where(unresolved)[0]:
                            neigh = idx_np[i]
                            neigh = neigh[neigh != i]
                            valid = neigh[clusters[neigh] >= 0]
                            if valid.size > 0:
                                clusters[i] = clusters[valid[0]]
                        # Any still unresolved fall back to noise label.
                        clusters[clusters < 0] = -1
                    elif missing_policy == "singleton":
                        start = clusters.max() + 1 if (clusters >= 0).any() else 0
                        clusters[clusters < 0] = np.arange(start, start + missing, dtype=np.int32)
                    else:  # noise
                        clusters[clusters < 0] = -1
                
                self.umap_embedding = cp.asnumpy(emb)
                self.clusters = clusters.astype(int)
                valid = self.clusters[self.clusters >= 0]
                print("n_clusters (excluding -1):", int(np.unique(valid).size) if valid.size else 0)

            except Exception as e:
                if backend == "gpu":
                    raise
                print(f"⚠️ GPU UMAP/Leiden failed -> fallback to scanpy CPU. Reason: {e}")
                use_gpu = False
                self.last_clustering_backend = "cpu"

        if not use_gpu:
            adata = sc.AnnData(X=X)
            adata.raw = None
            sc.pp.neighbors(adata, random_state=seed, n_neighbors=n_neighbors, use_rep="X")
            sc.tl.umap(adata, min_dist=min_dist, random_state=seed)
            sc.tl.leiden(adata, resolution=resolution, key_added="leiden", random_state=seed)

            self.umap_embedding = adata.obsm["X_umap"]
            self.clusters = adata.obs["leiden"].astype(int).values
            self.adata = adata

            print("✅ UMAP+Leiden on CPU (scanpy)")

        if cell_type_data is not None:
            # store alongside for plotting convenience
            if not hasattr(self, "adata"):
                self.adata = sc.AnnData(X=X)
                self.adata.obsm["X_umap"] = self.umap_embedding
                self.adata.obs["leiden"] = pd.Categorical(self.clusters.astype(str))
            self.adata.obs["cell_type"] = pd.Categorical(np.asarray(cell_type_data).astype(str))

        valid = self.clusters[self.clusters >= 0]
        n_clusters = int(np.unique(valid).size) if valid.size else 0
        print(f"Number of clusters (excluding -1): {n_clusters}")
        return self.umap_embedding, self.clusters

    # -------------------------
    # 5) Scanpy-style plot
    # -------------------------
    def visualize_results(
        self,
        figsize=(15, 18),
        max_clusters_to_show=30,   # ← 多すぎる時は上位だけ
        min_cluster_size=0,       # ← 小さすぎるクラスタはOtherへ
        point_size=0.5,
        alpha=0.6,
    ):
    
        if not hasattr(self, "clusters") or self.clusters is None:
            raise RuntimeError("self.clusters not found. Run perform_umap_clustering() first.")
        if not hasattr(self, "umap_embedding") or self.umap_embedding is None:
            raise RuntimeError("self.umap_embedding not found. Run perform_umap_clustering() first.")
        if not hasattr(self, "latent_features") or self.latent_features is None:
            raise RuntimeError("self.latent_features not found. Run extract_latent_features() first.")
        
        clusters = np.asarray(self.clusters)
        print(clusters.dtype, clusters.min(), clusters.max(), np.unique(clusters[:10]))
        
        # 強制的に int に
        clusters = clusters.astype(np.int32)
        self.clusters = clusters
        clusters_raw = np.asarray(self.clusters)
        K = int(clusters.max())  # 0..K-1 を想定（Other=-1があるなら別処理）
        # ----- sanitize cluster labels -----
        # allow -1, non-contiguous labels
        valid_mask = clusters_raw >= 0
        if valid_mask.sum() == 0:
            raise RuntimeError("All clusters are -1 (missing). Graph/Leiden may have failed.")
    
        # counts (safe even if labels are huge / non-contiguous)
        vc = pd.Series(clusters_raw[valid_mask]).value_counts().sort_values(ascending=False)
    
        # Decide which clusters to show
        keep = vc[vc >= min_cluster_size].index[:max_clusters_to_show].to_numpy()
    
        # Make "display labels": keep as-is, others -> -1 (Other)
        disp = clusters_raw.copy()
        is_keep = np.isin(disp, keep)
        disp[~is_keep] = -1  # Other
    
        # remap display labels to 0..K for coloring
        # Other -> -1 stays -1
        keep_list = list(keep)
        map_dict = {cid: i for i, cid in enumerate(keep_list)}
        disp_mapped = np.full(disp.shape, -1, dtype=np.int32)
        for cid, new_id in map_dict.items():
            disp_mapped[disp == cid] = new_id
    
        n_keep = len(keep_list)
        n_other = int((disp_mapped < 0).sum())
        total = len(disp_mapped)
    
        # choose colormap
        if n_keep <= 20:
            cmap = plt.get_cmap("tab20", K)  # K色に離散化
        elif n_keep <= 60:
            cmap = plt.get_cmap("tab20b", K)  # K色に離散化
        else:
            cmap = "hsv"
    
        fig, axes = plt.subplots(3, 2, figsize=figsize)
    
        # 1. Training loss
        if hasattr(self, "losses") and self.losses is not None:
            axes[0, 0].plot(self.losses)
            axes[0, 0].set_title("VAE Training Loss")
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("Loss")
        else:
            axes[0, 0].text(0.5, 0.5, "No losses recorded", ha="center", va="center")
            axes[0, 0].set_axis_off()
    
        # helper for scatter
        def _scatter(ax, x, y, c, title, xlabel, ylabel):
            sca = ax.scatter(
                x, y,
                c=c,
                s=point_size,
                alpha=alpha,
                cmap=cmap,
                rasterized=True,   # huge speedup for many points in vector backends
                linewidths=0,
            )
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            return sca
    
        # 2. Spatial distribution
        sca1 = _scatter(
            axes[0, 1],
            self.coords[:, 1], self.coords[:, 0],
            disp_mapped,
            f"Spatial Distribution (top {n_keep} clusters; Other={n_other:,}/{total:,})",
            "Array Column", "Array Row",
        )
        plt.colorbar(sca1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
        # 3. UMAP embedding
        sca2 = _scatter(
            axes[1, 0],
            self.umap_embedding[:, 0], self.umap_embedding[:, 1],
            disp_mapped,
            "UMAP Embedding (Colored by Cluster)",
            "UMAP 1", "UMAP 2",
        )
        plt.colorbar(sca2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
        # 4. Cluster distribution (use original counts)
        top_show = vc.head(max_clusters_to_show)
        axes[1, 1].bar(range(len(top_show)), top_show.values)
        axes[1, 1].set_title(f"Cluster Size Distribution (top {len(top_show)})")
        axes[1, 1].set_xlabel("Ranked Cluster")
        axes[1, 1].set_ylabel("Number of Cells")
        axes[1, 1].set_xticks(range(len(top_show)))
        axes[1, 1].set_xticklabels([str(x) for x in top_show.index], rotation=90, fontsize=8)
    
        # 5. Latent feature distribution (Dim0 vs Dim1)
        sca3 = _scatter(
            axes[2, 0],
            self.latent_features[:, 0], self.latent_features[:, 1],
            disp_mapped,
            "Latent Features (Dim 0 vs 1)",
            "Latent Dim 0", "Latent Dim 1",
        )
        plt.colorbar(sca3, ax=axes[2, 0], fraction=0.046, pad=0.04)
    
        # 6. Cluster mean latent features heatmap (top clusters only)
        # Compute means only for kept clusters (avoid 100k x 32)
        means = []
        labels = []
        for cid in keep_list:
            m = (clusters_raw == cid)
            if m.sum() == 0:
                continue
            means.append(self.latent_features[m].mean(axis=0))
            labels.append(str(cid))
        if len(means) > 0:
            cluster_means = np.vstack(means)  # (K, latent_dim)
            im = axes[2, 1].imshow(cluster_means, aspect="auto", cmap="viridis")
            axes[2, 1].set_title("Cluster Mean Latent Features (top clusters)")
            axes[2, 1].set_xlabel("Latent Dimension")
            axes[2, 1].set_ylabel("Cluster ID")
            axes[2, 1].set_yticks(range(len(labels)))
            axes[2, 1].set_yticklabels(labels, fontsize=8)
            plt.colorbar(im, ax=axes[2, 1], fraction=0.046, pad=0.04)
        else:
            axes[2, 1].text(0.5, 0.5, "No clusters to show", ha="center", va="center")
            axes[2, 1].set_axis_off()
    
        plt.tight_layout()
        plt.show()

    def visualize_scanpy_results(self):
        if not hasattr(self, "adata"):
            # build minimal adata if GPU path didn't create it
            X = self.latent_features
            ad = sc.AnnData(X=X)
            ad.obsm["X_umap"] = self.umap_embedding
            ad.obs["leiden"] = pd.Categorical(self.clusters.astype(str))
            self.adata = ad

        if "cell_type" in self.adata.obs.columns:
            sc.pl.umap(self.adata, color=["cell_type"], wspace=0.4)
            sc.pl.umap(self.adata, color=["leiden"], wspace=0.4)
        else:
            sc.pl.umap(self.adata, color="leiden")