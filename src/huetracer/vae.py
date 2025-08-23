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

# Helper function to format time in HH:MM:SS or MM:SS format
def format_time(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"
    
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """VAE loss function (reconstruction error + KL divergence)"""
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
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


class SpatialMicroenvironmentAnalyzer:
    """Class for spatial transcriptomics microenvironment analysis"""
    
    def __init__(self, coords, expression_data, k_neighbors=30, device = torch.device('cpu')):
        """
        Parameters:
        coords: numpy array of shape (n_cells, 2) - XY coordinates
        expression_data: numpy array of shape (n_cells, n_genes) - Expression data
        k_neighbors: int - Number of neighboring cells
        """
        self.coords = coords
        self.expression_data = expression_data
        self.k_neighbors = k_neighbors
        self.n_cells, self.n_genes = expression_data.shape
        self.device = device
        print(f"Number of cells: {self.n_cells:,}")
        print(f"Number of genes: {self.n_genes:,}")
        
    def build_microenvironment_data(self):
        """Construct microenvironment data for each cell"""
        print("Executing k-NN search...")
        
        # Build k-NN model
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors, 
                                algorithm='ball_tree').fit(self.coords)
        
        # Get neighboring cells for each cell
        distances, indices = nbrs.kneighbors(self.coords)
        
        # Initialize microenvironment data
        # microenv_data = np.zeros((self.n_cells, self.k_neighbors * self.n_genes))
        microenv_data = np.zeros((self.n_cells, self.n_genes))
        
        print("Constructing microenvironment data...")
        for i in tqdm(range(self.n_cells)):
            # Indices of neighboring cells
            neighbor_indices = indices[i, 0:]  # The first one is the cell itself, don't exclude
            #neighbor_indices = indices[i, 1:]  # Exclude the first one since it's the cell itself
            
            # Get expression data of neighboring cells
            neighbor_expr = self.expression_data[neighbor_indices]
            
            # # Sort each gene in descending order
            # sorted_expr = np.sort(neighbor_expr, axis=0)[::-1]  # Descending
            
            # # Flatten and store in microenv_data
            # microenv_data[i] = sorted_expr.flatten()
            microenv_data[i] = np.mean(neighbor_expr, axis=0)

        microenv_data = microenv_data / microenv_data.max()
        
        self.microenv_data = microenv_data
        print(f"Microenvironment data shape: {microenv_data.shape}")
        
        return indices, microenv_data
    
    def train_vae(self, dim_1 = 1024, dim_2 = 256, latent_dim=32, epochs=100, batch_size=256, lr=1e-3, weight_decay=1e-4, beta=1.0):
        """Train VAE"""
        print("Starting VAE training...")
        
        # Prepare data loader
        cpu_count = multiprocessing.cpu_count()
        num_workers = min(cpu_count - 1, cpu_count // os.cpu_count() if os.cpu_count() else 1)
        dataset = TensorDataset(torch.FloatTensor(self.microenv_data))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, multiprocessing_context="fork")
        
        # Initialize model
        # input_dim = self.k_neighbors * self.n_genes
        input_dim = self.n_genes
        self.vae = MicroenvironmentVAE(dim_1 = dim_1, dim_2 = dim_2, input_dim=input_dim, latent_dim=latent_dim).to(self.device)
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Training loop
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
        """Extract latent features (mu)"""
        print("Extracting latent features...")
        
        # Data extraction
        if isinstance(self.microenv_data, pd.DataFrame):
            microenv_data_np = self.microenv_data.values
        else:
            microenv_data_np = self.microenv_data
        
        self.vae.eval() # Set model to evaluation mode
        latent_features = [] # List to store extracted latent features

        # Extract latent features in batches
        batch_size = 1024 # Define batch size
        
        # Process microenv_data_np in batches
        for i in range(0, len(microenv_data_np), batch_size):
            # Get the current batch as a NumPy array
            batch_np = microenv_data_np[i:i+batch_size]
            
            # Convert batch to a PyTorch FloatTensor and send to device
            batch_tensor = torch.FloatTensor(batch_np).to(self.device)
            
            with torch.no_grad(): # Disable gradient calculation (inference mode)
                encoded = self.vae.encoder(batch_tensor) # Encode with the encoder
                mu = self.vae.mu_layer(encoded) # Extract latent features mu from the mu layer
                latent_features.append(mu.cpu().numpy()) # Convert to NumPy array and append to list

        # Vertically stack the latent features from all batches
        self.latent_features = np.vstack(latent_features)
        
        print(f"Latent features shape: {self.latent_features.shape}")
        return self.latent_features
    
    def reconstruct_from_latent(self):
        """Get reconstructed data from latent features"""
        print("Generating reconstructed data from latent features...")
    
        # Set model to evaluation mode
        self.vae.eval()
    
        # Convert latent features to a Tensor
        latent_tensor = torch.FloatTensor(self.latent_features).to(self.device)
    
        # Reconstruct through the decoder
        with torch.no_grad():
            recon_x = self.vae.decoder(latent_tensor)
    
        # Convert to NumPy array and return
        self.reconstructed_data = recon_x.cpu().numpy()
        print(f"Reconstructed data shape: {self.reconstructed_data.shape}")
        return self.reconstructed_data
    
    def perform_umap_clustering(self, n_neighbors=15, min_dist=0.5, n_components=2, 
                                cell_type_data=None, seed=42):
        """UMAP dimensionality reduction and Leiden clustering"""
        print("Creating AnnData object...")
        
        # Create AnnData object
        adata = sc.AnnData(X=self.latent_features)
        
        # Pre-process data (impute missing values)
        print("Preprocessing data...")
        if hasattr(adata.X, "toarray"):
            X_dense = adata.X.toarray()
        else:
            X_dense = adata.X
        
        imputer = SimpleImputer(strategy='constant', fill_value=0)
        adata.X = imputer.fit_transform(X_dense)
        adata.raw = None
        
        # Build neighborhood graph
        print("Building neighborhood graph...")
        sc.pp.neighbors(adata, random_state=seed, n_neighbors=n_neighbors, use_rep='X')
        
        # UMAP
        print("Executing UMAP dimensionality reduction...")
        sc.tl.umap(adata, min_dist=min_dist, random_state=seed)
        
        # Leiden clustering
        print("Executing Leiden clustering...")
        sc.tl.leiden(adata, resolution=0.3, key_added='leiden', random_state=seed)
        
        # Add Cell type information (if provided)
        if cell_type_data is not None:
            adata.obs['cell_type'] = cell_type_data.tolist()
        
        self.clusters = adata.obs['leiden'].values.astype(int)
        self.adata = adata
        self.umap_embedding = adata.obsm['X_umap']
        
        print(f"Number of clusters: {len(np.unique(self.clusters))}")
        
        return self.umap_embedding, self.clusters
    
    def visualize_results(self, figsize=(15, 18)):
        # Basic visualization
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        
        # 1. Training loss
        axes[0, 0].plot(self.losses)
        axes[0, 0].set_title('VAE Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        
        # 2. Spatial distribution
        scatter = axes[0, 1].scatter(self.coords[:, 1], self.coords[:, 0], 
                                     c=self.clusters, cmap='tab20', s=0.5)
        axes[0, 1].set_title('Spatial Distribution (Colored by Cluster)')
        axes[0, 1].set_xlabel('Array Column')
        axes[0, 1].set_ylabel('Array Row')
        plt.colorbar(scatter, ax=axes[0, 1])
        
        # 3. UMAP embedding
        scatter2 = axes[1, 0].scatter(self.umap_embedding[:, 0], 
                                     self.umap_embedding[:, 1],
                                     c=self.clusters, cmap='tab20', s=0.5)
        axes[1, 0].set_title('UMAP Embedding (Colored by Cluster)')
        axes[1, 0].set_xlabel('UMAP 1')
        axes[1, 0].set_ylabel('UMAP 2')
        plt.colorbar(scatter2, ax=axes[1, 0])
        
        # 4. Cluster distribution
        cluster_counts = np.bincount(self.clusters)
        axes[1, 1].bar(range(len(cluster_counts)), cluster_counts)
        axes[1, 1].set_title('Cluster Size Distribution')
        axes[1, 1].set_xlabel('Cluster ID')
        axes[1, 1].set_ylabel('Number of Cells')
        
        # 5. Latent feature distribution (first 2 dimensions)
        axes[2, 0].scatter(self.latent_features[:, 0], 
                           self.latent_features[:, 1],
                           c=self.clusters, cmap='tab20', s=0.5, alpha=0.6)
        axes[2, 0].set_title('Latent Features (Dim 0 vs 1)')
        axes[2, 0].set_xlabel('Latent Feature Dimension 0')
        axes[2, 0].set_ylabel('Latent Feature Dimension 1')
        
        # 6. Latent feature heatmap (mean for each cluster)
        cluster_means = []
        for cluster_id in np.unique(self.clusters):
            mask = self.clusters == cluster_id
            cluster_mean = self.latent_features[mask].mean(axis=0)
            cluster_means.append(cluster_mean)
        
        cluster_means = np.array(cluster_means)
        im = axes[2, 1].imshow(cluster_means, aspect='auto', cmap='viridis')
        axes[2, 1].set_title('Cluster Mean Latent Features')
        axes[2, 1].set_xlabel('Latent Dimension')
        axes[2, 1].set_ylabel('Cluster ID')
        plt.colorbar(im, ax=axes[2, 1])
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def visualize_scanpy_results(self):
        """Detailed visualization in Scanpy format"""
        if not hasattr(self, 'adata'):
            print("Please run perform_umap_clustering() first")
            return
        
        # Basic UMAP visualization
        print("Basic UMAP visualization:")
        if 'cell_type' in self.adata.obs.columns:
            sc.pl.umap(self.adata, color=['cell_type', 'leiden'], wspace=0.4)
        else:
            sc.pl.umap(self.adata, color='leiden')