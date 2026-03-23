import numpy as np
import matplotlib
matplotlib.use('Agg') # Sử dụng backend không hiển thị để tránh treo terminal
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    data = np.load(file_path)
    return data['mu']

def plot_latent_time_series(mu, num_samples=5):
    """
    Visualize latent mu over time for a few random samples.
    mu shape: (Batch, Time, Z_dim)
    """
    B, T, Z = mu.shape
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples), sharex=True)
    if num_samples == 1: axes = [axes]
    
    indices = np.random.choice(B, num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        for z in range(Z):
            axes[i].plot(mu[idx, :, z], label=f'Dim {z}' if i == 0 else "")
        axes[i].set_title(f"Sample Index: {idx}")
        axes[i].set_ylabel("Mu Value")
        
    axes[0].legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))
    plt.xlabel("Time Step")
    plt.tight_layout()
    plt.savefig("results/latent_trajectories.png")
    print("Saved latent trajectories to results/latent_trajectories.png")

def plot_latent_space_distribution(mu):
    """
    Visualize the distribution of the latent space (mean across time) 
    using PCA and t-SNE.
    """
    B, T, Z = mu.shape
    # Flatten or average over time? 
    # Let's use the mean mu across the sequence as a representation of each window
    mu_flat = mu.mean(axis=1) # (B, Z)
    
    # 1. PCA
    pca = PCA(n_components=2)
    mu_pca = pca.fit_transform(mu_flat)
    
    # 2. t-SNE
    print("Running t-SNE (this might take a moment)...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    mu_tsne = tsne.fit_transform(mu_flat)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot PCA
    ax1.scatter(mu_pca[:, 0], mu_pca[:, 1], alpha=0.5, c='blue', s=10)
    ax1.set_title(f"PCA of Latent Space (Var explained: {sum(pca.explained_variance_ratio_):.2f})")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    
    # Plot t-SNE
    ax2.scatter(mu_tsne[:, 0], mu_tsne[:, 1], alpha=0.5, c='red', s=10)
    ax2.set_title("t-SNE of Latent Space")
    ax2.set_xlabel("Dimension 1")
    ax2.set_ylabel("Dimension 2")
    
    plt.tight_layout()
    plt.savefig("results/latent_space_cluster.png")
    print("Saved latent space visualization to results/latent_space_cluster.png")

def plot_dimension_variance(mu):
    """
    Check for latent collapse by looking at the variance of each dimension.
    """
    # Calculate variance across all batches and all time steps for each dimension
    variances = mu.reshape(-1, mu.shape[-1]).var(axis=0)
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(variances)), variances, color='teal')
    plt.title("Variance per Latent Dimension (Check for Collapse)")
    plt.xlabel("Dimension Index")
    plt.ylabel("Variance")
    plt.xticks(range(len(variances)))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("results/latent_variance.png")
    print("Saved dimension variance plot to results/latent_variance.png")

if __name__ == "__main__":
    file_path = "results/latent_results.npz"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Please run 'uv run python -m evaluate.infer_z' first.")
    else:
        mu = load_data(file_path)
        print(f"Loaded mu with shape: {mu.shape}")
        
        # 1. Visualize trajectories over time
        plot_latent_time_series(mu)
        
        # 2. Visualize global distribution (PCA/t-SNE)
        plot_latent_space_distribution(mu)
        
        # 3. Check for posterior collapse
        plot_dimension_variance(mu)
