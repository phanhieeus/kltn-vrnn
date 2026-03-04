import torch
from torch.utils.data import DataLoader
from model import VRNN
from data_utils import FinanceDataset
from train import train
from torch.optim import Adam

def main():
    # Hyperparameters
    x_dim = 1
    z_dim = 16
    h_dim = 64
    n_layers = 2
    T = 40
    batch_size = 32
    lr = 1e-3
    epochs = 100
    file_path = r"data\FPT Corp Stock Price History.csv"
    feature_columns = ["Change %"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset & DataLoader
    dataset = FinanceDataset(T=T, file_path=file_path, feature_columns=feature_columns)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Model & Optimizer
    model = VRNN(x_dim=x_dim, z_dim=z_dim, h_dim=h_dim, n_layers=n_layers).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    # Start Training    
    model, history = train(model, dataloader, optimizer, epochs=epochs, device=device)
    
    print("\n--- Training Complete ---")
    print(f"Final Total Loss: {history['total_loss'][-1]:.4f}")
    print(f"Final Recon Loss: {history['recon_loss'][-1]:.4f}")
    print(f"Final KLD Loss: {history['kld_loss'][-1]:.4f}")

if __name__ == "__main__":
    main()
