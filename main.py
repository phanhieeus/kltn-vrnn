import os
import random
import wandb
from dotenv import load_dotenv

import numpy as np

import torch
from torch.utils.data import DataLoader
from model import VRNN
from data_utils import FinanceDataset
from train import train
from torch.optim import Adam
from logger_utils import setup_logging, get_logger

logger = get_logger(__name__)



def main():
    load_dotenv()
    setup_logging()
    # Ensure deterministic behavior


    torch.backends.cudnn.deterministic = True
    random.seed(hash("setting random seeds") % 2**32 - 1)
    np.random.seed(hash("improves reproducibility") % 2**32 - 1)
    torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")


    # Hyperparameters 
    config = {
        "x_dim": 1,
        "z_dim": 16,
        "h_dim": 64,
        "n_layers": 2,
        "T": 40,
        "file_path": r"data\FPT Corp Stock Price History.csv",
        "feature_columns": ["Change %"],
        "normalize_dataset": True,
        "batch_size": 32,
        "lr": 1e-3,
        "epochs": 100,
        "level": "one-stock",
        "stock_name": "FPT",
        "category": "retail"
    }

    wandb.init(project="vrnn", config=config)
    config = wandb.config
    # Dataset & DataLoader
    dataset = FinanceDataset(T=config.T, file_path=config.file_path, feature_columns=config.feature_columns, normalize=config.normalize_dataset)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    # Model & Optimizer
    model = VRNN(x_dim=config.x_dim, z_dim=config.z_dim, h_dim=config.h_dim, n_layers=config.n_layers).to(device)
    optimizer = Adam(model.parameters(), lr=config.lr)
    
    # Start Training    
    model = train(model, dataloader, optimizer, epochs=config.epochs, device=device)

    logger.info("Training Complete")

    wandb.finish()


if __name__ == "__main__":
    main()
