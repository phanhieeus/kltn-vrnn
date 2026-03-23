import os
import random
import wandb
from dotenv import load_dotenv

import numpy as np

import torch
from torch.utils.data import DataLoader
from .trainer import train
from .model import VRNN
from .data_utils import FinanceDataset
from .logger_utils import setup_logging, get_logger
from torch.optim import Adam


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
        "x_dim": 15,
        "z_dim": 8,
        "h_dim": 32,
        "n_layers": 2,
        "T": 20,
        "file_path": r"data\FPT_features.csv",
        "feature_columns": "all",
        "clean_data": False,
        "normalize_dataset": False,
        "batch_size": 32,
        "lr": 1e-3,
        "epochs": 1000,
        "warmup_epochs": 200,
        "level": "one-stock",
        "stock_name": "FPT",
        "category": "retail"
    }

    wandb.init(project="vrnn", config=config)
    config = wandb.config

    # Log Dataset as Artifact
    if os.path.exists(config.file_path):
        dataset_artifact = wandb.Artifact(
            name=f"finance-dataset-{config.stock_name}", 
            type="dataset",
            description=f"Stock price history for {config.stock_name}",
            metadata=dict(config)
        )
        dataset_artifact.add_file(config.file_path)
        wandb.log_artifact(dataset_artifact)

    # Dataset & DataLoader
    dataset = FinanceDataset(T=config.T, file_path=config.file_path, feature_columns=config.feature_columns, clean=config.clean_data, normalize=config.normalize_dataset)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)


    # Model & Optimizer
    model = VRNN(x_dim=config.x_dim, z_dim=config.z_dim, h_dim=config.h_dim, n_layers=config.n_layers).to(device)
    optimizer = Adam(model.parameters(), lr=config.lr)
    
    # Start Training    
    model = train(model, dataloader, optimizer, epochs=config.epochs, warmup_epochs=config.warmup_epochs, device=device)

    logger.info("Training Complete")

    wandb.finish()


if __name__ == "__main__":
    main()
