import os
import torch
import numpy as np
import pandas as pd
import logging
from torch.utils.data import DataLoader
from training.model import VRNN
from training.data_utils import FinanceDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def infer_all_z(model: VRNN, dataloader: DataLoader, device: torch.device):
    """
    Sử dụng method infer_latent từ model để trích xuất các biến ẩn
    """
    model.eval()
    all_z = []
    all_mu = []
    all_logvar = []
    
    with torch.no_grad():
        for x in dataloader:
            x = x.to(device)
            # x shape: (B, T, x_dim) từ FinanceDataset
            # VRNN model mong đợi (T, B, x_dim)
            x = x.permute(1, 0, 2)
            
            z_t, mu_t, logvar_t = model.infer_latent(x)
            
            # Chuyển về (B, T, dim) để dễ quản lý theo batch
            all_z.append(z_t.permute(1, 0, 2).cpu().numpy())
            all_mu.append(mu_t.permute(1, 0, 2).cpu().numpy())
            all_logvar.append(logvar_t.permute(1, 0, 2).cpu().numpy())
            
    return (np.concatenate(all_z, axis=0), 
            np.concatenate(all_mu, axis=0), 
            np.concatenate(all_logvar, axis=0))

def main():
    # Cấu hình khớp với training/main.py
    config = {
        "x_dim": 15,
        "z_dim": 8,
        "h_dim": 32,
        "n_layers": 2,
        "T": 20,
        "file_path": r"data/FPT_features.csv",
        "feature_columns": "all",
        "clean_data": False,
        "normalize_dataset": False,
        "batch_size": 32,
        "model_path": "checkpoints/vrnn_final.pth", # Cập nhật tên file thực tế
        "output_dir": "results"
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Khởi tạo Model
    model = VRNN(
        x_dim=config["x_dim"],
        z_dim=config["z_dim"],
        h_dim=config["h_dim"],
        n_layers=config["n_layers"]
    ).to(device)
    
    # 2. Load trọng số
    if os.path.exists(config["model_path"]):
        model.load_state_dict(torch.load(config["model_path"], map_location=device))
        logger.info(f"Loaded model weights from {config['model_path']}")
    else:
        logger.warning(f"Model path {config['model_path']} not found. Running with random weights!")

    # 3. Chuẩn bị Data
    dataset = FinanceDataset(
        T=config["T"],
        file_path=config["file_path"],
        feature_columns=config["feature_columns"],
        clean=config["clean_data"],
        normalize=config["normalize_dataset"]
    )
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
    
    # 4. Suy luận (Inference)
    logger.info("Starting latent inference...")
    z_all, mu_all, logvar_all = infer_all_z(model, dataloader, device)
    
    # Lấy danh sách ngày
    all_dates = dataset.get_dates()
    
    # 5. Lưu kết quả vào 1 file duy nhất
    os.makedirs(config["output_dir"], exist_ok=True)
    output_path = os.path.join(config["output_dir"], "latent_results.npz")
    
    # Tính variance từ logvar: var = exp(logvar)
    var_all = np.exp(logvar_all)
    
    # Lưu tất cả vào 1 file .npz
    save_dict = {
        "z": z_all, 
        "mu": mu_all, 
        "var": var_all
    }
    if all_dates is not None:
        save_dict["dates"] = all_dates

    np.savez(output_path, **save_dict)
    
    logger.info(f"Inference complete. All results saved to: {output_path}")
    if all_dates is not None:
        logger.info(f"Number of dates saved: {len(all_dates)}")
    logger.info(f"Data shape: {z_all.shape} (Batch, Time, Z_dim)")

if __name__ == "__main__":
    main()
