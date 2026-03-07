import torch
import logging
from torch.utils.data import Dataset, DataLoader

import pandas as pd

class FinanceDataset(Dataset):
    def __init__(self, T: int, file_path: str, feature_columns: list[str], normalize: bool = True):
        self.T = T
        self.file_path = file_path
        self.feature_columns = feature_columns
        self.normalize = normalize

        df = pd.read_csv(file_path)
        df = df[self.feature_columns]

        # Clean string columns: remove '%' and ',' then convert to float
        for col in self.feature_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].astype(str).str.replace('%', '', regex=False)
                df[col] = df[col].str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any rows with NaN values
        df = df.dropna()
        
        self.data = df.values.astype('float32')

        if self.normalize:
            self.data = (self.data - self.data.mean(axis=0)) / (self.data.std(axis=0) + 1e-8)

    def __len__(self):
        return len(self.data) - self.T + 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.T]
        return torch.tensor(x, dtype=torch.float32)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    dataset = FinanceDataset(T=40, file_path=r"data\FPT Corp Stock Price History.csv", feature_columns=["Change %"])
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    logger.info(f"Dataset shape: {dataset.data.shape}")
    logger.info(f"Dataloader length: {len(dataloader)}")
    for x in dataloader:
        logger.info(f"Batch shape: {x.shape}")
        break  # Just check one shape
