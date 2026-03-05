# Latent Sentiment Analysis of Vietnamese Stock Market using VRNN 🚀📈

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-ee4c2c.svg)](https://pytorch.org/)
[![W&B](https://img.shields.io/badge/Weights_%26_Biases-Enabled-gold.svg)](https://wandb.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This research project explores the extraction of **latent sentiment** from the Vietnamese stock market using **Variational Recurrent Neural Networks (VRNN)**. The model captures temporal dependencies and stochastic nature of stock prices to identify underlying market moods.

---

## 🌟 Key Features

- **VRNN Architecture**: Stochastic modeling for high-volatility financial time series.
- **W&B Integration**: 
    - Real-time metric tracking (Loss, Reconstruction, KLD).
    - **Artifacts Management**: Version control for datasets and model checkpoints.
- **Professional Logging**: Standardized Python logging instead of simple prints.
- **Environment Management**: Fully powered by `uv` for reproducible environments.
- **GPU Accelerated**: Built-in support for CUDA.

---

## 📂 Project Structure

```text
.
├── checkpoints/            # Local storage for model checkpoints (.pth)
├── data/                   # Historical stock market datasets (CSV)
├── main.py                 # Project entry point (orchestrates training)
├── train.py                # Core training loop and W&B logging logic
├── model.py                # VRNN architecture implementation
├── data_utils.py           # Preprocessing and DataLoader utilities
├── logger_utils.py         # Standardized logging configuration
├── test_cuda.py            # Utility to verify GPU acceleration
├── .env                    # Environment variables (WANDB_API_KEY)
├── pyproject.toml          # Project dependencies
└── README.md               # You are here!
```

---

## 🛠️ Installation & Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

### 1. Clone the repository
```bash
git clone https://github.com/phanhieeus/kltn-vrnn.git
cd kltn-vrnn
```

### 2. Install dependencies
```bash
uv sync
```

### 3. Configure Weights & Biases
Create a `.env` file in the root directory and add your W&B API Key:
```env
WANDB_API_KEY=your_api_key_here
```

---

## 🚀 How to Run

### 1. Training the Model
To start the training process with full W&B logging and artifact tracking:
```bash
uv run python main.py
```
This command will:
1. Load configurations and data.
2. Upload the dataset as a **W&B Artifact**.
3. Train the VRNN while logging metrics to the cloud.
4. Save and upload model checkpoints periodically.

### 2. Verify Hardware Acceleration
To check if PyTorch can access your GPU:
```bash
uv run python test_cuda.py
```

### 3. Model Architecture Test
Run a quick forward pass test:
```bash
uv run python model.py
```

---

## 📊 Monitoring with W&B

All training metrics and artifacts are available on your W&B Dashboard.
- **Metrics**: Track `total_loss`, `recon_loss`, and `kld_loss` live.
- **Artifacts**: Access and download specific model versions from the "Artifacts" tab.
- **System**: Monitor GPU/CPU utilization during the training run.

---

## 🧠 Model Architecture

The **VRNN** extends the standard RNN by including a latent random variable $z_t$ at each time step, allowing for better handling of uncertainty in financial data.
- **Encoder**: $q(z_t | x_{\leq t}, z_{< t})$
- **Prior**: $p(z_t | x_{< t}, z_{< t})$
- **Decoder**: $p(x_t | z_t, x_{< t}, z_{< t})$

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Created for Vietnamese FinTech Research.*

