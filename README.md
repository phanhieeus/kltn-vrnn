# Latent Sentiment Analysis of Vietnamese Stock Market using VRNN 🚀📈

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This research project explores the extraction of **latent sentiment** from the Vietnamese stock market using **Variational Recurrent Neural Networks (VRNN)**. By leveraging the temporal dependencies and stochastic nature of stock prices, the model aims to capture underlying market moods that traditional deterministic models might miss.

---

## 🌟 Key Features

- **Variational Recurrent Neural Networks (VRNN)**: Combines the power of Recurrent Neural Networks (RNNs) with the stochasticity of Variational Autoencoders (VAEs).
- **Stochastic Modeling**: Specifically designed for high-volatility time series like stock market data.
- **Latent Sentiment Extraction**: Aims to identify non-linear patterns and "hidden" sentiment transitions in VN30 and VN-Index.
- **PyTorch Implementation**: High-performance model training with CUDA support.

---

## 📂 Project Structure

```text
.
├── data/                   # Historical stock market datasets (CSV)
│   ├── Dữ liệu Lịch sử VN 30.csv
│   └── Dữ liệu Lịch sử VN Index.csv
├── model.py                # Core VRNN Model architecture implementation
├── data_utils.py           # Data loading, cleaning, and preprocessing utilities
├── test_cuda.py            # Utility to verify GPU acceleration availability
├── pyproject.toml          # Project dependencies and metadata
└── README.md               # You are here!
```

---

## 🛠️ Installation

This project uses [uv](https://github.com/astral-sh/uv) for lightning-fast dependency management.

### 1. Clone the repository
```bash
git clone https://github.com/your-username/kltn-vrnn.git
cd kltn-vrnn
```

### 2. Set up the environment
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 3. Verify CUDA (Optional - for GPU training)
```bash
python test_cuda.py
```

---

## 🚀 Usage

Currently, the model architecture is defined in `model.py`. You can run a quick check of the model with:

```bash
python model.py
```

> [!NOTE]
> The `main.py` entry point and full training pipeline are currently under development as part of the research phase.

---

## 📊 Dataset Description

The research focuses on the following primary indices from the Vietnamese market:
- **VN30 Index**: Top 30 large-cap stocks on the Ho Chi Minh City Stock Exchange (HOSE).
- **VN-Index**: Represents the performance of all stocks listed on HOSE.

Data includes historical price points (Open, High, Low, Close) and Volume.

---

## 🧠 Model Architecture

The **VRNN** extends the standard RNN by including a latent random variable $z_t$ at each time step.
- **Encoder**: $q(z_t | x_{\leq t}, z_{< t})$
- **Prior**: $p(z_t | x_{< t}, z_{< t})$
- **Decoder**: $p(x_t | z_t, x_{< t}, z_{< t})$

This allows the model to handle the inherent uncertainty and complex dynamics of financial sentiment.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/your-username/kltn-vrnn/issues).

---
*Created with ❤️ for Vietnamese FinTech Research.*
