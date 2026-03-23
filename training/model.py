import torch
from torch import nn
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class VRNN(nn.Module):
    """
    Variational Recurrent Neural Network (VRNN) implementation.
    Reference: Chung et al., 2015 - 'A Recurrent Latent Variable Model for Sequential Data'
    """

    def __init__(
        self, 
        x_dim: int, 
        z_dim: int, 
        h_dim: int, 
        n_layers: int, 
        bias: bool = False
    ):
        super(VRNN, self).__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.n_layers = n_layers

        # ========================
        # Feature extractors
        # ========================
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )

        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU()
        )

        # ========================
        # Encoder q(z_t | x_t, h_{t-1})
        # ========================
        self.enc = nn.Sequential(
            nn.Linear(2 * h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        self.enc_mu = nn.Linear(h_dim, z_dim)
        self.enc_logvar = nn.Linear(h_dim, z_dim)

        # ========================
        # Prior p(z_t | h_{t-1})
        # ========================
        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        self.prior_mu = nn.Linear(h_dim, z_dim)
        self.prior_logvar = nn.Linear(h_dim, z_dim)

        # ========================
        # Decoder p(x_t | z_t, h_{t-1})
        # ========================
        self.dec = nn.Sequential(
            nn.Linear(2 * h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        self.dec_mu = nn.Linear(h_dim, x_dim)
        self.dec_logvar = nn.Linear(h_dim, x_dim)

        # ========================
        # Recurrent dynamics
        # ========================
        self.rnn = nn.GRU(2 * h_dim, h_dim, n_layers, bias=bias)

    # ======================================================
    # Utilities
    # ======================================================

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Thực hiện reparameterization trick: z = mu + eps * std
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_gaussian(
        self, 
        mu_q: torch.Tensor, 
        logvar_q: torch.Tensor, 
        mu_p: torch.Tensor, 
        logvar_p: torch.Tensor
    ) -> torch.Tensor:
        """
        Tính KL Divergence giữa hai phân phối Gaussian diagonal: KL( q || p )
        Trả về kết quả có shape (B,)
        """
        return 0.5 * torch.sum(
            logvar_p - logvar_q
            + (logvar_q.exp() + (mu_q - mu_p).pow(2)) / logvar_p.exp()
            - 1,
            dim=1
        )

    def nll_gaussian(self, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Tính Negative Log Likelihood của Gaussian phân phối: -log p(x|z)
        Trả về kết quả có shape (B,)
        """
        # Hằng số 0.5 * log(2 * pi) khoảng 0.9189
        return 0.5 * torch.sum(
            np.log(2 * np.pi) + logvar + (x - mu).pow(2) / logvar.exp(),
            dim=1
        )

    def encode(self, phi_x_t: torch.Tensor, h_last: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encoder q(z_t | x_t, h_{t-1})
        Inputs:
            phi_x_t: (B, h_dim) - Đặc trưng của x_t
            h_last: (B, h_dim) - Hidden state cuối cùng của RNN
        Returns:
            mu_q, logvar_q: (B, z_dim)
        """
        enc_input = torch.cat([phi_x_t, h_last], dim=1)
        enc_hidden = self.enc(enc_input)
        mu_q = self.enc_mu(enc_hidden)
        logvar_q = self.enc_logvar(enc_hidden)
        logvar_q = torch.clamp(logvar_q, -6, 6)
        return mu_q, logvar_q

    def decode(self, phi_z_t: torch.Tensor, h_last: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decoder p(x_t | z_t, h_{t-1})
        Inputs:
            phi_z_t: (B, h_dim) - Đặc trưng của z_t
            h_last: (B, h_dim) - Hidden state cuối cùng của RNN
        Returns:
            dec_mu, dec_logvar: (B, x_dim)
        """
        dec_input = torch.cat([phi_z_t, h_last], dim=1)
        dec_hidden = self.dec(dec_input)
        dec_mu = self.dec_mu(dec_hidden)
        dec_logvar = self.dec_logvar(dec_hidden)
        dec_logvar = torch.clamp(dec_logvar, -6, 6)
        return dec_mu, dec_logvar

    def get_prior(self, h_last: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tính Prior p(z_t | h_{t-1})
        Inputs:
            h_last: (B, h_dim) - Hidden state cuối cùng của RNN
        Returns:
            mu_p, logvar_p: (B, z_dim)
        """
        prior_hidden = self.prior(h_last)
        mu_p = self.prior_mu(prior_hidden)
        logvar_p = self.prior_logvar(prior_hidden)
        logvar_p = torch.clamp(logvar_p, -6, 6)
        return mu_p, logvar_p

    # ======================================================
    # Forward
    # ======================================================

    def forward(self, x: torch.Tensor, beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Thực hiện lan truyền tiến qua toàn bộ chuỗi thời gian
        Inputs:
            x: (T, B, x_dim) - Dữ liệu đầu vào
            beta: Hệ số cân bằng cho KL Divergence
        Returns:
            total_loss, recon_loss, kld_loss (tất cả đều được chia cho T)
        """
        T, B, _ = x.size()
        device = x.device

        # Khởi tạo hidden state: (num_layers, B, h_dim)
        h = torch.zeros(self.n_layers, B, self.h_dim, device=device)

        kld_loss = 0.0
        recon_loss = 0.0

        for t in range(T):
            x_t = x[t]  # (B, x_dim)

            # 1. Trích xuất đặc trưng x_t và lấy hidden state lớp cuối
            phi_x_t = self.phi_x(x_t)           # (B, h_dim)
            h_last = h[-1]                      # (B, h_dim)

            # 2. Tính Prior p(z_t | h_{t-1})
            mu_p, logvar_p = self.get_prior(h_last)

            # 3. Tính Encoder q(z_t | x_t, h_{t-1})
            mu_q, logvar_q = self.encode(phi_x_t, h_last)

            # 4. Lấy mẫu z_t dùng Reparameterization Trick
            z_t = self.reparameterize(mu_q, logvar_q)
            phi_z_t = self.phi_z(z_t)

            # 5. Tính Decoder p(x_t | z_t, h_{t-1})
            dec_mu, dec_logvar = self.decode(phi_z_t, h_last)

            # 6. Cập nhật Hidden State (RNN update)
            rnn_input = torch.cat([phi_x_t, phi_z_t], dim=1)
            rnn_input = rnn_input.unsqueeze(0)  # (1, B, 2*h_dim)
            _, h = self.rnn(rnn_input, h)

            # 7. Tích luỹ Loss
            kld_loss += self.kl_gaussian(mu_q, logvar_q, mu_p, logvar_p).mean()
            recon_loss += self.nll_gaussian(x_t, dec_mu, dec_logvar).mean()

        total_loss = (beta * kld_loss + recon_loss) / T

        return total_loss, recon_loss / T, kld_loss / T

    def infer_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Infer latent variables z_t for the entire sequence
        Inputs:
            x: (T, B, x_dim) - Dữ liệu đầu vào
        Returns:
            z: (T, B, z_dim) - Latent variables
            mu_q: (T, B, z_dim) - Mean of latent variables
            logvar_q: (T, B, z_dim) - Log variance of latent variables
        """
        T, B, _ = x.size()
        device = x.device

        # Khởi tạo hidden state: (num_layers, B, h_dim)
        h = torch.zeros(self.n_layers, B, self.h_dim, device=device)

        z_list = []
        mu_q_list = []
        logvar_q_list = []

        for t in range(T):
            x_t = x[t]  # (B, x_dim)

            # 1. Trích xuất đặc trưng x_t và lấy hidden state lớp cuối
            phi_x_t = self.phi_x(x_t)           # (B, h_dim)
            h_last = h[-1]                      # (B, h_dim)

            # 3. Tính Encoder q(z_t | x_t, h_{t-1})
            mu_q, logvar_q = self.encode(phi_x_t, h_last)

            # 4. Lấy mẫu z_t dùng Reparameterization Trick
            z_t = self.reparameterize(mu_q, logvar_q)
            phi_z_t = self.phi_z(z_t)

            # 5. Cập nhật Hidden State (RNN update)
            rnn_input = torch.cat([phi_x_t, phi_z_t], dim=1)
            rnn_input = rnn_input.unsqueeze(0)  # (1, B, 2*h_dim)
            _, h = self.rnn(rnn_input, h)

            z_list.append(z_t)
            mu_q_list.append(mu_q)
            logvar_q_list.append(logvar_q)

        return torch.stack(z_list), torch.stack(mu_q_list), torch.stack(logvar_q_list)


if __name__ == "__main__":
    # Test model
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Đang sử dụng thiết bị: {device}")

    # Khởi tạo model example
    batch_size = 32
    seq_len = 16
    x_dim = 1
    
    model = VRNN(x_dim=x_dim, z_dim=16, h_dim=64, n_layers=2).to(device)
    
    # Tạo tensor ngẫu nhiên: (T, B, x_dim)
    x = torch.randn(seq_len, batch_size, x_dim).to(device)
    
    loss, recon_loss, kld_loss = model(x)
    
    logger.info(f"Kết quả Test - Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, KLD: {kld_loss.item():.4f}")
