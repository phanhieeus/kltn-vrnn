import torch
from torch import nn


class VRNN(nn.Module):
    def __init__(self, x_dim, z_dim, h_dim, n_layers, bias=False):
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

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_gaussian(self, mu_q, logvar_q, mu_p, logvar_p):
        """
        KL( q || p ) cho Gaussian diagonal
        trả về (B,)
        """
        return 0.5 * torch.sum(
            logvar_p - logvar_q
            + (logvar_q.exp() + (mu_q - mu_p).pow(2)) / logvar_p.exp()
            - 1,
            dim=1
        )

    def nll_gaussian(self, x, mu, logvar):
        """
        Gaussian negative log likelihood
        trả về (B,)
        """
        return 0.5 * torch.sum(
            logvar + (x - mu).pow(2) / logvar.exp(),
            dim=1
        )

    # ======================================================
    # Forward
    # ======================================================

    def forward(self, x, beta=1.0):
        """
        x: (T, B, x_dim)
        """

        T, B, _ = x.size()
        device = x.device

        # Hidden state: (num_layers, B, h_dim)
        h = torch.zeros(self.n_layers, B, self.h_dim, device=device)

        kld_loss = 0.0
        recon_loss = 0.0

        for t in range(T):

            x_t = x[t]  # (B, x_dim)

            # ========================
            # Feature extraction
            # ========================
            phi_x_t = self.phi_x(x_t)           # (B, h_dim)
            h_last = h[-1]                      # (B, h_dim)

            # ========================
            # Encoder
            # ========================
            enc_input = torch.cat([phi_x_t, h_last], dim=1)
            enc_hidden = self.enc(enc_input)

            mu_q = self.enc_mu(enc_hidden)
            logvar_q = self.enc_logvar(enc_hidden)
            logvar_q = torch.clamp(logvar_q, -6, 6)

            # ========================
            # Prior
            # ========================
            prior_hidden = self.prior(h_last)

            mu_p = self.prior_mu(prior_hidden)
            logvar_p = self.prior_logvar(prior_hidden)
            logvar_p = torch.clamp(logvar_p, -6, 6)

            # ========================
            # Sample z_t
            # ========================
            z_t = self.reparameterize(mu_q, logvar_q)
            phi_z_t = self.phi_z(z_t)

            # ========================
            # Decoder
            # ========================
            dec_input = torch.cat([phi_z_t, h_last], dim=1)
            dec_hidden = self.dec(dec_input)

            dec_mu = self.dec_mu(dec_hidden)
            dec_logvar = self.dec_logvar(dec_hidden)
            dec_logvar = torch.clamp(dec_logvar, -6, 6)

            # ========================
            # RNN update
            # ========================
            rnn_input = torch.cat([phi_x_t, phi_z_t], dim=1)
            rnn_input = rnn_input.unsqueeze(0)  # (1, B, 2*h_dim)

            _, h = self.rnn(rnn_input, h)

            # ========================
            # Loss accumulation
            # ========================
            kld_loss += self.kl_gaussian(
                mu_q, logvar_q, mu_p, logvar_p
            ).mean()

            recon_loss += self.nll_gaussian(
                x_t, dec_mu, dec_logvar
            ).mean()

        total_loss = (beta * kld_loss + recon_loss) / T

        return total_loss, recon_loss / T, kld_loss / T

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = VRNN(x_dim=1, z_dim=16, h_dim=64, n_layers=2).to(device)
    x = torch.randn(32, 16, 1).to(device)
    loss, recon_loss, kld_loss = model(x)
    print(loss, recon_loss, kld_loss)