"""
Conditional VAE for mixed-type tabular data (PyTorch).

  encoder q(z | x, c)  ->  latent z (dim 16-32)
  decoder p(x | z, c)  ->  one head per generated variable:
      continuous   : Gaussian (mean, log-variance)
      log_glucose  : K-component Gaussian mixture (K=1 is a plain Gaussian head)
      categorical  : softmax logits

Condition c = learned embedding of each condition variable (+ optional state
embedding, dim 8). Loss = reconstruction NLL + beta * KL, with KL annealing in
the training script.

Variants (T8, optional course-concept ablations):
  mlp : MLP encoder + MLP decoder (main model)
  gru : MLP encoder + GRU decoder that emits one variable per step
  cnn : 1D-CNN encoder over the input vector + MLP decoder
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

LOGVAR_MIN, LOGVAR_MAX = -7.0, 5.0
LOG_2PI = math.log(2 * math.pi)


def mlp(d_in: int, dims: List[int]) -> nn.Sequential:
    layers, d = [], d_in
    for h in dims:
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU()]
        d = h
    return nn.Sequential(*layers)


class ConditionEmbedding(nn.Module):
    def __init__(self, cards: List[int], n_states: int = 0, emb_dim: int = 4, state_dim: int = 8):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(k, min(emb_dim, k)) for k in cards])
        self.state = nn.Embedding(n_states, state_dim) if n_states else None
        self.out_dim = sum(min(emb_dim, k) for k in cards) + (state_dim if n_states else 0)

    def forward(self, cond: torch.Tensor, state: Optional[torch.Tensor] = None):
        parts = [e(cond[:, i]) for i, e in enumerate(self.embs)]
        if self.state is not None:
            parts.append(self.state(state))
        return torch.cat(parts, dim=1)


class CNNEncoder(nn.Module):
    def __init__(self, d_in: int, hidden: int, channels: int = 16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, channels, 3, padding=1), nn.ReLU(),
            nn.Conv1d(channels, channels, 3, padding=1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(channels * d_in, hidden), nn.LayerNorm(hidden), nn.ReLU())

    def forward(self, x):
        return self.fc(self.conv(x.unsqueeze(1)).flatten(1))


class CVAE(nn.Module):
    def __init__(self, cond_cards: List[int], n_cont: int, cat_cards: List[int],
                 glucose_index: Optional[int], n_states: int = 0, latent_dim: int = 32,
                 hidden_dims=(128, 64), glucose_components: int = 3, variant: str = "mlp",
                 state_dim: int = 8):
        super().__init__()
        self.hparams = dict(cond_cards=list(cond_cards), n_cont=n_cont, cat_cards=list(cat_cards),
                            glucose_index=glucose_index, n_states=n_states, latent_dim=latent_dim,
                            hidden_dims=list(hidden_dims), glucose_components=glucose_components,
                            variant=variant, state_dim=state_dim)
        self.n_cont, self.cat_cards = n_cont, list(cat_cards)
        self.gi, self.K, self.variant = glucose_index, glucose_components, variant
        self.latent_dim = latent_dim
        self.cemb = ConditionEmbedding(cond_cards, n_states, state_dim=state_dim)
        d_x = n_cont + sum(cat_cards)
        d_enc_in = d_x + self.cemb.out_dim
        h_last = hidden_dims[-1]
        if variant == "cnn":
            self.encoder = nn.Sequential(CNNEncoder(d_enc_in, hidden_dims[0]),
                                         mlp(hidden_dims[0], list(hidden_dims[1:])))
        else:
            self.encoder = mlp(d_enc_in, list(hidden_dims))
        self.mu = nn.Linear(h_last, latent_dim)
        self.logvar = nn.Linear(h_last, latent_dim)

        d_dec_in = latent_dim + self.cemb.out_dim
        n_vars = n_cont + len(cat_cards)
        if variant == "gru":
            self.dec_init = nn.Linear(d_dec_in, h_last)
            self.gru = nn.GRU(d_dec_in + n_vars, h_last, batch_first=True)
            self.n_vars = n_vars
        else:
            self.decoder = mlp(d_dec_in, list(reversed(hidden_dims)))
            h_last = list(reversed(hidden_dims))[-1]
        heads = []
        for j in range(n_cont):
            heads.append(nn.Linear(h_last, 3 * self.K if j == self.gi else 2))
        for k in cat_cards:
            heads.append(nn.Linear(h_last, k))
        self.heads = nn.ModuleList(heads)

    # ── pieces ───────────────────────────────────────────────────────────────
    def _x_in(self, cont, cat):
        parts = [cont] + [F.one_hot(cat[:, j], k).float() for j, k in enumerate(self.cat_cards)]
        return torch.cat(parts, 1)

    def encode(self, cont, cat, c):
        h = self.encoder(torch.cat([self._x_in(cont, cat), c], 1))
        return self.mu(h), self.logvar(h).clamp(LOGVAR_MIN, LOGVAR_MAX)

    def decode(self, z, c) -> List[torch.Tensor]:
        zc = torch.cat([z, c], 1)
        if self.variant == "gru":
            B = zc.shape[0]
            steps = torch.eye(self.n_vars, device=zc.device).unsqueeze(0).expand(B, -1, -1)
            inp = torch.cat([zc.unsqueeze(1).expand(-1, self.n_vars, -1), steps], 2)
            out, _ = self.gru(inp, torch.tanh(self.dec_init(zc)).unsqueeze(0).contiguous())
            return [head(out[:, j]) for j, head in enumerate(self.heads)]
        h = self.decoder(zc)
        return [head(h) for head in self.heads]

    def forward(self, cont, cat, cond, state=None):
        c = self.cemb(cond, state)
        mu, logvar = self.encode(cont, cat, c)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z, c), mu, logvar

    # ── loss ─────────────────────────────────────────────────────────────────
    def recon_nll(self, outs, cont, cat) -> torch.Tensor:
        """Per-row negative log-likelihood (summed over variables)."""
        nll = torch.zeros(cont.shape[0], device=cont.device)
        for j in range(self.n_cont):
            o, x = outs[j], cont[:, j]
            if j == self.gi:
                logits, mean, lv = o[:, :self.K], o[:, self.K:2 * self.K], o[:, 2 * self.K:].clamp(LOGVAR_MIN, LOGVAR_MAX)
                comp = -0.5 * (LOG_2PI + lv + (x.unsqueeze(1) - mean) ** 2 / lv.exp())
                nll = nll - torch.logsumexp(F.log_softmax(logits, 1) + comp, 1)
            else:
                mean, lv = o[:, 0], o[:, 1].clamp(LOGVAR_MIN, LOGVAR_MAX)
                nll = nll + 0.5 * (LOG_2PI + lv + (x - mean) ** 2 / lv.exp())
        for j in range(len(self.cat_cards)):
            nll = nll + F.cross_entropy(outs[self.n_cont + j], cat[:, j], reduction="none")
        return nll

    @staticmethod
    def kl(mu, logvar) -> torch.Tensor:
        return -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), 1)

    def loss(self, cont, cat, cond, state=None, beta: float = 1.0) -> Dict[str, torch.Tensor]:
        outs, mu, logvar = self(cont, cat, cond, state)
        rec = self.recon_nll(outs, cont, cat).mean()
        kl = self.kl(mu, logvar).mean()
        return {"loss": rec + beta * kl, "recon": rec, "kl": kl, "elbo": -(rec + kl)}

    # ── sampling ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def sample(self, cond, state=None, generator: Optional[torch.Generator] = None):
        """Fresh draws from the decoder: z ~ N(0, I), then each head is sampled.
        Returns (cont [n, n_cont] normalised, cat [n, n_cat] level index)."""
        n = cond.shape[0]
        dev = cond.device
        c = self.cemb(cond, state)
        z = torch.randn(n, self.latent_dim, generator=generator, device=dev)
        outs = self.decode(z, c)
        cont = torch.empty(n, self.n_cont, device=dev)
        for j in range(self.n_cont):
            o = outs[j]
            if j == self.gi:
                logits, mean, lv = o[:, :self.K], o[:, self.K:2 * self.K], o[:, 2 * self.K:].clamp(LOGVAR_MIN, LOGVAR_MAX)
                k = torch.multinomial(F.softmax(logits, 1), 1, generator=generator).squeeze(1)
                m, s = mean.gather(1, k[:, None]).squeeze(1), (0.5 * lv.gather(1, k[:, None]).squeeze(1)).exp()
            else:
                m, s = o[:, 0], (0.5 * o[:, 1].clamp(LOGVAR_MIN, LOGVAR_MAX)).exp()
            cont[:, j] = m + s * torch.randn(n, generator=generator, device=dev)
        cats = [torch.multinomial(F.softmax(outs[self.n_cont + j], 1), 1, generator=generator).squeeze(1)
                for j in range(len(self.cat_cards))]
        cat = torch.stack(cats, 1) if cats else torch.zeros(n, 0, dtype=torch.long, device=dev)
        return cont, cat


def build_model(spec, model_cfg: dict, variant: str = "mlp", glucose_head: str = "mixture") -> CVAE:
    K = int(model_cfg.get("glucose_mixture_components", 3)) if glucose_head == "mixture" else 1
    gi = spec.cont_cols.index("log_glucose") if "log_glucose" in spec.cont_cols else None
    return CVAE(cond_cards=[len(spec.cond_levels[c]) for c in spec.cond_names],
                n_cont=len(spec.cont_cols),
                cat_cards=[len(spec.cat_levels[c]) for c in spec.cat_cols],
                glucose_index=gi,
                n_states=spec.n_states if spec.use_state else 0,
                latent_dim=int(model_cfg.get("latent_dim", 32)),
                hidden_dims=tuple(model_cfg.get("hidden_dims", [128, 64])),
                glucose_components=K, variant=variant,
                state_dim=int(model_cfg.get("state_embedding_dim", 8)))
