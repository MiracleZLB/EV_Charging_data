from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    hist_steps: int = 12
    future_steps: int = 12
    dt_minutes: int = 5

@dataclass
class ModelConfig:
    d_model: int = 64
    n_heads: int = 4
    d_attr: int = 16
    d_time: int = 16
    d_exo: int = 16
    gcn_hops: int = 2
    st_blocks: int = 3
    piecewise_bins: int = 16
    mol_components: int = 8  # Mixture-of-Logistics components for marginals
    global_temporal_layers: int = 1

@dataclass
class TrainConfig:
    batch_size: int = 16
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-5
    random_drop_past: float = 0.0  # set >0.0 
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_samples_train: int = 1     

class WindowedDataset(torch.utils.data.Dataset):
    def __init__(self, X: Tensor, M: Tensor, U: Tensor, W: Tensor, H: int, Fw: int, t_start: int, t_end: int):
        assert X.ndim == 3 and M.shape == X.shape
        assert U.shape[0] == X.shape[0] and W.shape[0] == X.shape[0]
        self.X, self.M, self.U, self.W = X, M, U, W
        self.H, self.Fw = H, Fw
        self.t_start, self.t_end = t_start, t_end
        self.indices: List[int] = []
        for t in range(t_start + H, t_end - Fw + 1):
            self.indices.append(t)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        t = self.indices[idx]
        H, Fw = self.H, self.Fw
        X_in = self.X[t - H:t]
        M_in = self.M[t - H:t]
        U_in = self.U[t - H:t]
        W_in = self.W[t - H:t]
        X_tar = self.X[t:t + Fw]
        return X_in, M_in, U_in, W_in, X_tar


def build_splits(
    X: Tensor, M: Tensor, U: Tensor, W: Tensor, cfg: DataConfig,
    train_ratio: float = 0.6, val_ratio: float = 0.2
) -> Dict[str, torch.utils.data.Dataset]:
    T = X.shape[0]
    t_train_end = int(T * train_ratio)
    t_val_end = int(T * (train_ratio + val_ratio))
    ds_train = WindowedDataset(X, M, U, W, cfg.hist_steps, cfg.future_steps, 0, t_train_end)
    ds_val   = WindowedDataset(X, M, U, W, cfg.hist_steps, cfg.future_steps, t_train_end - cfg.hist_steps, t_val_end)
    ds_test  = WindowedDataset(X, M, U, W, cfg.hist_steps, cfg.future_steps, t_val_end - cfg.hist_steps, T)
    return {"train": ds_train, "val": ds_val, "test": ds_test}


# Layers & Blocks

class MLP(nn.Module):
    def __init__(self, inp: int, out: int, hidden: int = 128, depth: int = 2):
        super().__init__()
        layers: List[nn.Module] = []
        d = inp
        for _ in range(depth - 1):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers += [nn.Linear(d, out)]
        self.net = nn.Sequential(*layers)
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

class GCN_K(nn.Module):

    def __init__(self, d: int, hops: int = 2):
        super().__init__()
        self.hops = hops
        self.lin = nn.Linear(d, d)
    def forward(self, A: Tensor, H: Tensor) -> Tensor:
        # A: [B?, n, n] or [n, n], H: [B?, n, d] or [n, d]
        # Add self loops
        I = torch.eye(A.shape[-1], device=A.device)
        A_hat = A + I
        # Symmetric normalization: D^{-1/2} A D^{-1/2}
        D_inv_sqrt = torch.pow(A_hat.sum(-1).clamp(min=1e-6), -0.5)
        if A_hat.ndim == 3:
            D_inv_sqrt = torch.diag_embed(D_inv_sqrt)
            A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
        else:
            A_norm = (D_inv_sqrt[..., None] * A_hat) * D_inv_sqrt[..., None, :]
        X = H
        for _ in range(self.hops):
            X = A_norm @ X
            X = self.lin(X)
            X = F.relu(X)
        return X

class LocalTemporalEncoder(nn.Module):

    def __init__(self, d_model: int, n_heads: int, n_layers: int = 1, dim_ff: int = 128):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff, batch_first=False)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
    def forward(self, seq: Tensor) -> Tensor:
        # seq: [T_in, d_model]
        return self.enc(seq)

class GlobalTemporalEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int = 1, dim_ff: int = 128):
        super().__init__()
        if n_layers <= 0:
            self.enc = None
        else:
            enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff, batch_first=False)
            self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
    def forward(self, seqTNAD: Tensor) -> Tensor:
        # seqTNAD: [T, N, A, d]
        if self.enc is None:
            return seqTNAD
        T, N, A, d = seqTNAD.shape
        x = seqTNAD.view(T, N * A, d)  # treat each token separately across time
        x = self.enc(x)
        return x.view(T, N, A, d)


# AMSTG Model

class AMSTG(nn.Module):
    def __init__(self, N: int, A: int, model_cfg: ModelConfig):
        super().__init__()
        self.N, self.A = N, A
        self.cfg = model_cfg

        # Trainable attribute embeddings per (i,j)
        self.c_attr = nn.Parameter(torch.randn(N, A, model_cfg.d_attr) * 0.02)

        # Token embedding MLP
        # inputs: x (1) + mask (1) + attr(d_attr) + time(d_time) + exo(d_exo)
        self.embed_mlp = MLP(1 + 1 + model_cfg.d_attr + model_cfg.d_time + model_cfg.d_exo, model_cfg.d_model, hidden=128, depth=2)

        # Temporal encoders
        self.local_temp = LocalTemporalEncoder(model_cfg.d_model, model_cfg.n_heads, n_layers=1, dim_ff=128)
        self.global_temp = GlobalTemporalEncoder(model_cfg.d_model, model_cfg.n_heads, n_layers=model_cfg.global_temporal_layers, dim_ff=128)

        # Graph components
        self.W1_attr = nn.Linear(model_cfg.d_model, model_cfg.d_model, bias=False)
        self.W2_attr = nn.Linear(model_cfg.d_model, model_cfg.d_model, bias=False)
        self.W1_spa = nn.Linear(model_cfg.d_model, model_cfg.d_model, bias=False)
        self.W2_spa = nn.Linear(model_cfg.d_model, model_cfg.d_model, bias=False)
        self.gcn_attr = GCN_K(model_cfg.d_model, hops=model_cfg.gcn_hops)
        self.gcn_spa = GCN_K(model_cfg.d_model, hops=model_cfg.gcn_hops)

        # Learnable fusion weights (logits -> softmax)
        self.attr_fuse_logits = nn.Parameter(torch.zeros(2))   # [w_dyn, w_sta]
        self.spa_fuse_logits  = nn.Parameter(torch.zeros(3))   # [w_dyn, w_sta, w_geo]

        # Token fusion post-graph
        self.fuse_mlp = MLP(model_cfg.d_model, model_cfg.d_model, hidden=128, depth=2)



        # Marginal MoL parameters from token rep
        d = model_cfg.d_model
        K = model_cfg.mol_components
        self.mol_head = MLP(d, 3 * K, hidden=128, depth=2)  # outputs [logits, loc, log_scale] per component

        # Copula: attention + piecewise constant density
        self.ctx_proj = MLP(1 + model_cfg.d_model, model_cfg.d_model, hidden=128, depth=2)  # for (z_r, h_r)
        self.q_proj   = nn.Linear(model_cfg.d_model, model_cfg.d_model)
        self.copula_bins = model_cfg.piecewise_bins
        self.copula_head = MLP(model_cfg.d_model, self.copula_bins, hidden=128, depth=2)

    # ---------------------------- Utilities ----------------------------
    @staticmethod
    def _softmax_normalize(A: Tensor) -> Tensor:
        # Row-wise softmax normalize adjacency to [0,1] and sum=1 per row
        return F.softmax(A, dim=-1)

    def _build_token_embeddings(self, X_in: Tensor, M_in: Tensor, U_in: Tensor, W_in: Tensor) -> Tensor:
        B, H, N, A = X_in.shape
        d_attr = self.cfg.d_attr
        # expand attr embeddings to batch/time
        C = self.c_attr[None, None, :, :, :]  # [1,1,N,A,d_attr]
        C = C.expand(B, H, N, A, d_attr)

        # Broadcast time/exo to stations/attrs
        U = U_in[:, :, None, None, :]  # [B,H,1,1,d_time]
        U = U.expand(B, H, N, A, self.cfg.d_time)
        W = W_in[:, :, None, None, :]  # [B,H,1,1,d_exo]
        W = W.expand(B, H, N, A, self.cfg.d_exo)

        x = X_in.unsqueeze(-1)  # [B,H,N,A,1]
        m = M_in.unsqueeze(-1)  # [B,H,N,A,1]
        feat = torch.cat([x, m, C, U, W], dim=-1)  # [B,H,N,A, 1+1+d_attr+d_time+d_exo]
        H0 = self.embed_mlp(feat)
        return H0  # [B,H,N,A,d]

    def _temporal_local(self, H0: Tensor) -> Tensor:
        B, H, N, A, d = H0.shape
        out = torch.empty_like(H0)
        for b in range(B):
            for i in range(N):
                for j in range(A):
                    seq = H0[b, :, i, j, :]  # [H,d]
                    out[b, :, i, j, :] = self.local_temp(seq)
        return out

    def _intra_attr_graph(self, H_t: Tensor) -> Tensor:
        """Intra-station over attributes at a single time t, batched over B and N.
        H_t: [B, N, A, d]
        Returns: [B, N, A, d]
        """
        B, N, A, d = H_t.shape
        # Dynamic adjacency from features per station
        H_flat = H_t.view(B * N, A, d)
        A_dyn = F.relu(self.W1_attr(H_flat) @ self.W2_attr(H_flat).transpose(1, 2))  # [B*N, A, A]
        # Static from attr embeddings
        C = self.c_attr[None, :, :, :].expand(B, N, A, self.cfg.d_attr).contiguous().view(B * N, A, self.cfg.d_attr)
        A_sta = F.relu(C @ C.transpose(1, 2))
        # Fuse
        w = F.softmax(self.attr_fuse_logits, dim=0)
        S = w[0] * A_dyn + w[1] * A_sta
        # GCN
        H_out = self.gcn_attr(S, H_flat)
        return H_out.view(B, N, A, d)

    def _inter_spatial_graph(self, H_t: Tensor, A_geo: Tensor) -> Tensor:
        """Inter-station over stations per attribute at time t.
        H_t: [B, N, A, d]  A_geo: [N, N]
        Returns: [B, N, A, d]
        """
        B, N, A, d = H_t.shape
        H_spa_out = torch.empty_like(H_t)
        for j in range(A):
            H_attrj = H_t[:, :, j, :]         # [B,N,d]
            H_flat = H_attrj
            # Dynamic N×N from features
            A_dyn = F.relu(self.W1_spa(H_flat) @ self.W2_spa(H_flat).transpose(1, 2))  # [B,N,N]
            # Static from attr embeddings per j
            Cj = self.c_attr[:, j, :]  # [N,d_attr]
            A_sta = F.relu(Cj @ Cj.t())  # [N,N]
            # Fuse weights
            w = F.softmax(self.spa_fuse_logits, dim=0)
            S = w[0] * A_dyn + w[1] * A_sta + w[2] * A_geo  # broadcasting over B
            # GCN per batch
            H_out = self.gcn_spa(S, H_flat)  # [B,N,d]
            H_spa_out[:, :, j, :] = H_out
        return H_spa_out

    def _st_block(self, H_in: Tensor, A_geo: Tensor) -> Tensor:
        # Local temporal per token
        B, H, N, A, d = H_in.shape
        H_temp = self._temporal_local(H_in)  # [B,H,N,A,d]
        # For each time step, apply graphs
        out = torch.empty_like(H_temp)
        for t in range(H):
            H_t = H_temp[:, t]  # [B,N,A,d]
            H_attr = self._intra_attr_graph(H_t)
            H_spa  = self._inter_spatial_graph(H_t, A_geo)
            fused = self.fuse_mlp(H_attr + H_spa)
            out[:, t] = fused
        return out

    # ----------------- Marginals---------------
    def mol_params(self, h: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        
        K = self.cfg.mol_components
        raw = self.mol_head(h)  # [..., 3K]
        logits, loc, log_s = torch.split(raw, K, dim=-1)
        pi = F.softmax(logits, dim=-1)
        scale = F.softplus(log_s) + 1e-3
        return pi, loc, scale

    @staticmethod
    def logistic_cdf(x: Tensor, loc: Tensor, scale: Tensor) -> Tensor:
        return torch.sigmoid((x - loc) / scale)

    @staticmethod
    def logistic_pdf(x: Tensor, loc: Tensor, scale: Tensor) -> Tensor:
        z = (x - loc) / scale
        s = torch.sigmoid(z)
        return (s * (1 - s)) / scale

    def mol_cdf_pdf(self, x: Tensor, h: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        pi, loc, scale = self.mol_params(h)
        # broadcast x to mixture shape
        xE = x.unsqueeze(-1)
        Fk = self.logistic_cdf(xE, loc, scale)
        fk = self.logistic_pdf(xE, loc, scale)
        F_mix = (pi * Fk).sum(-1)
        f_mix = (pi * fk).sum(-1) + 1e-10
        z = F_mix.clamp(1e-6, 1 - 1e-6)
        return F_mix, f_mix, z

    def mol_icdf(self, u: Tensor, h: Tensor, x_min: float = -8.0, x_max: float = 8.0, iters: int = 40) -> Tensor:
    
        lo = torch.full_like(u, x_min)
        hi = torch.full_like(u, x_max)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            F_mid, _, _ = self.mol_cdf_pdf(mid, h)
            go_left = (F_mid > u)
            hi = torch.where(go_left, mid, hi)
            lo = torch.where(go_left, lo, mid)
        return 0.5 * (lo + hi)

    # ---------------------------- Copula ----------------------------
    def copula_log_density(self, z_list: List[Tensor], h_list: List[Tensor]) -> Tensor:
        """
        z_list: list of [B] tensors in [0,1]
        h_list: list of [B,d] token reps
        Returns: [B] log density
        """
        B = z_list[0].shape[0]
        log_c = torch.zeros(B, device=z_list[0].device)
        P = self.copula_bins
        for k in range(len(z_list)):
            z_k = z_list[k].unsqueeze(-1)  # [B,1]
            if k == 0:
                # no conditioning: uniform density on [0,1]
                w = torch.full((B, P), 1.0 / P, device=z_k.device)
            else:
                # Build bank from previous variables
                bank = []
                for r in range(k):
                    z_r = z_list[r].unsqueeze(-1)  # [B,1]
                    h_r = h_list[r]                # [B,d]
                    bank.append(self.ctx_proj(torch.cat([z_r, h_r], dim=-1)))  # [B,d]
                Kprev = torch.stack(bank, dim=1)  # [B, k, d]
                q = self.q_proj(h_list[k]).unsqueeze(1)  # [B,1,d]
                att = (q * Kprev).sum(-1) / math.sqrt(Kprev.shape[-1])  # [B,k]
                alpha = F.softmax(att, dim=1)
                ctx = (alpha.unsqueeze(-1) * Kprev).sum(1)  # [B,d]
                w = F.softmax(self.copula_head(ctx), dim=-1)  # [B,P]
            # piecewise constant pdf: density = w_bin * P inside its bin
            bin_idx = torch.clamp((z_k * P).long(), max=P-1)
            w_sel = w.gather(-1, bin_idx)
            pdf = w_sel.squeeze(-1) * P + 1e-10
            log_c = log_c + torch.log(pdf)
        return log_c

    def forward(self, X_in: Tensor, M_in: Tensor, U_in: Tensor, W_in: Tensor, A_geo: Tensor) -> Tensor:
        """
        Returns H_last: [B, N, A, d]
        """
        H0 = self._build_token_embeddings(X_in, M_in, U_in, W_in)  # [B,H,N,A,d]
        H = H0
        for _ in range(self.cfg.st_blocks):
            H = H + self._st_block(H, A_geo)
        H = self.global_temp(H)
        H_last = H[:, -1]  # [B,N,A,d]
        return H_last

    # ---------------------------- Losses ----------------------------
    def joint_log_prob(self, X_tar: Tensor, H_last: Tensor) -> Tensor:
        """
        X_tar: [B, F, N, A] (normalized future values)
        H_last: [B, N, A, d]  — we condition all futures on last hidden (one-shot)
        Returns: [B] batch log-likelihood
        """
        B, Fh, N, A = X_tar.shape
        # build lists in a fixed autoregressive order 
        x_list: List[Tensor] = []
        h_list: List[Tensor] = []
        for f in range(Fh):
            for i in range(N):
                for j in range(A):
                    x_list.append(X_tar[:, f, i, j])  # [B]
                    h_list.append(H_last[:, i, j, :]) # [B,d]
        # marginals
        z_list: List[Tensor] = []
        log_marg = torch.zeros(B, device=X_tar.device)
        for x, h in zip(x_list, h_list):
            F_x, f_x, z_x = self.mol_cdf_pdf(x, h)
            z_list.append(z_x)
            log_marg = log_marg + torch.log(f_x)
        # copula density
        log_c = self.copula_log_density(z_list, h_list)
        return log_marg + log_c

    # ---------------------------- Sampling / Forecast ----------------------------
    def sample_forecast(self, H_last: Tensor, Fh: int, samples: int) -> Tensor:
        """
        Returns: [S, Fh, N, A]
        """
        B, N, A, d = H_last.shape
        # draw for B=1 at a time . If B>1, loop.
        assert B == 1
        S = samples
        out = torch.empty(S, Fh, N, A, device=H_last.device)
        # Build h_list in the same order as training
        h_list: List[Tensor] = []
        for f in range(Fh):
            for i in range(N):
                for j in range(A):
                    h_list.append(H_last[:, i, j, :].squeeze(0))  # [d] -> treat as [1,d]
        # Repeat h_list for S and wrap to [S,d]
        h_listB: List[Tensor] = [h.unsqueeze(0).repeat(S, 1) for h in h_list]

        # Sequentially sample with the copula conditionals
        z_samples: List[Tensor] = []
        for k in range(len(h_listB)):
            if k == 0:
                w = torch.full((S, self.copula_bins), 1.0 / self.copula_bins, device=H_last.device)
            else:
                bank = []
                for r in range(k):
                    z_r = z_samples[r].unsqueeze(-1)  # [S,1]
                    h_r = h_listB[r]                  # [S,d]
                    bank.append(self.ctx_proj(torch.cat([z_r, h_r], dim=-1)))
                Kprev = torch.stack(bank, dim=1)  # [S,k,d]
                q = self.q_proj(h_listB[k]).unsqueeze(1)  # [S,1,d]
                att = (q * Kprev).sum(-1) / math.sqrt(Kprev.shape[-1])  # [S,k]
                alpha = F.softmax(att, dim=1)
                ctx = (alpha.unsqueeze(-1) * Kprev).sum(1)  # [S,d]
                w = F.softmax(self.copula_head(ctx), dim=-1)  # [S,P]
            # sample bin then uniform inside bin
            P = self.copula_bins
            cat = torch.distributions.Categorical(w)
            bins = cat.sample()                       # [S]
            u_in_bin = torch.rand(S, device=w.device)
            z = (bins.float() + u_in_bin) / P         # [S]
            z_samples.append(z)
        # Invert marginals per variable
        x_samples = []
        idx = 0
        for f in range(Fh):
            for i in range(N):
                for j in range(A):
                    h = h_listB[idx]
                    u = z_samples[idx]
                    x = self.mol_icdf(u, h)  # [S]
                    x_samples.append(x)
                    idx += 1
        # Stitch back to [S,Fh,N,A]
        idx = 0
        for f in range(Fh):
            for i in range(N):
                for j in range(A):
                    out[:, f, i, j] = x_samples[idx]
                    idx += 1
        return out


# Geo utilities

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def geo_proximity(coords: Tensor, sigma_km: float = 5.0) -> Tensor:
    """RBF kernel over haversine distances. coords: [N,2] (lat,lon) degrees."""
    N = coords.shape[0]
    A = torch.zeros(N, N, device=coords.device)
    for i in range(N):
        for j in range(N):
            d = haversine(float(coords[i,0]), float(coords[i,1]), float(coords[j,0]), float(coords[j,1]))
            A[i, j] = math.exp(- (d * d) / (2 * sigma_km * sigma_km))
    # Normalize rows
    A = F.softmax(A, dim=-1)
    return A


# Training / Evaluation helpers

@torch.no_grad()
def point_from_samples(samples: Tensor) -> Tensor:
   
    return samples.mean(dim=0)


def mae_rmse(y_true: Tensor, y_pred: Tensor) -> Tuple[float, float]:
    mae = (y_true - y_pred).abs().mean().item()
    rmse = torch.sqrt(((y_true - y_pred) ** 2).mean()).item()
    return mae, rmse

@torch.no_grad()
def crps_from_samples(y_true: Tensor, samples: Tensor) -> float:
    """
      y_true: [F,N,A]
      samples: [S,F,N,A]
    """
    S = samples.shape[0]
    y = y_true.unsqueeze(0).expand_as(samples)
    term1 = (samples - y).abs().mean()
    # pairwise |X - X'|
    X1 = samples.unsqueeze(1)  # [S,1,...]
    X2 = samples.unsqueeze(0)  # [1,S,...]
    term2 = (X1 - X2).abs().mean()
    crps = (term1 - 0.5 * term2).item()
    return crps


# train / validate / test 

class AMSTGSystem:
    def __init__(self, N: int, A: int, coords: Optional[Tensor], mcfg: ModelConfig, tcfg: TrainConfig):
        self.device = tcfg.device
        self.model = AMSTG(N, A, mcfg).to(self.device)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
        self.tcfg = tcfg
        # Precompute A_geo 
        if coords is None:
            self.A_geo = torch.eye(N, device=self.device)
        else:
            self.A_geo = geo_proximity(coords.to(self.device))

    def step_batch(self, batch: Tuple[Tensor, ...]) -> Tuple[Tensor, Dict[str, float]]:
        X_in, M_in, U_in, W_in, X_tar = [x.to(self.device) for x in batch]
        H_last = self.model(X_in, M_in, U_in, W_in, self.A_geo)
        loglik = self.model.joint_log_prob(X_tar, H_last)  # [B]
        loss = -loglik.mean()
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optim.step()
        return loss.detach(), {"nll": loss.item()}

    @torch.no_grad()
    def eval_batch(self, batch: Tuple[Tensor, ...], samples: int = 50) -> Dict[str, float]:
        X_in, M_in, U_in, W_in, X_tar = [x.to(self.device) for x in batch]
        H_last = self.model(X_in, M_in, U_in, W_in, self.A_geo)
        # Forecast S scenarios 
        B, Fh, N, A = X_tar.shape
        maes, rmses, crps_list = [], [], []
        for b in range(B):
            H_b = H_last[b:b+1]
            samp = self.model.sample_forecast(H_b, Fh, samples)  # [S,F,N,A]
            mean_pred = point_from_samples(samp)
            mae, rmse = mae_rmse(X_tar[b], mean_pred)
            crps = crps_from_samples(X_tar[b], samp)
            maes.append(mae); rmses.append(rmse); crps_list.append(crps)
        return {"MAE": float(sum(maes)/len(maes)), "RMSE": float(sum(rmses)/len(rmses)), "CRPS": float(sum(crps_list)/len(crps_list))}

    def fit(self, loaders: Dict[str, torch.utils.data.DataLoader]) -> None:
        for epoch in range(1, self.tcfg.epochs + 1):
            self.model.train()
            losses = []
            for batch in loaders["train"]:
                loss, _ = self.step_batch(batch)
                losses.append(loss.item())
            self.model.eval()
            val_metrics = self.eval_epoch(loaders["val"]) if "val" in loaders else {}
            print(f"Epoch {epoch:03d} | Train NLL: {sum(losses)/max(1,len(losses)):.4f} | Val: {val_metrics}")

    @torch.no_grad()
    def eval_epoch(self, loader: torch.utils.data.DataLoader, samples: int = 50) -> Dict[str, float]:
        agg = {"MAE": [], "RMSE": [], "CRPS": []}
        for batch in loader:
            m = self.eval_batch(batch, samples=samples)
            for k in agg.keys():
                agg[k].append(m[k])
        return {k: float(sum(v)/len(v)) for k, v in agg.items() if v}
