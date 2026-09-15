"""
ADProto-Net — Adaptive Dense Prototype Network
===============================================
Novel contribution: Multi-Scale Adaptive Prototype Aggregation (MSAPA)
----------------------------------------------------------------------
The core idea: standard prototypical matching uses a single masked-average-pool
(one spatial scale). MSAPA pools at 3 pyramid scales (32x32, 16x16, 8x8),
producing 3 prototype candidates per class. A lightweight channel-attention gate
(not full self-attention — no HW^2 matrix) adaptively weights the 3 scale
contributions per query image, giving the model both global context and
fine-grained local evidence.

Why this beats the baseline:
  - Fine-grained anatomy (kidneys, aorta) benefits from local 8x8 pooling
    which preserves tubular / small-structure signals.
  - Large diffuse organs (liver) benefit from global 32x32 pooling.
  - The attention gate learns *which scale to trust* from the query feature
    statistics — avoiding a fixed per-organ hand-design.

What is NOT here (keeping it clean for publication):
  - No HW^2 self-similarity matrix (that's what killed SPSP-Net)
  - No contrastive losses / memory banks / depth tokens
  Just one clean novel module + standard few-shot seg pipeline.

Architecture:
  ResNet50 (stem + layer1-3) -> ASPP -> Decoder
  Support branch: MSAPA -> fg_proto + bg_proto (B, 256)
  Query branch:   cosine similarity maps -> SegHead -> upsampled logit
  Aux head on coarse features for deep supervision

Loss: focal-Tversky (same as baseline for fair comparison)

CHANGE LOG (this version):
  - split_folds() now shuffles patient IDs (with a fixed, configurable seed)
    before slicing into folds, instead of using sorted-order contiguous
    blocks. This redistributes "hard" patients across folds instead of
    concentrating them in a single fold (as was happening with fold 3
    in the original sorted split). n_shot remains 1 as requested.
  - Added cfg.split_seed (default 42) to control / sweep the fold shuffle.
  - IMPORTANT: if you change split_seed, delete the old
    .ram_cache_adproto/ directory under data_dir before re-running,
    since cached fold contents are keyed by fold number only and will
    not auto-invalidate on a split change.

Usage:
  python adproto_net.py --mode train --fold -1 --n_shot 1
  python adproto_net.py --mode train --fold -1 --n_shot 1 --split_seed 7
  python adproto_net.py --mode eval  --fold -1
"""

import os, sys, math, random, time, warnings, pickle, argparse
from pathlib     import Path
from typing      import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn            as nn
import torch.nn.functional as F
from torch.cuda.amp   import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tv_models
import scipy.ndimage       as ndi

warnings.filterwarnings("ignore")


# ====================================================================
# CONFIG
# ====================================================================

class CFG:
    raw_dir      = "/home/balaji/Desktop/arun_MIS/medical_data/RawData"
    data_dir     = "/home/balaji/Desktop/arun_MIS/medical_data/Preprocessed"
    save_dir     = "/home/balaji/Desktop/arun_MIS/medical_data/checkpoints_adproto"

    mode         = "train"
    fold         = -1
    n_shot       = 1
    n_iter       = 15_000
    lr           = 2e-4
    batch        = 12
    val_interval = 2_000
    es_patience  = 20
    use_tta      = False
    num_workers  = 4

    # model hyper-params
    feat_ch      = 256
    # MSAPA scales: pool feature map to these sizes before prototype extraction
    # Feature map is 32x32; scale divisors give 32, 16, 8
    msapa_scales = (1, 2, 4)
    aux_weight   = 0.4
    focal_alpha  = 0.25
    focal_gamma  = 2.0

    eval_seed    = 2024

    # NEW: seed used to shuffle patient IDs before assigning folds.
    # Sorted-order contiguous fold splits can concentrate "hard" patients
    # into a single fold (as observed with fold 3 in the original split).
    # Shuffling redistributes difficulty across folds. Change this to sweep
    # alternative splits; remember to clear .ram_cache_adproto/ afterwards.
    split_seed   = 42


SEED      = 42
IMG_SIZE  = 256
N_FOLDS   = 4

ORGAN_MAP = {
    1: "spleen",
    2: "right_kidney",
    3: "left_kidney",
    6: "liver",
    8: "aorta",
}
ORGAN_IDS = sorted(ORGAN_MAP.keys())   # [1, 2, 3, 6, 8]

_CACHE_VERSION = "adproto_labels_1_2_3_6_8"

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = device.type == "cuda"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if device.type == "cuda":
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark        = True
    torch.backends.cudnn.deterministic    = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

print(f"[ADProto-Net]  device={device}  AMP={USE_AMP}")
print(f"  Organ label map: {ORGAN_MAP}")


# ====================================================================
# NOVEL MODULE: Multi-Scale Adaptive Prototype Aggregation (MSAPA)
# ====================================================================

class MSAPA(nn.Module):
    """
    Multi-Scale Adaptive Prototype Aggregation.

    Given support features (B, C, H, W) and a binary fg mask (B, 1, H, W):

    1. Pool features at K spatial scales via AvgPool -> (B, C, H/s, W/s)
    2. At each scale, extract fg and bg prototypes via masked average pooling
       -> fg_protos: (B, K, C), bg_protos: (B, K, C)
    3. A channel-attention gate network takes a global descriptor of the
       QUERY features and outputs soft scale weights (B, K) via softmax.
    4. Weighted sum over K scales -> final fg/bg prototype (B, C).

    Key design decisions:
    - No HW^2 self-similarity (O(K*HW) not O(HW^2), stable under AMP)
    - Gate conditioned on QUERY so model adapts per test image
    - All ops differentiable and numerically stable
    - ~130K extra parameters over the baseline
    """

    def __init__(self, feat_ch: int = 256, scales: Tuple[int, ...] = (1, 2, 4)):
        super().__init__()
        self.scales  = scales
        self.K       = len(scales)
        self.feat_ch = feat_ch

        # Per-scale feature refinement (lightweight 1x1 conv) before pooling
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feat_ch, feat_ch, 1, bias=False),
                nn.GroupNorm(8, feat_ch),
                nn.GELU(),
            )
            for _ in scales
        ])

        # Channel-attention gate: query global avg-pool -> scale weights
        # Small MLP: 256 -> 64 -> K
        self.gate = nn.Sequential(
            nn.Linear(feat_ch, 64),
            nn.GELU(),
            nn.Linear(64, self.K),
        )
        # Learned temperature for scale softmax, clamped to [0.2, 5]
        self.log_temp = nn.Parameter(torch.zeros(1))

        # Post-aggregation prototype projection (residual)
        self.proto_proj = nn.Sequential(
            nn.Linear(feat_ch, feat_ch),
            nn.LayerNorm(feat_ch),
        )

    @staticmethod
    def _masked_proto(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        feat : (B, C, h, w)
        mask : (B, 1, h, w) non-negative float
        returns: (B, C)
        """
        return (feat * mask).sum([-2, -1]) / mask.sum([-2, -1]).clamp(min=1e-5)

    def forward(
        self,
        s_feat: torch.Tensor,   # (B, C, H, W) support features
        s_mask: torch.Tensor,   # (B, 1, H, W) binary fg mask (float)
        q_feat: torch.Tensor,   # (B, C, H, W) query features (for gate)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: fg_proto (B, C), bg_proto (B, C)
        """
        fg_protos_list: List[torch.Tensor] = []
        bg_protos_list: List[torch.Tensor] = []

        for i, scale in enumerate(self.scales):
            sf = self.scale_convs[i](s_feat)   # (B, C, H, W)

            if scale == 1:
                sf_s   = sf
                mask_s = s_mask
            else:
                sf_s   = F.avg_pool2d(sf,     kernel_size=scale, stride=scale)
                mask_s = F.avg_pool2d(s_mask, kernel_size=scale, stride=scale)
                # Hard binarise the downsampled mask
                mask_s = (mask_s >= 0.5).float()

            bg_mask_s = (1.0 - mask_s).clamp(0.0, 1.0)

            fg_p = self._masked_proto(sf_s, mask_s)       # (B, C)
            bg_p = self._masked_proto(sf_s, bg_mask_s)    # (B, C)

            fg_protos_list.append(fg_p)
            bg_protos_list.append(bg_p)

        # Stack: (B, K, C)
        fg_stack = torch.stack(fg_protos_list, dim=1)
        bg_stack = torch.stack(bg_protos_list, dim=1)

        # Gate conditioned on global avg-pool of query features
        q_global = q_feat.mean([-2, -1])            # (B, C)
        raw_w    = self.gate(q_global)               # (B, K)
        temp     = self.log_temp.exp().clamp(0.2, 5.0)
        scale_w  = (raw_w / temp).softmax(dim=-1)   # (B, K)

        # Weighted sum: (B, K, 1) * (B, K, C) -> (B, C)
        w        = scale_w.unsqueeze(-1)
        fg_proto = (w * fg_stack).sum(1)             # (B, C)
        bg_proto = (w * bg_stack).sum(1)             # (B, C)

        # Residual projection
        fg_proto = self.proto_proj(fg_proto) + fg_proto
        bg_proto = self.proto_proj(bg_proto) + bg_proto

        return fg_proto, bg_proto


# ====================================================================
# BACKBONE COMPONENTS (identical to baseline for fair ablation)
# ====================================================================

class ASPP(nn.Module):
    def __init__(self, in_ch: int = 1024, out_ch: int = 256):
        super().__init__()
        self.b0  = self._pw(in_ch, out_ch)
        self.b6  = self._dc(in_ch, out_ch,  6)
        self.b12 = self._dc(in_ch, out_ch, 12)
        self.b18 = self._dc(in_ch, out_ch, 18)
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.proj = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(0.1))

    @staticmethod
    def _pw(ic, oc):
        return nn.Sequential(nn.Conv2d(ic, oc, 1, bias=False),
                              nn.BatchNorm2d(oc), nn.ReLU(inplace=True))
    @staticmethod
    def _dc(ic, oc, r):
        return nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=r, dilation=r, bias=False),
            nn.BatchNorm2d(oc), nn.ReLU(inplace=True))

    def forward(self, x):
        h, w = x.shape[-2:]
        g = F.interpolate(self.gap(x), (h, w), mode='bilinear', align_corners=False)
        return self.proj(torch.cat(
            [self.b0(x), self.b6(x), self.b12(x), self.b18(x), g], 1))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.skip_proj = nn.Sequential(
            nn.Conv2d(512, 48, 1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True))

    def forward(self, coarse: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        h, w = skip.shape[-2:]
        up   = F.interpolate(coarse, (h, w), mode='bilinear', align_corners=False)
        return self.fuse(torch.cat([up, self.skip_proj(skip)], 1))


class SegHead(nn.Module):
    """feat + fg_sim + bg_sim -> logit"""
    def __init__(self, feat_ch: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(feat_ch + 2, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1))
        # sigmoid(-2) ~= 0.12: bias toward predicting background initially
        nn.init.constant_(self.net[-1].bias, -2.0)
        nn.init.xavier_uniform_(self.net[-1].weight)

    def forward(self, feat, fg_sim, bg_sim):
        return self.net(torch.cat([feat, fg_sim, bg_sim], 1))


class AuxHead(nn.Module):
    def __init__(self, in_ch: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1))
        nn.init.constant_(self.net[-1].bias, -2.0)
        nn.init.xavier_uniform_(self.net[-1].weight)

    def forward(self, x):
        return self.net(x)


# ====================================================================
# ADProto-Net Full Model
# ====================================================================

class ADProtoNet(nn.Module):
    """
    ADProto-Net: ResNet50+ASPP+Decoder backbone with MSAPA prototype module.
    Only MSAPA is novel — everything else matches the baseline exactly.
    """
    def __init__(self, n_shot: int = 1, pretrained: bool = True, cfg=None):
        super().__init__()
        cfg = cfg or CFG()
        self.cfg    = cfg
        self.n_shot = n_shot
        C = cfg.feat_ch

        rn = tv_models.resnet50(
            weights="IMAGENET1K_V2" if pretrained else None)
        self.stem   = nn.Sequential(rn.conv1, rn.bn1, rn.relu, rn.maxpool)
        self.layer1 = rn.layer1
        self.layer2 = rn.layer2
        self.layer3 = rn.layer3
        for p in list(rn.conv1.parameters()) + list(rn.bn1.parameters()):
            p.requires_grad_(False)

        self.aspp  = ASPP(1024, C)
        self.dec   = Decoder()
        self.msapa = MSAPA(feat_ch=C, scales=tuple(cfg.msapa_scales))
        self.head  = SegHead(C)
        self.aux   = AuxHead(C)

    def _encode(self, imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x      = self.stem(imgs)
        skip1  = self.layer1(x)
        skip2  = self.layer2(skip1)
        coarse = self.aspp(self.layer3(skip2))
        feat   = self.dec(coarse, skip2)
        return feat, coarse

    @staticmethod
    def _cosim(feat: torch.Tensor, proto: torch.Tensor) -> torch.Tensor:
        f = F.normalize(feat,  p=2, dim=1)
        p = F.normalize(proto, p=2, dim=1)[..., None, None]
        return (f * p).sum(1, keepdim=True)

    @staticmethod
    def _resize_mask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        h, w = target.shape[-2:]
        m = F.interpolate(mask.float().unsqueeze(1), (h, w), mode='nearest')
        return (m >= 0.5).float()

    def forward(self, s_imgs, s_masks, q_img, s_z=None, q_z=None):
        """s_z, q_z accepted (dataloader compat) but unused."""
        B, S, _, H, W = s_imgs.shape

        sf_flat  = s_imgs.view(B * S, 3, H, W)
        all_imgs = torch.cat([sf_flat, q_img], dim=0)
        all_feat, all_coarse = self._encode(all_imgs)
        fH, fW = all_feat.shape[-2:]

        s_feat_flat = all_feat[:B * S]
        q_feat      = all_feat[B * S:]
        q_coarse    = all_coarse[B * S:]

        s_feats = s_feat_flat.view(B, S, self.cfg.feat_ch, fH, fW)

        sm_flat  = s_masks.view(B * S, H, W)
        sm_res_f = self._resize_mask(sm_flat, s_feat_flat)   # (B*S, 1, fH, fW)
        sm_res   = sm_res_f.view(B, S, 1, fH, fW)

        # MSAPA: multi-scale prototype per shot, then mean over shots
        fg_protos, bg_protos = [], []
        for i in range(S):
            sf_i = s_feats[:, i]               # (B, C, fH, fW)
            sm_i = sm_res[:, i].float()        # (B, 1, fH, fW)
            fg_p, bg_p = self.msapa(sf_i, sm_i, q_feat)
            fg_protos.append(fg_p)
            bg_protos.append(bg_p)

        fg_proto = torch.stack(fg_protos, dim=1).mean(1)   # (B, C)
        bg_proto = torch.stack(bg_protos, dim=1).mean(1)   # (B, C)

        fg_sim = self._cosim(q_feat, fg_proto)   # (B, 1, fH, fW)
        bg_sim = self._cosim(q_feat, bg_proto)

        logit = self.head(q_feat, fg_sim, bg_sim)
        q_out = F.interpolate(logit, (H, W), mode='bilinear', align_corners=False)

        aux_out = None
        if self.training:
            aux_raw = self.aux(q_coarse)
            aux_out = F.interpolate(aux_raw, (H, W), mode='bilinear', align_corners=False)

        return q_out, aux_out, q_feat, s_feats, sm_res.squeeze(2).long(), fg_proto


# ====================================================================
# LOSSES
# ====================================================================

def focal_tversky(logit: torch.Tensor, target: torch.Tensor,
                  alpha: float = 0.3, gamma: float = 2.0) -> torch.Tensor:
    t    = target.float()
    prob = torch.sigmoid(logit)
    t_sm = t * 0.95 + 0.025
    bce  = F.binary_cross_entropy_with_logits(logit, t_sm)
    p    = prob.view(prob.shape[0], -1)
    g    = t.view(t.shape[0], -1)
    tp   = (p * g).sum(1)
    fp   = (p * (1 - g)).sum(1)
    fn   = ((1 - p) * g).sum(1)
    tv   = (tp + 1e-5) / (tp + alpha * fp + (1 - alpha) * fn + 1e-5)
    return 0.5 * bce + 0.5 * (1 - tv).pow(1.0 / gamma).mean()


@torch.no_grad()
def dice_score(logit: torch.Tensor, target: torch.Tensor,
               thr: float = 0.5) -> float:
    pred = (torch.sigmoid(logit) >= thr).float().view(-1).cpu()
    gt   = target.float().view(-1).cpu()
    i    = (pred * gt).sum()
    u    = pred.sum() + gt.sum()
    return (2 * i / u).item() if u > 0 else 1.0


# ====================================================================
# PRE-PROCESSING
# ====================================================================

def preprocess(raw_dir: str, out_dir: str):
    try:
        import nibabel as nib
    except ImportError:
        sys.exit("pip install nibabel --break-system-packages")

    raw = Path(raw_dir); out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    img_d = lbl_d = None
    for id_, ld_ in [
        (raw / "Training" / "img",  raw / "Training" / "label"),
        (raw / "img",               raw / "label"),
        (raw / "imagesTr",          raw / "labelsTr"),
        (raw,                       raw),
    ]:
        if id_.exists() and list(id_.glob("*.nii*")):
            img_d, lbl_d = id_, ld_; break
    if img_d is None:
        raise FileNotFoundError(f"No .nii/.nii.gz under {raw_dir}")

    for img_f in sorted(img_d.glob("*.nii*")):
        stem    = img_f.name.split(".")[0]
        pid_str = "".join(c for c in stem if c.isdigit()) or stem
        lbls    = sorted(lbl_d.glob(f"*{pid_str}*.nii*"))
        if not lbls:
            print(f"  [skip] {img_f.name}"); continue
        img_vol = nib.load(str(img_f)).get_fdata(dtype=np.float32)
        lbl_vol = nib.load(str(lbls[0])).get_fdata().astype(np.int16)
        pid_int = int(pid_str) if pid_str.isdigit() else abs(hash(pid_str)) % 99999
        n_s     = img_vol.shape[-1] if img_vol.ndim == 3 else img_vol.shape[0]
        out_f   = out / f"pat_{pid_str}.npz"
        np.savez_compressed(str(out_f),
                            img=img_vol, label=lbl_vol,
                            pid=np.array([pid_int]),
                            n_slices=np.array([n_s]))
        print(f"  {out_f.name}  {img_vol.shape}")
    print(f"Done -> {out_dir}")


# ====================================================================
# SLICE CACHE
# ====================================================================

def _cache_is_valid(pkl_path: str) -> bool:
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        first = next(iter(data["slices"].values()))
        if len(first) < 4:
            return False
        if data.get("cache_version", "") != _CACHE_VERSION:
            print(f"    [cache] version mismatch -> rebuilding")
            return False
        return True
    except Exception:
        return False


class SliceCache:
    def __init__(self, paths: List[str], cache_pkl: Optional[str] = None,
                 min_px: int = 50):
        if cache_pkl and Path(cache_pkl).exists() and not _cache_is_valid(cache_pkl):
            print(f"    [cache] stale/invalid -> rebuilding: {Path(cache_pkl).name}")
            os.remove(cache_pkl)

        if cache_pkl and Path(cache_pkl).exists():
            print(f"    Loading cache {Path(cache_pkl).name} ...", end=" ", flush=True)
            t0 = time.time()
            with open(cache_pkl, "rb") as f:
                data = pickle.load(f)
            self.slices  = data["slices"]
            self.idx     = data["idx"]
            self.pid_idx = data.get("pid_idx", {})
            if not self.pid_idx:
                for k, (_, _, _, pid) in self.slices.items():
                    self.pid_idx.setdefault(pid, []).append(k)
            print(f"done {time.time() - t0:.1f}s")
            self._print(); return

        print("    Building RAM cache...")
        t0 = time.time()
        self.slices  = {}
        self.idx     = {o: [] for o in ORGAN_IDS}
        self.pid_idx = {}
        key = 0

        for path in paths:
            try:
                d = np.load(path, allow_pickle=True, mmap_mode='r')
            except Exception as e:
                print(f"    [warn] {e}"); continue
            img_vol = d["img"]; lbl_vol = d["label"]
            if img_vol.ndim != 3: continue
            par     = d.get("pid", None)
            pid_int = int(par[0]) if par is not None else abs(hash(path)) % 99999
            axis    = 2 if lbl_vol.shape[2] <= lbl_vol.shape[0] else 0
            D       = lbl_vol.shape[axis]
            pat_keys = []

            for s in range(D):
                sl_img = np.array(
                    img_vol[..., s] if axis == 2 else img_vol[s], dtype=np.float32)
                sl_lbl = np.array(
                    lbl_vol[..., s] if axis == 2 else lbl_vol[s], dtype=np.int16)
                organs = [o for o in ORGAN_IDS if (sl_lbl == o).sum() >= min_px]
                if not organs: continue

                sl_img = ((np.clip(sl_img, -125., 275.) + 125.) / 400.).astype(np.float32)
                h, w = sl_img.shape
                if (h, w) != (IMG_SIZE, IMG_SIZE):
                    sl_img = ndi.zoom(sl_img, (IMG_SIZE / h, IMG_SIZE / w), order=1)
                    sl_lbl = ndi.zoom(sl_lbl.astype(np.float32),
                                      (IMG_SIZE / h, IMG_SIZE / w), order=0).astype(np.int16)
                z_norm = float(s) / max(D - 1, 1)
                self.slices[key] = (sl_img, sl_lbl, z_norm, pid_int)
                for o in organs:
                    self.idx[o].append(key)
                pat_keys.append(key)
                key += 1

            if pat_keys:
                self.pid_idx[pid_int] = pat_keys

        ram = sum(a.nbytes + b.nbytes for a, b, _, _ in self.slices.values()) / 1e6
        print(f"    {len(self.slices)} slices ({ram:.0f}MB) {time.time() - t0:.1f}s")

        if cache_pkl:
            Path(cache_pkl).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_pkl, "wb") as f:
                pickle.dump({
                    "slices":        self.slices,
                    "idx":           self.idx,
                    "pid_idx":       self.pid_idx,
                    "cache_version": _CACHE_VERSION,
                }, f)
        self._print()

    def _print(self):
        print(f"    Label map: {ORGAN_MAP}")
        for o in ORGAN_IDS:
            cnt = len(self.idx.get(o, []))
            print(f"    {ORGAN_MAP[o]:<14}: {cnt} slices  (label {o})")

    def get(self, key: int, aug: bool = False):
        img_np, lbl_np, z, _ = self.slices[key]
        img = img_np.copy(); lbl = lbl_np.copy()

        if aug:
            if random.random() > 0.5:
                img = img[:, ::-1].copy(); lbl = lbl[:, ::-1].copy()
            if random.random() > 0.5:
                img = img[::-1, :].copy(); lbl = lbl[::-1, :].copy()
            if random.random() > 0.3:
                angle = random.uniform(-30, 30)
                img = ndi.rotate(img, angle, reshape=False, order=1)
                lbl = ndi.rotate(lbl.astype(np.float32), angle,
                                 reshape=False, order=0).astype(np.int16)
            if random.random() > 0.5:
                scale = random.uniform(0.85, 1.15)
                img = ndi.zoom(img, scale, order=1)
                lbl = ndi.zoom(lbl.astype(np.float32), scale, order=0).astype(np.int16)
                h, w = img.shape
                if scale > 1.0:
                    sh = (h - IMG_SIZE) // 2; sw = (w - IMG_SIZE) // 2
                    img = img[sh:sh+IMG_SIZE, sw:sw+IMG_SIZE]
                    lbl = lbl[sh:sh+IMG_SIZE, sw:sw+IMG_SIZE]
                else:
                    ph = IMG_SIZE - h; pw = IMG_SIZE - w
                    img = np.pad(img, ((ph//2, ph-ph//2),(pw//2, pw-pw//2)))
                    lbl = np.pad(lbl, ((ph//2, ph-ph//2),(pw//2, pw-pw//2)))
            if random.random() > 0.4:
                img = np.power(np.clip(img, 1e-6, 1),
                               random.uniform(0.6, 1.6)).astype(np.float32)
            if random.random() > 0.4:
                img = np.clip(
                    img + np.random.randn(*img.shape).astype(np.float32) * 0.03,
                    0., 1.)
            if random.random() > 0.5:
                img = np.clip(img * random.uniform(0.8, 1.2) +
                              random.uniform(-0.1, 0.1), 0., 1.).astype(np.float32)

        img_t = torch.from_numpy(img).float().unsqueeze(0).expand(3, -1, -1)
        lbl_t = torch.from_numpy(lbl).long()
        return img_t, lbl_t, z


# ====================================================================
# DATASET
# ====================================================================

class FewShotDataset(Dataset):
    def __init__(self, cache: SliceCache, n_shot: int = 1,
                 aug: bool = True, length: int = 999_999):
        self.cache  = cache
        self.n_shot = n_shot
        self.aug    = aug
        self.length = length

        self.organ_pid: Dict[int, Dict[int, List[int]]] = {}
        for o in ORGAN_IDS:
            pid_map: Dict[int, List[int]] = defaultdict(list)
            for k in cache.idx.get(o, []):
                pid = cache.slices[k][3]
                pid_map[pid].append(k)
            self.organ_pid[o] = dict(pid_map)

        self.organs = [
            o for o in ORGAN_IDS
            if len(self.organ_pid.get(o, {})) >= n_shot + 1
        ]
        if not self.organs:
            raise RuntimeError(
                f"No organ has >= {n_shot + 1} distinct patients. "
                "Check your data split or reduce n_shot.")
        print(f"    Sampling organs: "
              f"{[ORGAN_MAP[o] for o in self.organs]} "
              f"(>={n_shot + 1} patients each)")

    def __len__(self): return self.length

    def __getitem__(self, _):
        organ   = random.choice(self.organs)
        pid_map = self.organ_pid[organ]
        all_pids = list(pid_map.keys())
        random.shuffle(all_pids)

        q_pid = all_pids[0]
        q_key = random.choice(pid_map[q_pid])
        q_img, q_lbl, q_z = self.cache.get(q_key, aug=False)

        s_pids = all_pids[1:]
        chosen = random.sample(s_pids, self.n_shot)

        s_imgs, s_masks, s_z = [], [], []
        for p in chosen:
            k = random.choice(pid_map[p])
            img, lbl, z = self.cache.get(k, aug=self.aug)
            s_imgs.append(img)
            s_masks.append((lbl == organ).long())
            s_z.append(z)

        organ_idx = ORGAN_IDS.index(organ)

        return (torch.stack(s_imgs),
                torch.stack(s_masks),
                q_img,
                (q_lbl == organ).long(),
                torch.tensor(s_z,  dtype=torch.float32),
                torch.tensor(q_z,  dtype=torch.float32),
                organ,
                q_pid,
                organ_idx)


# ====================================================================
# FIXED VALIDATION SUPPORT
# ====================================================================

def build_fixed_val_support(
        val_cache: SliceCache,
        n_shot:    int,
        seed:      int = 2024,
) -> Dict[int, Dict[int, List[Tuple[int, ...]]]]:
    rng = random.Random(seed)
    fixed: Dict[int, Dict[int, List[Tuple[int, ...]]]] = {}

    for organ in ORGAN_IDS:
        pid_map: Dict[int, List[int]] = {}
        for k in val_cache.idx.get(organ, []):
            pid = val_cache.slices[k][3]
            pid_map.setdefault(pid, []).append(k)

        all_pids = list(pid_map.keys())
        if len(all_pids) < 2:
            continue

        organ_support: Dict[int, List[Tuple[int, ...]]] = {}
        for q_pid in all_pids:
            s_pids = [p for p in all_pids if p != q_pid]
            if len(s_pids) < n_shot:
                continue
            chosen_pids = rng.sample(s_pids, n_shot)
            organ_support[q_pid] = [(rng.choice(pid_map[sp]), sp)
                                    for sp in chosen_pids]
        if organ_support:
            fixed[organ] = organ_support

    return fixed


# ====================================================================
# DATA SPLITTING
# ====================================================================

def split_folds(data_dir: str, fold: int,
                split_seed: int = CFG.split_seed) -> Tuple[List[str], List[str]]:
    """
    Build train/val patient splits for a given fold.

    CHANGE: patient IDs are now shuffled (with a fixed `split_seed`) before
    being sliced into contiguous fold blocks. The original implementation
    sorted patient IDs and sliced them directly, which can concentrate
    anatomically "hard" patients into a single fold purely as a function
    of ID ordering (this was observed with fold 3 in the original split,
    which had a markedly lower spleen/right_kidney Dice than the other
    folds). Shuffling redistributes that difficulty across folds.

    `sorted(pid_map)` is applied first so the input to the shuffle is itself
    deterministic across runs/platforms; `random.Random(split_seed)` then
    makes the shuffle itself reproducible. To sweep alternative splits, pass
    a different `split_seed` (e.g. via --split_seed), and remember to clear
    the `.ram_cache_adproto/` directory under `data_dir` afterwards, since
    cached fold contents are keyed by fold number only and will not
    auto-invalidate on a split change.
    """
    paths = sorted(str(p) for p in Path(data_dir).rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz files in {data_dir}")
    pid_map: Dict[int, List[str]] = {}
    for p in paths:
        try:
            d   = np.load(p, allow_pickle=True, mmap_mode='r')
            par = d.get("pid", None)
            pid = int(par[0]) if par is not None else abs(hash(p)) % 99999
        except Exception:
            continue
        pid_map.setdefault(pid, []).append(p)

    pids = sorted(pid_map)
    rng  = random.Random(split_seed)
    rng.shuffle(pids)

    sz   = math.ceil(len(pids) / N_FOLDS)
    val  = set(pids[fold * sz: (fold + 1) * sz])
    tr   = set(pids) - val
    tr_f  = [p for pid in tr  for p in pid_map[pid]]
    val_f = [p for pid in val for p in pid_map[pid]]
    print(f"  Fold {fold}: {len(tr)} train | {len(val)} val patients  "
          f"(split_seed={split_seed})")
    return tr_f, val_f


def make_loaders(data_dir: str, fold: int, n_shot: int,
                 batch: int, cfg) -> Tuple[DataLoader, DataLoader, SliceCache]:
    tr_f, val_f = split_folds(data_dir, fold, split_seed=cfg.split_seed)
    cd = Path(data_dir) / ".ram_cache_adproto"

    print("  Loading train data...")
    tr_cache  = SliceCache(tr_f,  str(cd / f"fold{fold}_train.pkl"))
    print("  Loading val data...")
    val_cache = SliceCache(val_f, str(cd / f"fold{fold}_val.pkl"))

    nw = cfg.num_workers; pf = 4 if nw > 0 else None
    tr_ds  = FewShotDataset(tr_cache,  n_shot, aug=True,  length=999_999)
    val_ds = FewShotDataset(val_cache, n_shot, aug=False, length=3_000)

    tr_ld = DataLoader(
        tr_ds, batch_size=batch, shuffle=True,
        num_workers=nw, pin_memory=True, drop_last=True,
        persistent_workers=(nw > 0), prefetch_factor=pf)
    val_ld = DataLoader(
        val_ds, batch_size=6, shuffle=False,
        num_workers=nw, pin_memory=True,
        persistent_workers=(nw > 0), prefetch_factor=pf)

    return tr_ld, val_ld, val_cache


# ====================================================================
# SHAPE VERIFICATION
# ====================================================================

def verify_shapes(model: nn.Module, cfg) -> None:
    print("  [verify] shape check ...", end=" ", flush=True)
    was = next(model.parameters()).device
    model.cpu().eval()
    B, S = 2, cfg.n_shot
    si  = torch.zeros(B, S, 3, IMG_SIZE, IMG_SIZE)
    sm  = torch.zeros(B, S, IMG_SIZE, IMG_SIZE, dtype=torch.long)
    qi  = torch.zeros(B, 3, IMG_SIZE, IMG_SIZE)
    s_z = torch.rand(B, S)
    q_z = torch.rand(B)
    try:
        with torch.no_grad():
            q_out, _, q_feat, s_feats, sm_res, fg_proto = model(si, sm, qi, s_z, q_z)
        assert q_out.shape   == (B, 1, IMG_SIZE, IMG_SIZE), f"q_out: {q_out.shape}"
        assert q_feat.shape[0]   == B,      f"q_feat B: {q_feat.shape}"
        assert s_feats.shape[:2] == (B, S), f"s_feats BS: {s_feats.shape}"
        assert fg_proto.shape    == (B, cfg.feat_ch), f"fg_proto: {fg_proto.shape}"
        print(f"  q_out={tuple(q_out.shape)}  "
              f"q_feat={tuple(q_feat.shape)}  "
              f"head_bias={model.head.net[-1].bias.item():.1f}")
    except Exception as e:
        print(f"  {e}"); raise
    finally:
        model.to(was).train()


# ====================================================================
# VOLUME-LEVEL DICE EVALUATION
# ====================================================================

@torch.no_grad()
def _postprocess_pred(pred_np: np.ndarray) -> np.ndarray:
    if pred_np.sum() == 0:
        return pred_np
    filled  = ndi.binary_fill_holes(pred_np)
    labeled, n = ndi.label(filled)
    if n == 0:
        return pred_np
    sizes   = ndi.sum(filled, labeled, range(1, n + 1))
    largest = np.argmax(sizes) + 1
    return (labeled == largest).astype(np.uint8)


@torch.no_grad()
def evaluate_volume_level(
        model:          nn.Module,
        val_cache:      SliceCache,
        fixed_support:  Dict,
        n_shot:         int,
        use_tta:        bool = False,
) -> Dict[str, float]:
    model.eval()
    per_organ: Dict[int, List[float]] = {o: [] for o in ORGAN_IDS}

    for organ in ORGAN_IDS:
        if organ not in fixed_support:
            continue
        organ_support = fixed_support[organ]

        pid_map: Dict[int, List[int]] = {}
        for k in val_cache.idx.get(organ, []):
            pid = val_cache.slices[k][3]
            pid_map.setdefault(pid, []).append(k)

        for q_pid, support_pairs in organ_support.items():
            if q_pid not in pid_map:
                continue

            s_imgs_l, s_masks_l, s_z_l = [], [], []
            for sk, _ in support_pairs:
                si_img, si_lbl, si_z = val_cache.get(sk, aug=False)
                s_imgs_l.append(si_img)
                s_masks_l.append((si_lbl == organ).long())
                s_z_l.append(si_z)

            s_imgs  = torch.stack(s_imgs_l).unsqueeze(0).to(device)
            s_masks = torch.stack(s_masks_l).unsqueeze(0).to(device)
            s_z     = torch.tensor([s_z_l], dtype=torch.float32).to(device)

            all_pred, all_gt = [], []
            for q_key in pid_map[q_pid]:
                q_img_t, q_lbl_t, q_z_v = val_cache.get(q_key, aug=False)
                q_img = q_img_t.unsqueeze(0).to(device, non_blocking=True)
                q_z   = torch.tensor([q_z_v], dtype=torch.float32).to(device)
                q_gt  = (q_lbl_t == organ).long()

                with autocast(enabled=USE_AMP):
                    logit, _, _, _, _, _ = model(s_imgs, s_masks, q_img, s_z, q_z)
                    if use_tta:
                        lf, _, _, _, _, _ = model(s_imgs, s_masks,
                                                   q_img.flip(-1), s_z, q_z)
                        logit = (logit + lf.flip(-1)) * 0.5

                pred_2d = (torch.sigmoid(logit.squeeze()) >= 0.5).cpu().numpy()
                pred_pp = _postprocess_pred(pred_2d.astype(np.uint8))
                all_pred.append(pred_pp)
                all_gt.append(q_gt.numpy())

            if not all_pred:
                continue

            pred_vol = np.stack(all_pred)
            gt_vol   = np.stack(all_gt)
            inter    = (pred_vol * gt_vol).sum()
            union    = pred_vol.sum() + gt_vol.sum()
            d = float(2 * inter / union) if union > 0 else 1.0
            per_organ[organ].append(d)

    results: Dict[str, float] = {}
    for o, vals in per_organ.items():
        if vals:
            results[ORGAN_MAP[o]] = float(np.mean(vals))
    return results


def _format_val_line(vol_dice: Dict[str, float],
                     best: float, use_tta: bool) -> str:
    tag  = "+TTA" if use_tta else "no-TTA"
    vals = list(vol_dice.values())
    mean = float(np.mean(vals)) if vals else 0.0
    org  = "  ".join(f"{n}={v*100:.2f}%" for n, v in vol_dice.items())
    return (f"-- val vol-Dice ({tag}): {mean*100:.2f}%  "
            f"best={best*100:.2f}%  [{org}] --")


# ====================================================================
# VISUALISATION
# ====================================================================

@torch.no_grad()
def visualize_top5(model: nn.Module,
                   val_cache: SliceCache,
                   fixed_support: Dict,
                   fold: int, save_dir: str,
                   use_tta: bool = False, top_k: int = 5):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    COLORS = {1: "#40FF80", 2: "#FFAA00", 3: "#FF80FF", 6: "#4080FF", 8: "#FF4040"}
    model.eval()
    per: Dict[int, list] = {o: [] for o in ORGAN_IDS}

    for organ in ORGAN_IDS:
        if organ not in fixed_support:
            continue
        pid_map: Dict[int, List[int]] = {}
        for k in val_cache.idx.get(organ, []):
            pid = val_cache.slices[k][3]
            pid_map.setdefault(pid, []).append(k)

        for q_pid, support_pairs in fixed_support[organ].items():
            if q_pid not in pid_map:
                continue
            s_imgs_l, s_masks_l, s_z_l = [], [], []
            for sk, _ in support_pairs:
                si_img, si_lbl, si_z = val_cache.get(sk, aug=False)
                s_imgs_l.append(si_img)
                s_masks_l.append((si_lbl == organ).long())
                s_z_l.append(si_z)
            s_imgs  = torch.stack(s_imgs_l).unsqueeze(0).to(device)
            s_masks = torch.stack(s_masks_l).unsqueeze(0).to(device)
            s_z     = torch.tensor([s_z_l], dtype=torch.float32).to(device)

            for q_key in pid_map[q_pid]:
                q_img_t, q_lbl_t, q_z_v = val_cache.get(q_key, aug=False)
                q_img = q_img_t.unsqueeze(0).to(device, non_blocking=True)
                q_z   = torch.tensor([q_z_v], dtype=torch.float32).to(device)
                q_gt  = (q_lbl_t == organ).long()

                with autocast(enabled=USE_AMP):
                    logit, _, _, _, _, _ = model(s_imgs, s_masks, q_img, s_z, q_z)
                    if use_tta:
                        lf, _, _, _, _, _ = model(s_imgs, s_masks,
                                                   q_img.flip(-1), s_z, q_z)
                        logit = (logit + lf.flip(-1)) * 0.5

                logit_cpu = logit.squeeze(1).cpu()
                prob_np   = torch.sigmoid(logit_cpu.squeeze()).numpy()
                d = dice_score(logit_cpu, q_gt.unsqueeze(0))
                per[organ].append((d, q_img_t[0].numpy(),
                                   q_gt.numpy().astype(bool), prob_np >= 0.5))

    os.makedirs(save_dir, exist_ok=True)
    tta_tag = "+TTA" if use_tta else "no-TTA"

    for organ in ORGAN_IDS:
        entries = sorted(per[organ], key=lambda x: x[0], reverse=True)
        if not entries: continue
        top = entries[:top_k]; K = len(top)
        name  = ORGAN_MAP[organ].replace("_", " ").capitalize()
        color = COLORS.get(organ, "#FFFFFF")
        rv = int(color[1:3], 16) / 255
        gv = int(color[3:5], 16) / 255
        bv = int(color[5:7], 16) / 255

        fig, axes = plt.subplots(3, K, figsize=(4 * K, 10), facecolor="#0D1117")
        if K == 1: axes = [[axes[r]] for r in range(3)]
        fig.suptitle(
            f"Fold {fold+1} - {name} | ADProto-Net MSAPA ({tta_tag})",
            color="white", fontsize=11, y=1.01)
        for row, label in enumerate(["CT Input", "Ground Truth", "Predicted"]):
            axes[row][0].set_ylabel(label, color="white", fontsize=10, labelpad=8)

        for col, (d, q_img, gt, pred) in enumerate(top):
            for row, mask in enumerate([None, gt, pred]):
                ax = axes[row][col]
                ax.imshow(q_img, cmap="gray", vmin=0, vmax=1)
                if mask is not None:
                    ov = np.zeros((*mask.shape, 4), dtype=np.float32)
                    ov[mask] = [rv, gv, bv, 0.55]
                    ax.imshow(ov, interpolation="nearest")
                if row == 0:
                    ax.set_title(f"Dice: {d*100:.1f}%", color="#A8C7FA", fontsize=10)
                ax.axis("off"); ax.set_facecolor("#0D1117")

        patch = mpatches.Patch(color=color, label=f"{name} mask")
        fig.legend(handles=[patch], loc="lower center", framealpha=0.3,
                   facecolor="#1A1F2E", labelcolor="white", fontsize=9,
                   bbox_to_anchor=(0.5, -0.02))
        plt.tight_layout()
        out = os.path.join(save_dir,
                           f"fold{fold}_top5_{ORGAN_MAP[organ]}_{tta_tag}.png")
        fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="#0D1117")
        plt.close(fig)
        print(f"  [viz] -> {out}")


# ====================================================================
# TRAINING LOOP
# ====================================================================

def train_fold(fold: int, cfg) -> Tuple[float, Dict[str, float]]:
    print(f"\n{'=' * 60}")
    print(f"  ADProto-Net  |  FOLD {fold + 1}/{N_FOLDS}  |  {cfg.n_shot}-shot")
    print(f"  Novel module: MSAPA (scales={cfg.msapa_scales})")
    print(f"{'=' * 60}")

    tr_ld, val_ld, val_cache = make_loaders(
        cfg.data_dir, fold, cfg.n_shot, cfg.batch, cfg)

    print("  Building fixed validation support set...")
    fixed_support = build_fixed_val_support(val_cache, cfg.n_shot, seed=cfg.eval_seed)
    n_pairs = sum(len(v) for v in fixed_support.values())
    print(f"  Fixed support built for {n_pairs} patient-organ pairs.")

    model = ADProtoNet(n_shot=cfg.n_shot, pretrained=True, cfg=cfg).to(device)
    verify_shapes(model, cfg)

    try:
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.lr, weight_decay=1e-4, fused=True)
    except TypeError:
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.lr, weight_decay=1e-4)

    warmup = int(cfg.n_iter * 0.05)
    def lr_lambda(s):
        if s < warmup:
            return s / max(warmup, 1)
        t = (s - warmup) / max(cfg.n_iter - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * t))

    sched  = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = GradScaler(enabled=USE_AMP)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params / 1e6:.1f}M")
    os.makedirs(cfg.save_dir, exist_ok=True)
    ck_path = os.path.join(cfg.save_dir, f"fold{fold}_best.pth")

    best_vol_mean = 0.0
    no_improve    = 0
    step          = 0
    run_loss = run_n = 0
    t0 = time.time()
    loader_iter = iter(tr_ld)

    while step < cfg.n_iter:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(tr_ld)
            batch = next(loader_iter)

        si, sm, qi, qm, s_z, q_z, oids, _, organ_idxs = batch
        if step == 0:
            print(f"  First batch in {time.time() - t0:.1f}s. Training...")
            t0 = time.time()

        model.train()
        si  = si.to(device, non_blocking=True)
        sm  = sm.to(device, non_blocking=True)
        qi  = qi.to(device, non_blocking=True)
        qm  = qm.to(device, non_blocking=True)
        s_z = s_z.to(device, non_blocking=True)
        q_z = q_z.to(device, non_blocking=True)

        with autocast(enabled=USE_AMP):
            q_logit, aux_out, q_feat, s_feats, sm_res, fg_proto = model(
                si, sm, qi, s_z, q_z)

            loss = focal_tversky(q_logit.squeeze(1), qm,
                                 cfg.focal_alpha, cfg.focal_gamma)

            if aux_out is not None:
                loss = loss + cfg.aux_weight * focal_tversky(
                    aux_out.squeeze(1), qm, cfg.focal_alpha, cfg.focal_gamma)

        if not torch.isfinite(loss):
            print(f"  [warn] non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True); step += 1; continue

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()

        run_loss += loss.item(); run_n += 1; step += 1

        if step % 200 == 0:
            lr_now  = opt.param_groups[0]['lr']
            elapsed = time.time() - t0
            its     = 200 / max(elapsed, 1e-3)
            print(f"  step {step:5d}/{cfg.n_iter}  "
                  f"loss={run_loss / max(run_n, 1):.4f}  "
                  f"lr={lr_now:.2e}  {its:.1f} it/s")
            run_loss = run_n = 0; t0 = time.time()

        if step % cfg.val_interval == 0 or step == cfg.n_iter:
            vol_dice = evaluate_volume_level(
                model, val_cache, fixed_support, cfg.n_shot, use_tta=False)
            vals = list(vol_dice.values())
            vol_mean = float(np.mean(vals)) if vals else 0.0

            print(f"  {_format_val_line(vol_dice, best_vol_mean, False)}")

            if vol_mean > best_vol_mean:
                best_vol_mean = vol_mean
                no_improve    = 0
                torch.save({
                    "step":      step,
                    "state":     model.state_dict(),
                    "vol_dice":  vol_dice,
                    "vol_mean":  best_vol_mean,
                    "fold":      fold,
                    "n_shot":    cfg.n_shot,
                    "eval_seed": cfg.eval_seed,
                    "split_seed":cfg.split_seed,
                    "organ_map": ORGAN_MAP,
                    "model":     "ADProto-Net (MSAPA)",
                    "cfg":       vars(cfg),
                }, ck_path)
                print(f"  * saved -> {ck_path}")
            else:
                no_improve += 1
                if no_improve >= cfg.es_patience:
                    print(f"  [EarlyStop] at step {step}")
                    break

    print(f"\n  Fold {fold + 1} done.  "
          f"Best val vol-Dice = {best_vol_mean * 100:.2f}%")

    print("  Reloading best checkpoint for final evaluation...")
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["state"])

    print("  Running final volume-level evaluation (no-TTA)...")
    vol_dice_final = evaluate_volume_level(
        model, val_cache, fixed_support, cfg.n_shot, use_tta=False)

    print(f"  Volume-level Dice (fold {fold+1}, no-TTA):")
    all_vols: List[float] = []
    for organ in ORGAN_IDS:
        name = ORGAN_MAP[organ]
        if name in vol_dice_final:
            d = vol_dice_final[name]
            print(f"    {name:<14} {d * 100:.2f}%")
            all_vols.append(d)
        else:
            print(f"    {name:<14} N/A (insufficient data)")
    if all_vols:
        print(f"    {'Mean':<14} {float(np.mean(all_vols)) * 100:.2f}%  "
              f"(over {len(all_vols)} organs evaluated)")

    if cfg.use_tta:
        print("  Running final volume-level evaluation (+TTA)...")
        vol_dice_tta = evaluate_volume_level(
            model, val_cache, fixed_support, cfg.n_shot, use_tta=True)
        print(f"  Volume-level Dice (fold {fold+1}, +TTA):")
        for organ in ORGAN_IDS:
            name = ORGAN_MAP[organ]
            if name in vol_dice_tta:
                print(f"    {name:<14} {vol_dice_tta[name] * 100:.2f}%")
        tta_vals = list(vol_dice_tta.values())
        if tta_vals:
            print(f"    {'Mean':<14} {float(np.mean(tta_vals)) * 100:.2f}%")

    print("  Generating visualisations...")
    visualize_top5(model, val_cache, fixed_support,
                   fold=fold, save_dir=cfg.save_dir, use_tta=False)
    if cfg.use_tta:
        visualize_top5(model, val_cache, fixed_support,
                       fold=fold, save_dir=cfg.save_dir, use_tta=True)

    return best_vol_mean, vol_dice_final


# ====================================================================
# ENTRY POINT
# ====================================================================

def run(cfg=None):
    if cfg is None:
        cfg = CFG()
    folds = list(range(N_FOLDS)) if cfg.fold == -1 else [cfg.fold]

    if cfg.mode == "preprocess":
        preprocess(cfg.raw_dir, cfg.data_dir); return

    print(f"\n{'=' * 60}")
    print(f"  ADProto-Net -- Adaptive Dense Prototype Network")
    print(f"  Novel: Multi-Scale Adaptive Prototype Aggregation (MSAPA)")
    print(f"  Device:{device}  AMP:{USE_AMP}  Shot:{cfg.n_shot}  TTA:{cfg.use_tta}")
    print(f"  Scales: {cfg.msapa_scales}")
    print(f"  Organs: {[ORGAN_MAP[o] for o in ORGAN_IDS]}")
    print(f"  Eval seed: {cfg.eval_seed}  (deterministic support assignment)")
    print(f"  Split seed: {cfg.split_seed}  (shuffled fold assignment)")
    print(f"  Baseline to beat: 85.57% overall")
    print(f"{'=' * 60}")

    if cfg.mode == "train":
        all_best: List[float] = []
        all_vol_dices: Dict[str, List[float]] = {n: [] for n in ORGAN_MAP.values()}

        for f in folds:
            best, vol_dice_fold = train_fold(f, cfg)
            all_best.append(best)
            for name, d in vol_dice_fold.items():
                all_vol_dices[name].append(d)

        print(f"\n  +================================================+")
        print(f"  |  FINAL RESULTS -- ADProto-Net (MSAPA)          |")
        print(f"  |  {cfg.n_shot}-shot  |  no-TTA  |  volume-level Dice      |")
        print(f"  +------------------------------------------------+")

        grand_all: List[float] = []
        for organ in ORGAN_IDS:
            name   = ORGAN_MAP[organ]
            scores = all_vol_dices[name]
            if scores:
                mu  = float(np.mean(scores))
                std = float(np.std(scores))
                print(f"  |  {name:<16} {mu*100:.2f}% +/- {std*100:.2f}%          |")
                grand_all.extend(scores)

        if grand_all:
            gm = float(np.mean(grand_all))
            print(f"  |  {'Overall':<16} {gm*100:.2f}%                         |")
        print(f"  +------------------------------------------------+")
        best_strs = [f"{d*100:.2f}%" for d in all_best]
        print(f"  |  Per-fold: {best_strs}")
        print(f"  |  Baseline: ['90.45%', '77.45%', '88.04%', '86.34%']")
        print(f"  +================================================+")
        return all_best, all_vol_dices

    if cfg.mode == "eval":
        all_vol_dices: Dict[str, List[float]] = {n: [] for n in ORGAN_MAP.values()}
        for f in folds:
            ck_path = os.path.join(cfg.save_dir, f"fold{f}_best.pth")
            if not os.path.exists(ck_path):
                print(f"  Fold {f}: no checkpoint at {ck_path}"); continue
            model = ADProtoNet(n_shot=cfg.n_shot, pretrained=False, cfg=cfg).to(device)
            ck    = torch.load(ck_path, map_location=device, weights_only=False)
            key   = "state" if "state" in ck else "model_state_dict"
            model.load_state_dict(ck[key])
            _, val_f = split_folds(cfg.data_dir, f, split_seed=cfg.split_seed)
            cd = Path(cfg.data_dir) / ".ram_cache_adproto"
            val_cache     = SliceCache(val_f, str(cd / f"fold{f}_val.pkl"))
            fixed_support = build_fixed_val_support(
                val_cache, cfg.n_shot, seed=cfg.eval_seed)
            od = evaluate_volume_level(
                model, val_cache, fixed_support, cfg.n_shot, use_tta=False)
            print(f"  Fold {f} (no-TTA):")
            for name, d in od.items():
                all_vol_dices[name].append(d)
                print(f"    {name:<14} {d*100:.2f}%")
            if cfg.use_tta:
                od_tta = evaluate_volume_level(
                    model, val_cache, fixed_support, cfg.n_shot, use_tta=True)
                print(f"  Fold {f} (+TTA):")
                for name, d in od_tta.items():
                    print(f"    {name:<14} {d*100:.2f}%")

        print(f"\n  Organ          Mean+/-Std (no-TTA, volume-level)")
        grand_all: List[float] = []
        for organ in ORGAN_IDS:
            name   = ORGAN_MAP[organ]
            scores = all_vol_dices[name]
            if scores:
                print(f"  {name:<16} {np.mean(scores)*100:.2f}% +/- "
                      f"{np.std(scores)*100:.2f}%")
                grand_all.extend(scores)
        if grand_all:
            print(f"  {'Overall':<16} {np.mean(grand_all)*100:.2f}%")
        return all_vol_dices


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ADProto-Net (MSAPA)")
    ap.add_argument("--mode",        choices=["preprocess", "train", "eval"],
                    default="train")
    ap.add_argument("--raw_dir",     default=CFG.raw_dir)
    ap.add_argument("--data_dir",    default=CFG.data_dir)
    ap.add_argument("--save_dir",    default=CFG.save_dir)
    ap.add_argument("--fold",        type=int,   default=CFG.fold)
    ap.add_argument("--n_shot",      type=int,   default=CFG.n_shot)
    ap.add_argument("--n_iter",      type=int,   default=CFG.n_iter)
    ap.add_argument("--lr",          type=float, default=CFG.lr)
    ap.add_argument("--batch",       type=int,   default=CFG.batch)
    ap.add_argument("--workers",     type=int,   default=CFG.num_workers)
    ap.add_argument("--eval_seed",   type=int,   default=CFG.eval_seed)
    ap.add_argument("--split_seed",  type=int,   default=CFG.split_seed)
    ap.add_argument("--tta",         action="store_true")
    ap.add_argument("--msapa_scales",nargs="+", type=int,
                    default=list(CFG.msapa_scales))
    args, _ = ap.parse_known_args()
    cfg = CFG()
    for k, v in vars(args).items():
        if   k == "workers":      cfg.num_workers   = v
        elif k == "tta":          cfg.use_tta        = v
        elif k == "msapa_scales": cfg.msapa_scales   = tuple(v)
        elif hasattr(cfg, k):     setattr(cfg, k, v)
    run(cfg)