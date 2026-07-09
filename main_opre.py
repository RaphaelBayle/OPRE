#!/usr/bin/env python3
"""
OPRE reproduction script
========================
Minimal, self-contained reproduction of the OPRE experiments (CIFAR-10 /
CIFAR-100, plus the synthetic linearly-separated data of Appendix B via
--dataset synthetic) from the paper, plus two matched-memory-budget baselines:
GDumb (greedy class-balanced buffer + retraining from scratch) and
ER (online Experience Replay with a reservoir buffer).

Outputs, per configuration: final accuracy (mean & SD over seeds),
number of stored patches, and memory cost in the Table-4 format
"total (data | model)" in MB (1 MB = 1e6 bytes).

See README_opre.md for full documentation.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# --------------------------------------------------------------------------
# Configurations from the article
# --------------------------------------------------------------------------

QUALITY = {
    "low": {"eps": 0.4, "levels": 6},  # the single OPRE setting of the article
}

# Data-memory budgets (MB) reported in Table 4 of the article, used as the
# default budget for GDumb / ER when --budget-mb is not given and when the
# matched OPRE run is not available (standalone mode).
ARTICLE_BUDGET_MB = {
    ("cifar10", "low"): 35.57,
    ("cifar100", "low"): 36.08,
}

RAW_IMAGE_BYTES = 3 * 32 * 32          # 8-bit RGB CIFAR image
NUM_CLASSES = {"cifar10": 10, "cifar100": 100, "synthetic": 2}
ID_BITS = 32                           # one int32 per patch ID (article convention)
PATCHES_PER_IMAGE = 64                 # 32x32 image, 4x4 patches, no overlap
PATCH_DIM = 3 * 4 * 4                  # 48


def patch_bits(levels: int) -> int:
    """Bits needed to store one 3x4x4 patch, using the article's packing:
    four 2x2x3 sub-patches, each packed into one integer (32 or 64 bits
    depending on whether levels**12 fits in 2**32)."""
    sub_states = levels ** 12
    bits_per_sub = 32 if sub_states <= 2 ** 32 else 64
    return 4 * bits_per_sub


def mb(n_bytes: float) -> float:
    return n_bytes / 1e6


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_cifar(name: str, root: str):
    """Returns train_x, train_y, test_x, test_y with pixel values in [0, 1]
    (float32, no normalization -- epsilon is therefore expressed in [0,1]
    pixel units, as in the original implementation)."""
    cls = torchvision.datasets.CIFAR10 if name == "cifar10" else torchvision.datasets.CIFAR100
    tr = cls(root, train=True, download=True)
    te = cls(root, train=False, download=True)
    train_x = torch.from_numpy(tr.data).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    train_y = torch.tensor(tr.targets, dtype=torch.long)
    test_x = torch.from_numpy(te.data).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    test_y = torch.tensor(te.targets, dtype=torch.long)
    return train_x, train_y, test_x, test_y


def load_synthetic(gen_seed: int = 0):
    """Random linearly-separated data of Appendix B of the article.

    - v_hyperplane: a (32,32) tensor ~ N(0,1), duplicated over the 3
      channels, then flattened (3072 values);
    - images: (3,32,32) pure noise, each pixel ~ N(0,1);
    - label (2 classes): sign of <v_hyperplane, flattened image>, computed
      on the ORIGINAL N(0,1) pixels;
    - 50,000 images, 80/20 train/test split (40,000 / 10,000);
    - pixels are then mapped into [0,1] via clip((x+3)/6, 0, 1) so that the
      L-level quantization and epsilon keep their [0,1] meaning. The affine
      map preserves linear separability (the clip affects ~0.3% of pixels).

    The generation seed is fixed independently of the run seed, so every
    run/seed uses the same dataset (only the stream order changes).
    Memory accounting keeps the 8-bit raw-size convention (RAW_IMAGE_BYTES),
    as for CIFAR."""
    g = torch.Generator().manual_seed(gen_seed)
    v = torch.randn((32, 32), generator=g)
    v = v.unsqueeze(0).repeat(3, 1, 1).reshape(-1)              # (3072,)
    x = torch.randn((50000, 3, 32, 32), generator=g)
    y = (x.reshape(len(x), -1) @ v > 0).long()                  # classes {0,1}
    x = ((x + 3.0) / 6.0).clamp_(0.0, 1.0)
    return x[:40000].contiguous(), y[:40000], x[40000:].contiguous(), y[40000:]


def load_data(name: str, root: str, args):
    if name == "synthetic":
        return load_synthetic(args.synthetic_seed)
    return load_cifar(name, root)


def stream_order(y: torch.Tensor, mode: str, seed: int, shuffle_within_class: bool) -> torch.Tensor:
    """Order in which training samples arrive.
    - 'class': class-incremental (class 0, then class 1, ...), as stated in
      the article. Within-class order is the dataset order unless
      shuffle_within_class is set (then it is seeded).
    - 'native': dataset native order (classes mixed) -- reproduces the
      historical runs done with shuffle=False.
    - 'shuffle': fully shuffled stream (seeded)."""
    g = torch.Generator().manual_seed(seed)
    n = len(y)
    if mode == "native":
        return torch.arange(n)
    if mode == "shuffle":
        return torch.randperm(n, generator=g)
    # class-incremental
    parts = []
    for c in sorted(torch.unique(y).tolist()):
        idx = (y == c).nonzero(as_tuple=True)[0]
        if shuffle_within_class:
            idx = idx[torch.randperm(len(idx), generator=g)]
        parts.append(idx)
    return torch.cat(parts)


def stratified_truncate(order: torch.Tensor, y: torch.Tensor, limit: int) -> torch.Tensor:
    """Truncates the stream while preserving class coverage: keeps the first
    limit/num_classes samples of each class, in stream order. (A naive
    truncation of a class-incremental stream would keep a single class,
    making the classification task degenerate.)"""
    if limit is None or limit >= len(order):
        return order
    classes = torch.unique(y).tolist()
    per_class = max(1, limit // len(classes))
    counts = {c: 0 for c in classes}
    keep = []
    for i, c in zip(order.tolist(), y[order].tolist()):
        if counts[c] < per_class:
            counts[c] += 1
            keep.append(i)
    return torch.tensor(keep, dtype=torch.long)


# --------------------------------------------------------------------------
# Model (Appendix A of the article)
# --------------------------------------------------------------------------

def _block(cin, cout):
    return [nn.Conv2d(cin, cout, kernel_size=3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True)]


class SimpleNet(nn.Module):
    """The ~10M-parameter CNN of Appendix A."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        layers = []
        layers += _block(3, 64) + _block(64, 64) + _block(64, 64)
        layers += [nn.MaxPool2d(2, 2)]
        layers += _block(64, 128) + _block(128, 128) + _block(128, 128)
        layers += [nn.MaxPool2d(2, 2)]
        layers += _block(128, 256) + _block(256, 256) + _block(256, 256)
        layers += [nn.MaxPool2d(2, 2)]
        layers += _block(256, 512) + _block(512, 512) + _block(512, 512) + _block(512, 512)
        layers += [nn.AdaptiveAvgPool2d((1, 1))]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def model_param_bytes(num_classes: int) -> int:
    m = SimpleNet(num_classes)
    return sum(p.numel() for p in m.parameters()) * 4  # float32 weights


# --------------------------------------------------------------------------
# OPRE core
# --------------------------------------------------------------------------

def quantize(x: torch.Tensor, levels: int) -> torch.Tensor:
    """Uniform discretization of pixel values in [0,1] to `levels` values."""
    return torch.round(x * (levels - 1)) / (levels - 1)


def patchify(images: torch.Tensor) -> torch.Tensor:
    """(b,3,32,32) -> (b*64, 48), non-overlapping 4x4 patches."""
    p = F.unfold(images, kernel_size=4, stride=4)      # (b, 48, 64)
    return p.permute(0, 2, 1).reshape(-1, PATCH_DIM)   # (b*64, 48)


class PatchMemory:
    """Append-only memory of distinct patches, discretized on arrival then
    epsilon-deduplicated.

    Every incoming patch is first discretized (its pixel values quantized to
    `levels` uniform values per channel) *before* any comparison with the
    memory takes place, so the buffer only ever contains discretized patches
    (grid points) and epsilon is measured between discretized patches
    (Algorithm 1 of the article: discretize -> dedup).

    VRAM-frugal design inspired by the historical implementation:
    - distances are computed blockwise as *squared* Euclidean distances via
      the identity ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>, accumulated
      in place, so the only large temporary is one (m, block) float32 matrix
      whose size is capped by `dist_mb` megabytes (no torch.cdist, whose
      internal temporaries caused OOM on 16 GB GPUs);
    - squared norms of stored patches are cached;
    - patches can be stored in float16 (`--patch-dtype float16`, halving the
      buffer like the historical bfloat16 storage); distance arithmetic is
      always carried out in float32 so that epsilon and the discretization
      remain the only sources of information loss."""

    def __init__(self, eps: float, levels: int, device: torch.device,
                 dtype: torch.dtype = torch.float32,
                 compute_dtype: torch.dtype = torch.float32,
                 dist_mb: float = 256.0, init_capacity: int = 1 << 17):
        if device.type == "cpu":
            dtype = compute_dtype = torch.float32  # half matmul is poor on CPU
        self.eps2 = float(eps) ** 2
        self.levels = int(levels)
        self.device = device
        self.dtype = dtype
        self.cdtype = compute_dtype
        self.dist_mb = dist_mb
        self.buf = torch.empty((init_capacity, PATCH_DIM), dtype=dtype, device=device)
        self.sqn = torch.empty((init_capacity,), dtype=torch.float32, device=device)
        self.count = 0

    def _ensure_capacity(self, extra: int):
        needed = self.count + extra
        cap = self.buf.shape[0]
        if needed <= cap:
            return
        while cap < needed:
            cap *= 2
        new_buf = torch.empty((cap, PATCH_DIM), dtype=self.dtype, device=self.device)
        new_sqn = torch.empty((cap,), dtype=torch.float32, device=self.device)
        new_buf[: self.count] = self.buf[: self.count]
        new_sqn[: self.count] = self.sqn[: self.count]
        self.buf, self.sqn = new_buf, new_sqn

    def _append(self, x: torch.Tensor):
        """x: (k, 48) float32 on device."""
        self._ensure_capacity(len(x))
        self.buf[self.count: self.count + len(x)] = x.to(self.dtype)
        self.sqn[self.count: self.count + len(x)] = (x * x).sum(dim=1)
        self.count += len(x)

    @torch.no_grad()
    def _scan(self, xc: torch.Tensor, start: int, end: int):
        """Blockwise min over stored patches in [start, end) of
        g_ij = ||y_j||^2 - 2<x_i, y_j>  ( = d2_ij - ||x_i||^2 ).
        The row-constant ||x_i||^2 does not affect the min/argmin and is
        reapplied at threshold time, which removes one full traversal of the
        big matrix. torch.addmm fuses the y^2 add and the -2 scaling into the
        GEMM epilogue, so the (m, block) matrix is written once and read once
        (by the min reduction)."""
        m = len(xc)
        elt = 2 if self.cdtype in (torch.float16, torch.bfloat16) else 4
        best = torch.full((m,), float("inf"), dtype=torch.float32, device=self.device)
        arg = torch.zeros(m, dtype=torch.long, device=self.device)
        block = max(1024, int(self.dist_mb * 1e6 / (m * elt)))
        for s in range(start, end, block):
            e = min(s + block, end)
            yb = self.buf[s:e]
            if yb.dtype != self.cdtype:
                yb = yb.to(self.cdtype)
            y2 = self.sqn[s:e].to(self.cdtype)
            g = torch.addmm(y2.unsqueeze(0), xc, yb.T, beta=1.0, alpha=-2.0)
            mn, am = g.min(dim=1)
            del g
            mn = mn.float()
            upd = mn < best
            best[upd] = mn[upd]
            arg[upd] = am[upd] + s
        return best, arg

    @torch.no_grad()
    def add(self, x: torch.Tensor) -> torch.Tensor:
        """Discretization on arrival, then two-step epsilon-deduplication
        (equivalent to sequential processing), then returns the ID (index of
        the nearest stored patch) for every input patch.

        Each incoming patch is discretized *first* (quantized to
        `self.levels` values per channel); deduplication is then carried out
        between discretized patches only. A (discretized) patch is redundant
        iff its distance to a stored patch is < eps (compared as squared
        distances: g >= eps^2 - x^2 means non-redundant)."""
        x = quantize(x.to(self.device, dtype=torch.float32), self.levels)
        x2 = (x * x).sum(dim=1)
        xc = x if self.cdtype == torch.float32 else x.to(self.cdtype)
        thr = self.eps2 - x2
        if self.count == 0:
            self._append(x[:1])
        count0 = self.count
        gmin, argmin = self._scan(xc, 0, count0)
        cand_mask = gmin >= thr
        if bool(cand_mask.any()):
            cand = xc[cand_mask]
            c2 = x2[cand_mask]
            # within-batch dedup: keep first occurrence (sequential semantics)
            d = torch.addmm(c2.to(self.cdtype).unsqueeze(0), cand, cand.T,
                            beta=1.0, alpha=-2.0)
            d.add_(torch.triu(torch.full_like(d, float("inf"))))  # mask j >= i
            keep = d.min(dim=1).values.float() >= (self.eps2 - c2)
            del d
            self._append(x[cand_mask][keep])
        if self.count > count0:
            # merge with distances to the patches just added (cheap: few cols)
            gnew, argnew = self._scan(xc, count0, self.count)
            upd = gnew < gmin
            argmin[upd] = argnew[upd]
        return argmin.to(torch.int32)


@torch.no_grad()
def run_opre(images_ordered: torch.Tensor, eps: float, levels: int,
             device: torch.device, chunk: int = 32,
             dist_mb: float = 256.0, patch_dtype: torch.dtype = torch.float32,
             dist_dtype: torch.dtype = torch.float32, verbose: bool = True):
    """Streams images (already in arrival order) through OPRE.
    Order of Algorithm 1: every new patch is discretized on arrival, inside
    PatchMemory.add, *before* being compared with the (already discretized)
    stored patches; the patch memory therefore only ever contains
    discretized patches.
    Returns (PatchMemory, ids tensor (n,64) int32)."""
    n = len(images_ordered)
    mem = PatchMemory(eps=eps, levels=levels, device=device, dtype=patch_dtype,
                      compute_dtype=dist_dtype, dist_mb=dist_mb)
    ids_all = torch.empty((n, PATCHES_PER_IMAGE), dtype=torch.int32)
    t0 = time.time()
    for s in range(0, n, chunk):
        x = images_ordered[s: s + chunk].to(device)
        ids = mem.add(patchify(x))
        ids_all[s: s + len(x)] = ids.reshape(len(x), PATCHES_PER_IMAGE).cpu()
        if verbose and (s // chunk) % 200 == 0:
            print(f"  [opre] {s + len(x):>6}/{n} images | stored patches: "
                  f"{mem.count:>8} | {time.time() - t0:6.1f}s", flush=True)
    if verbose:
        print(f"  [opre] done: {mem.count} stored patches "
              f"({mem.count / (n * PATCHES_PER_IMAGE) * 100:.1f}% of "
              f"{n * PATCHES_PER_IMAGE}) in {time.time() - t0:.1f}s")
    return mem, ids_all


@torch.no_grad()
def reconstruct(ids_all: torch.Tensor, mem: PatchMemory, chunk: int = 2048) -> torch.Tensor:
    """Rebuilds the compressed images from patch IDs. Returns (n,3,32,32)
    float16 on CPU."""
    n = len(ids_all)
    out = torch.empty((n, 3, 32, 32), dtype=torch.float16)
    for s in range(0, n, chunk):
        ids = ids_all[s: s + chunk].to(mem.device).long()
        patches = mem.buf[ids].float()               # (b, 64, 48)
        patches = patches.permute(0, 2, 1)           # (b, 48, 64)
        imgs = F.fold(patches, output_size=(32, 32), kernel_size=4, stride=4)
        out[s: s + len(ids)] = imgs.to("cpu", dtype=torch.float16)
    return out

# --------------------------------------------------------------------------
# Classifier training (used by OPRE, GDumb and the no-compression bound)
# --------------------------------------------------------------------------

def augment_batch(x: torch.Tensor) -> torch.Tensor:
    """Random crop (pad 4) + horizontal flip, on a GPU batch."""
    b = x.size(0)
    xp = F.pad(x, (4, 4, 4, 4))
    ii = torch.randint(0, 9, (b,))
    jj = torch.randint(0, 9, (b,))
    out = torch.stack([xp[k, :, ii[k]: ii[k] + 32, jj[k]: jj[k] + 32] for k in range(b)])
    flip = torch.rand(b, device=x.device) < 0.5
    out[flip] = out[flip].flip(-1)
    return out


@torch.no_grad()
def evaluate(model: nn.Module, test_x: torch.Tensor, test_y: torch.Tensor,
             device: torch.device, batch: int = 512) -> float:
    model.eval()
    correct = 0
    for s in range(0, len(test_x), batch):
        xb = test_x[s: s + batch].to(device, dtype=torch.float32)
        pred = model(xb).argmax(dim=1).cpu()
        correct += (pred == test_y[s: s + batch]).sum().item()
    return 100.0 * correct / len(test_x)


def train_classifier(train_x: torch.Tensor, train_y: torch.Tensor, num_classes: int,
                     test_x: torch.Tensor, test_y: torch.Tensor, device: torch.device,
                     epochs: int, lr: float, batch_size: int, augment: bool,
                     seed: int, optimizer: str = "sgd", verbose: bool = True) -> float:
    """Trains SimpleNet from scratch (random init) on (train_x, train_y),
    as in the GDumb protocol, and returns the test accuracy (%).
    optimizer: 'sgd' (CIFAR protocol, cosine schedule) or 'adam'
    (Appendix B protocol for the synthetic data, no schedule)."""
    set_seed(seed)
    model = SimpleNet(num_classes).to(device)
    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = None
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                              weight_decay=5e-4, nesterov=True)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = len(train_x)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for s in range(0, n, batch_size):
            idx = perm[s: s + batch_size]
            xb = train_x[idx].to(device, dtype=torch.float32)
            yb = train_y[idx].to(device)
            if augment:
                xb = augment_batch(xb)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        if sched is not None:
            sched.step()
        if verbose and (ep + 1) % max(1, epochs // 6) == 0:
            print(f"  [train] epoch {ep + 1:>3}/{epochs} | loss {loss.item():.3f} "
                  f"| {time.time() - t0:6.1f}s", flush=True)
    return evaluate(model, test_x, test_y, device)


# --------------------------------------------------------------------------
# Baselines: GDumb (matched budget) and ER (matched budget)
# --------------------------------------------------------------------------

def budget_to_images(budget_mb: float) -> int:
    return int(budget_mb * 1e6 // RAW_IMAGE_BYTES)


def gdumb_select(labels: torch.Tensor, order: torch.Tensor, n_buf: int, seed: int) -> torch.Tensor:
    """GDumb's greedy class-balanced sampler over the stream.
    Returns the dataset indices kept in the buffer."""
    rng = random.Random(seed)
    buf_idx, counts, members = [], {}, {}
    for pos in order.tolist():
        c = int(labels[pos])
        counts.setdefault(c, 0)
        members.setdefault(c, [])
        if len(buf_idx) < n_buf:
            members[c].append(len(buf_idx))
            buf_idx.append(pos)
            counts[c] += 1
        else:
            cmax = max(counts, key=counts.get)
            if counts[c] < counts[cmax] and members[cmax]:
                slot = members[cmax].pop(rng.randrange(len(members[cmax])))
                buf_idx[slot] = pos
                counts[cmax] -= 1
                counts[c] += 1
                members[c].append(slot)
    return torch.tensor(buf_idx, dtype=torch.long)


def run_er(train_x, train_y, order, n_buf, num_classes, test_x, test_y, device,
           seed, lr, stream_bs, replay_bs, verbose=True) -> float:
    """Online Experience Replay: single pass over the stream, one gradient
    step per incoming mini-batch combined with a replay mini-batch sampled
    from a reservoir buffer of n_buf raw images."""
    set_seed(seed)
    model = SimpleNet(num_classes).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    buf_x = torch.empty((n_buf, 3, 32, 32), dtype=torch.float16)
    buf_y = torch.empty((n_buf,), dtype=torch.long)
    filled, n_seen = 0, 0
    rng = random.Random(seed)
    model.train()
    t0 = time.time()
    for s in range(0, len(order), stream_bs):
        idx = order[s: s + stream_bs]
        xb = train_x[idx].to(device, dtype=torch.float32)
        yb = train_y[idx].to(device)
        if filled > 0:
            r = torch.randint(0, filled, (min(replay_bs, filled),))
            xb = torch.cat([xb, buf_x[r].to(device, dtype=torch.float32)])
            yb = torch.cat([yb, buf_y[r].to(device)])
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()
        for i in idx.tolist():  # reservoir update
            if filled < n_buf:
                buf_x[filled] = train_x[i].half()
                buf_y[filled] = train_y[i]
                filled += 1
            else:
                j = rng.randint(0, n_seen)
                if j < n_buf:
                    buf_x[j] = train_x[i].half()
                    buf_y[j] = train_y[i]
            n_seen += 1
        if verbose and (s // stream_bs) % 1000 == 0:
            print(f"  [er] {s:>6}/{len(order)} | loss {loss.item():.3f} "
                  f"| {time.time() - t0:6.1f}s", flush=True)
    return evaluate(model, test_x, test_y, device)


# --------------------------------------------------------------------------
# Experiment driver
# --------------------------------------------------------------------------

def run_one(method, dataset, quality, seed, data, args, budget_mb=None) -> dict:
    """Runs one (method, dataset, quality, seed) configuration and returns a
    result record."""
    train_x, train_y, test_x, test_y = data
    num_classes = NUM_CLASSES[dataset]
    device = get_device(args.device)
    # Per-dataset training protocol. CIFAR: SGD 0.1, batch 128, augmentation.
    # Synthetic (Appendix B): Adam 0.001, batch 64, and augmentation FORCED
    # OFF -- the label is the sign of a fixed linear form of the pixels, so
    # flips/crops change the label.
    synth = dataset == "synthetic"
    lr = args.lr if args.lr is not None else (0.001 if synth else 0.1)
    batch_size = args.batch_size if args.batch_size is not None else (64 if synth else 128)
    optimizer = "adam" if synth else "sgd"
    augment = (not args.no_augment) and not synth
    if synth and not args.no_augment:
        print("  [note] synthetic data: augmentation disabled (labels are not "
              f"flip/crop-invariant); Adam lr={lr}, batch={batch_size}")
    order = stream_order(train_y, args.stream, seed, args.shuffle_within_class)
    if args.limit_train:
        order = stratified_truncate(order, train_y, args.limit_train)
    n_img = len(order)
    model_mb = mb(model_param_bytes(num_classes))
    t0 = time.time()
    rec = {"dataset": dataset, "method": method, "quality": quality, "seed": seed,
           "n_images": n_img, "model_mb": round(model_mb, 2)}

    if method == "opre":
        eps = args.eps if args.eps is not None else QUALITY[quality]["eps"]
        levels = args.levels if args.levels is not None else QUALITY[quality]["levels"]
        _dt = {"float32": torch.float32, "float16": torch.float16,
               "bfloat16": torch.bfloat16}
        mem, ids = run_opre(train_x[order], eps, levels,
                            device, chunk=args.chunk, dist_mb=args.dist_mb,
                            patch_dtype=_dt[args.patch_dtype],
                            dist_dtype=_dt[args.dist_dtype])
        stored = mem.count
        data_mb = mb(stored * patch_bits(levels) / 8 + n_img * PATCHES_PER_IMAGE * ID_BITS / 8)
        recon = reconstruct(ids, mem)
        del mem
        if device.type == "cuda":
            torch.cuda.empty_cache()
        acc = train_classifier(recon, train_y[order], num_classes, test_x, test_y,
                               device, args.epochs, lr, batch_size,
                               augment, seed, optimizer=optimizer)
        rec.update(stored_patches=stored, eps=eps, levels=levels)

    elif method == "none":
        data_mb = mb(n_img * RAW_IMAGE_BYTES)
        acc = train_classifier(train_x[order], train_y[order], num_classes, test_x,
                               test_y, device, args.epochs, lr,
                               batch_size, augment, seed, optimizer=optimizer)
        rec.update(stored_patches=n_img * PATCHES_PER_IMAGE)

    elif method in ("gdumb", "er"):
        if budget_mb is None:
            budget_mb = args.budget_mb or ARTICLE_BUDGET_MB.get((dataset, quality))
            if budget_mb is None:
                raise SystemExit(f"No default budget for dataset '{dataset}': run "
                                 "OPRE first and pass its data size via --budget-mb.")
        n_buf = budget_to_images(budget_mb)
        data_mb = mb(n_buf * RAW_IMAGE_BYTES)
        rec.update(budget_mb=round(budget_mb, 2), buffer_images=n_buf)
        if method == "gdumb":
            keep = gdumb_select(train_y, order, n_buf, seed)
            acc = train_classifier(train_x[keep], train_y[keep], num_classes,
                                   test_x, test_y, device, args.epochs, lr,
                                   batch_size, augment, seed, optimizer=optimizer)
        else:
            acc = run_er(train_x, train_y, order, n_buf, num_classes, test_x,
                         test_y, device, seed, args.er_lr, args.er_stream_bs,
                         args.er_replay_bs)
        rec.update(stored_patches=None)
    else:
        raise ValueError(method)

    rec.update(accuracy=round(acc, 2), data_mb=round(data_mb, 2),
               total_mb=round(data_mb + model_mb, 2),
               elapsed_s=round(time.time() - t0, 1))
    print(f"[done] {dataset} | {method}({quality}) | seed {seed} | "
          f"acc {acc:.2f}% | data {data_mb:.2f} MB | total {data_mb + model_mb:.1f} MB "
          f"| {rec['elapsed_s']}s")
    return rec


def append_result(path: Path, rec: dict):
    results = json.loads(path.read_text()) if path.exists() else []
    results.append(rec)
    path.write_text(json.dumps(results, indent=1))


# --------------------------------------------------------------------------
# Report (Tables 2, 3 and 4 of the article)
# --------------------------------------------------------------------------

def _label(rec):
    if rec["method"] in ("opre", "gdumb", "er"):
        return f"{rec['method']}_{rec['quality']}"
    return rec["method"]


def _mean_sd(vals):
    m = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals)) if len(vals) > 1 else 0.0
    return m, sd


def report(path: Path):
    if not path.exists():
        print(f"No results file at {path}")
        return
    results = json.loads(path.read_text())
    datasets = sorted({r["dataset"] for r in results})
    labels_order = ["none", "opre_low", "gdumb_low", "er_low"]
    labels = [l for l in labels_order if any(_label(r) == l for r in results)]
    labels += sorted({_label(r) for r in results} - set(labels))

    def cell(ds, lab, fn, fmt):
        runs = [r for r in results if r["dataset"] == ds and _label(r) == lab]
        return fmt(runs) if runs else "-"

    col = 28
    print("\n=== Final accuracy (%) -- mean (SD) over runs ===")
    print(f"{'method':<{col}}" + "".join(f"{d:>20}" for d in datasets))
    for lab in labels:
        row = f"{lab:<{col}}"
        for ds in datasets:
            row += f"{cell(ds, lab, None, lambda rs: '%.2f (%.2f) [n=%d]' % (*_mean_sd([r['accuracy'] for r in rs]), len(rs))):>20}"
        print(row)

    print("\n=== Number of stored patches (mean, M) ===")
    print(f"{'method':<{col}}" + "".join(f"{d:>20}" for d in datasets))
    for lab in labels:
        row = f"{lab:<{col}}"
        for ds in datasets:
            def fmt(rs):
                vals = [r["stored_patches"] for r in rs if r.get("stored_patches")]
                if not vals:
                    return "-"
                return f"{sum(vals) / len(vals) / 1e6:.3f} M"
            row += f"{cell(ds, lab, None, fmt):>20}"
        print(row)

    print("\n=== Memory cost (MB) -- total (data | model), Table-4 format ===")
    print(f"{'method':<{col}}" + "".join(f"{d:>28}" for d in datasets))
    for lab in labels:
        row = f"{lab:<{col}}"
        for ds in datasets:
            def fmt(rs):
                d_, _ = _mean_sd([r["data_mb"] for r in rs])
                m_, _ = _mean_sd([r["model_mb"] for r in rs])
                return f"{d_ + m_:.1f} ({d_:.2f} | {m_:.2f})"
            row += f"{cell(ds, lab, None, fmt):>28}"
        print(row)
    print()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="OPRE reproduction (see README_opre.md)")
    p.add_argument("--dataset", choices=["cifar10", "cifar100", "synthetic"], default=None,
                   help="dataset (default: cifar10; with --all: restricts the "
                        "full protocol to this dataset instead of both CIFARs)")
    p.add_argument("--synthetic-seed", type=int, default=0,
                   help="generation seed of the Appendix-B synthetic data "
                        "(fixed across run seeds)")
    p.add_argument("--method", choices=["opre", "none", "gdumb", "er"], default="opre")
    p.add_argument("--quality", choices=["low"], default="low")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--all", action="store_true",
                   help="run every configuration of the article + matched-budget "
                        "GDumb and ER, on both datasets")
    p.add_argument("--report", action="store_true", help="print the tables and exit")
    p.add_argument("--results", type=str, default="results.json")
    # OPRE
    p.add_argument("--eps", type=float, default=None, help="override epsilon")
    p.add_argument("--levels", type=int, default=None, help="override discretization levels")
    p.add_argument("--chunk", type=int, default=32, help="images per OPRE batch "
                   "(2048 patches at 32; lower it if VRAM is very tight)")
    p.add_argument("--dist-mb", type=float, default=256.0,
                   help="VRAM budget (MB) for the blockwise distance matrix")
    p.add_argument("--patch-dtype", choices=["float32", "float16", "bfloat16"],
                   default="float32",
                   help="storage dtype of the patch memory (float16 halves the "
                        "buffer, like the historical bfloat16 storage)")
    p.add_argument("--dist-dtype", choices=["float32", "float16", "bfloat16"],
                   default="float32",
                   help="dtype of the distance GEMM. float32 (default) makes "
                        "epsilon and the discretization the only information "
                        "losses; float16/bfloat16 roughly halves runtime and "
                        "memory traffic but adds noise to the epsilon threshold "
                        "(the historical implementation computed distances in "
                        "bfloat16). Pair with --patch-dtype of the same type "
                        "to avoid per-block casts.")
    # stream
    p.add_argument("--stream", choices=["class", "native", "shuffle"], default="class")
    p.add_argument("--shuffle-within-class", action="store_true")
    # classifier
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=None,
                   help="learning rate (default: 0.1 SGD for CIFAR, "
                        "0.001 Adam for synthetic)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="batch size (default: 128 for CIFAR, 64 for synthetic)")
    p.add_argument("--no-augment", action="store_true",
                   help="disable random crop + horizontal flip during CNN training")
    # baselines
    p.add_argument("--budget-mb", type=float, default=None,
                   help="data budget (MB) for gdumb/er; default: Table-4 value "
                        "for the chosen dataset/quality, or the measured OPRE "
                        "data size in --all mode")
    p.add_argument("--er-lr", type=float, default=0.05)
    p.add_argument("--er-stream-bs", type=int, default=10)
    p.add_argument("--er-replay-bs", type=int, default=10)
    # misc
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--limit-train", type=int, default=None, help="debug: truncate stream")
    p.add_argument("--quick", action="store_true",
                   help="smoke test: 1 seed, 5 epochs, 4000 images")
    return p


def main():
    args = build_parser().parse_args()
    if args.quick:
        args.seeds = args.seeds[:1]
        args.epochs = 5
        args.limit_train = args.limit_train or 4000
    res_path = Path(args.results)
    if args.report:
        report(res_path)
        return

    if args.all:
        datasets = [args.dataset] if args.dataset else ["cifar10", "cifar100"]
    else:
        datasets = [args.dataset or "cifar10"]
    for ds in datasets:
        data = load_data(ds, args.data_root, args)
        if args.all:
            # 1) all OPRE runs first
            opre_data_mb = []
            for seed in args.seeds:
                rec = run_one("opre", ds, "low", seed, data, args)
                append_result(res_path, rec)
                opre_data_mb.append(rec["data_mb"])
            budget = sum(opre_data_mb) / len(opre_data_mb)
            print(f"[budget] {ds}: mean OPRE data size over {len(opre_data_mb)} "
                  f"run(s) = {budget:.2f} MB -> used as GDumb/ER budget")
            # 2) no-compression upper bound
            for seed in args.seeds:
                append_result(res_path, run_one("none", ds, "low", seed, data, args))
            # 3) matched-budget baselines, budget = mean OPRE data size
            for bl in ("gdumb", "er"):
                for seed in args.seeds:
                    append_result(res_path,
                                  run_one(bl, ds, "low", seed, data, args,
                                          budget_mb=budget))
        else:
            for seed in args.seeds:
                append_result(res_path,
                              run_one(args.method, ds, args.quality, seed, data, args))
    report(res_path)


if __name__ == "__main__":
    main()
