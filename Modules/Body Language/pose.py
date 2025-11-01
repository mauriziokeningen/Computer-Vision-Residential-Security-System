import os, re, glob, random, zipfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --------------------- Configuración local ---------------------
ZIP_NAME       = "nturgbd_skeletons_s001_to_s017.zip"  
EXTRACT_DIR    = "NTU_dataset"                         
LIMIT_N_FILES  = 10000                                 

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(1)  

DATA_DIR = None
TARGET_FILE = None   # Ej: "S001C001P001R001A024.skeleton" o None para elegir aleatorio del test

# --------------------- Clases de seguridad ---------------------
ALLOWED_ACTIONS = {
    24, 26, 27, 28, 41, 42,     # kick/hit/punch/push/staggering/falling
    63, 65, 66, 67, 80, 81,     # variantes
    117, 119, 120               # variantes
}
OPTIONAL_ACTIONS = {23, 62, 116, 30, 69, 85}  # saludar/pointing (opcionales)
KEEP = ALLOWED_ACTIONS | OPTIONAL_ACTIONS

A2TEXT = {
  24:"Kicking", 26:"Hitting", 27:"Punching/Slapping", 28:"Pushing",
  41:"Staggering", 42:"Falling",
  63:"Kicking (var)", 65:"Hitting (var)", 66:"Punching/Slapping (var)", 67:"Pushing (var)",
  80:"Staggering (var)", 81:"Falling (var)",
  117:"Kicking (v2)", 119:"Hitting (v2)", 120:"Punching/Slapping (v2)",
  23:"Hand waving", 62:"Hand waving (v2)", 116:"Hand waving (v2)",
  30:"Pointing", 69:"Pointing (var)", 85:"Pointing (v2)"
}

# Grafo NTU-25
NTU25_EDGES = [
    (0,1),(1,20),(20,2),(2,3),                 # Columna/neck/head
    (20,4),(4,5),(5,6),(6,7),(7,21),(7,22),    # Brazo izquierdo + tips
    (20,8),(8,9),(9,10),(10,11),(11,23),(11,24),# Brazo derecho + tips
    (0,12),(12,13),(13,14),(14,15),            # Pierna izquierda
    (0,16),(16,17),(17,18),(18,19)             # Pierna derecha
]

# --------------------- Utilidades ---------------------
def ensure_extracted_from_zip():
    """
    Extrae del ZIP local los .skeleton (hasta LIMIT_N_FILES si se definió).
    Deja todo en EXTRACT_DIR y retorna la ruta base que contiene los .skeleton.
    """
    zip_path = Path(__file__).parent / ZIP_NAME
    out_dir  = Path(__file__).parent / EXTRACT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        raise FileNotFoundError(f"No se encontró el ZIP: {zip_path}")

    # Detectar si ya hay suficientes archivos extraídos
    existing = list(out_dir.rglob("*.skeleton"))
    if LIMIT_N_FILES is not None and len(existing) >= LIMIT_N_FILES:
        print(f"✅ Ya hay {len(existing)} skeletons en {out_dir}. No se vuelve a extraer.")
    elif LIMIT_N_FILES is None and len(existing) > 0:
        print(f"✅ Ya hay skeletons en {out_dir}. No se vuelve a extraer.")
    else:
        print("⏳ Extrayendo .skeleton desde el ZIP local...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            members = [m for m in z.namelist() if m.endswith(".skeleton")]
            if LIMIT_N_FILES is not None:
                members = members[:LIMIT_N_FILES]
            if not members:
                raise RuntimeError("No se encontraron .skeleton dentro del ZIP.")
            z.extractall(out_dir, members=members)
        print(f"✅ Archivos extraídos en: {out_dir}")

    # DATA_DIR esperado
    candidate = out_dir / "nturgb+d_skeletons"
    return str(candidate if candidate.exists() else out_dir)

def action_id_from_filename(path: str) -> int:
    m = re.search(r"A(\d{3})", os.path.basename(path))
    return int(m.group(1)) if m else -1

def list_skeleton_files(data_dir: str):
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.skeleton"), recursive=True))
    files = [f for f in files if action_id_from_filename(f) in KEEP]
    return files

# --------------------- Parser NTU ---------------------
def read_ntu_skeleton(path, max_persons=2, n_joints=25):
    """Devuelve x[T,P,25,3] con hasta `max_persons` y coords x,y,z."""
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip() != ""]
    cursor = 0
    n_frames = int(lines[cursor]); cursor += 1
    seq = np.zeros((n_frames, max_persons, n_joints, 3), dtype=np.float32)

    for t in range(n_frames):
        n_bodies = int(lines[cursor]); cursor += 1
        nb = min(n_bodies, max_persons)
        for _ in range(nb):
            cursor += 1  # body info
            n_j = int(lines[cursor]); cursor += 1
            for j in range(min(n_j, n_joints)):
                vals = lines[cursor].split()
                x, y, z = map(float, vals[:3])
                seq[t, _, j] = (x, y, z)
                cursor += 1
            # Joints extra (si los hay)
            for _skip in range(max(0, n_j - n_joints)):
                cursor += 1
        # Cuerpos extra (si los hay)
        for _ in range(max(0, n_bodies - nb)):
            cursor += 1  # body info
            n_j_skip = int(lines[cursor]); cursor += 1
            cursor += n_j_skip
    return seq

def valid_frame_mask(x):
    return (np.abs(x).sum(axis=(1,2,3)) > 0)

def active_person_index(x):
    T,P,J,C = x.shape
    scores = []
    for p in range(P):
        xp = x[:, p, :, :2]
        if np.abs(xp).sum() == 0:
            scores.append(-np.inf); continue
        v = np.diff(xp, axis=0)                      # [T-1, J, 2]
        scores.append(float(np.nanmean(np.linalg.norm(v, axis=2))))
    best = int(np.argmax(scores)) if len(scores) else 0
    return best if scores[best] != -np.inf else 0

# ---------- Normalización ----------
def center_scale(x):
    """Centra en pelvis(0) y escala por mediana distancia hombros(4,8)."""
    mask = valid_frame_mask(x)
    if not mask.any():
        return x.copy()
    x = x.copy()
    base = x[..., 0, :]                   # pelvis
    x = x - base[..., None, :]
    L = np.linalg.norm(x[..., 4, :] - x[..., 8, :], axis=-1)  # [T,1]
    L_valid = L[mask]
    scale = np.median(L_valid) if L_valid.size > 0 else 1.0
    scale = max(scale, 1e-6)
    return x / scale

def standardize_per_sample(x):
    x = x.copy()
    mask = valid_frame_mask(x)
    if not mask.any():
        return x
    xv = x[mask]                         # [Tv,1,25,3]
    joint_mask = (np.abs(xv).sum(axis=-1) > 0)  # [Tv,1,25]
    vals = xv[joint_mask]
    if vals.size == 0:
        return x
    vals = vals.reshape(-1, 3)
    mu = vals.mean(axis=0, keepdims=True)
    sigma = vals.std(axis=0, keepdims=True) + 1e-6
    x = (x - mu) / sigma
    return x

def preprocess(x):
    x = center_scale(x)
    x = standardize_per_sample(x)
    return x

# ---------------------- Aumentos ----------------------
def aug_yaw_scale(x, max_deg=10.0, scale_jitter=0.05):
    if np.abs(x).sum() == 0:
        return x
    theta = np.deg2rad(np.random.uniform(-max_deg, max_deg))
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    xy = x[..., :2]
    xy = xy @ R.T
    sc = 1.0 + np.random.uniform(-scale_jitter, scale_jitter)
    xy = xy * sc
    x2 = x.copy()
    x2[..., :2] = xy
    return x2

def pad_with_last_valid(x, T):
    if x.shape[0] >= T:
        return x[:T]
    mask = valid_frame_mask(x)
    last_idx = np.where(mask)[0]
    last_idx = int(last_idx[-1]) if last_idx.size else x.shape[0]-1
    pad_frames = np.repeat(x[last_idx:last_idx+1], T - x.shape[0], axis=0)
    return np.concatenate([x, pad_frames], axis=0)

def best_frame_safe(raw1: np.ndarray) -> int:
    arr = np.asarray(raw1, dtype=np.float32)
    if arr.ndim != 4 or arr.shape[0] <= 1:
        return 0
    v = np.diff(arr[..., :2], axis=0)  # [T-1,1,25,2]
    if v.size == 0:
        return 0
    energ = np.sqrt((v ** 2).sum(axis=-1))  # [T-1,1,25]
    energ = np.sum(energ, axis=(1, 2))      # [T-1]
    if energ.size == 0 or np.all(~np.isfinite(energ)):
        return 0
    return int(np.nanargmax(energ))

# ---------------------- Dataset ----------------------
class NTUDataset(Dataset):
    def __init__(self, files, action2idx, max_len=64, augment=True):
        self.files      = files
        self.action2idx = action2idx  # <<--- evita globals en workers
        self.max_len    = max_len
        self.augment    = augment

    def __len__(self): 
        return len(self.files)

    def _load(self, path):
        raw = read_ntu_skeleton(path, max_persons=2)  # [T,P,25,3]
        pidx = active_person_index(raw)
        x = raw[:, pidx:pidx+1]                      # [T,1,25,3]
        T = x.shape[0]

        # recorte/padding temporal con jitter
        if T >= self.max_len:
            start = 0
            if self.augment:
                start = np.random.randint(0, T - self.max_len + 1)
            x = x[start:start+self.max_len]
        else:
            x = pad_with_last_valid(x, self.max_len)

        # aumentos geométricos
        if self.augment:
            x = aug_yaw_scale(x)

        x = preprocess(x)
        x = np.transpose(x, (0,2,3,1))  # [T,25,3,1]
        return x

    def __getitem__(self, idx):
        path = self.files[idx]
        a = action_id_from_filename(path)
        y = self.action2idx[a]
        try:
            x = self._load(path)
        except Exception:
            x = np.zeros((64,25,3,1), dtype=np.float32)  # fallback defensivo
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long), path

def make_splits(files, action2idx, test_size=0.2):
    labels = [action2idx[action_id_from_filename(p)] for p in files]
    ok_to_stratify = all(v >= 2 for v in Counter(labels).values())
    if ok_to_stratify:
        X_tr, X_te = train_test_split(
            files, test_size=test_size, random_state=SEED, stratify=labels
        )
    else:
        X_tr, X_te = train_test_split(files, test_size=test_size, random_state=SEED, shuffle=True)
    return X_tr, X_te

# ---------------------- Modelo ----------------------
class TinyPoseBiGRU(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        C, J, P = 3, 25, 1
        in_dim = C * J * P
        self.fc_in = nn.Linear(in_dim, 128)
        self.rnn   = nn.GRU(128, 128, batch_first=True, bidirectional=True)
        self.norm  = nn.LayerNorm(256)
        self.drop  = nn.Dropout(0.3)
        self.head  = nn.Linear(256*2, n_classes)

    def forward(self, x):            # x: [B,T,25,3,1]
        B,T,_,_,_ = x.shape
        x = x.contiguous().view(B, T, -1)   # [B,T,75]
        x = torch.relu(self.fc_in(x))
        out,_ = self.rnn(x)                 # [B,T,256]
        out = self.norm(out)
        # temporal pooling
        mean_pool = out.mean(dim=1)
        max_pool  = out.max(dim=1).values
        h = torch.cat([mean_pool, max_pool], dim=1)
        h = self.drop(h)
        return self.head(h)

# ---------------------- Entrenamiento / Evaluación ----------------------
def run_epoch(model, loader, criterion, optimizer, train=True, device="cpu"):
    model.train(train)
    total = correct = 0
    loss_sum = 0.0
    for X, y, _ in loader:
        X = X.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True)
        logits = model(X)
        loss = criterion(logits, y)
        if train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        preds = logits.argmax(1)
        total  += y.size(0)
        correct += (preds == y).sum().item()
        loss_sum += loss.item() * y.size(0)
    return loss_sum/total, correct/total

def evaluate_preds(model, loader, device="cpu"):
    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for X, y, _ in loader:
            X = X.to(device).float()
            y = y.to(device)
            logits = model(X)
            p = logits.argmax(1)
            all_y.extend(y.cpu().tolist())
            all_p.extend(p.cpu().tolist())
    return np.array(all_y), np.array(all_p)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # 1) Extraer del ZIP local (si hace falta) y fijar DATA_DIR
    DATA_DIR = ensure_extracted_from_zip()
    print("DATA_DIR =", DATA_DIR)

    # 2) Descubrir archivos y clases presentes
    all_files = list_skeleton_files(DATA_DIR)
    if TARGET_FILE is not None:
        tgt = os.path.join(DATA_DIR, TARGET_FILE)
        if os.path.exists(tgt) and tgt not in all_files:
            all_files.append(tgt)

    present_actions = sorted({action_id_from_filename(f) for f in all_files} & KEEP)
    if len(present_actions) == 0:
        raise RuntimeError("No hay acciones válidas en los archivos filtrados. Revisa DATA_DIR y KEEP.")
    ACTIONS     = present_actions
    ACTION2IDX  = {a:i for i,a in enumerate(ACTIONS)}
    IDX2ACTION  = {i:a for a,i in ACTION2IDX.items()}

    print("Acciones presentes:", {a: A2TEXT.get(a, f"A{a:03d}") for a in ACTIONS})
    print("#Archivos filtrados:", len(all_files))

    # 3) Train / Test split
    train_files, test_files = make_splits(all_files, ACTION2IDX, test_size=0.2)
    train_ds = NTUDataset(train_files, action2idx=ACTION2IDX, max_len=64, augment=True)
    test_ds  = NTUDataset(test_files,  action2idx=ACTION2IDX, max_len=64, augment=False)

    # 4) Pesos por clase y sampler balanceado
    n_classes = len(ACTIONS)
    counts = np.zeros(n_classes, dtype=np.int64)
    for p in train_files:
        ai = ACTION2IDX[action_id_from_filename(p)]
        counts[ai] += 1

    counts_safe = counts.copy()
    if (counts_safe > 0).any():
        min_pos = counts_safe[counts_safe > 0].min()
        counts_safe[counts_safe == 0] = min_pos
    else:
        counts_safe[:] = 1

    class_weights = (counts_safe.sum() / (n_classes * counts_safe)).astype(np.float32)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32)

    sample_weights = np.array(
        [float(class_weights[ACTION2IDX[action_id_from_filename(p)]]) for p in train_files],
        dtype=np.float64
    )
    sample_weights_t = torch.as_tensor(sample_weights, dtype=torch.double)
    sampler = WeightedRandomSampler(weights=sample_weights_t,
                                    num_samples=len(sample_weights_t),
                                    replacement=True)

    pin = torch.cuda.is_available()
    # Si en tu Windows prefieres evitar workers, pon num_workers=0.
    train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler, shuffle=False,
                              num_workers=2, pin_memory=pin, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False,
                              num_workers=2, pin_memory=pin, drop_last=False)

    # 5) Modelo / optimizadores
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyPoseBiGRU(n_classes).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_t.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    # 6) Entrenamiento
    EPOCHS = 15
    for ep in range(1, EPOCHS+1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, True, device)
        te_loss, te_acc = run_epoch(model, test_loader,  criterion, optimizer, False, device)
        scheduler.step()
        print(f"[Ep {ep:02d}] train_loss={tr_loss:.3f} train_acc={tr_acc:.2f} | val_loss={te_loss:.3f} val_acc={te_acc:.2f}")

    # 7) Reporte de clasificación en el set de prueba
    all_y, all_p = evaluate_preds(model, test_loader, device)
    target_names = [A2TEXT.get(a, f"A{a:03d}") for a in ACTIONS]
    print("\nReporte de clasificación:\n",
          classification_report(all_y, all_p, digits=3, target_names=target_names))

    # 8) Predicción + visualización en un archivo del test
    import random as _r
    assert len(test_files) > 0, "No hay archivos de test."
    if TARGET_FILE is None:
        sysrand = _r.SystemRandom()
        target = os.path.basename(sysrand.choice(test_files))
    else:
        target = os.path.basename(TARGET_FILE)

    target_path = os.path.join(DATA_DIR, target)
    print("\nArchivo objetivo (demo):", os.path.basename(target_path))

    true_A    = action_id_from_filename(target_path)
    true_text = A2TEXT.get(true_A, f"A{true_A:03d}")
    print(f"Real: A{true_A:03d} ({true_text})")

    tmp_ds = NTUDataset([target_path], action2idx=ACTION2IDX, max_len=64, augment=False)
    x, y_true, _ = tmp_ds[0]
    with torch.no_grad():
        logits   = model(x.unsqueeze(0).to(device).float())
        prob     = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        top3     = prob.argsort()[-3:][::-1]
        pred_idx = int(top3[0])

    # asegurar mapeo actualizado
    IDX2ACTION = {i:a for a,i in ACTION2IDX.items()}
    pred_A    = IDX2ACTION[pred_idx]
    pred_text = A2TEXT.get(pred_A, f"A{pred_A:03d}")
    print("Top-3 predicciones:")
    for k, i in enumerate(top3):
        a = IDX2ACTION[i]
        print(f"  {k+1}) idx={i:2d}  A{a:03d}  {A2TEXT.get(a, str(a))}  p={prob[i]:.3f}")

    # Visualizar un frame representativo
    raw  = read_ntu_skeleton(target_path)
    pidx = active_person_index(raw)
    raw1 = raw[:, pidx:pidx+1]
    t0   = best_frame_safe(raw1)
    pts  = preprocess(raw1)
    pts0 = pts[t0,0,:,:2]

    plt.figure(figsize=(5,6))
    plt.scatter(pts0[:,0], pts0[:,1])
    for i,j in NTU25_EDGES:
        plt.plot([pts0[i,0], pts0[j,0]], [pts0[i,1], pts0[j,1]])
    plt.gca().invert_yaxis()
    plt.title(f"{os.path.basename(target_path)} | frame={t0}\nPred: A{pred_A:03d} ({pred_text}) | Real: A{true_A:03d} ({true_text})")
    plt.tight_layout(); plt.show()
