# %% [markdown]
# # UAVid — veri incelemeden ilk modele
# Aynı `.ipynb` yerel Jupyter ve Kaggle'da bağımsız çalışır; repo modüllerini import etmez.
# Hücreleri yukarıdan aşağı çalıştırın. Başlangıçta eğitim kapalıdır.
# Kernel → Restart Kernel and Run All Cells, değişkenleri temizleyerek başlamak içindir.
# Kaggle: bu dosyayı **Import Notebook** ile yükleyin, `emirhanuludogan/paaaaa` verisini ekleyin.
# GPU isterseniz oturum ayarlarında açın. Yerelde masaüstündeki Jupyter başlatıcısını kullanın.
#
# Problem: her piksele 8 sınıftan birini atamak. Referans etiket bir model tahmini değildir.
# Resmî kaynak: https://uavid.nl/ — veri CC BY-NC-SA 4.0; burada veri dağıtılmaz.
# Atıf: Lyu ve diğerleri (2020), DOI: https://doi.org/10.1016/j.isprsjprs.2020.05.009
# 23 Kasım topluluk kopyası sonuçları bu notebook'un resmî veri sonuçları değildir.
# Kullanıcının önceki Kaggle çalışmasında 3 epoch tamamlandı; en iyi val mIoU %4,6915 (epoch 1).
# Bu aktarılmış sonuç bu yeni dosyanın yürütülmüş çıktısı değildir; checkpoint yerelde yoktur.
# Ayrıntılar repo günlüğünde bulunur. Bu dosya yeniden deney yapmaya ve hataları incelemeye yarar.

# %% [markdown]
# ## 1. Ortam ve açık ayarlar
# Bu hücre paket kurmaz. Eksik bağımlılıkta yerelde `python -m pip install -r requirements-notebook.txt`
# kullanın. Torch eksikse https://pytorch.org/get-started/locally/ üzerinden Windows/CPU veya
# uygun CUDA paketini seçin. Kaggle'ın çalışan Torch kurulumunu değiştirmeyin.
# `VERI_KOKU` boşsa yalnız bu projenin bilinen veri yolları aranır; eski topluluk kopyası aranmaz.
# Birden fazla kaynak varsa otomatik seçim durur; yolu açıkça yazın.

# %%
import os
import sys
import csv
import copy
import json
import math
import random
import hashlib
import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

try:
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
except ImportError as exc:
    raise RuntimeError("Eksik paket: yerelde requirements-notebook.txt kurulumunu tamamlayın; Kaggle'da Python ortamını kontrol edin.") from exc

VERI_KOKU = ""  # İsteğe bağlı: r"D:\veri\uavid"; train ve val içeren kök.
CIHAZ_SECIMI = "auto"  # auto, cpu veya cuda:0
BOYUT = (512, 320)  # (genişlik, yükseklik); ikisi de 4'ün katı olmalı.
YOK_SAY = 255
TOHUM = 42
BATCH = 4
EPOCH_SAYISI = 3
OGRENME_ORANI = 0.001
TAM_DENETIM = False  # True: 270 görüntü ve etiketinin tamamını oku, RGB say.
SABIT_REFERANSI_HESAPLA = True  # Model eğitmeden validation üzerinde sürekli bina tahmini.
EGITIMI_BASLAT = False  # Yalnız hazır olduğunuzda True yapıp eğitim hücresini çalıştırın.
CALISMA_MODU = "reset"  # reset: sıfırdan; resume: aşağıdaki kendi checkpoint'inizden.
DEVAM_CHECKPOINT = ""
HARICI_CHECKPOINT = ""  # Bu notebook biçiminde, güvendiğiniz kendi checkpoint'inizin yolu.

SINIFLAR = ("clutter", "building", "road", "tree", "low_vegetation", "moving_car", "static_car", "human")
RENKLER = ((0, 0, 0), (128, 0, 0), (128, 64, 128), (0, 128, 0),
           (128, 128, 0), (64, 0, 128), (192, 0, 192), (64, 64, 0))
SINIF_HARITASI = {name: {"id": i, "rgb": list(RENKLER[i])} for i, name in enumerate(SINIFLAR)}
KAGGLE = Path("/kaggle/input").is_dir()
CIKTI_KOKU = Path("/kaggle/working/uavid-runs") if KAGGLE else Path.cwd().resolve() / "outputs" / "uavid-runs"
if CIHAZ_SECIMI == "auto":
    cihaz = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    cihaz = torch.device(CIHAZ_SECIMI)
    if cihaz.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA seçildi ama Torch CUDA kullanamıyor; cpu/auto seçin veya kurulumu kontrol edin.")
print({"python": sys.version.split()[0], "torch": torch.__version__, "cuda_build": torch.version.cuda,
       "cihaz": str(cihaz), "gpu": torch.cuda.get_device_name(0) if cihaz.type == "cuda" else None})
print("Çıktı kökü:", CIKTI_KOKU, "| Eğitim varsayılan olarak kapalı.")

# %% [markdown]
# ## 2. Kaynağı ve çiftleri bul
# Orijinal train/val ayrımı korunur. Rastgele kare bölme yapılmaz.
# Resmî kopyada beklenen: train 20 sekans/200 çift; val 7 sekans/70 çift.
# Paketlerin iki kez iç içe açılması desteklenir, birden çok geçerli kopya varsa hata verilir.

# %%
if "SINIF_HARITASI" not in globals():
    raise RuntimeError("Önce 1. Ortam ve açık ayarlar hücresini çalıştırın.")


def split_klasoru(root, split):
    name = f"uavid_{split}"
    candidates = [root] if root.name == name else []
    candidates += [p for p in root.rglob(name) if p.is_dir()]
    candidates = {p.resolve() for p in candidates
                  if any(q.is_dir() and q.name.startswith("seq") and (q / "Images").is_dir()
                         for q in p.iterdir())}
    if len(candidates) != 1:
        raise ValueError(f"{split}: tek bir geçerli {name} bekleniyor, bulundu: {sorted(map(str, candidates))}")
    return next(iter(candidates))


def veri_kokunu_bul(explicit=""):
    override = explicit.strip() or os.environ.get("UAVID_DATA_ROOT", "").strip()
    if override:
        root = Path(override).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Veri kökü yok: {root}. Önce arşivleri hazırlayın; kullanım belgesine bakın.")
        split_klasoru(root, "train")
        split_klasoru(root, "val")
        return root
    if KAGGLE:
        paths = [Path("/kaggle/input/datasets/emirhanuludogan/paaaaa"), Path("/kaggle/input/paaaaa")]
    else:
        paths = [p / "data" / "official" for p in (Path.cwd(), *Path.cwd().parents)]
    candidates = {p.resolve() for p in paths if p.is_dir()}
    if len(candidates) != 1:
        raise RuntimeError(f"Tek veri kökü bulunamadı: {sorted(map(str, candidates))}. VERI_KOKU ayarını doldurun; Kaggle'da emirhanuludogan/paaaaa ekleyin.")
    root = next(iter(candidates))
    split_klasoru(root, "train")
    split_klasoru(root, "val")
    return root


def ciftleri_bul(root, split):
    split_root = split_klasoru(root, split)
    pairs, counts = [], {}
    for seq in sorted(p for p in split_root.iterdir() if p.is_dir() and p.name.startswith("seq")):
        images, labels = seq / "Images", seq / "Labels"
        if not images.is_dir() or not labels.is_dir():
            raise ValueError(f"Eksik Images/Labels: {seq}")
        image_map = {p.name: p for p in images.iterdir() if p.is_file() and p.suffix.lower() == ".png"}
        label_map = {p.name: p for p in labels.iterdir() if p.is_file() and p.suffix.lower() == ".png"}
        if not image_map or image_map.keys() != label_map.keys():
            raise ValueError(f"Boş veya eşleşmeyen sekans {seq.name}: görüntüsüz={sorted(label_map.keys()-image_map.keys())}, etiketsiz={sorted(image_map.keys()-label_map.keys())}")
        counts[seq.name] = len(image_map)
        pairs += [(image_map[n], label_map[n]) for n in sorted(image_map)]
    if not pairs:
        raise ValueError(f"Boş split: {split}")
    return pairs, counts


veri_koku = veri_kokunu_bul(VERI_KOKU)
train_pairs, train_seqs = ciftleri_bul(veri_koku, "train")
val_pairs, val_seqs = ciftleri_bul(veri_koku, "val")
if set(train_seqs) & set(val_seqs):
    raise ValueError("Train/val sekans adları örtüşüyor; kaynağı kontrol edin.")
expected_train = {f"seq{i}" for i in [*range(1, 16), *range(31, 36)]}
expected_val = {f"seq{i}" for i in [*range(16, 21), 36, 37]}
if (set(train_seqs) != expected_train or set(val_seqs) != expected_val
        or set(train_seqs.values()) != {10} or set(val_seqs.values()) != {10}):
    raise ValueError("Bu notebook için beklenen resmî 200/70 çift ve sekans manifesti uyuşmuyor; kaynağı araştırın.")
print("Kullanılan kök:", veri_koku)
print("Train:", len(train_seqs), "sekans", len(train_pairs), "çift", train_seqs)
print("Val:", len(val_seqs), "sekans", len(val_pairs), "çift", val_seqs)
# This is a portable structure fingerprint, NOT an image-content hash or leak test.
manifest = [(split, p.parent.parent.name, p.name, p.stat().st_size, q.stat().st_size)
            for split, pairs in (("train", train_pairs), ("val", val_pairs)) for p, q in pairs]
VERI_MANIFESTI = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()

# %% [markdown]
# ## 3. İsteğe bağlı tam denetim ve görsel örnek
# `TAM_DENETIM=True` bütün dosyaları açar ve sınıf piksel alanlarını sayar; nesne sayısı değildir.
# Kontrol sızıntısız ayrımı veya her anotasyonun doğruluğunu kanıtlamaz.
# Dört örnekte overlay yalnız kaba hizalama kontrolüdür. Tam denetim kapalıyken
# bu oturumun bütün maskeleri doğruladığı söylenemez.

# %%
if "train_pairs" not in globals():
    raise RuntimeError("Önce 1 ve 2 numaralı hücreleri sırayla çalıştırın.")


def rgb_sinif(mask_rgb):
    # Packed RGB lookup is faster than eight full RGB comparisons for 4K masks.
    lut = np.full(256**3, YOK_SAY, dtype=np.uint8)
    for i, (r, g, b) in enumerate(RENKLER):
        lut[(r << 16) | (g << 8) | b] = i
    packed = mask_rgb[..., 0].astype(np.uint32) << 16
    packed |= mask_rgb[..., 1].astype(np.uint32) << 8
    packed |= mask_rgb[..., 2].astype(np.uint32)
    ids = lut[packed]
    if np.any(ids == YOK_SAY):
        raise ValueError(f"Tanımsız RGB etiketi: {int(np.sum(ids == YOK_SAY))} piksel")
    return ids


def tam_denetle(pairs):
    histogram = np.zeros(8, dtype=np.int64)
    sizes = Counter()
    for image_path, label_path in pairs:
        with Image.open(image_path) as image, Image.open(label_path) as label:
            image.load()
            label.load()
            if image.size != label.size:
                raise ValueError(f"Boyut farkı: {image_path}")
            if label.mode not in ("RGB", "P"):
                raise ValueError(f"Desteklenmeyen etiket modu: {label.mode}")
            sizes[f"{image.width}x{image.height}"] += 1
            ids = rgb_sinif(np.asarray(label.convert("RGB")))
            histogram += np.bincount(ids.ravel(), minlength=8)
    return {"pairs": len(pairs), "sizes": dict(sizes), "class_pixels": histogram.tolist(),
            "class_percent": dict(zip(SINIFLAR, (histogram / histogram.sum() * 100).tolist())),
            "unknown_pixels": 0, "read_errors": 0, "size_mismatches": 0}


denetim = {}
if TAM_DENETIM:
    for split, pairs in (("train", train_pairs), ("val", val_pairs)):
        denetim[split] = tam_denetle(pairs)
    print(json.dumps(denetim, ensure_ascii=False, indent=2))
else:
    print("Tam dosya/RGB denetimi bu oturumda çalıştırılmadı.")

fig, axes = plt.subplots(2, 2, figsize=(12, 7))
for ax, (split, seq) in zip(axes.flat, (("train", "seq1"), ("train", "seq9"), ("val", "seq16"), ("val", "seq37"))):
    pairs = train_pairs if split == "train" else val_pairs
    p, q = next((p, q) for p, q in pairs if p.parent.parent.name == seq)
    with Image.open(p) as image, Image.open(q) as label:
        if image.size != label.size:
            raise ValueError(f"Boyut farkı: {p}")
        im = np.asarray(image.convert("RGB").resize((640, 360), Image.Resampling.BILINEAR))
        mask = np.asarray(label.convert("RGB").resize((640, 360), Image.Resampling.NEAREST))
    ax.imshow((0.65 * im + 0.35 * mask).astype(np.uint8))
    ax.set_title(f"{split}/{seq}/{p.name} — referans overlay")
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Oranı koruyan ortak dönüşüm
# Genişlik/yükseklik sırası `(512, 320)`. Görüntü BILINEAR, sınıf maskesi NEAREST ile küçülür.
# Aynı merkezli letterbox yerleşimi uygulanır. Görüntü dolgusu siyah, hedef dolgusu 255'tir.
# 255 bir sınıf değildir; kayıp ve metrik dışında kalır. İnce nesneler küçülürken kaybolabilir.
# RGB→ID sırası bu projenin sabit sırasıdır; toolkit'in enumerate sırasını kullanmayın.

# %%
if "rgb_sinif" not in globals() or "train_pairs" not in globals():
    raise RuntimeError("Önce ortam, veri ve denetim hücrelerini sırayla çalıştırın.")


def donustur(image, label):
    if image.size != label.size:
        raise ValueError("Kaynak görüntü/etiket boyutları farklı.")
    if label.mode not in ("RGB", "P"):
        raise ValueError(f"Desteklenmeyen etiket modu: {label.mode}")
    tw, th = BOYUT
    if tw % 4 or th % 4 or min(tw, th) <= 0:
        raise ValueError("Hedef boyutlar pozitif ve 4'ün katı olmalı.")
    # Validate ALL source pixels before resizing can hide an unknown color.
    source_ids = rgb_sinif(np.asarray(label.convert("RGB")))
    scale = min(tw / image.width, th / image.height)
    w, h = max(1, round(image.width * scale)), max(1, round(image.height * scale))
    left, top = (tw - w) // 2, (th - h) // 2
    resized_image = np.asarray(image.convert("RGB").resize((w, h), Image.Resampling.BILINEAR))
    resized_ids = np.asarray(Image.fromarray(source_ids).resize((w, h), Image.Resampling.NEAREST))
    x = np.zeros((th, tw, 3), dtype=np.float32)
    y = np.full((th, tw), YOK_SAY, dtype=np.int64)
    x[top:top+h, left:left+w] = resized_image.astype(np.float32) / 255.0
    y[top:top+h, left:left+w] = resized_ids
    return torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.from_numpy(y)


class UAVidDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = list(pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        p, q = self.pairs[index]
        with Image.open(p) as image, Image.open(q) as label:
            return donustur(image, label)


def tohumla():
    random.seed(TOHUM)
    np.random.seed(TOHUM)
    torch.manual_seed(TOHUM)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TOHUM)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def yukleyiciler():
    generator = torch.Generator().manual_seed(TOHUM)
    train = DataLoader(UAVidDataset(train_pairs), batch_size=BATCH, shuffle=True, num_workers=0, generator=generator)
    val = DataLoader(UAVidDataset(val_pairs), batch_size=BATCH, shuffle=False, num_workers=0)
    return train, val, generator


train_loader, val_loader, _ = yukleyiciler()
xb, yb = next(iter(train_loader))
assert xb.dtype == torch.float32 and yb.dtype == torch.int64
assert tuple(xb.shape[1:]) == (3, BOYUT[1], BOYUT[0])
assert tuple(yb.shape[1:]) == (BOYUT[1], BOYUT[0])
assert 0 <= xb.min() <= xb.max() <= 1
assert set(yb.unique().tolist()) <= set(range(8)) | {YOK_SAY}
print("x:", tuple(xb.shape), xb.dtype, "y:", tuple(yb.shape), yb.dtype, "ID:", yb.unique().tolist())

# %% [markdown]
# ## 5. Aynı başlangıç modeli ve değerlendirme
# Sıfırdan öğrenen 34.008 parametreli küçük ağ; ön eğitim yoktur. CE padding'i yok sayar.
# Confusion matrix: satır gerçek sınıf, sütun tahmin. IoU = TP/(TP+FP+FN).
# Birleşimi 0 olan sınıf NaN, mIoU yalnız birleşimi pozitif sınıfların ortalamasıdır.
# Kayıp geçerli piksel sayısıyla ağırlıklandırılır; batch ortalamalarının basit ortalaması değildir.

# %%
if "xb" not in globals():
    raise RuntimeError("Önce 4. Ortak dönüşüm ve yükleyici hücresini çalıştırın.")


class BaslangicModeli(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 2, stride=2), nn.ReLU(),
            nn.Conv2d(16, 8, 1))


def confusion_ekle(matrix, target, prediction):
    valid = target != YOK_SAY
    t, p = target[valid], prediction[valid]
    if t.numel() and (t.min() < 0 or t.max() > 7 or p.min() < 0 or p.max() > 7):
        raise ValueError("Metrikte geçersiz sınıf ID'si.")
    matrix += torch.bincount(8 * t + p, minlength=64).reshape(8, 8).cpu()


def iou_hesapla(matrix):
    matrix = matrix.to(torch.float64)
    diagonal = matrix.diag()
    union = matrix.sum(0) + matrix.sum(1) - diagonal
    iou = torch.full((8,), float("nan"), dtype=torch.float64)
    present = union > 0
    iou[present] = diagonal[present] / union[present]
    return iou, iou[present].mean().item() if present.any() else float("nan")


def bir_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, valid_count = 0.0, 0
    matrix = torch.zeros((8, 8), dtype=torch.int64)
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(cihaz), y.to(cihaz)
            count = int((y != YOK_SAY).sum().item())
            if count == 0:
                continue
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Sonlu olmayan kayıp.")
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * count
            valid_count += count
            confusion_ekle(matrix, y, logits.detach().argmax(1))
    if valid_count == 0:
        raise ValueError("Epoch'ta geçerli hedef piksel yok.")
    iou, miou = iou_hesapla(matrix)
    return {"loss": total_loss / valid_count, "miou": miou,
            "iou": iou.tolist(), "confusion": matrix.tolist(), "valid_pixels": valid_count}


tohumla()
kontrol_modeli = BaslangicModeli().to(cihaz)
assert sum(p.numel() for p in kontrol_modeli.parameters()) == 34008
with torch.no_grad():
    logits = kontrol_modeli(xb.to(cihaz))
    kontrol_loss = nn.CrossEntropyLoss(ignore_index=YOK_SAY)(logits, yb.to(cihaz)).item()
assert tuple(logits.shape) == (len(xb), 8, BOYUT[1], BOYUT[0])
print("Parametre: 34008 | Çıkış:", tuple(logits.shape), "| Eğitimsiz CE:", kontrol_loss)
del kontrol_modeli, logits

# %% [markdown]
# ## 6. Sürekli bina tahmini: basit bir karşılaştırma
# Model yerine her geçerli piksele bina ID=1 veriyoruz. Bu bir öğrenilmiş model değildir.
# Aynı 512×320 dönüşümü ve ignore=255 kullanıldığı için yeni eğitimle metrikler karşılaştırılabilir.
# Özgün maskelerdeki %33,976 bina oranından hesaplanan yaklaşık mIoU %4,247'dir (8 sınıf varsa).
# Aşağıdaki gerçek hesap, küçültme nedeniyle bu yaklaşık sayıyla biraz farklı olabilir.
# 3 epoch Kaggle aktarımı: val mIoU %4,6915 → %4,2865 → %4,2831; son epoch'ta
# bina IoU %34,27, diğer 7 sınıf IoU 0. Kayıp azalması kalite artışı demek değildir.
# Bina ağırlıklı tahmin şüphesi histogram ve tahmin görselleriyle sınanmalıdır.

# %%
if "UAVidDataset" not in globals() or "iou_hesapla" not in globals():
    raise RuntimeError("Önce veri dönüşümü ve model/metrik hücrelerini çalıştırın.")


def sabit_bina_referansi(pairs):
    matrix = torch.zeros((8, 8), dtype=torch.int64)
    for _, target in UAVidDataset(pairs):
        confusion_ekle(matrix, target, torch.ones_like(target))
    iou, miou = iou_hesapla(matrix)
    return {"miou": miou, "iou": dict(zip(SINIFLAR, iou.tolist())), "confusion": matrix.tolist()}


if SABIT_REFERANSI_HESAPLA:
    sabit_sonuc = sabit_bina_referansi(val_pairs)
    print("Öğrenme yok — sabit bina validation referansı:", json.dumps(sabit_sonuc, ensure_ascii=False))
else:
    print("Sabit bina referansı bu oturumda hesaplanmadı.")

# %% [markdown]
# ## 7. Eğitim — açıkça başlat, reset veya resume seç
# `EGITIMI_BASLAT=True`, `CALISMA_MODU="reset"`: her çalıştırmada yeni ağ, Adam ve
# benzersiz bir çıktı klasörü kurulur. Hücreyi tekrar çalıştırmak sessizce eğitime devam etmez.
# `resume`: yalnız kendinizin ürettiği checkpoint yolunu yazın; 3, **toplam hedef epoch** sayısıdır.
# Örneğin epoch 2'den devam ve hedef 3 ise bir epoch daha çalışır. Sürdürme yeni çıktı klasörüne
# yazılır; eski sonuçlar ezilmez. GPU/CPU farkı nedeniyle bit düzeyinde aynı sonuç garanti edilmez.
# En iyi model validation mIoU ile seçilir. Validation bağımsız test başarısı değildir.
# Çıktılar: `history.csv`, `last.pt`, `best.pt`, `config.json`, `audit.json` (denetim açıldıysa).
# Eğitim süresi boyunca hücre aktiftir; her epoch sonunda kayıp ve bütün sınıfların IoU'su basılır.

# %%
if "BaslangicModeli" not in globals() or "VERI_MANIFESTI" not in globals():
    raise RuntimeError("Önce veri ve model hücrelerini yukarıdan aşağı çalıştırın.")


def checkpoint_uyumunu_kontrol_et(checkpoint):
    if (checkpoint["class_map"] != SINIF_HARITASI or tuple(checkpoint["target_size"]) != BOYUT
            or checkpoint["ignore_index"] != YOK_SAY or checkpoint["data_manifest"] != VERI_MANIFESTI
            or checkpoint.get("seed") != TOHUM or checkpoint.get("batch_size") != BATCH):
        raise ValueError("Checkpoint sınıf/boyut/veri/tohum/batch ayarları uyuşmuyor.")


def csv_yaz(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def egit():
    if CALISMA_MODU not in ("reset", "resume"):
        raise ValueError("CALISMA_MODU reset veya resume olmalı.")
    if EPOCH_SAYISI < 1:
        raise ValueError("Epoch sayısı pozitif olmalı.")
    tohumla()
    model = BaslangicModeli().to(cihaz)
    optimizer = torch.optim.Adam(model.parameters(), lr=OGRENME_ORANI)
    criterion = nn.CrossEntropyLoss(ignore_index=YOK_SAY)
    train, val, generator = yukleyiciler()
    start_epoch, best, rows, best_weights, best_epoch, best_snapshot = 0, -math.inf, [], None, 0, None
    if CALISMA_MODU == "resume":
        if not DEVAM_CHECKPOINT:
            raise ValueError("Resume için DEVAM_CHECKPOINT gerekli.")
        checkpoint = torch.load(DEVAM_CHECKPOINT, map_location="cpu", weights_only=True)
        checkpoint_uyumunu_kontrol_et(checkpoint)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch, best = checkpoint["epoch"], checkpoint["best_miou"]
        rows = checkpoint["history"]
        best_weights = checkpoint["best_model"]
        best_epoch = checkpoint.get("best_epoch", checkpoint["epoch"])
        best_snapshot = checkpoint.get("best_snapshot")
        if best_snapshot is None:
            raise ValueError("Bu checkpoint en iyi epoch'un optimizer/RNG durumunu içermiyor; bu notebook'un güncel kayıt biçimini kullanın.")
        generator.set_state(checkpoint["loader_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if cihaz.type == "cuda" and checkpoint.get("cuda_rng"):
            if len(checkpoint["cuda_rng"]) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    if start_epoch >= EPOCH_SAYISI:
        raise ValueError("Checkpoint zaten hedef epoch'a ulaşmış; hedefi artırın veya reset seçin.")
    run_name = f"{'kaggle' if KAGGLE else 'local'}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    run_dir = CIKTI_KOKU / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {"run": run_name, "device": str(cihaz), "python": sys.version.split()[0],
              "torch": str(torch.__version__), "numpy": np.__version__, "platform": platform.platform(),
              "cuda_build": torch.version.cuda, "class_map": SINIF_HARITASI, "target_size": BOYUT,
              "ignore_index": YOK_SAY, "seed": TOHUM, "batch_size": BATCH,
              "learning_rate": OGRENME_ORANI, "epochs": EPOCH_SAYISI,
              "mode": CALISMA_MODU, "data_manifest": VERI_MANIFESTI,
              "manifest_note": "Dosya adı ve boyut özeti; içerik hash'i/sızıntı kanıtı değildir.",
              "resume_from": str(DEVAM_CHECKPOINT) if CALISMA_MODU == "resume" else None}
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    if denetim:
        (run_dir / "audit.json").write_text(json.dumps(denetim, indent=2), encoding="utf-8")
    print("Çıktı:", run_dir)
    for epoch in range(start_epoch + 1, EPOCH_SAYISI + 1):
        train_result = bir_epoch(model, train, criterion, optimizer)
        val_result = bir_epoch(model, val, criterion)
        row = {"epoch": epoch, "train_loss": train_result["loss"], "val_loss": val_result["loss"],
               "train_miou": train_result["miou"], "val_miou": val_result["miou"]}
        for split, result in (("train", train_result), ("val", val_result)):
            row.update({f"{split}_iou_{name}": value for name, value in zip(SINIFLAR, result["iou"])})
        rows.append(row)
        improved = val_result["miou"] > best
        if improved:
            best = val_result["miou"]
            best_epoch = epoch
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                      "class_map": SINIF_HARITASI, "target_size": BOYUT, "ignore_index": YOK_SAY,
                      "seed": TOHUM, "batch_size": BATCH, "data_manifest": VERI_MANIFESTI,
                      "best_miou": best, "best_epoch": best_epoch, "best_model": best_weights, "history": rows,
                      "torch_rng": torch.get_rng_state(), "loader_rng": generator.get_state(),
                      "cuda_rng": torch.cuda.get_rng_state_all() if cihaz.type == "cuda" else [],
                      "val_confusion": val_result["confusion"]}
        if improved:
            best_snapshot = {key: copy.deepcopy(checkpoint[key]) for key in
                             ("model", "optimizer", "epoch", "torch_rng", "loader_rng", "cuda_rng", "val_confusion")}
        checkpoint["best_snapshot"] = best_snapshot
        # Replace only this run's own checkpoint; a partial write cannot corrupt last.pt.
        pending = run_dir / "pending.pt"
        torch.save(checkpoint, pending)
        pending.replace(run_dir / "last.pt")
        if improved or not (run_dir / "best.pt").exists():
            # Even after resume without improvement, best.pt must contain matching
            # weights, optimizer, RNG and epoch from the same best validation step.
            best_checkpoint = dict(checkpoint)
            best_checkpoint.update(best_snapshot)
            best_checkpoint["history"] = [r for r in rows if r["epoch"] <= best_epoch]
            torch.save(best_checkpoint, run_dir / "best.pt")
        csv_yaz(run_dir / "history.csv", rows)
        print(json.dumps(row, ensure_ascii=False, allow_nan=True))
    return model, run_dir, rows


egitim_sonucu = None
if EGITIMI_BASLAT:
    egitim_sonucu = egit()
else:
    print("Eğitim başlatılmadı. Hazır olduğunuzda ayar hücresinde EGITIMI_BASLAT=True yapın ve bu hücreyi çalıştırın.")

# %% [markdown]
# ## 8. Eğriler ve en iyi modelin örnek tahmini
# Eğitim kapalıysa bu hücre tahmin üretmez. Tahmin yalnız bu çalışmanın `best.pt` dosyasından yüklenir.
# Padding koyu gri gösterilir; metrikte yok sayılır. Renkler sabit sınıf haritasındadır.
# Örnek görsel kalite kontrolüdür; bağımsız bir test veya tüm veri başarısının kanıtı değildir.

# %%
if "egitim_sonucu" not in globals():
    raise RuntimeError("Önce 7. Eğitim hücresini çalıştırın; eğitim kapalı kalabilir.")


if egitim_sonucu is None:
    print("Henüz eğitim çıktısı yok; eğri ve tahmin çizilmedi.")
else:
    _, run_dir, rows = egitim_sonucu
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for split in ("train", "val"):
        axes[0].plot([r["epoch"] for r in rows], [r[f"{split}_loss"] for r in rows], label=split)
        axes[1].plot([r["epoch"] for r in rows], [r[f"{split}_miou"] for r in rows], label=split)
    for ax, title in zip(axes, ("Geçerli piksel başına CE", "mIoU")):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "learning-curves.png", dpi=150)
    plt.show()
    best_checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    checkpoint_uyumunu_kontrol_et(best_checkpoint)
    best_model = BaslangicModeli().to(cihaz)
    best_model.load_state_dict(best_checkpoint["best_model"])
    best_model.eval()
    x, y = UAVidDataset(val_pairs)[0]
    with torch.no_grad():
        prediction = best_model(x.unsqueeze(0).to(cihaz)).argmax(1)[0].cpu().numpy()
    palette = np.asarray(RENKLER, dtype=np.uint8)
    target_rgb = np.full((*y.shape, 3), 48, dtype=np.uint8)
    valid = y.numpy() != YOK_SAY
    target_rgb[valid] = palette[y.numpy()[valid]]
    predicted_rgb = palette[prediction]
    predicted_rgb[~valid] = 48
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, picture, title in zip(axes, (x.permute(1, 2, 0).numpy(), target_rgb, predicted_rgb),
                                  ("Görüntü", "Referans", "Model tahmini")):
        ax.imshow(picture)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(run_dir / "validation-example.png", dpi=150)
    plt.show()
    print("En iyi validation mIoU:", best_checkpoint["best_miou"], "| Bağımsız test sonucu değildir.")

# %% [markdown]
# ## 9. Kayıtlı checkpoint'i ayrıca incele
# Eğitim yapmadan kendi checkpoint'inizi `HARICI_CHECKPOINT` ayarına yazabilirsiniz.
# Bu hücre, bu notebook'un doğrulanabilir sınıf/boyut/veri meta verilerini içeren biçimini kabul eder.
# Önceki sohbet hücresinin `/kaggle/working/baslangic_modeli_en_iyi.pt` dosyası yerelde yoktur;
# dosya getirildiğinde anahtarları/sınıf sırası görülmeden biçimi varsayılmaz.
# Seçilen dosyanın gerçek model ağırlıkları değerlendirilir; `last.pt` son epoch, `best.pt` en iyi epoch'tur.
# Piksel tahmin histogramı, tek sınıfa yönelimi gözlemlemeye yardımcı olur.

# %%
if "checkpoint_uyumunu_kontrol_et" not in globals():
    raise RuntimeError("Önce veri/model ve 7. Eğitim tanımları hücresini çalıştırın.")


def kayitli_modeli_incele(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"model", "class_map", "target_size", "ignore_index", "data_manifest", "seed", "batch_size"}
    if not isinstance(checkpoint, dict) or not required <= checkpoint.keys():
        keys = sorted(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint).__name__
        raise ValueError(f"Bu notebook biçimiyle eşleşmeyen checkpoint; anahtarlar={keys}. Eski dosyanın sınıf sırası ve meta verileri incelenmeden yüklenmez.")
    checkpoint_uyumunu_kontrol_et(checkpoint)
    model = BaslangicModeli().to(cihaz)
    model.load_state_dict(checkpoint["model"])
    loader = DataLoader(UAVidDataset(val_pairs), batch_size=BATCH, shuffle=False, num_workers=0)
    result = bir_epoch(model, loader, nn.CrossEntropyLoss(ignore_index=YOK_SAY))
    matrix = torch.tensor(result["confusion"], dtype=torch.int64)
    predicted_counts = matrix.sum(0).numpy()
    result["predicted_pixel_percent"] = dict(zip(SINIFLAR, (100 * predicted_counts / predicted_counts.sum()).tolist()))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    x, y = UAVidDataset(val_pairs)[0]
    model.eval()
    with torch.no_grad():
        pred = model(x.unsqueeze(0).to(cihaz)).argmax(1)[0].cpu().numpy()
    valid = y.numpy() != YOK_SAY
    palette = np.asarray(RENKLER, dtype=np.uint8)
    target_rgb = np.full((*y.shape, 3), 48, dtype=np.uint8)
    target_rgb[valid] = palette[y.numpy()[valid]]
    pred_rgb = palette[pred]
    pred_rgb[~valid] = 48
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, picture, title in zip(axes, (x.permute(1, 2, 0).numpy(), target_rgb, pred_rgb),
                                  ("Görüntü", "Referans", "Checkpoint tahmini")):
        ax.imshow(picture)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    plt.show()
    return result


if HARICI_CHECKPOINT:
    harici_sonuc = kayitli_modeli_incele(HARICI_CHECKPOINT)
else:
    print("Haricî checkpoint seçilmedi; önceki Kaggle dosyasına bu oturumdan erişilmedi.")

# %% [markdown]
# ## 10. Deney günlüğüne ne yazılmalı?
# Çalışma klasörü adı, veri manifesti, cihaz/sürümler, tohum, 512×320 letterbox,
# batch=4, Adam=0.001, epoch sayısı, bütün sınıfların IoU'su, mIoU, kayıp ve hata örnekleri.
# Bu ilk 3 epoch bir başlangıç denemesidir; başarı iddiası değildir. Sonraki deneyde
# yalnız bir değişkeni değiştirin. Kaggle ve yerel çalışmanın aynı ayarlarda olduğunu
# `config.json` ile karşılaştırın; aynı tohum farklı cihazda aynı sayıyı garanti etmez.
# Kaggle: Save Version ile notebook ve `/kaggle/working/uavid-runs` çıktılarını saklayın.
# Git'e yalnız çıktı temizlenmiş notebook, kod ve paylaşılması uygun özetleri ekleyin.
# Checkpoint, veri ve kişisel yolları içeren ham çıktı dosyaları repoya gönderilmez.
