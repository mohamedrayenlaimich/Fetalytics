"""Phase A diagnostics: mask format, pairing, frame-seek accuracy, augmentation alignment.
Writes contact sheets + a JSON summary to results/diagnostics/. No training."""
import csv, json, sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(r"c:\Users\MSI\OneDrive\Documents\Fetalytics")
sys.path.insert(0, str(ROOT))
from AI.preprocessing.pipeline import PreprocessingConfig, encode_segmentation_mask
from AI.training.trainer import TrainingSample, UltrasoundMultiTaskDataset

VAL = ROOT / "tmp_ingest/DatasetV3/extracted/val"
OUT = ROOT / "results/diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

seg_info = {}
with open(VAL / "seg/seg_info.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        seg_info[r["filename"]] = int(r["labeled_frame_index"])

summary = {"pairs": [], "value_hist": {}, "issues": []}
tiles = []
for vname, fidx in sorted(seg_info.items()):
    stem = Path(vname).stem
    mpath = VAL / "seg" / f"{stem}.png"
    vpath = VAL / "videos" / vname
    if not mpath.exists() or not vpath.exists():
        summary["issues"].append(f"missing pair for {vname}")
        continue
    m = cv2.imread(str(mpath), cv2.IMREAD_UNCHANGED)
    vals, counts = np.unique(m, return_counts=True)
    for v, c in zip(vals.tolist(), counts.tolist()):
        summary["value_hist"][str(v)] = summary["value_hist"].get(str(v), 0) + c

    # frame by seeking vs. by sequential decode
    cap = cv2.VideoCapture(str(vpath))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx); ok1, f_seek = cap.read(); cap.release()
    cap = cv2.VideoCapture(str(vpath)); f_seq = None
    for i in range(fidx + 1):
        ok2, fr = cap.read()
        if not ok2: break
        if i == fidx: f_seq = fr
    cap.release()
    seek_diff = float(np.abs(f_seek.astype(int) - f_seq.astype(int)).mean()) if (ok1 and f_seq is not None) else None

    # component analysis per label value
    comp = {}
    for v in (127, 255):
        b = (m == v).astype(np.uint8)
        area = int(b.sum())
        cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        ecc = None
        if cnts and len(max(cnts, key=cv2.contourArea)) >= 5:
            (_, _), (a1, a2), _ = cv2.fitEllipse(max(cnts, key=cv2.contourArea))
            ecc = round(min(a1, a2) / max(a1, a2), 3)
        comp[v] = {"area_px": area, "n_components": len(cnts), "axis_ratio": ecc}

    # mean image intensity inside each label (head = bright skull ring / soft tissue)
    gray = cv2.cvtColor(f_seq, cv2.COLOR_BGR2GRAY)
    inten = {v: (round(float(gray[m == v].mean()), 1) if (m == v).any() else None) for v in (127, 255)}

    summary["pairs"].append({
        "video": vname, "frame": fidx, "frames_total": n,
        "img_shape": list(f_seq.shape), "mask_shape": list(m.shape),
        "seek_vs_sequential_mean_abs_diff": seek_diff,
        "label_127": comp[127], "label_255": comp[255], "mean_intensity": inten,
    })

    ov = f_seq.copy()
    ov[m == 255] = (0.5 * ov[m == 255] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    ov[m == 127] = (0.5 * ov[m == 127] + 0.5 * np.array([0, 140, 255])).astype(np.uint8)
    mask_vis = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    tile = np.hstack([cv2.resize(f_seq, (256, 256)), cv2.resize(mask_vis, (256, 256), interpolation=cv2.INTER_NEAREST), cv2.resize(ov, (256, 256))])
    cv2.putText(tile, f"{stem} f{fidx}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    tiles.append(tile)

# contact sheets: 10 rows per sheet (image | mask | overlay  green=255, orange=127)
for s in range(0, len(tiles), 10):
    cv2.imwrite(str(OUT / f"pairs_sheet_{s // 10 + 1}.png"), np.vstack(tiles[s:s + 10]))

# --- augmentation alignment check on the ACTUAL training dataset class ---
cfg = PreprocessingConfig()
p0 = summary["pairs"][0]
cap = cv2.VideoCapture(str(VAL / "videos" / p0["video"])); cap.set(cv2.CAP_PROP_POS_FRAMES, p0["frame"]); _, img0 = cap.read(); cap.release()
m0 = cv2.imread(str(VAL / "seg" / f"{Path(p0['video']).stem}.png"), cv2.IMREAD_UNCHANGED)
mcls = encode_segmentation_mask(m0, (cfg.target_height, cfg.target_width))
sample = TrainingSample("s", p0["video"], p0["frame"], img0, mcls, 1)
gt_head_frac = float((mcls == 2).mean())
aug_fracs, aug_centroids = [], []
ds = UltrasoundMultiTaskDataset([sample], cfg, augment=True, seed=0)
for _ in range(10):
    mk = ds[0]["mask"].numpy()
    aug_fracs.append(float((mk == 2).mean()))
    ys, xs = np.nonzero(mk == 2)
    aug_centroids.append([float(xs.mean()), float(ys.mean())] if len(xs) else None)
ys, xs = np.nonzero(mcls == 2)
summary["augmentation_check"] = {
    "image_shape_fed_to_aug": list(img0.shape), "mask_shape_fed_to_aug": list(mcls.shape),
    "gt_head_area_fraction": round(gt_head_frac, 4),
    "gt_head_centroid_xy": [round(float(xs.mean()), 1), round(float(ys.mean()), 1)],
    "augmented_head_area_fractions": [round(a, 4) for a in aug_fracs],
    "augmented_head_centroids_xy": aug_centroids,
}
noaug = UltrasoundMultiTaskDataset([sample], cfg, augment=False)[0]["mask"].numpy()
summary["augmentation_check"]["no_aug_matches_gt"] = bool((noaug == mcls).all())

# class-frequency (for loss weighting review)
all_cls = []
for p in summary["pairs"]:
    mm = cv2.imread(str(VAL / "seg" / f"{Path(p['video']).stem}.png"), cv2.IMREAD_UNCHANGED)
    all_cls.append(encode_segmentation_mask(mm, (256, 256)))
stack = np.stack(all_cls)
summary["class_fraction_at_256"] = {str(c): round(float((stack == c).mean()), 4) for c in (0, 1, 2)}

(OUT / "diagnostics_phaseA.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

# console digest
a = summary["augmentation_check"]
print("pairs:", len(summary["pairs"]), "issues:", summary["issues"])
print("value_hist:", summary["value_hist"])
print("class_fraction_at_256:", summary["class_fraction_at_256"])
sd = [p["seek_vs_sequential_mean_abs_diff"] for p in summary["pairs"]]
print("seek diff max:", max(sd), "nonzero:", sum(1 for d in sd if d and d > 0.5))
r127 = [p["label_127"]["area_px"] for p in summary["pairs"]]; r255 = [p["label_255"]["area_px"] for p in summary["pairs"]]
print("area 127 median:", int(np.median(r127)), " area 255 median:", int(np.median(r255)))
print("axis ratio 127 median:", np.median([p["label_127"]["axis_ratio"] or 0 for p in summary["pairs"]]),
      " 255 median:", np.median([p["label_255"]["axis_ratio"] or 0 for p in summary["pairs"]]))
print("n_components 127:", sorted(set(p["label_127"]["n_components"] for p in summary["pairs"])),
      " 255:", sorted(set(p["label_255"]["n_components"] for p in summary["pairs"])))
print("AUG:", json.dumps(a))

