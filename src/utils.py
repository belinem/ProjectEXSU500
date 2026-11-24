"""very important ! shared functions for the notebooks for data loading, post-processing and metrics.
if you want to add a new function, please add it here and not in the notebooks, so that it can be reused by all notebooks."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_multiotsu

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "Task02_Heart"
RESULTS = ROOT / "results"
SPLITS = json.loads((ROOT / "splits" / "splits.json").read_text())


#data

def load_case(case):
    """Returns the image (float32), the LA mask (bool) and the voxel spacing in mm."""
    img = nib.load(DATA_ROOT / "imagesTr" / f"{case}.nii.gz")
    x = img.get_fdata(dtype=np.float32)
    y = np.asarray(nib.load(DATA_ROOT / "labelsTr" / f"{case}.nii.gz").dataobj) > 0
    return x, y, tuple(float(s) for s in img.header.get_zooms()[:3])


def normalize(x):
    # z-score on non-zero voxels, same as MONAI NormalizeIntensityd(nonzero=True)
    out = x.copy()
    m = x != 0
    out[m] = (x[m] - x[m].mean()) / x[m].std()
    return out


def fold_cases(k):
    """Training and validation volumes of cross-validation fold k."""
    val = SPLITS["folds"][k]
    return [c for c in SPLITS["dev"] if c not in val], val


def la_prior(labels, margin=15):
    """Mean LA centroid (voxels) and a box around all the training LAs, in-plane only."""
    centroid = np.mean([np.argwhere(y).mean(0) for y in labels], axis=0)
    lo = np.min([np.argwhere(y).min(0) for y in labels], axis=0) - margin
    hi = np.max([np.argwhere(y).max(0) for y in labels], axis=0) + margin
    # the LA position along the 3rd axis changes a lot between volumes, so the box keeps all slices
    box = (slice(lo[0], hi[0] + 1), slice(lo[1], hi[1] + 1), slice(None))
    return centroid, box


#post-processing

def keep_largest_component(mask):
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (np.argmax(sizes) + 1)


def keep_closest_component(mask, point):
    """Keeps the connected component with the voxel closest to `point`."""
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    grid = np.indices(mask.shape, sparse=True)
    dist = np.sqrt(sum((g - p) ** 2 for g, p in zip(grid, point)))
    closest = ndimage.minimum(dist, lab, range(1, n + 1))
    return lab == (np.argmin(closest) + 1)


def opening(mask, radius):
    # opening with a ball, computed with distance transforms (much faster than with a large structuring element)
    if radius == 0:
        return mask
    eroded = ndimage.distance_transform_edt(mask) > radius
    return (ndimage.distance_transform_edt(~eroded) <= radius) & mask


#models

def otsu_segment(x, centroid, box, n_classes, radius):
    """Baseline: brightest Otsu class in the box, opening, component closest to the mean LA centroid."""
    roi = x[box]
    mask = roi > threshold_multiotsu(roi, n_classes)[-1]
    mask = opening(mask, radius)
    # centroid in the coordinates of the box
    offset = [s.start or 0 for s in box]
    mask = keep_closest_component(mask, np.array(centroid) - offset)
    pred = np.zeros(x.shape, dtype=bool)
    pred[box] = mask
    return pred


FEATURE_NAMES = ["intensity", "smooth_1", "smooth_2", "smooth_4", "gradient_1", "gradient_2", "pos_0", "pos_1", "pos_2"]


def rf_features(x, box, pad=16):
    """Handcrafted features of every voxel of the box, shape (n_voxels, 9)."""
    a, b = box[0], box[1]
    # filters are computed on a slightly larger crop to avoid border effects
    xp = x[a.start - pad:a.stop + pad, b.start - pad:b.stop + pad]
    feats = [xp] + [ndimage.gaussian_filter(xp, s) for s in (1, 2, 4)] \
                 + [ndimage.gaussian_gradient_magnitude(xp, s) for s in (1, 2)]
    feats = [f[pad:-pad, pad:-pad] for f in feats]
    i, j, k = np.indices(feats[0].shape, dtype=np.float32)
    feats += [i + a.start, j + b.start, k]
    return np.stack(feats, axis=-1).reshape(-1, len(FEATURE_NAMES)).astype(np.float32)


def sample_training_set(volumes, box, n_per_class, seed):
    """Balanced sample of LA and background voxels in the box of each (image, mask) training volume."""
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for x, y in volumes:
        feats, labels = rf_features(x, box), y[box].ravel()
        for cls in (True, False):
            idx = rng.choice(np.flatnonzero(labels == cls), n_per_class, replace=False)
            X.append(feats[idx])
            Y.append(labels[idx])
    return np.concatenate(X), np.concatenate(Y)


def rf_segment(model, x, box, threshold=0.5):
    """Random forest prediction in the box, then largest component. Also returns the probability map."""
    prob = np.zeros(x.shape, dtype=np.float32)
    prob[box] = model.predict_proba(rf_features(x, box))[:, 1].reshape(prob[box].shape)
    return keep_largest_component(prob >= threshold), prob


#metrics

def dice(pred, ref):
    return 2 * (pred & ref).sum() / (pred.sum() + ref.sum())


def hd95(pred, ref, spacing):
    """95th percentile of the symmetric surface distance, in mm."""
    if not pred.any():
        return np.nan
    # crop to the union of both masks to speed up the distance transforms
    idx = np.argwhere(pred | ref)
    crop = tuple(slice(a, b + 2) for a, b in zip(np.maximum(idx.min(0) - 1, 0), idx.max(0)))
    pred, ref = pred[crop], ref[crop]
    pred_border = pred ^ ndimage.binary_erosion(pred)
    ref_border = ref ^ ndimage.binary_erosion(ref)
    d_pred = ndimage.distance_transform_edt(~ref_border, sampling=spacing)[pred_border]
    d_ref = ndimage.distance_transform_edt(~pred_border, sampling=spacing)[ref_border]
    return float(np.percentile(np.concatenate([d_pred, d_ref]), 95))


def evaluate(pred, ref, spacing):
    """All the metrics of the protocol for one volume."""
    voxel_ml = np.prod(spacing) / 1000
    tp = (pred & ref).sum()
    return {
        "dice": dice(pred, ref),
        "hd95_mm": hd95(pred, ref, spacing),
        "sensitivity": tp / ref.sum(),
        "precision": tp / pred.sum() if pred.any() else np.nan,
        "volume_ml": pred.sum() * voxel_ml,
        "ref_volume_ml": ref.sum() * voxel_ml,
        "volume_error_ml": (pred.sum() - ref.sum()) * voxel_ml,
    }


def bootstrap_ci(values, n_boot=10000, seed=0):
    """Mean and 95% bootstrap confidence interval (resampling patients)."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values))).mean(1)
    return values.mean(), *np.percentile(means, [2.5, 97.5])
