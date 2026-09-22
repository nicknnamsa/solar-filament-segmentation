"""Precompute limb-darkening-corrected copies of the train and test images.

The sun is darker towards its edge. For each image we estimate the disc, measure the average brightness at every
distance from its centre, and divide that profile out so a filament looks the same anywhere on the disc. Outside
the disc is set to 0. Output is lossless 8-bit PNG so training reads plain files.

    python scripts/preprocess.py                 # writes data/processed/limb/{train,test}_images/
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage as ndi
from skimage import filters, measure

LIMB_SMOOTH = 8            # smoothing of the radial brightness profile
EDGE_STRENGTH = 0.3        # a drop counts as the limb if it is at least this fraction of the ray's steepest drop
DISC_MARGIN = 0.985        # treat the outer 1.5% of the radius as edge, to avoid limb artefacts


def _rough_disc(img):
    """Rough disc + halo from a low brightness threshold on a 4x downsample.

    The halo around the disc is more symmetric than the disc interior (which can have strong brightness gradients),
    so its centre is a good starting point. Its radius is larger than the disc's.
    """
    small = ndi.gaussian_filter(img[::4, ::4], 2)
    thr = filters.threshold_multiotsu(small, classes=3)[0]
    biggest = max(measure.regionprops(measure.label(small > thr)), key=lambda r: r.area)
    cy, cx = biggest.centroid
    return cy * 4, cx * 4, np.sqrt(biggest.area / np.pi) * 4


def _fit_circle(xs, ys):
    """Least-squares circle through points; returns (cy, cx, radius)."""
    A = np.column_stack([xs, ys, np.ones_like(xs)])
    (a, b, c), *_ = np.linalg.lstsq(A, xs ** 2 + ys ** 2, rcond=None)
    cx, cy = a / 2, b / 2
    return cy, cx, np.sqrt(c + cx ** 2 + cy ** 2)


def _limb_pass(smooth, cy0, cx0, r_lo, r_hi, n_rays=180):
    """Search rays from (cy0, cx0) between r_lo and r_hi for the limb edge; fit a circle robustly."""
    radii = np.arange(r_lo, r_hi, 1.0)
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    ys = cy0 + np.outer(np.sin(angles), radii)
    xs = cx0 + np.outer(np.cos(angles), radii)
    valid = (ys >= 0) & (ys <= smooth.shape[0] - 1) & (xs >= 0) & (xs <= smooth.shape[1] - 1)
    profiles = ndi.map_coordinates(smooth, [ys.ravel(), xs.ravel()], order=1, mode="nearest").reshape(ys.shape)
    grad = np.gradient(profiles, axis=1)
    grad[~valid] = 0
    # The limb is the INNERMOST strong outward drop; the steepest one can be the outer edge of the gray halo.
    edge = np.zeros(n_rays, int)
    for i in range(n_rays):
        g = grad[i]
        first = np.nonzero(g <= EDGE_STRENGTH * g.min())[0][0]
        edge[i] = first + g[first:first + 15].argmin()           # settle on the steepest point of that edge
    ex = cx0 + np.cos(angles) * radii[edge]
    ey = cy0 + np.sin(angles) * radii[edge]
    keep = np.ones(n_rays, bool)
    for _ in range(3):                                           # refit, dropping outliers (prominences etc.)
        cy, cx, r = _fit_circle(ex[keep], ey[keep])
        resid = np.abs(np.hypot(ex - cx, ey - cy) - r)
        keep = resid < max(3 * np.median(resid[keep]), 3.0)
    return cy, cx, r


def find_disc(img):
    """Find the solar disc (centre y, centre x, radius) from the sharp brightness drop at the limb."""
    smooth = ndi.gaussian_filter(img, 2)
    cy0, cx0, r_halo = _rough_disc(img)
    cy, cx, r = _limb_pass(smooth, cy0, cx0, 0.7 * r_halo, 1.05 * r_halo)   # coarse: wide search
    return _limb_pass(smooth, cy, cx, 0.9 * r, 1.12 * r)                    # fine: narrow search around it


def normalize(x, mask, lo=1, hi=99):
    a, b = np.percentile(x[mask], [lo, hi])
    return np.clip((x - a) / max(b - a, 1e-8), 0, 1)


def limb_correct(img, dist, radius, smooth=LIMB_SMOOTH):
    inside = dist < radius
    bins = dist[inside].astype(int)
    profile = np.bincount(bins, weights=img[inside]) / np.maximum(np.bincount(bins), 1)
    profile = ndi.gaussian_filter1d(profile, smooth, mode="nearest")
    flat = np.where(inside, img / np.maximum(profile[np.minimum(dist.astype(int), len(profile) - 1)], 1e-3), 0)
    return flat * img[inside].mean() / max(flat[inside].mean(), 1e-8)


def process_one(job):
    src, dst = job
    img = np.asarray(Image.open(src).convert("L"), dtype=np.float32) / 255.0
    cy, cx, radius = find_disc(img)
    yy, xx = np.indices(img.shape)
    dist = np.hypot(yy - cy, xx - cx)
    disc = dist < radius * DISC_MARGIN
    out = normalize(limb_correct(img, dist, radius), disc) * disc
    Image.fromarray((out * 255).round().astype(np.uint8)).save(dst, compress_level=1)
    return src.name, cx, cy, radius


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="V2/configs/maskrcnn_v2.yaml", help="config with the RAW image folders")
    parser.add_argument("--out", default="data/processed/limb")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    with open(args.config) as f:
        d = yaml.safe_load(f)["data"]

    jobs = []
    for key in ("train_images", "test_images"):
        out_dir = Path(args.out) / Path(d[key]).name
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(Path(d[key]).glob("*.jp*g")):
            jobs.append((src, out_dir / (src.stem + ".png")))
    print(f"processing {len(jobs)} images -> {args.out}")

    with Pool(args.workers) as pool:
        stats = pool.map(process_one, jobs, chunksize=8)

    radii = np.array([s[3] for s in stats])
    med = np.median(radii)
    print(f"disc radius: median {med:.0f}px, min {radii.min():.0f}, max {radii.max():.0f}")
    odd = [(n, r) for n, _, _, r in stats if abs(r - med) > 0.1 * med]
    print(f"{len(odd)} images with a disc radius more than 10% from the median (check these by eye):")
    for n, r in odd[:20]:
        print(f"    {n}  radius {r:.0f}")


if __name__ == "__main__":
    main()
