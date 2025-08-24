#!/usr/bin/env python3
from pathlib import Path
import cv2
import numpy as np
import argparse
from collections import defaultdict


def get_depth_config(video_path: Path, verbose=False):
    video_name = video_path.stem
    depth_dir = video_path.parent / "depth_anything_v2" / video_name
    if not depth_dir.is_dir():
        if verbose: print(f"[Depth] folder missing: {depth_dir}")
        return None, "", 0
    for f in depth_dir.iterdir():
        if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
            pad = len(f.stem) if f.stem.isdigit() else 0
            return depth_dir, f.suffix, pad
    if verbose: print(f"[Depth] no images found in {depth_dir}")
    return None, "", 0


def get_dino_config(video_path: Path, verbose=False):
    video_name = video_path.stem
    dino_dir = video_path.parent / "dinos" / video_name
    if not dino_dir.is_dir():
        if verbose: print(f"[DINO] folder missing: {dino_dir}")
        return None, 0
    files = sorted(dino_dir.glob("*.npy"))
    if not files:
        if verbose: print(f"[DINO] no .npy files in {dino_dir}")
        return None, 0
    pad = len(files[0].stem) if files[0].stem.isdigit() else 0
    return dino_dir, pad


def get_mask_files(sam2_dir: Path, video_name: str, verbose=False):
    """Return a sorted list of mask image paths, or empty."""
    if not sam2_dir:
        return []
    candidate = sam2_dir / video_name
    base = candidate if candidate.is_dir() else sam2_dir
    files = sorted(p for p in base.iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not files and verbose:
        print(f"[Mask] no images found in {base}")
    return files


def prepare_tracking(track_dir: Path):
    """Return frame->list of .npy paths and a color map for each (query,track_idx)."""
    if not track_dir or not track_dir.is_dir():
        return defaultdict(list), {}
    files = list(track_dir.glob("*.npy"))
    frame_map = defaultdict(list)
    for p in files:
        stem = p.stem
        if "_" in stem:
            q, t = stem.split("_", 1)
            if q.isdigit() and t.isdigit():
                frame_map[int(t)].append(p)
    color_map = {}
    for p in files:
        if "_" not in p.stem:
            continue
        q, t = p.stem.split("_", 1)
        if q == t:
            arr = np.load(str(p))
            for i in range(len(arr)):
                color_map[(int(q), i)] = tuple(np.random.randint(0, 256, 3).tolist())
    return frame_map, color_map


def load_depth_frame(idx, depth_dir, ext, pad, size):
    h, w = size
    blank = np.zeros((h, w, 3), dtype=np.uint8)
    if not depth_dir:
        return blank
    name = str(idx).zfill(pad) if pad else str(idx)
    p = depth_dir / f"{name}{ext}"
    if not p.is_file():
        return blank
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        return blank
    if img.dtype == np.uint16:
        img8 = cv2.convertScaleAbs(img, alpha=255.0/65535.0)
    else:
        img8 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    cmap = cv2.applyColorMap(img8, cv2.COLORMAP_JET)
    return cv2.resize(cmap, (w, h), interpolation=cv2.INTER_NEAREST)


def load_tracks_frame(idx, frame, frame_map, color_map):
    disp = frame.copy()
    for p in frame_map.get(idx, []):
        data = np.load(str(p))
        if data.size == 0:
            continue
        q = int(p.stem.split("_", 1)[0])
        for i, pt in enumerate(data):
            x, y = int(pt[0]), int(pt[1])
            occl = pt[2] if pt.shape[0] > 2 else 0
            if occl < 0.5:
                color = color_map.get((q, i), (0, 255, 0))
                cv2.circle(disp, (x*2, y*2), 3, color, -1)
    return disp


def load_dino_frame(idx, dino_dir, pad, size):
    h, w = size
    blank = np.zeros((h, w, 3), dtype=np.uint8)
    if not dino_dir:
        return blank
    name = str(idx).zfill(pad) if pad else str(idx)
    p = dino_dir / f"{name}.npy"
    if not p.is_file():
        return blank
    feat = np.load(str(p))
    mag = np.linalg.norm(feat, axis=2) if feat.ndim == 3 else feat
    # norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    norm = cv2.normalize(mag.astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX)
    cmap = cv2.applyColorMap(norm, cv2.COLORMAP_HOT)
    return cv2.resize(cmap, (w, h), interpolation=cv2.INTER_NEAREST)


def load_mask_frame(idx, mask_files, size):
    h, w = size
    blank = np.zeros((h, w, 3), dtype=np.uint8)
    if idx >= len(mask_files):
        return blank
    img = cv2.imread(str(mask_files[idx]), cv2.IMREAD_UNCHANGED)
    if img is None:
        return blank
    if img.ndim == 2:
        disp = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        for v in np.unique(img):
            if v == 0:
                continue
            disp[img == v] = tuple(np.random.randint(0, 256, 3).tolist())
    else:
        disp = img
    return cv2.resize(disp, (w, h), interpolation=cv2.INTER_NEAREST)


def setup_windows(types, size):
    names = []
    titles = {
        "orig": "Original",
        "depth": "Depth Map",
        "tracks": "Tracking",
        "dino": "DINO",
        "mask": "Mask"
    }
    w, h = size
    for i, t in enumerate(types):
        name = titles[t]
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, w, h)
        cv2.moveWindow(name, (i % 3) * w, (i // 3) * h)
        names.append(name)
    return names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video_path", type=Path, default=Path(r"C:\Workspace\ChimpanzeesThesis\data_local_copy\per_signal_videos_fixed_interlacing\10_7_2017_a__index_0.mp4"))
    p.add_argument("--sam2_dir", type=Path, default=Path(r'C:\Workspace\ChimpanzeesThesis\data_local_copy\per_signal_videos_fixed_interlacing\sam2dir/initial_preds'))
    p.add_argument("--motion_seg_dir", type=Path, default=None)
    p.add_argument("--play", action="store_true", default=True)
    p.add_argument("--verbose", action="store_true", default=True)
    args = p.parse_args()

    cap = cv2.VideoCapture(str(args.video_path))
    if not cap.isOpened():
        print("Cannot open", args.video_path)
        return

    depth_dir, depth_ext, depth_pad = get_depth_config(args.video_path, args.verbose)
    dino_dir, dino_pad = get_dino_config(args.video_path, args.verbose)
    track_base = args.motion_seg_dir or (args.video_path.parent / "bootstapir")
    frame_map, color_map = prepare_tracking(track_base / args.video_path.stem)
    masks = get_mask_files(args.sam2_dir, args.video_path.stem, args.verbose)

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return
    h, w = frame.shape[:2]

    types = ["orig"]
    if depth_dir:
        types.append("depth")
    if frame_map:
        types.append("tracks")
    if dino_dir:
        types.append("dino")
    if masks:
        types.append("mask")

    types = ["tracks"]
    # types = ["mask", "dino", "tracks"]

    win_names = setup_windows(types, (w, h))
    idx = 0
    delay = int(1000 / cap.get(cv2.CAP_PROP_FPS)) if args.play and cap.get(cv2.CAP_PROP_FPS) else 1

    while ret:
        outputs = {
            "orig": frame,
            "depth": load_depth_frame(idx, depth_dir, depth_ext, depth_pad, (h, w)),
            "tracks": load_tracks_frame(idx, frame, frame_map, color_map),
            "dino": load_dino_frame(idx, dino_dir, dino_pad, (h, w)),
            "mask": load_mask_frame(idx, masks, (h, w)),
        }
        for t, name in zip(types, win_names):
            cv2.imshow(name, outputs[t])

        if cv2.waitKey(delay) & 0xFF in (ord('q'), 27):
            break
        idx += 1
        ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
