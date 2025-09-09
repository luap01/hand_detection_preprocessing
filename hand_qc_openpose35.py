#!/usr/bin/env python3
"""
Hand QC Fast Start (OpenPose 0.35 schema + Azure/Kinect-like JSON calibration)

Layout this script expects:

Images:     data/input/<DATASET>/<CAM>/
Calib:      data/input/<DATASET>/camera<CAM>.json      (tries also camera_<CAM>.json, <CAM>.json, camera{CAM:02d}.json)
Keypoints:  data/output/<DATASET>/openpose_0.35/<CAM>/json/<FRAME>.json

OpenPose 0.35 hand JSON (per-hand shift):
{
  "people": [{
     "hand_left_shift":  [sx, sy],
     "hand_left_keypoints_2d":  [x1,y1,s1, x2,y2,s2, ... x21,y21,s21],
     "hand_right_shift": [sx, sy],
     "hand_right_keypoints_2d": [ ... ],
     "hand_left_conf": [...],   # ignored
     "hand_right_conf": [...]   # ignored
  }]
}
Absolute pixel coords = (shift_x + x, shift_y + y). Only x,y are used.

Calibration JSON (per cam, e.g., camera01.json):
- Intrinsics from value0.color_parameters: fov_x, fov_y, c_x, c_y  -> K
- WORLD->DEPTH from value0.camera_pose (quaternion + translation)
- COLOR->DEPTH from value0.color2depth_transform (quaternion + translation)

We compute WORLD->COLOR as:
  R_dc = R_cd^T
  R_wc = R_dc R_wd
  t_wc = R_dc (t_wd - t_cd)

Then we build F_ij = K_j^{-T} [t_ji]_x R_ji K_i^{-1} and use epipolar consistency to detect left/right swaps.

Outputs:
- qc_out/qc_metrics.csv
- qc_out/keypoints/<CAM>/<FRAME>.json (corrected JSONs if --auto-swap; others copied to mirror input)
- qc_out/sheets/...   (only if --save-sheets)

Example:
python hand_qc_openpose35.py \
  --dataset OR1 \
  --cams 01 02 03 \
  --pairs 01,02 01,03 \
  --ref-cam 01 \
  --auto-swap \
  --save-sheets \
  --roi data/input/OR1/rois.yaml
"""

import argparse, os, sys, json, glob
from pathlib import Path
import numpy as np
import cv2
import yaml
import pandas as pd
from tqdm import tqdm

# ------------------ geometry helpers ------------------

def hat(v):
    x,y,z = v
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]], dtype=float)

def quat_to_R(x, y, z, w):
    """Quaternion (x,y,z,w) -> 3x3 rotation matrix."""
    n = np.sqrt(x*x + y*y + z*z + w*w) + 1e-12
    x, y, z, w = x/n, y/n, z/n, w/n
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)    ],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)    ],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)]
    ], dtype=float)
    return R

def fundamental_from_pair(Ki, Kj, Ri, ti, Rj, tj):
    """Build Fundamental matrix from world->cam extrinsics and intrinsics."""
    Rji = Rj @ Ri.T
    tji = tj - Rji @ ti
    E = hat(tji) @ Rji
    F = np.linalg.inv(Kj).T @ E @ np.linalg.inv(Ki)
    return F

def sampson_distance(F, x1, x2):
    """Median Sampson distance over joints. x1,x2: (N,2)."""
    if x1 is None or x2 is None or len(x1)==0 or len(x2)==0:
        return np.inf
    N = min(len(x1), len(x2))
    x1 = np.asarray(x1[:N], dtype=float)
    x2 = np.asarray(x2[:N], dtype=float)
    x1h = np.c_[x1, np.ones(N)]
    x2h = np.c_[x2, np.ones(N)]
    Fx1 = (F @ x1h.T).T
    Ftx2 = (F.T @ x2h.T).T
    denom = Fx1[:,0]**2 + Fx1[:,1]**2 + Ftx2[:,0]**2 + Ftx2[:,1]**2 + 1e-12
    num = np.sum(x2h * (F @ x1h.T).T, axis=1)**2
    d = num / denom
    d = d[np.isfinite(d)]
    if len(d)==0:
        return np.inf
    return float(np.median(d))

# ------------------ calibration loading ------------------

def load_calib_json_for_cam(calib_dir, cam_id):
    """
    Load per-camera calibration JSON and convert to color-camera intrinsics/extrinsics.
    Tries: camera{cam}.json, camera{cam:02d}.json, {cam}.json, camera_{cam}.json
    Returns dict { 'K':3x3, 'R':3x3, 't':(3,), 'size':(W,H) } in WORLD->COLOR convention.
    """
    candidates = [
        f"camera{cam_id}.json",
        (f"camera{int(cam_id):02d}.json" if str(cam_id).isdigit() else None),
        f"{cam_id}.json",
        f"camera_{cam_id}.json"
    ]
    candidates = [c for c in candidates if c]
    path = None
    for c in candidates:
        p = os.path.join(calib_dir, c)
        if os.path.exists(p):
            path = p; break
    if path is None:
        raise FileNotFoundError(f"Calibration JSON not found for cam {cam_id} in {calib_dir} (tried {candidates})")

    with open(path, 'r') as f:
        data = json.load(f)
    v0 = data.get('value0', data)

    # Intrinsics (color)
    cp = v0.get('color_parameters', {})
    fx = cp.get('fov_x', cp.get('intrinsics_matrix', {}).get('m00', None))
    fy = cp.get('fov_y', cp.get('intrinsics_matrix', {}).get('m11', None))
    cx = cp.get('c_x',   cp.get('intrinsics_matrix', {}).get('m20', None))
    cy = cp.get('c_y',   cp.get('intrinsics_matrix', {}).get('m22', None))
    if None in (fx, fy, cx, cy):
        raise ValueError(f"Intrinsics missing in {path} (looked in color_parameters.*)")
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1.0]], dtype=float)
    W = int(cp.get('width', 1920))
    H = int(cp.get('height', 1080))

    # WORLD->DEPTH pose
    pose = v0.get('camera_pose', {})
    t_wd = np.array([pose.get('translation',{}).get('m00',0.0),
                     pose.get('translation',{}).get('m10',0.0),
                     pose.get('translation',{}).get('m20',0.0)], dtype=float)
    qd = pose.get('rotation', {})
    R_wd = quat_to_R(qd.get('x',0.0), qd.get('y',0.0), qd.get('z',0.0), qd.get('w',1.0))

    # COLOR->DEPTH transform
    c2d = v0.get('color2depth_transform', {})
    tc = np.array([c2d.get('translation',{}).get('m00',0.0),
                   c2d.get('translation',{}).get('m10',0.0),
                   c2d.get('translation',{}).get('m20',0.0)], dtype=float)
    qc = c2d.get('rotation', {})
    R_cd = quat_to_R(qc.get('x',0.0), qc.get('y',0.0), qc.get('z',0.0), qc.get('w',1.0))

    # WORLD->COLOR
    R_dc = R_cd.T
    R_wc = R_dc @ R_wd
    t_wc = R_dc @ (t_wd - tc)

    return {'K': K, 'R': R_wc, 't': t_wc, 'size': (W,H), 'path': path}

def load_all_calibs(calib_root, cam_ids):
    out = {}
    for c in cam_ids:
        out[c] = load_calib_json_for_cam(calib_root, c)
    return out

# ------------------ keypoint loading (OpenPose 0.35) ------------------

def parse_openpose35_with_shift(obj):
    """
    Returns dict {'L': (21,2) or None, 'R': (21,2) or None} using per-hand shift.
    """
    if not isinstance(obj, dict): return {'L':None,'R':None}
    people = obj.get('people', [])
    if not people:
        return {'L':None,'R':None}
    p = people[0]
    out = {'L':None,'R':None}
    for side in ('left','right'):
        k_field = f'hand_{side}_keypoints_2d'
        s_field = f'hand_{side}_shift'
        key = 'L' if side=='left' else 'R'
        if k_field not in p or s_field not in p:
            out[key] = None; continue
        arr = p[k_field]
        sx, sy = p[s_field][0], p[s_field][1]
        if not isinstance(arr, list) or len(arr) < 3:
            out[key] = None; continue
        pts = np.array(arr, dtype=float).reshape(-1,3)[:21,:2]  # ignore score
        pts[:,0] += float(sx)
        pts[:,1] += float(sy)
        out[key] = pts
    return out

def swap_in_openpose35(obj):
    """Swap left/right fields in-place in an OpenPose 0.35 JSON object."""
    if not isinstance(obj, dict): return obj
    people = obj.get('people', [])
    if not people: return obj
    p = people[0]
    left_keys  = ['hand_left_keypoints_2d','hand_left_shift','hand_left_conf']
    right_keys = ['hand_right_keypoints_2d','hand_right_shift','hand_right_conf']
    tmp = {k: p.get(k, None) for k in left_keys}
    for lk, rk in zip(left_keys, right_keys):
        p[lk] = p.get(rk, None)
    for lk, rk in zip(left_keys, right_keys):
        p[rk] = tmp.get(lk, None)
    people[0] = p
    obj['people'] = people
    return obj

# ------------------ ROI helpers ------------------

def point_in_poly_mask(img_wh, poly_xy):
    W,H = img_wh
    mask = np.zeros((H,W), dtype=np.uint8)
    pts = np.array(poly_xy, dtype=np.int32).reshape(-1,1,2)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)

def load_rois(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    rois = data.get('rois', {})
    out = {}
    for cam_id, r in rois.items():
        polys = r.get('polygons', [])
        W,H = None, None
        if 'image_size' in r:
            W,H = r['image_size']
        entry = {}
        if W and H:
            entry['masks'] = [point_in_poly_mask((W,H), poly) for poly in polys]
            entry['size'] = (W,H)
        else:
            entry['polygons'] = polys
        out[cam_id] = entry
    return out

def roi_score(kp_xy, roi_entry):
    if kp_xy is None: return 0.0
    pts = np.asarray(kp_xy, dtype=float)
    pts = pts[~np.isnan(pts).any(axis=1)]
    if len(pts)==0: return 0.0
    if roi_entry is None: return 1.0
    if 'masks' in roi_entry:
        W,H = roi_entry.get('size',(None,None))
        if W is None or H is None: return 1.0
        inside = np.zeros(len(pts), dtype=bool)
        xy = np.round(pts).astype(int)
        valid = (xy[:,0]>=0)&(xy[:,0]<W)&(xy[:,1]>=0)&(xy[:,1]<H)
        for i,v in enumerate(valid):
            if not v: continue
            for m in roi_entry['masks']:
                if m[xy[i,1], xy[i,0]]:
                    inside[i] = True; break
        return float(np.mean(inside))
    inside_any = []
    for p in pts:
        inside = False
        for poly in roi_entry.get('polygons', []):
            poly_pts = np.array(poly, dtype=np.float32)
            if cv2.pointPolygonTest(poly_pts, (float(p[0]),float(p[1])), measureDist=False) >= 0:
                inside = True; break
        inside_any.append(inside)
    return float(np.mean(inside_any)) if inside_any else 0.0

# ------------------ drawing ------------------

HAND_EDGES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

def draw_hand(img, pts, color, thickness=2):
    if pts is None: return
    pts = np.asarray(pts)
    for a,b in HAND_EDGES:
        if a < len(pts) and b < len(pts):
            pa, pb = pts[a], pts[b]
            if not (np.any(np.isnan(pa)) or np.any(np.isnan(pb))):
                cv2.line(img, tuple(np.round(pa).astype(int)), tuple(np.round(pb).astype(int)), color, thickness)
    for p in pts:
        if not np.any(np.isnan(p)):
            cv2.circle(img, tuple(np.round(p).astype(int)), 2, color, -1)

def make_contact_sheet(imgA, imgB, text_lines, out_path):
    H = max(imgA.shape[0], imgB.shape[0])
    W = imgA.shape[1] + imgB.shape[1]
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:imgA.shape[0], :imgA.shape[1]] = imgA
    canvas[:imgB.shape[0], imgA.shape[1]:] = imgB
    y = 20
    for line in text_lines:
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1, cv2.LINE_AA)
        y += 22
    os.makedirs(os.path.dirname(out_path), exist_ok=True
    )
    cv2.imwrite(out_path, canvas)

# ------------------ IO helpers for this dataset ------------------

def build_paths(dataset, cams, img_ext):
    img_root = os.path.join('data','input',dataset)
    kp_root  = os.path.join('data','output',dataset,'mediapipe_0.35')
    calib_root = os.path.join('data','input',dataset,'calib')
    cam_dirs = {c: os.path.join(img_root, c) for c in cams}
    kp_dirs  = {c: os.path.join(kp_root, c, 'json') for c in cams}
    return img_root, kp_root, calib_root, cam_dirs, kp_dirs

def list_common_frames(kp_dirs, basename_glob):
    sets = []
    for c, d in kp_dirs.items():
        files = glob.glob(os.path.join(d, basename_glob))
        bases = {Path(f).stem for f in files}
        sets.append(bases)
    common = set.intersection(*sets) if sets else set()
    return sorted(common)

def load_kp_openpose(path):
    try:
        with open(path,'r') as f:
            obj = json.load(f)
        kp = parse_openpose35_with_shift(obj)
        return kp, obj
    except Exception:
        return {'L':None,'R':None}, None

def find_image(img_dir, frame_stem):
    for ext in ('.jpg','.png','.jpeg','.bmp'):
        p = os.path.join(img_dir, frame_stem + ext)
        if os.path.exists(p): return p
    return None

# ------------------ main ------------------

def parse_args():
    ap = argparse.ArgumentParser(description="QC + auto-swap for OpenPose 0.35 hand keypoints with Azure/Kinect JSON calib")
    ap.add_argument('--dataset', required=True, help='Dataset name under data/input and data/output')
    ap.add_argument('--cams', nargs='+', required=True, help='Camera IDs (as subfolder names), e.g., 01 02 03')
    ap.add_argument('--pairs', nargs='+', required=True, help='Space-separated list of cam pairs \"01,02\" ...')
    ap.add_argument('--ref-cam', required=True, help='Reference camera ID to trust (never swapped)')
    ap.add_argument('--out', default='qc_out', help='Output root')
    ap.add_argument('--basename-glob', default='*.json', help='Keypoint JSON filename glob (default: *.json)')
    ap.add_argument('--roi', help='Optional ROI YAML to check hand-in-region')
    ap.add_argument('--roi-min', type=float, default=0.4, help='Flag when fraction of joints in ROI is below this')
    ap.add_argument('--auto-swap', action='store_true', help='Write corrected JSONs (non-ref cam)')
    ap.add_argument('--save-sheets', action='store_true', help='Write side-by-side images for flagged frames')
    ap.add_argument('--sheet-limit', type=int, default=300, help='Max contact sheets to save')
    ap.add_argument('--min-pts', type=int, default=7, help='Require ≥ this many joints per hand to attempt geometry')
    ap.add_argument('--margin', type=float, default=0.7, help='Swap margin: crossed ≤ margin × noncrossed')
    ap.add_argument('--max-frames', type=int, default=0, help='Process at most N frames (0=all)')
    ap.add_argument('--quiet', action='store_true', help='Less progress output')
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Paths per convention
    img_root, kp_root, calib_root, cam_dirs, kp_dirs = build_paths(args.dataset, args.cams, '.jpg')

    # Load calibrations (WORLD->COLOR) and sizes
    calibs = load_all_calibs(calib_root, args.cams)

    # Precompute F for pairs
    pair_list = []
    for pair_str in args.pairs:
        a,b = pair_str.split(',')
        if a not in calibs or b not in calibs:
            print(f"[WARN] Unknown camera id in pair {pair_str}", file=sys.stderr); continue
        Ca, Cb = calibs[a], calibs[b]
        F = fundamental_from_pair(Ca['K'], Cb['K'], Ca['R'], Ca['t'], Cb['R'], Cb['t'])
        pair_list.append((a,b,F))
    if not pair_list:
        print("No valid pairs. Abort."); sys.exit(1)

    # Load ROI masks if provided
    rois = load_rois(args.roi) if args.roi else {}

    # Build frame list (intersection)
    frames = list_common_frames(kp_dirs, args.basename_glob)
    if args.max_frames and len(frames) > args.max_frames:
        frames = frames[:args.max_frames]

    rows = []
    sheet_count = 0
    it = frames if args.quiet else tqdm(frames, desc="Frames")

    for frame in it:
        # Load per-cam keypoints and original JSON
        kp_by_cam = {}
        orig_by_cam = {}
        for c in args.cams:
            path = os.path.join(kp_dirs[c], frame + ".json")
            if not os.path.exists(path):
                kp_by_cam[c] = {'L':None,'R':None}; orig_by_cam[c]=None; continue
            kp, obj = load_kp_openpose(path)
            kp_by_cam[c] = kp
            orig_by_cam[c] = obj

        # ROI scores
        roi_scores = {}
        for c in args.cams:
            roi_entry = rois.get(c, None)
            roi_scores[c] = {
                'L': roi_score(kp_by_cam[c].get('L'), roi_entry),
                'R': roi_score(kp_by_cam[c].get('R'), roi_entry),
            }

        # ROI low flags (sheets + CSV rows)
        if args.roi:
            for c in args.cams:
                l_in = roi_scores[c]['L']; r_in = roi_scores[c]['R']
                if (l_in < args.roi_min) or (r_in < args.roi_min):
                    if args.save_sheets and sheet_count < args.sheet_limit:
                        img_path = find_image(cam_dirs[c], frame)
                        if img_path:
                            im = cv2.imread(img_path)
                            if im is not None:
                                draw_hand(im, kp_by_cam[c]['L'], (0,255,0)); draw_hand(im, kp_by_cam[c]['R'], (0,255,255))
                                lines = [f'ROI LOW cam={c} frame={frame}', f'roiL={l_in:.2f} roiR={r_in:.2f} (min {args.roi_min})']
                                out_img = os.path.join(args.out, 'sheets', f'ROI_{c}', f'{frame}.jpg')
                                make_contact_sheet(im, im, lines, out_img)
                                sheet_count += 1
                    rows.append({
                        'frame': frame, 'pair': f'{c}',
                        'd_LL': np.nan, 'd_RR': np.nan, 'd_LR': np.nan, 'd_RL': np.nan,
                        'crossed_better': False, 'margin': np.nan,
                        'swap_applied': False, 'swap_target': '', 'reason': 'roi_low',
                        'roiL_a': l_in, 'roiR_a': r_in,
                        'roiL_b': np.nan, 'roiR_b': np.nan,
                    })

        # Pairwise swap detection using epipolar geometry
        for (a,b,F) in pair_list:
            Ai, Aj = kp_by_cam[a], kp_by_cam[b]
            def count_good(arr):
                if arr is None: return 0
                return int(np.sum(~np.isnan(arr[:,0])))
            if min(count_good(Ai['L']), count_good(Ai['R']), count_good(Aj['L']), count_good(Aj['R'])) < args.min_pts:
                rows.append({
                    'frame': frame, 'pair': f'{a},{b}',
                    'd_LL': np.nan, 'd_RR': np.nan, 'd_LR': np.nan, 'd_RL': np.nan,
                    'crossed_better': False, 'margin': np.nan,
                    'swap_applied': False, 'swap_target': '', 'reason': 'too_few_points',
                    'roiL_a': roi_scores[a]['L'], 'roiR_a': roi_scores[a]['R'],
                    'roiL_b': roi_scores[b]['L'], 'roiR_b': roi_scores[b]['R'],
                })
                continue

            d_LL = sampson_distance(F, Ai['L'], Aj['L'])
            d_RR = sampson_distance(F, Ai['R'], Aj['R'])
            d_LR = sampson_distance(F, Ai['L'], Aj['R'])
            d_RL = sampson_distance(F, Ai['R'], Aj['L'])
            crossed_cost = d_LR + d_RL
            noncross_cost = d_LL + d_RR
            margin = crossed_cost / (noncross_cost + 1e-9)
            crossed_better = crossed_cost < noncross_cost and margin <= args.margin

            swap_target = None
            applied = False
            reason = ''

            if crossed_better:
                ref = args.ref_cam
                if ref == a:
                    swap_target = b
                elif ref == b:
                    swap_target = a
                else:
                    roi_a = max(roi_scores[a]['L'], roi_scores[a]['R'])
                    roi_b = max(roi_scores[b]['L'], roi_scores[b]['R'])
                    swap_target = a if roi_a < roi_b else b
                reason = 'crossed_assignment'

                # Apply swap and write JSON under out/keypoints/<cam>/<frame>.json
                if args.auto_swap and orig_by_cam.get(swap_target) is not None:
                    new_obj = swap_in_openpose35(json.loads(json.dumps(orig_by_cam[swap_target])))
                    out_json = os.path.join(args.out, 'keypoints', swap_target, frame + '.json')
                    os.makedirs(os.path.dirname(out_json), exist_ok=True)
                    with open(out_json, 'w') as f:
                        json.dump(new_obj, f, separators=(',',':'))
                    applied = True

                # Contact sheet
                if args.save_sheets and sheet_count < args.sheet_limit:
                    img_a = find_image(cam_dirs[a], frame)
                    img_b = find_image(cam_dirs[b], frame)
                    if img_a and img_b:
                        ia = cv2.imread(img_a); ib = cv2.imread(img_b)
                        if ia is not None and ib is not None:
                            draw_hand(ia, Ai['L'], (0,255,0)); draw_hand(ia, Ai['R'], (0,255,255))
                            draw_hand(ib, Aj['L'], (0,255,0)); draw_hand(ib, Aj['R'], (0,255,255))
                            lines = [
                                f'Pair {a},{b}  Frame {frame}',
                                f'd_LL={d_LL:.3f}  d_RR={d_RR:.3f}  d_LR={d_LR:.3f}  d_RL={d_RL:.3f}  margin={margin:.3f}',
                                f'SWAP → {swap_target}   reason={reason}'
                            ]
                            out_img = os.path.join(args.out, 'sheets', f'{a}_{b}', f'{frame}.jpg')
                            make_contact_sheet(ia, ib, lines, out_img)
                            sheet_count += 1

            rows.append({
                'frame': frame, 'pair': f'{a},{b}',
                'd_LL': d_LL, 'd_RR': d_RR, 'd_LR': d_LR, 'd_RL': d_RL,
                'crossed_better': bool(crossed_better), 'margin': float(margin),
                'swap_applied': bool(applied), 'swap_target': swap_target or '',
                'reason': reason or '',
                'roiL_a': roi_scores[a]['L'], 'roiR_a': roi_scores[a]['R'],
                'roiL_b': roi_scores[b]['L'], 'roiR_b': roi_scores[b]['R'],
            })

        # Mirror originals for cams not swapped this frame (so out/ mirrors input)
        if args.auto_swap:
            for c in args.cams:
                out_json = os.path.join(args.out, 'keypoints', c, frame + '.json')
                if os.path.exists(out_json):
                    continue
                in_json = os.path.join(kp_dirs[c], frame + '.json')
                if os.path.exists(in_json):
                    with open(in_json,'r') as f:
                        obj = json.load(f)
                    os.makedirs(os.path.dirname(out_json), exist_ok=True)
                    with open(out_json,'w') as f:
                        json.dump(obj, f, separators=(',',':'))

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out, 'qc_metrics.csv')
    df.to_csv(csv_path, index=False)
    if not args.quiet:
        print(f"\nWrote metrics: {csv_path}")
        if args.auto_swap:
            print(f"Corrected keypoints (and copies) under: {os.path.join(args.out,'keypoints')}")
        if args.save_sheets:
            print(f"Contact sheets under: {os.path.join(args.out,'sheets')}")

if __name__ == '__main__':
    main()
