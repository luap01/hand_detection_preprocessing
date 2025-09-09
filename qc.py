#!/usr/bin/env python3
"""
QC left/right hand keypoints in frames with colored overlays.

Assumptions:
- Left hand: red dots (keypoints), green skeleton lines.
- Right hand: blue dots and yellow skeleton lines.
- Anatomical left appears on the viewer's RIGHT side.

Outputs:
- CSV with per-frame issues and centroid info
- Optional overlays for flagged frames
"""

import os
import sys
import argparse
import cv2
import numpy as np
import pandas as pd

def color_masks(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Red (two hue bands)
    lower_red1 = np.array([0, 120, 80]); upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 80]); upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    # Blue
    lower_blue = np.array([95, 120, 80]); upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # Green (left-hand skeleton)
    lower_green = np.array([40, 70, 70]); upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    return red_mask, blue_mask, green_mask

def find_centroid(mask):
    ys, xs = np.where(mask > 0)
    count = xs.size
    if count == 0:
        return None, 0
    return (float(xs.mean()), float(ys.mean())), int(count)

def analyze_frame(path, red_thresh=30, blue_thresh=30, green_thresh=10):
    img = cv2.imread(path)
    if img is None:
        return {"frame": os.path.basename(path), "issues": "failed_to_load"}
    h, w = img.shape[:2]
    red_mask, blue_mask, green_mask, yellow_mask = color_masks(img)

    red_c, red_px = find_centroid(red_mask)
    blue_c, blue_px = find_centroid(blue_mask)
    green_c, green_px = find_centroid(green_mask)
    yellow_c, yellow_px = find_centroid(yellow_mask)

    issues = []
    if red_px < red_thresh: issues.append("missing_or_too_few_red_left")
    if blue_px < blue_thresh: issues.append("missing_or_too_few_blue_right")

    # Anatomical-left (red) should be to the viewer's right of blue
    if red_px >= red_thresh and blue_px >= blue_thresh and red_c and blue_c:
        red_x, blue_x = red_c[0], blue_c[0]
        if red_x <= blue_x:
            issues.append("red_not_to_right_of_blue_expected")
        if red_x < w*0.5:
            issues.append("left_hand_on_left_side")
        if blue_x > w*0.5:
            issues.append("right_hand_on_right_side")
        if abs(red_x - blue_x) < w*0.05:
            issues.append("hands_overlap_or_crossing")

    # If red is present but green skeleton is barely detected
    if red_px >= red_thresh and green_px < green_thresh:
        issues.append("green_skeleton_missing_for_left")

    return {
        "frame": os.path.basename(path),
        "width": w, "height": h,
        "red_pixels": int(red_px), "blue_pixels": int(blue_px), "green_pixels": int(green_px), "yellow_pixels": int(yellow_px),
        "red_cx": float("nan") if not red_c else float(red_c[0]),
        "red_cy": float("nan") if not red_c else float(red_c[1]),
        "blue_cx": float("nan") if not blue_c else float(blue_c[0]),
        "blue_cy": float("nan") if not blue_c else float(blue_c[1]),
        "issues": ";".join(issues) if issues else ""
    }

def save_overlay(img, red_mask, blue_mask, green_mask, red_c, blue_c, issues, out_path):
    overlay = img.copy()
    # Draw contours
    for mask, color in [(red_mask, (0,0,255)), (blue_mask, (255,0,0)), (green_mask, (0,255,0))]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)
    # Centroids
    if red_c:
        cv2.circle(overlay, (int(red_c[0]), int(red_c[1])), 6, (0,0,255), -1)
    if blue_c:
        cv2.circle(overlay, (int(blue_c[0]), int(blue_c[1])), 6, (255,0,0), -1)
    if issues:
        txt = "; ".join(issues)[:220]
        cv2.putText(overlay, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    cv2.imwrite(out_path, overlay)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_dir", help="Directory with frames (jpg/png)")
    ap.add_argument("--out_csv", default="hand_keypoint_qc.csv")
    ap.add_argument("--overlay_dir", default=None, help="Directory to write overlays for flagged frames")
    ap.add_argument("--max_overlays", type=int, default=100)
    ap.add_argument("--red_thresh", type=int, default=30)
    ap.add_argument("--blue_thresh", type=int, default=30)
    ap.add_argument("--green_thresh", type=int, default=10)
    args = ap.parse_args()

    files = [os.path.join(args.frames_dir, f) for f in os.listdir(args.frames_dir)
             if f.lower().endswith((".jpg",".jpeg",".png"))]
    files.sort()
    if not files:
        print("No images found.")
        sys.exit(1)

    rows = []
    overlay_count = 0
    if args.overlay_dir:
        os.makedirs(args.overlay_dir, exist_ok=True)

    for path in files:
        img = cv2.imread(path)
        if img is None:
            rows.append({"frame": os.path.basename(path), "issues": "failed_to_load"})
            continue
        red_mask, blue_mask, green_mask = color_masks(img)
        h, w = img.shape[:2]
        red_c, red_px = find_centroid(red_mask)
        blue_c, blue_px = find_centroid(blue_mask)
        _, green_px = find_centroid(green_mask)

        issues = []
        if red_px < args.red_thresh: issues.append("missing_or_too_few_red_left")
        if blue_px < args.blue_thresh: issues.append("missing_or_too_few_blue_right")
        if red_px >= args.red_thresh and blue_px >= args.blue_thresh and red_c and blue_c:
            if red_c[0] <= blue_c[0]: issues.append("red_not_to_right_of_blue_expected")
            if red_c[0] < w*0.5: issues.append("left_hand_on_left_side")
            if blue_c[0] > w*0.5: issues.append("right_hand_on_right_side")
            if abs(red_c[0] - blue_c[0]) < w*0.05: issues.append("hands_overlap_or_crossing")
        if red_px >= args.red_thresh and green_px < args.green_thresh:
            issues.append("green_skeleton_missing_for_left")

        rows.append({
            "frame": os.path.basename(path),
            "width": w, "height": h,
            "red_pixels": int(red_px), "blue_pixels": int(blue_px), "green_pixels": int(green_px),
            "red_cx": float("nan") if not red_c else float(red_c[0]),
            "red_cy": float("nan") if not red_c else float(red_c[1]),
            "blue_cx": float("nan") if not blue_c else float(blue_c[0]),
            "blue_cy": float("nan") if not blue_c else float(blue_c[1]),
            "issues": ";".join(issues) if issues else ""
        })

        if issues and args.overlay_dir and overlay_count < args.max_overlays:
            out_path = os.path.join(args.overlay_dir, os.path.basename(path))
            save_overlay(img, red_mask, blue_mask, green_mask, red_c, blue_c, issues, out_path)
            overlay_count += 1

    df = pd.DataFrame(rows).sort_values("frame")
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote CSV: {args.out_csv} with {len(df)} rows. Overlays saved: {overlay_count}")

if __name__ == "__main__":
    main()
