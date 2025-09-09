import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np

import cv2


HAND_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]


def load_json(json_path: Path) -> dict:
    with open(json_path, "r") as f:
        return json.load(f)


def save_json(json_path: Path, data: dict, make_backup: bool = True) -> None:
    """Save JSON with optional backup."""
    if make_backup:
        bak = json_path.with_suffix(json_path.suffix + ".bak")
        if not bak.exists():
            try:
                with open(bak, "w") as f:
                    json.dump(load_json(json_path), f, indent=2)
            except Exception:
                pass
    tmp = json_path.with_suffix(json_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, json_path)


def extract_points_and_shift(person: dict, side: str) -> Tuple[List[Tuple[float, float]], Tuple[int, int]]:
    """Extract 21 (x, y) points and (x_shift, y_shift) from the mediapipe export format.

    The format stores flattened triplets (x, y, z_or_score) for 21 keypoints in
    `hand_{side}_keypoints_2d`, and a `hand_{side}_shift` array with [x_shift, y_shift].
    """
    key = f"hand_{side}_keypoints_2d"
    shift_key = f"hand_{side}_shift"

    if key not in person or shift_key not in person:
        return [], (0, 0)

    flat = person[key]
    if not isinstance(flat, list) or len(flat) < 63:  # 21 * 3
        return [], (0, 0)

    points: List[Tuple[float, float]] = []
    for i in range(0, len(flat), 3):
        x = float(flat[i])
        y = float(flat[i + 1])
        points.append((x, y))

    shift = person.get(shift_key, [0, 0])
    if not isinstance(shift, list) or len(shift) != 2:
        shift = [0, 0]

    return points[:21], (int(shift[0]), int(shift[1]))


def has_hand_data(person: dict, side: str) -> bool:
    """Check if a person has valid hand data for the given side."""
    key = f"hand_{side}_keypoints_2d"
    shift_key = f"hand_{side}_shift"
    
    if key not in person or shift_key not in person:
        return False
    
    flat = person[key]
    if not isinstance(flat, list) or len(flat) < 63:  # 21 * 3
        return False
    
    # Check if keypoints are not all zeros (indicating actual detection)
    for i in range(0, len(flat), 3):
        if flat[i] != 0 or flat[i + 1] != 0:  # If any point is non-zero
            return True
    
    return False


def hands_overlap(person: dict, threshold: float = 50.0) -> bool:
    """Check if left and right hands are overlapping based on centroid distance."""
    if not (has_hand_data(person, "left") and has_hand_data(person, "right")):
        return False
    
    left_points, left_shift = extract_points_and_shift(person, "left")
    right_points, right_shift = extract_points_and_shift(person, "right")
    
    if not left_points or not right_points:
        return False
    
    # Calculate centroids with shifts applied
    left_centroid = np.mean([(x + left_shift[0], y + left_shift[1]) for x, y in left_points], axis=0)
    right_centroid = np.mean([(x + right_shift[0], y + right_shift[1]) for x, y in right_points], axis=0)
    
    # Calculate distance between centroids
    distance = np.linalg.norm(left_centroid - right_centroid)
    return distance < threshold


def zero_out_hand(person: dict, side: str) -> None:
    """Zero out hand data for the specified side."""
    keys_to_zero = [
        f"hand_{side}_keypoints_2d",
        f"hand_{side}_shift",
        f"hand_{side}_conf"
    ]
    
    for key in keys_to_zero:
        if key in person:
            if "keypoints_2d" in key:
                person[key] = [0.0] * 63  # 21 points * 3 values each
            elif "shift" in key:
                person[key] = [0, 0]
            elif "conf" in key:
                person[key] = [0.0]


def swap_hands_in_person(person: dict) -> bool:
    """
    Swap left/right hand data. Works even when only one hand is detected.
    Returns True if any swap occurred.
    """
    left_keys = {
        "kps": "hand_left_keypoints_2d",
        "shift": "hand_left_shift", 
        "conf": "hand_left_conf",
    }
    right_keys = {
        "kps": "hand_right_keypoints_2d",
        "shift": "hand_right_shift",
        "conf": "hand_right_conf",
    }

    # Check which hands have actual data
    has_left_data = has_hand_data(person, "left")
    has_right_data = has_hand_data(person, "right")
    
    # If no hands detected at all, nothing to swap
    if not (has_left_data or has_right_data):
        return False

    # Store current values (including None/empty values)
    tmp_storage = {}
    for key_type, left_key in left_keys.items():
        tmp_storage[f"left_{key_type}"] = person.get(left_key)
    for key_type, right_key in right_keys.items():
        tmp_storage[f"right_{key_type}"] = person.get(right_key)

    # Clear existing keys completely
    for left_key in left_keys.values():
        person.pop(left_key, None)
    for right_key in right_keys.values():
        person.pop(right_key, None)

    # Swap: left becomes right, right becomes left
    for key_type in ["kps", "shift", "conf"]:
        left_val = tmp_storage[f"left_{key_type}"]
        right_val = tmp_storage[f"right_{key_type}"]
        
        # Set right side to what was on left (even if None/empty)
        if left_val is not None:
            person[right_keys[key_type]] = left_val
            
        # Set left side to what was on right (even if None/empty)  
        if right_val is not None:
            person[left_keys[key_type]] = right_val

    return True


def draw_hand(img, points: List[Tuple[float, float]], shift: Tuple[int, int], dot_color, line_color):
    if not points:
        return
    sx, sy = shift
    # draw points
    for (x, y) in points:
        cv2.circle(img, (int(x) + sx, int(y) + sy), 3, dot_color, -1)
    # draw connections
    for a, b in HAND_CONNECTIONS:
        if a < len(points) and b < len(points):
            x1, y1 = int(points[a][0]) + sx, int(points[a][1]) + sy
            x2, y2 = int(points[b][0]) + sx, int(points[b][1]) + sy
            cv2.line(img, (x1, y1), (x2, y2), line_color, 1)


def overlay_text(img, lines, origin=(10, 22)):
    """Overlay text with outline for better visibility."""
    x, y = origin
    for line in lines:
        cv2.putText(img, line, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22


def keycode(k: int) -> str:
    """Normalize key codes to actions."""
    if k == 27: return "ESC"
    if k in (ord('q'), ord('Q')): return "q"
    if k in (ord('d'), ord('D')): return "next"
    if k in (ord('a'), ord('A')): return "prev"
    if k in (ord('s'), ord('S')): return "swap"
    if k in (ord('r'), ord('R')): return "reload"
    if k in (ord('z'), ord('Z')): return "zero_left"
    if k in (ord('x'), ord('X')): return "zero_right"
    if k in (13, 32): return "next"           # Enter or Space
    if k == 81: return "prev"                 # Left arrow
    if k == 83: return "next"                 # Right arrow
    if k == 8:  return "prev"                 # Backspace
    return ""


def interactive_viewer(
    output_root: Path,
    dataset: str,
    variant: str,
    input_root: Path,
    save_overlay: bool = False,
) -> None:
    """
    Interactive visualization with navigation, swapping, and overlap detection.
    """
    dataset_out_dir = output_root / dataset / variant
    dataset_in_dir = input_root / dataset

    if not dataset_out_dir.exists():
        print(f"Output directory not found: {dataset_out_dir}")
        return
    if not dataset_in_dir.exists():
        print(f"Input images directory not found: {dataset_in_dir}")
        return

    cameras = sorted([d for d in dataset_out_dir.iterdir() if d.is_dir() and d.name.startswith("camera")])
    if not cameras:
        print(f"No camera directories found in: {dataset_out_dir}")
        return

    cv2.namedWindow("hand_viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("hand_viewer", 1400, 900)

    for cam_dir in cameras:
        cam_name = cam_dir.name
        json_dir = cam_dir / "json"
        if not json_dir.exists():
            print(f"Skipping {cam_name}: no json directory at {json_dir}")
            continue

        img_in_dir = dataset_in_dir / cam_name
        if not img_in_dir.exists():
            print(f"Skipping {cam_name}: input image directory missing at {img_in_dir}")
            continue

        json_files = sorted([p for p in json_dir.glob("*.json")])
        if not json_files:
            print(f"No JSON frames in {json_dir}")
            continue

        idx = 0
        modified_flags = [False] * len(json_files)  # Track if a frame was modified & saved
        cached_data = {}  # Cache loaded data to track modifications

        while True:
            # Handle bounds properly
            if idx < 0:
                idx = 0
            elif idx >= len(json_files):
                break  # Exit camera loop when we go past the last frame

            jf = json_files[idx]
            frame_key = str(jf)
            img_path = img_in_dir / f"{jf.stem}.jpg"
            
            # Load data (use cache if available, otherwise load from disk)
            try:
                if frame_key not in cached_data:
                    cached_data[frame_key] = load_json(jf)
                data = cached_data[frame_key]
            except Exception as e:
                print(f"  Failed to read {jf}: {e}")
                # Skip this frame
                idx += 1
                continue

            people = data.get("people", [])
            # Choose first person if multiple
            person = people[0] if people else {}

            # Check for overlapping hands and auto-zero one if needed
            if hands_overlap(person):
                print(f"  Overlapping hands detected in {jf.stem}, zeroing out left hand")
                zero_out_hand(person, "left")
                # Save the change immediately
                try:
                    save_json(jf, data, make_backup=True)
                    modified_flags[idx] = True
                    cached_data[frame_key] = load_json(jf)
                except Exception as e:
                    print(f"  Failed to save overlap fix to {jf}: {e}")

            # Prepare image
            img = cv2.imread(str(img_path)) if img_path.exists() else None
            if img is None:
                # Create placeholder canvas to still allow working with JSON without image
                img = np.full((720, 1280, 3), 255, dtype=np.uint8)

            # Extract and draw
            left_points, left_shift = extract_points_and_shift(person, "left")
            right_points, right_shift = extract_points_and_shift(person, "right")

            # Draw both hands with different colors
            # Due to mediapipe's convention of looking at the image not anatomically, the right hand is the left hand and the left hand is the right hand
            draw_hand(img, right_points, right_shift, (0, 0, 255), (0, 255, 0))   # right: red dots, green lines
            draw_hand(img, left_points, left_shift, (255, 0, 0), (0, 255, 255))   # left: blue dots, yellow lines

            # Overlay info
            status = "MODIFIED (unsaved)" if frame_key in cached_data and not modified_flags[idx] else \
                     "MODIFIED (saved)" if modified_flags[idx] else "Loaded"
            
            # Add hand detection status
            left_detected = "✓" if has_hand_data(person, "left") else "✗"
            right_detected = "✓" if has_hand_data(person, "right") else "✗"
            overlap_status = "OVERLAP!" if hands_overlap(person) else ""
            
            lines = [
                f"Camera: {cam_name}   Frame: {jf.stem}   [{idx+1}/{len(json_files)}]",
                "Controls: ←/a/backspace prev | →/d/space/enter next | s swap | r reload | z zero-left | x zero-right | q/ESC quit",
                f"JSON path: {jf.name}",
                f"Status: {status} {overlap_status}",
                f"Hands detected: Left {left_detected} | Right {right_detected}",
                "Legend: Right=red+green, Left=blue+yellow",
            ]
            overlay_text(img, lines, origin=(10, 26))

            cv2.imshow("hand_viewer", img)
            k = cv2.waitKey(0) & 0xFF
            action = keycode(k)

            if action in ("q", "ESC"):
                cv2.destroyAllWindows()
                return
            elif action == "swap":
                if not people:
                    # Create 'people' if missing so swap produces valid structure
                    data["people"] = [{}]
                    person = data["people"][0]
                    cached_data[frame_key] = data
                
                changed = swap_hands_in_person(person)
                if changed:
                    try:
                        save_json(jf, data, make_backup=True)
                        modified_flags[idx] = True
                        print(f"  Swapped and saved hands for {jf.stem}")
                        # Update cache with saved data
                        cached_data[frame_key] = load_json(jf)
                    except Exception as e:
                        print(f"  Failed to save swap to {jf}: {e}")
                else:
                    print(f"  Nothing to swap in {jf.stem} (no hand data present)")
                # After swap, continue loop to redraw with updated data
            elif action == "zero_left":
                zero_out_hand(person, "left")
                try:
                    save_json(jf, data, make_backup=True)
                    modified_flags[idx] = True
                    print(f"  Zeroed left hand for {jf.stem}")
                    cached_data[frame_key] = load_json(jf)
                except Exception as e:
                    print(f"  Failed to save zero-left to {jf}: {e}")
            elif action == "zero_right":
                zero_out_hand(person, "right")
                try:
                    save_json(jf, data, make_backup=True)
                    modified_flags[idx] = True
                    print(f"  Zeroed right hand for {jf.stem}")
                    cached_data[frame_key] = load_json(jf)
                except Exception as e:
                    print(f"  Failed to save zero-right to {jf}: {e}")
            elif action == "reload":
                # Force reload from disk, discarding cached changes
                try:
                    cached_data[frame_key] = load_json(jf)
                    modified_flags[idx] = False  # Reset modified flag
                    print(f"  Reloaded {jf.stem} from disk")
                except Exception as e:
                    print(f"  Failed to reload {jf}: {e}")
            elif action == "prev":
                idx -= 1
                # Don't clamp here, let the while loop handle bounds
            else:  # default to next for any other key or "next"
                idx += 1
                # Don't clamp here, let the while loop handle bounds

            # Optionally save an overlay image for auditing (not overwriting inputs)
            if save_overlay:
                audit_dir = cam_dir / "overlay_preview"
                audit_dir.mkdir(parents=True, exist_ok=True)
                out_path = audit_dir / f"{jf.stem}.jpg"
                try:
                    cv2.imwrite(str(out_path), img)
                except Exception:
                    pass

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive viewer for mediapipe keypoints with navigation, swapping, and overlap detection"
    )
    parser.add_argument(
        "--dataset",
        default="20250206_Testing",
        help="Dataset name under data roots (e.g., 20250206 or 20250206_Testing)",
    )
    parser.add_argument(
        "--variant",
        default="mediapipe_0.35",
        help="Output variant folder name under dataset (default: mediapipe_0.35)",
    )
    parser.add_argument(
        "--output-root",
        default="data/output",
        help="Root directory where JSON outputs live (default: data/output)",
    )
    parser.add_argument(
        "--input-root",
        default="data/input",
        help="Root directory where input images live (default: data/input)",
    )
    parser.add_argument(
        "--save-overlay",
        action="store_true",
        help="If set, stores a drawn preview image per frame under each camera/overlay_preview/",
    )

    args = parser.parse_args()

    interactive_viewer(
        output_root=Path(args.output_root),
        dataset=args.dataset,
        variant=args.variant,
        input_root=Path(args.input_root),
        save_overlay=args.save_overlay,
    )


if __name__ == "__main__":
    main()


