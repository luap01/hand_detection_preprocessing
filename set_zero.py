#!/usr/bin/env python3
"""
Script to set hand keypoints and confidences to zero for specific frames.

This script sets all keypoints to [0, 0] and all confidences to 0 in:
- results/<seq>/<hand>/<cam>/<frame>.json

Usage:
    python set_zero.py --seq SEQUENCE --hand HAND --cam CAMERA [--frame FRAME | --start-frame START --end-frame END] [--dry-run] [--visualize]
    
Arguments:
    --seq SEQUENCE       Sequence name (e.g., "20250519_Testing")
    --hand HAND          Hand type: "left" or "right"
    --cam CAMERA         Camera name (e.g., "camera05", "camera06")
    --frame FRAME        Single frame number to zero out (e.g., "001002" or "1002")
    --start-frame START  Starting frame number for range processing (e.g., "001000")
    --end-frame END      Ending frame number for range processing (e.g., "001010")
    --dry-run            Show what would be done without actually doing it
    --visualize          Show the result image with keypoints after processing
"""

import os
import sys
import argparse
import json
import cv2
import numpy as np
from pathlib import Path


def normalize_frame_number(frame):
    """Normalize frame number to standard format (6 digits with leading zeros)."""
    try:
        frame_num = int(frame)
        return f"{frame_num:06d}"
    except ValueError:
        return frame


def create_zero_keypoints():
    """Create a keypoint structure with all coordinates and confidences set to zero."""
    # 21 keypoints * 2 coordinates (x, y) = 42 values, all set to 0
    coordinates = [0] * 42
    
    # 21 confidence values, all set to 0
    confidences = [0] * 21
    
    return [coordinates, confidences]


def visualize_2d_points(points_2d, img, dot_colour, line_colour):
    """Visualize hand keypoints on an image."""
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),      # Index
        (0, 9), (9, 10), (10, 11), (11, 12), # Middle
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring
        (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
    ]

    # Draw keypoints
    for i, (x, y) in enumerate(points_2d):
        cv2.circle(img, (int(x), int(y)), 4, dot_colour, -1)

    # Draw connections
    for idx1, idx2 in HAND_CONNECTIONS:
        if idx1 < len(points_2d) and idx2 < len(points_2d):
            x1, y1 = int(points_2d[idx1][0]), int(points_2d[idx1][1])
            x2, y2 = int(points_2d[idx2][0]), int(points_2d[idx2][1])
            cv2.line(img, (x1, y1), (x2, y2), line_colour, 2)

    return img


def visualize_result(seq, hand, cam, frame):
    """Visualize the keypoints result after setting to zero."""
    # Color definitions
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    YELLOW = (0, 255, 255)
    BLUE = (255, 0, 0)
    
    frame_filename = f"{frame}.json"
    
    # Path to keypoint file
    kps_file = Path("results") / seq / hand / cam / frame_filename
    
    # Path to corresponding image (assuming it's in data/input structure)
    img_path = Path("data/input") / seq / cam / f"{frame}.jpg"
    
    # Alternative path if the first doesn't exist
    if not img_path.exists():
        img_path = Path("check_input_data") / cam / f"{frame}.jpg"
    
    try:
        # Load the image
        if not img_path.exists():
            print(f"Warning: Image file not found at {img_path}")
            print("Visualization skipped - image file needed for overlay")
            return False
            
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not load image from {img_path}")
            return False
        
        # Load keypoints
        with open(kps_file, 'r') as f:
            data = json.load(f)
        
        coordinates, confidences = data[0], data[1]
        
        # Reshape coordinates to (21, 2) format
        points_2d = np.array(coordinates, dtype=np.float32).reshape(21, 2)
        
        # Choose colors based on hand
        if hand == "left":
            dot_color, line_color = RED, GREEN
        else:
            dot_color, line_color = BLUE, YELLOW
        
        # Visualize keypoints
        img_with_kps = visualize_2d_points(points_2d, img.copy(), dot_color, line_color)
        
        # Create window and display
        window_name = f"Zero Keypoints: {seq}_{hand}_{cam}_{frame}"
        cv2.imshow(window_name, img_with_kps)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        
        print(f"\nVisualization displayed. All keypoints should be at (0,0).")
        print("Press any key to close the visualization window.")
        
        # Wait for key press and close
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return True
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        return False


def set_keypoints_to_zero(file_path, dry_run=False):
    """Set all keypoints and confidences in a JSON file to zero."""
    if not os.path.exists(file_path):
        return False, f"File does not exist: {file_path}"
    
    if dry_run:
        return True, "[DRY RUN] Would set keypoints to zero"
    
    try:
        # Create the zero keypoints structure
        zero_data = create_zero_keypoints()
        
        # Write the zero data to the file
        with open(file_path, 'w') as f:
            json.dump(zero_data, f, indent=4)
        
        return True, "Keypoints set to zero successfully"
        
    except Exception as e:
        return False, f"ERROR writing file: {e}"


def main():
    parser = argparse.ArgumentParser(description="Set hand keypoints and confidences to zero for specific frames")
    parser.add_argument("--seq", required=True, help="Sequence name (e.g., '20250519_Testing')")
    parser.add_argument("--hand", required=True, choices=["left", "right"], help="Hand type: left or right")
    parser.add_argument("--cam", required=True, help="Camera name (e.g., 'camera05', 'camera06')")
    parser.add_argument("--frame", help="Single frame number to zero out (e.g., '001002' or '1002')")
    parser.add_argument("--start-frame", help="Starting frame number for range processing (e.g., '001000')")
    parser.add_argument("--end-frame", help="Ending frame number for range processing (e.g., '001010')")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without actually doing it")
    parser.add_argument("--visualize", action="store_true", help="Show the result image with keypoints after processing")
    args = parser.parse_args()
    
    # Validate frame arguments
    if args.frame and (args.start_frame or args.end_frame):
        print("Error: Cannot use --frame with --start-frame/--end-frame. Choose either single frame or range processing.")
        return 1
    
    if not args.frame and not (args.start_frame and args.end_frame):
        print("Error: Must specify either --frame for single frame or both --start-frame and --end-frame for range processing.")
        return 1
    
    if args.start_frame and args.end_frame:
        start_num = int(normalize_frame_number(args.start_frame))
        end_num = int(normalize_frame_number(args.end_frame))
        if start_num > end_num:
            print("Error: start-frame must be less than or equal to end-frame.")
            return 1
    
    # Path to the keypoints directory
    base_path = Path("results")
    
    if not base_path.exists():
        print(f"Error: Directory {base_path} does not exist!")
        return 1
    
    # Check if sequence exists
    seq_path = base_path / args.seq
    if not seq_path.exists():
        print(f"Error: Sequence {args.seq} does not exist in {base_path}")
        return 1
    
    # Check if hand directory exists
    hand_path = seq_path / args.hand
    if not hand_path.exists():
        print(f"Error: Hand directory {args.hand} does not exist in sequence {args.seq}")
        return 1
    
    # Check if camera directory exists
    cam_path = hand_path / args.cam
    if not cam_path.exists():
        print(f"Error: Camera directory {args.cam} does not exist in {args.seq}/{args.hand}")
        return 1
    
    # Determine frames to process
    if args.frame:
        # Single frame processing
        frames_to_process = [normalize_frame_number(args.frame)]
        print(f"Processing single frame: {frames_to_process[0]}")
    else:
        # Range processing
        start_frame = normalize_frame_number(args.start_frame)
        end_frame = normalize_frame_number(args.end_frame)
        start_num = int(start_frame)
        end_num = int(end_frame)
        frames_to_process = [f"{i:06d}" for i in range(start_num, end_num + 1)]
        print(f"Processing frame range: {start_frame} to {end_frame} ({len(frames_to_process)} frames)")
    
    if args.dry_run:
        print("\n*** DRY RUN MODE - No actual changes will be made ***")
    
    # Process each frame
    total_processed = 0
    total_failed = 0
    successful_frames = []
    
    for frame in frames_to_process:
        frame_filename = f"{frame}.json"
        file_path = base_path / args.seq / args.hand / args.cam / frame_filename
        
        print(f"\nProcessing frame {frame}...")
        
        success, message = set_keypoints_to_zero(str(file_path), args.dry_run)
        
        if success:
            print(f"  ✓ {message}")
            total_processed += 1
            successful_frames.append(frame)
        else:
            print(f"  ✗ {message}")
            total_failed += 1
    
    # Summary
    print(f"\n--- Processing Summary ---")
    print(f"Successfully {'would process' if args.dry_run else 'processed'}: {total_processed} frames")
    print(f"Failed: {total_failed} frames")
    
    if not args.dry_run and total_processed > 0:
        print(f"\nKeypoints in {args.seq}/{args.hand}/{args.cam} have been set to zero for {total_processed} frame(s).")
        
        # Handle visualization for successful frames
        if args.visualize and successful_frames:
            if len(successful_frames) == 1:
                print("\nShowing visualization...")
                visualize_result(args.seq, "left" if args.hand == "right" else "right", args.cam, successful_frames[0])
            else:
                print(f"\nVisualization requested for {len(successful_frames)} frames.")
                response = input("Show visualization for each frame? (y/n/first/last): ").lower().strip()
                
                if response == 'y' or response == 'yes':
                    for frame in successful_frames:
                        print(f"\nShowing visualization for frame {frame}...")
                        visualize_result(args.seq, "left" if args.hand == "right" else "right", args.cam, frame)
                elif response == 'first':
                    print(f"\nShowing visualization for first frame ({successful_frames[0]})...")
                    visualize_result(args.seq, "left" if args.hand == "right" else "right", args.cam, successful_frames[0])
                elif response == 'last':
                    print(f"\nShowing visualization for last frame ({successful_frames[-1]})...")
                    visualize_result(args.seq, "left" if args.hand == "right" else "right", args.cam, successful_frames[-1])
                else:
                    print("Skipping visualization.")
    
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
