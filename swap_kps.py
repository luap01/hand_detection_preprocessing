#!/usr/bin/env python3
"""
Script to swap left and right hand keypoint files for specific frames in the OR dataset.

This script swaps individual frame files between:
- data/OR/rgb_2D_keypoints/<seq>/left/<cam>/<frame>.json <-> data/OR/rgb_2D_keypoints/<seq>/right/<cam>/<frame>.json

Usage:
    python swap_kps.py --frame FRAME [--seq SEQUENCE] [--cam CAMERA] [--dry-run]
    
Arguments:
    --frame FRAME     Frame number to swap (e.g., "017997" or "17997")
    --seq SEQUENCE    Only swap in a specific sequence (e.g., "20250519")
    --cam CAMERA      Only swap in a specific camera (e.g., "camera01")
    --dry-run         Show what would be swapped without actually doing it
"""

import os
import sys
import argparse
import shutil
import tempfile
from pathlib import Path


def get_sequences(base_path):
    """Get all sequence directories in the keypoints path."""
    sequences = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            sequences.append(item)
    return sorted(sequences)


def get_cameras(seq_path, hand):
    """Get all camera directories in a sequence/hand path."""
    hand_path = seq_path / hand
    if not hand_path.exists():
        return []
    
    cameras = []
    for item in os.listdir(hand_path):
        item_path = hand_path / item
        if item_path.is_dir() and not item.startswith('.'):
            cameras.append(item)
    return sorted(cameras)


def normalize_frame_number(frame):
    """Normalize frame number to standard format (6 digits with leading zeros)."""
    try:
        frame_num = int(frame)
        return f"{frame_num:06d}"
    except ValueError:
        return frame


def swap_files(left_file, right_file, dry_run=False):
    """Safely swap two files using temporary names."""
    if not os.path.exists(left_file) and not os.path.exists(right_file):
        return False, "Neither file exists"
    
    if not os.path.exists(left_file):
        return False, f"Left file {left_file} doesn't exist"
        
    if not os.path.exists(right_file):
        return False, f"Right file {right_file} doesn't exist"
    
    if dry_run:
        return True, "[DRY RUN] Would swap files"
    
    # Create temporary file in the same directory as left file
    temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(left_file), prefix='.swap_temp_', suffix='.json')
    os.close(temp_fd)  # Close the file descriptor immediately
    
    try:
        # Step 1: Copy left to temp (use copy to preserve original during swap)
        shutil.copy2(left_file, temp_path)
        
        # Step 2: Copy right to left
        shutil.copy2(right_file, left_file)
        
        # Step 3: Copy temp to right
        shutil.copy2(temp_path, right_file)
        
        # Step 4: Remove temp file
        os.remove(temp_path)
        
        return True, "Files swapped successfully"
        
    except Exception as e:
        # Clean up temp file if it exists
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return False, f"ERROR during swap: {e}"


def main():
    parser = argparse.ArgumentParser(description="Swap left and right hand keypoint files for specific frames")
    parser.add_argument("--frame", required=True, help="Frame number to swap (e.g., '017997' or '17997')")
    parser.add_argument("--seq", help="Only swap in a specific sequence (e.g., '20250519')")
    parser.add_argument("--cam", help="Only swap in a specific camera (e.g., 'camera01')")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without actually doing it")
    args = parser.parse_args()
    
    # Path to the keypoints directory
    base_path = Path("results/")
    
    if not base_path.exists():
        print(f"Error: Directory {base_path} does not exist!")
        return 1
    
    # Normalize frame number
    frame = normalize_frame_number(args.frame)
    frame_filename = f"{frame}.json"
    
    # Get sequences to process
    if args.seq:
        if (base_path / args.seq).exists():
            sequences = [args.seq]
        else:
            print(f"Error: Sequence {args.seq} does not exist!")
            return 1
    else:
        sequences = get_sequences(base_path)
    
    if not sequences:
        print("No sequences found to process.")
        return 0
    
    print(f"Swapping frame {frame} in {len(sequences)} sequence(s): {', '.join(sequences)}")
    
    if args.dry_run:
        print("\n*** DRY RUN MODE - No actual changes will be made ***")
    
    total_swapped = 0
    total_failed = 0
    
    for seq in sequences:
        seq_path = base_path / seq
        
        print(f"\nProcessing sequence: {seq}")
        
        # Check if both left and right directories exist
        left_dir = seq_path / "left"
        right_dir = seq_path / "right"
        
        if not left_dir.exists():
            print(f"  Left directory doesn't exist: {left_dir}")
            continue
            
        if not right_dir.exists():
            print(f"  Right directory doesn't exist: {right_dir}")
            continue
        
        # Get cameras to process
        if args.cam:
            cameras = [args.cam] if (left_dir / args.cam).exists() and (right_dir / args.cam).exists() else []
            if not cameras:
                print(f"  Camera {args.cam} doesn't exist in both left and right directories")
                continue
        else:
            # Get cameras that exist in both left and right
            left_cameras = set(get_cameras(seq_path, "left"))
            right_cameras = set(get_cameras(seq_path, "right"))
            cameras = sorted(left_cameras.intersection(right_cameras))
        
        if not cameras:
            print(f"  No cameras found to process in sequence {seq}")
            continue
        
        print(f"  Processing {len(cameras)} camera(s): {', '.join(cameras)}")
        
        seq_swapped = 0
        seq_failed = 0
        
        for cam in cameras:
            left_file = left_dir / cam / frame_filename
            right_file = right_dir / cam / frame_filename
            
            success, message = swap_files(str(left_file), str(right_file), args.dry_run)
            
            if success:
                print(f"    {cam}: ✓ {message}")
                seq_swapped += 1
            else:
                print(f"    {cam}: ✗ {message}")
                seq_failed += 1
        
        total_swapped += seq_swapped
        total_failed += seq_failed
        
        print(f"  Sequence {seq}: {seq_swapped} swapped, {seq_failed} failed")
    
    print(f"\nSummary:")
    print(f"  Successfully {'would swap' if args.dry_run else 'swapped'}: {total_swapped} files")
    print(f"  Failed: {total_failed} files")
    
    if not args.dry_run and total_swapped > 0:
        print(f"\nSwap complete! Frame {frame} keypoint files have been exchanged between left and right hands.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
