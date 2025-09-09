import os
import shutil
from pathlib import Path
from collections import defaultdict
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Export JSON files that exist across multiple cameras')
    parser.add_argument('--cameras', nargs='+', type=int, default=[1, 4, 5, 6], 
                       help='Camera indices to check for common frames')
    parser.add_argument('--hands', nargs='+', default=['left', 'right'],
                       help='Hand types to process (left, right)')
    parser.add_argument('--category', default='success', 
                       help='Categories to process (success, partial, failure)')
    parser.add_argument('--dataset', default="20250519_Testing")
    parser.add_argument('--base_path', default='./results', help='Input directory path')
    parser.add_argument('--export_path', default='./export', help='Export directory path')
    return parser.parse_args()

def extract_frame_number(filename):
    """Extract frame number from filename (e.g., '000765.json' -> '000765')"""
    return Path(filename).stem

def find_common_frames(base_path, dataset, cameras, category='success', hand_type='left'):
    """Find frame numbers that exist across all specified cameras for the given hand type"""
    
    # Dictionary to track which cameras have each frame
    frame_cameras = defaultdict(set)
    
    # Scan all cameras for the specified hand type
    for cam_idx in cameras:
        cam_dir = Path(base_path) / dataset / category / hand_type / f"camera{cam_idx:02d}"
        
        if not cam_dir.exists():
            print(f"Warning: Directory {cam_dir} does not exist")
            continue
                
        # Get all JSON files in this camera directory
        json_files = list(cam_dir.glob("*.json"))
        
        for json_file in json_files:
            frame_num = extract_frame_number(json_file.name)
            frame_cameras[frame_num].add(cam_idx)
    
    # Find frames that exist in ALL specified cameras
    target_cameras = set(cameras)
    common_frames = []
    
    for frame_num, available_cameras in frame_cameras.items():
        if target_cameras.issubset(available_cameras):
            common_frames.append(frame_num)
    
    return sorted(common_frames)

def copy_files(base_path, export_path, cameras, hands, common_frames, category, dataset):
    """Copy JSON files for common frames to export directory"""
    
    export_base = Path(export_path)
    stats = {
        'copied': 0,
        'skipped': 0,
        'errors': 0,
        'skipped_files': []
    }
    
    for hand_type in hands:
        for cam_idx in cameras:
            cam_dir = Path(base_path) / dataset / category / hand_type / f"camera{cam_idx:02d}"
            export_cam_dir = export_base / dataset / category / hand_type / f"camera{cam_idx:02d}"
            
            # Create export directory
            export_cam_dir.mkdir(parents=True, exist_ok=True)
            
            if not cam_dir.exists():
                continue
            
            # Copy files for common frames
            for frame_num in common_frames:
                source_file = cam_dir / f"{frame_num}.json"
                target_file = export_cam_dir / f"{frame_num}.json"
                
                if source_file.exists():
                    try:
                        shutil.copy2(source_file, target_file)
                        stats['copied'] += 1
                    except Exception as e:
                        print(f"Error copying {source_file}: {e}")
                        stats['errors'] += 1
                else:
                    stats['skipped'] += 1
                    stats['skipped_files'].append(str(source_file))
    
    return stats

def main():
    args = parse_args()
    print(f"Looking for common frames across cameras: {args.cameras}")
    print(f"Hand types: {args.hands}")
    print(f"Dataset: {args.dataset}")
    print(f"Category: {args.category}")
    print(f"Base path: {args.base_path}")
    print(f"Export path: {args.export_path}")
    print("-" * 60)
    
    # Find common frames for both left and right hands
    print("Scanning for common frames based on left hand files...")
    left_common_frames = find_common_frames(
        args.base_path, 
        args.dataset,
        args.cameras, 
        args.category,
        hand_type='left'
    )
    
    print(f"Found {len(left_common_frames)} common frames for left hand across all specified cameras")
    
    print("Scanning for common frames based on right hand files...")
    right_common_frames = find_common_frames(
        args.base_path, 
        args.dataset,
        args.cameras, 
        args.category,
        hand_type='right'
    )
    
    print(f"Found {len(right_common_frames)} common frames for right hand across all specified cameras")
    
    # Find intersection - frames that exist for both hands across all cameras
    common_frames = sorted(list(set(left_common_frames) & set(right_common_frames)))
    
    print(f"Found {len(common_frames)} frames that exist for BOTH hands across all specified cameras")
    
    if len(common_frames) == 0:
        print("No common frames found for both hands. Exiting.")
        return
    
    # Show first and last few frame numbers
    if len(common_frames) <= 10:
        print(f"Common frames: {common_frames}")
    else:
        print(f"Common frames: {common_frames[:5]} ... {common_frames[-5:]}")
    
    # Ask for confirmation
    response = input(f"\nProceed with copying {len(common_frames)} frames for {args.hands} hands? (y/n): ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        return
    
    # Copy files
    print("\nCopying files...")
    stats = copy_files(
        args.base_path,
        args.export_path,
        args.cameras,
        args.hands,
        common_frames,
        args.category,
        args.dataset
    )
    
    print("\nCopy operation completed!")
    print(f"Files copied: {stats['copied']}")
    print(f"Files skipped: {stats['skipped']}")
    print(f"Errors: {stats['errors']}")
    
    # Show skipped files if any
    if stats['skipped'] > 0:
        print(f"\nSkipped files:")
        for skipped_file in stats['skipped_files']:
            print(f"  - {skipped_file}")
    
    # Show export structure
    export_path = Path(args.export_path)
    if export_path.exists():
        print(f"\nExported files are in: {export_path}")
        print("Directory structure:")
        for hand_type in args.hands:
            for camera in args.cameras:
                cam_dir = export_path / args.category / hand_type / f"camera{camera:02d}"
                if cam_dir.exists():
                    file_count = len(list(cam_dir.glob("*.json")))
                    print(f"  {hand_type}/camera{camera:02d}/: {file_count} files")

if __name__ == "__main__":
    main() 