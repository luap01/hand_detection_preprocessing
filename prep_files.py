import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple, Optional


def find_camera_files(source_dir: str, pattern: str = r"color_(\d+)_camera(\d+)\.jpg") -> List[Tuple[str, str, str]]:
    """
    Find files matching the camera pattern and extract index and camera number.
    
    Args:
        source_dir: Directory to search for files
        pattern: Regex pattern to match files (default matches color_<idx>_camera<cam>.jpg)
    
    Returns:
        List of tuples: (original_filename, idx, cam_num)
    """
    source_path = Path(source_dir)
    matches = []
    
    if not source_path.exists():
        print(f"Source directory {source_dir} does not exist!")
        return matches
    
    pattern_re = re.compile(pattern)
    
    for file_path in source_path.iterdir():
        if file_path.is_file():
            match = pattern_re.match(file_path.name)
            if match:
                idx = match.group(1)
                cam_num = match.group(2)
                matches.append((str(file_path), idx, cam_num))
    
    return matches


def organize_camera_files(source_dir: str, 
                         output_dir: str, 
                         pattern: str = r"color_(\d+)_camera(\d+)\.jpg",
                         copy_files: bool = False,
                         dry_run: bool = False) -> None:
    """
    Organize camera files by renaming and moving them to camera-specific folders.
    
    Args:
        source_dir: Directory containing the original files
        output_dir: Base directory where camera folders will be created
        pattern: Regex pattern to match files
        copy_files: If True, copy files instead of moving them
        dry_run: If True, only show what would be done without actually doing it
    """
    matches = find_camera_files(source_dir, pattern)
    
    if not matches:
        print(f"No files matching pattern found in {source_dir}")
        return
    
    print(f"Found {len(matches)} files matching the pattern")
    
    output_path = Path(output_dir)
    
    # Group files by camera number for better organization
    camera_groups = {}
    for original_file, idx, cam_num in matches:
        if cam_num not in camera_groups:
            camera_groups[cam_num] = []
        camera_groups[cam_num].append((original_file, idx))
    
    for cam_num, files in camera_groups.items():
        # Create camera directory
        cam_dir = output_path / f"camera{cam_num.zfill(2)}"  # e.g., camera01, camera02
        
        if not dry_run:
            cam_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing camera{cam_num.zfill(2)} ({len(files)} files):")
        
        for original_file, idx in files:
            # New filename: just the index with .jpg extension
            new_filename = f"{idx}.jpg"
            new_file_path = cam_dir / new_filename
            
            if dry_run:
                action = "COPY" if copy_files else "MOVE"
                print(f"  {action}: {Path(original_file).name} -> {cam_dir.name}/{new_filename}")
            else:
                try:
                    if copy_files:
                        shutil.copy2(original_file, new_file_path)
                        print(f"  COPIED: {Path(original_file).name} -> {cam_dir.name}/{new_filename}")
                    else:
                        shutil.move(original_file, new_file_path)
                        print(f"  MOVED: {Path(original_file).name} -> {cam_dir.name}/{new_filename}")
                except Exception as e:
                    print(f"  ERROR: Failed to process {Path(original_file).name}: {e}")


def main():
    """
    Main function to demonstrate usage of the file organization functions.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Organize camera files by renaming and moving them")
    parser.add_argument("source_dir", help="Source directory containing the files")
    parser.add_argument("output_dir", help="Output directory where camera folders will be created")
    parser.add_argument("--pattern", default=r"color_(\d+)_camera(\d+)\.jpg", 
                       help="Regex pattern to match files (default: color_<idx>_camera<cam>.jpg)")
    parser.add_argument("--copy", action="store_true", 
                       help="Copy files instead of moving them")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be done without actually doing it")
    
    args = parser.parse_args()
    
    # Show preview first
    print("=== FILE ORGANIZATION PREVIEW ===")
    organize_camera_files(args.source_dir, args.output_dir, args.pattern, args.copy, dry_run=True)
    
    if not args.dry_run:
        print("\n=== PROCEEDING WITH ACTUAL OPERATION ===")
        response = input("Do you want to proceed? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            organize_camera_files(args.source_dir, args.output_dir, args.pattern, args.copy, dry_run=False)
            print("\nOperation completed!")
        else:
            print("Operation cancelled.")


if __name__ == "__main__":
    main()