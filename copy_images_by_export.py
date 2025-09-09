#!/usr/bin/env python3
"""
Script to copy images based on image indices present in export JSON files.
This script scans the export directory for JSON files and copies the corresponding
images from the input directory to a new images directory.
"""

import os
import shutil
import argparse
from pathlib import Path


def copy_images_by_export(export_base_dir, input_base_dir, output_dir, dataset_name, camera=None, side=None):
    """
    Copy images based on JSON files present in export directory.
    
    Args:
        export_base_dir: Base directory containing export data (e.g., './export')
        input_base_dir: Base directory containing input images (e.g., './data/input')
        output_dir: Directory where images should be copied to
        dataset_name: Name of the dataset (e.g., '20250519_Testing')
        camera: Specific camera to process (e.g., 'camera01'), or None for all cameras
        side: Specific side to process ('left' or 'right'), or None for both sides
    """
    
    export_dataset_dir = Path(export_base_dir) / dataset_name
    input_dataset_dir = Path(input_base_dir) / dataset_name
    output_path = Path(output_dir)
    
    # Check if export directory exists
    if not export_dataset_dir.exists():
        print(f"Export directory not found: {export_dataset_dir}")
        return
    
    # Check if input directory exists
    if not input_dataset_dir.exists():
        print(f"Input directory not found: {input_dataset_dir}")
        return
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine which sides to process
    sides_to_process = []
    if side:
        if side in ['left', 'right']:
            sides_to_process = [side]
        else:
            print(f"Invalid side '{side}'. Must be 'left' or 'right'.")
            return
    else:
        # Check which sides exist
        for potential_side in ['left', 'right']:
            if (export_dataset_dir / potential_side).exists():
                sides_to_process.append(potential_side)
    
    copied_count = 0
    missing_count = 0
    
    for current_side in sides_to_process:
        side_export_dir = export_dataset_dir / current_side
        
        if not side_export_dir.exists():
            print(f"Side directory not found: {side_export_dir}")
            continue
        
        # Determine which cameras to process
        cameras_to_process = []
        if camera:
            if (side_export_dir / camera).exists():
                cameras_to_process = [camera]
            else:
                print(f"Camera directory not found: {side_export_dir / camera}")
                continue
        else:
            # Get all camera directories
            cameras_to_process = [d.name for d in side_export_dir.iterdir() 
                                if d.is_dir() and d.name.startswith('camera')]
            cameras_to_process.sort()
        
        for current_camera in cameras_to_process:
            camera_export_dir = side_export_dir / current_camera
            camera_input_dir = input_dataset_dir / current_camera
            
            print(f"Processing {current_side}/{current_camera}...")
            
            # Check if camera input directory exists
            if not camera_input_dir.exists():
                print(f"  Warning: Input camera directory not found: {camera_input_dir}")
                continue
            
            # Create output subdirectory
            camera_output_dir = output_path / dataset_name / current_side / current_camera
            camera_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process JSON files in this camera directory
            json_files = list(camera_export_dir.glob("*.json"))
            print(f"  Found {len(json_files)} JSON files")
            
            for json_file in json_files:
                # Extract image index from JSON filename
                img_idx = json_file.stem  # filename without extension
                
                # Construct source image path
                source_img_path = camera_input_dir / f"{img_idx}.jpg"
                
                # Construct destination image path
                dest_img_path = camera_output_dir / f"{img_idx}.jpg"
                
                # Copy image if it exists
                if source_img_path.exists():
                    shutil.copy2(source_img_path, dest_img_path)
                    copied_count += 1
                else:
                    print(f"    Warning: Image not found: {source_img_path}")
                    missing_count += 1
            
            print(f"  Copied {len(json_files) - missing_count} images for {current_side}/{current_camera}")
    
    print(f"\nSummary:")
    print(f"  Total images copied: {copied_count}")
    print(f"  Missing images: {missing_count}")
    print(f"  Output directory: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Copy images based on export JSON files")
    parser.add_argument("--export-dir", default="./export", 
                       help="Base export directory (default: ./export)")
    parser.add_argument("--input-dir", default="./data/input", 
                       help="Base input directory (default: ./data/input)")
    parser.add_argument("--output-dir", default="./images", 
                       help="Output directory for copied images (default: ./images)")
    parser.add_argument("--dataset", default="20250519_Testing", 
                       help="Dataset name (default: 20250519_Testing)")
    parser.add_argument("--cam", 
                       help="Specific camera to process (e.g., camera01)")
    parser.add_argument("--side", choices=['left', 'right'], 
                       help="Specific side to process (left or right)")
    
    args = parser.parse_args()
    
    copy_images_by_export(
        export_base_dir=args.export_dir,
        input_base_dir=args.input_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        camera=f"camera0{args.cam}",
        side=args.side
    )


if __name__ == "__main__":
    main() 

# python3 copy_images_by_export.py --cam 6 --dataset 20250206_Testing