import numpy as np
import json
import os
import cv2
import argparse

def json_load(p):
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d

def save_file(data, path):
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)

def build_arr(data, hand_side):
    kps = data[f"hand_{hand_side}_keypoints_2d"]
    conf = data[f"hand_{hand_side}_conf"]
    shift = data[f"hand_{hand_side}_shift"]
    coords = []
    confs = []
    # LEFT_WRIST,
    # LEFT_PINKY,
    # LEFT_INDEX,
    # LEFT_THUMB,
    # wrist = conf[0]
    # pinky = conf[1]
    # index = conf[2]
    # thumb = conf[3]
    # middle = 0.7 * index + 0.3 * pinky
    # ring = 0.3 * index + 0.7 * pinky
    # confs = [
    #     wrist,                             # 0: wrist
    #     thumb, thumb, thumb, thumb,        # 1-4: thumb joints
    #     index, index, index, index,        # 5-8: index joints
    #     middle, middle, middle, middle,    # 9-12: middle joints
    #     ring, ring, ring, ring,            # 13-16: ring joints
    #     pinky, pinky, pinky, pinky         # 17-20: pinky joints
    # ]
    for idx in range(0, len(kps), 3):
        coords.append(kps[idx] + shift[0])
        coords.append(kps[idx+1] + shift[1])
        confs.append(conf[0])

    return [coords, confs]

def has_valid_hand_keypoints(data, hand_side):
    """Check if hand has valid keypoints (not empty/zero)"""
    kps = data.get(f"hand_{hand_side}_keypoints_2d", [])
    if not kps or len(kps) == 0:
        return False
    # Check if keypoints are not all zeros
    return any(kp != 0 for kp in kps)

def parse_args():
    parser = argparse.ArgumentParser(description='Convert MediaPipe keypoints to custom format')
    parser.add_argument('--conf', type=float, default=0.35, help='Confidence threshold level')
    parser.add_argument('--categories', nargs='+', default=['success'], 
                       help='Categories to process (success, partial, failure)')
    parser.add_argument('--dataset', default="20250519_Testing")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build paths relative to the script location or use provided paths
    base_path = os.path.join(script_dir, "output", args.dataset, f"mediapipe_{args.conf:.2f}")
    
    tar_base_path = os.path.join(script_dir, "results", args.dataset)
    
    print(f"Processing confidence level: {args.conf:.6f}")
    print(f"Categories: {args.categories}")
    print(f"Input path: {base_path}")
    print(f"Output path: {tar_base_path}")
    print("-" * 60)
    
    if not os.path.exists(base_path):
        print(f"Error: Base path {base_path} does not exist!")
        exit(1)
    
    directories = os.listdir(base_path)
    total_processed = 0
    total_errors = 0
    
    for camera_name in directories:
        if camera_name == "calib":
            continue

        if camera_name != "camera07":
            continue
            
        print(f"Processing {camera_name}...")
        camera_processed = 0
        camera_errors = 0
        
        for category in args.categories:
            category_path = os.path.join(base_path, camera_name, "json")
            
            if not os.path.exists(category_path):
                print(f"  Warning: {category_path} does not exist, skipping...")
                continue
            
            files = os.listdir(category_path)
            print(f"  Processing {len(files)} files from {category} category...")
            files = [f for f in files    if f.endswith(".json")]
            for idx, file in enumerate(files):
                try:
                    data = json_load(os.path.join(category_path, file))
                    files_saved = 0

                    if len(data["people"]) > 0:
                        person_data = data["people"][0]
                        
                        # # Check which hands are detected
                        has_left = has_valid_hand_keypoints(person_data, "left")
                        has_right = has_valid_hand_keypoints(person_data, "right")

                        if has_left:
                            res_left = build_arr(person_data, "left")
                            tar_pth_left = os.path.join(tar_base_path, category, "left", camera_name, file)
                            os.makedirs(os.path.dirname(tar_pth_left), exist_ok=True)
                            save_file(res_left, tar_pth_left)
                        
                        if has_right:
                            res_right = build_arr(person_data, "right")
                            tar_pth_right = os.path.join(tar_base_path, category, "right", camera_name, file)
                            os.makedirs(os.path.dirname(tar_pth_right), exist_ok=True)
                            save_file(res_right, tar_pth_right)
                        # if (category == "success") or (has_left and has_right):
                        #     # For success, save both hands (even if one might be empty)

                        #     res_left = build_arr(person_data, "left")
                        #     res_right = build_arr(person_data, "right")
                            
                        #     tar_pth_left = os.path.join(tar_base_path, category, "left", camera_name, file)
                        #     tar_pth_right = os.path.join(tar_base_path, category, "right", camera_name, file)
                            
                        #     os.makedirs(os.path.dirname(tar_pth_left), exist_ok=True)
                        #     os.makedirs(os.path.dirname(tar_pth_right), exist_ok=True)
                            
                        #     save_file(res_left, tar_pth_left)
                        #     save_file(res_right, tar_pth_right)
                        #     files_saved = 2
                            
                        # elif category == "partial" or (has_left and not has_right) or (has_right and not has_left):
                        #     # For partial, only save detected hands
                        #     detected_hands = []
                        #     if has_left:
                        #         res_left = build_arr(person_data, "left")
                        #         tar_pth_left = os.path.join(tar_base_path, category, "left", camera_name, file)
                        #         os.makedirs(os.path.dirname(tar_pth_left), exist_ok=True)
                        #         save_file(res_left, tar_pth_left)
                        #         files_saved += 1
                        #         detected_hands.append("left")
                                
                        #     if has_right:
                        #         res_right = build_arr(person_data, "right")
                        #         tar_pth_right = os.path.join(tar_base_path, category, "right", camera_name, file)
                        #         os.makedirs(os.path.dirname(tar_pth_right), exist_ok=True)
                        #         save_file(res_right, tar_pth_right)
                        #         files_saved += 1
                        #         detected_hands.append("right")
                            
                        #     # Debug output for partial files
                        #     if idx < 5:  # Only show first few files to avoid spam
                        #         hands_str = ", ".join(detected_hands) if detected_hands else "none"
                        #         print(f"    {file}: detected hands - {hands_str}")
                                
                        # elif category == "failure":
                        #     # For failure, create empty files only if explicitly needed
                        #    pass
                            
                    else:
                        pass
                        # For partial with no people, skip entirely
                    
                    camera_processed += files_saved
                    total_processed += files_saved

                except Exception as e:
                    print(f"    Error processing {camera_name}/{category}/{file}: {e}")
                    camera_errors += 1
                    total_errors += 1
        
        print(f"  {camera_name}: {camera_processed} output files created, {camera_errors} errors")
    
    print("-" * 60)
    print(f"Total output files created: {total_processed}")
    print(f"Total errors: {total_errors}")
    print(f"Output saved to: {tar_base_path}")
    print(f"Directory structure: {tar_base_path}/{{left,right}}/{{camera_name}}/{{file.json}}")