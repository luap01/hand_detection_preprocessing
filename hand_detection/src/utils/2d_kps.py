import numpy as np
import json
import os
import cv2

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
    for idx in range(0, len(kps), 3):
        coords.append(kps[idx])
        coords.append(kps[idx+1])
        confs.append(conf[0])
    return [coords, confs, shift]

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build paths relative to the script location
    base_path = os.path.join(script_dir, "..", "hand_detection", "output", "final_new", "mediapipe", "conf_0.35")
    # tar_base_path = os.path.join(script_dir, "..", "..", "HaMuCo", "data", "OR", "rgb_2D_keypoints")
    tar_base_path = os.path.join(script_dir, "..", "validate", "json")
    
    directories = os.listdir(base_path)
    for camera_name in directories:
        files = os.listdir(f"{base_path}/{camera_name}/json")
        count = 0
        idx = 0
        while idx < 20:
            try:
                data = json_load(os.path.join(base_path, camera_name, "json", f"{idx:06d}.json"))

                exepction_mask = (camera_name == "camera03" and idx > 7)
                if len(data["people"]) > 0:
                    res_left = build_arr(data["people"][0], "left" if camera_name == "camera03" and idx > 7 else "right")
                    res_right = build_arr(data["people"][0], "right" if camera_name == "camera03" and idx > 7 else "left")
                else:
                    count += 1
                    arr = [0] * 63
                    res_right = build_arr(arr)
                    res_left = build_arr(arr)
                
                # Create target paths relative to script location
                # camera_name = os.path.basename(base_path)
                # file_prefix = file.split('_')[0]
                
                tar_pth_right = os.path.join(tar_base_path, "right", camera_name, f"{idx:06d}.json")
                tar_pth_left = os.path.join(tar_base_path, "left", camera_name, f"{idx:06d}.json")
                
                # Create directories if they don't exist
                os.makedirs(os.path.dirname(tar_pth_right), exist_ok=True)
                os.makedirs(os.path.dirname(tar_pth_left), exist_ok=True)
                
                save_file(res_right, tar_pth_right)
                save_file(res_left, tar_pth_left)

            except:
                print("Error for:", camera_name, idx)
            
            idx += 1

        # pics = os.listdir(f"{base_path}/{camera_name}/blanks")
        # pic_tar = tar_base_path.replace("rgb_2D_keypoints", "orbbec")
        # idx = 0
        # while idx < 20:
        #     if any(substr in f"{idx:06d}" for substr in ['000004', '000012', '000018', '000019']):
        #         idx += 1
        #         continue
        #     try:
        #         res_right = cv2.imread(os.path.join(base_path, camera_name, "blanks", f"{idx:06d}_cropped_256_{'left' if camera_name != 'camera03' else 'right'}_blank.jpg"))
        #         res_left = cv2.imread(os.path.join(base_path, camera_name, "blanks", f"{idx:06d}_cropped_256_{'right' if camera_name != 'camera03' else 'left'}_blank.jpg"))
        #         orig = cv2.imread(os.path.join(base_path, camera_name, "original", f"{idx:06d}_test.jpg"))

        #         tar_pth_right = os.path.join(pic_tar, "right", camera_name, f"{idx:06d}.jpg")
        #         tar_pth_left = os.path.join(pic_tar, "left", camera_name, f"{idx:06d}.jpg")
        #         tar_pth_orig = os.path.join(pic_tar, "orig", camera_name, f"{idx:06d}.jpg")

        #         # Create directories if they don't exist
        #         os.makedirs(os.path.dirname(tar_pth_right), exist_ok=True)
        #         os.makedirs(os.path.dirname(tar_pth_left), exist_ok=True)
        #         os.makedirs(os.path.dirname(tar_pth_orig), exist_ok=True)
                
        #         cv2.imwrite(tar_pth_right, res_right)
        #         cv2.imwrite(tar_pth_left, res_left)
        #         cv2.imwrite(tar_pth_orig, orig)

        #     except IndexError:
        #         print("Error for:", idx)

        #     idx += 1


    print(count)
    print(idx)