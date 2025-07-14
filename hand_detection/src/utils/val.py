import cv2
import json
import os
import numpy as np
from datetime import datetime
from pathlib import Path

def _json_load(p):
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d


GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
BLUE = (255, 0, 0)

img_idx = "000100"


SCRIPT_DIR = Path(__file__).resolve().parent

# BASE_PATH = "test_bbox_larger_shift_into_opposite"
# BASE_PATH = "test_bbox"
BASE_PATH = SCRIPT_DIR / ".." / "hand_detection" / "output" / "final" / "mediapipe" / "conf_0.40" / "camera05"
# BASE_PATH = SCRIPT_DIR / ".." / ".." / "data" / "or_input_hamuco" / "mediapipe" / "cropped" / "2D_kps"
# BASE_PATH = SCRIPT_DIR / ".." / "hand_detection" / "src" / "models" / "openpose_impl" / "output_save"
OPENPOSE_PTH = SCRIPT_DIR / ".." / "hand_detection" / "src" / "models" / "openpose_impl" / "output" / "2025-07-07_11:58" / "json" / "right" / "camera02"
INPUT_PATH = SCRIPT_DIR / ".." / ".." / "data" / "or_input_hamuco" / "mediapipe" / "original" / "images" / "camera05"
folder = datetime.now().strftime('%Y-%m-%d_%H:%M')
tmp = str(BASE_PATH)

# BASE_PATH = "test_val/conf_0.7/camera04"
for i in range(2, 3):
    # img_idx = f"{i:06d}"
    for idx in range(0, 20):
        img_idx = f"{idx:06d}"
        camera = f"camera0{i}"

        BASE_PATH = tmp.replace("camera05", camera)    
        # if not os.path.exists(f"{BASE_PATH}/json/{img_idx}.json"):
        if not os.path.exists(OPENPOSE_PTH / f"{img_idx}.json"):
            print(f"Skipping {img_idx}: Json could not be loaded")
            continue

        left = cv2.imread(f"{BASE_PATH}/blanks/left/{img_idx}.jpg")
        right = cv2.imread(f"{BASE_PATH}/blanks/right/{img_idx}.jpg")
        orig = cv2.imread(f"{BASE_PATH}/original/{img_idx}.jpg")

        # Check if images were loaded successfully
        if orig is None:# if left is None or right is None or orig is None:
            print(f"Skipping {img_idx}: One or more images could not be loaded")
            continue

        ldata = _json_load(f"{BASE_PATH}/json/{img_idx}.json")
        r_shift = _json_load(f"{BASE_PATH}/json/{img_idx}.json")['people'][0]['hand_left_shift']
        rdata = _json_load(OPENPOSE_PTH / f"{img_idx}.json")
        lkps = np.array(ldata['people'][0]['hand_left_keypoints_2d']).reshape(-1, 3)[:, :2]
        rkps = np.array(rdata['people'][0]['hand_right_keypoints_2d']).reshape(-1, 3)[:, :2]

        if len(lkps) == 0 or len(rkps) == 0:
            print(f"Skipping {img_idx}: One or more keypoints could not be loaded")
            continue
        
        l_shift = np.array(ldata['people'][0]['hand_left_shift'])
        # r_shift = np.array(rdata['people'][0]['hand_right_shift'])

        def visualize_2d_points(points_2d, img, dot_colour, line_colour, offset):
            HAND_CONNECTIONS = [
                (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),      # Index
                (0, 9), (9, 10), (10, 11), (11, 12), # Middle
                (0, 13), (13, 14), (14, 15), (15, 16), # Ring
                (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
            ]

            x_shift, y_shift = offset[0], offset[1]
            for i, (x, y) in enumerate(points_2d):
                cv2.circle(img, (int(x) + x_shift, int(y) + y_shift), 4, dot_colour, -1)

            for idx1, idx2 in HAND_CONNECTIONS:
                x1, y1 = int(points_2d[idx1][0] + x_shift), int(points_2d[idx1][1] + y_shift)
                x2, y2 = int(points_2d[idx2][0] + x_shift), int(points_2d[idx2][1] + y_shift)
                cv2.line(img, (x1, y1), (x2, y2), line_colour, 1)

            return img


        l_img = visualize_2d_points(lkps, left, RED, GREEN, [0, 0])
        r_img = visualize_2d_points(rkps, left, BLUE, YELLOW, [0, 0])

        window_name = img_idx
        cv2.imshow(window_name, l_img)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, 0, -750)
        cv2.waitKey(0)
        cv2.imshow(window_name, r_img)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, 0, -750)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        orig = visualize_2d_points(lkps, orig, RED, GREEN, l_shift)
        orig = visualize_2d_points(rkps, orig, BLUE, YELLOW, r_shift)

        cv2.imshow(window_name, orig)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, 0, -750)
        cv2.resizeWindow(window_name, 1280, 720)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # os.makedirs(f"validate/results/{folder}/{camera}", exist_ok=True)
        # cv2.imwrite(f"validate/results/{folder}/{camera}/{idx:06d}.jpg", orig)