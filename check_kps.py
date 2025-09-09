import cv2
import json
import os
import numpy as np

def _json_load(p):
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d


GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
BLUE = (255, 0, 0)

img_idx = "017997"

# BASE_PATH = "test_bbox_larger_shift_into_opposite"
# BASE_PATH = "test_bbox"
# BASE_PATH = "results/20250519_Testing"
BASE_PATH = "data/output/20250206_Testing/mediapipe_0.35/camera01"
tmp = BASE_PATH
# BASE_PATH = "test_val/conf_0.7/camera04"

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
        if idx1 < len(points_2d):
            x1, y1 = int(points_2d[idx1][0] + x_shift), int(points_2d[idx1][1] + y_shift)
            x2, y2 = int(points_2d[idx2][0] + x_shift), int(points_2d[idx2][1] + y_shift)
            cv2.line(img, (x1, y1), (x2, y2), line_colour, 1)

    return img

def process_and_show_image(camera, idx):
    """Process and display a single image with keypoints"""
    img_idx = f"{idx:06d}"
    BASE_PATH_CAM = tmp.replace("camera05", camera)
    
    print(f"{BASE_PATH_CAM}/right/{camera}/{img_idx}.json")
    if not os.path.exists(f"{BASE_PATH_CAM}/right/{camera}/{img_idx}.json"):
        print(f"Skipping {img_idx}: Json could not be loaded")
        return False
    
    img_path = tmp.replace("results", "data/input")
    img = cv2.imread(f"{img_path}/{camera}/{img_idx}.jpg")

    # Check if images were loaded successfully
    if img is None:
        print(f"Skipping {img_idx}: One or more images could not be loaded")
        return False

    lkps, _ = _json_load(f"{BASE_PATH_CAM}/left/{camera}/{img_idx}.json")
    rkps, _ = _json_load(f"{BASE_PATH_CAM}/right/{camera}/{img_idx}.json")
    rkps = np.array(rkps, dtype=np.float32).reshape(21, 2) if len(rkps) > 0 else []
    lkps = np.array(lkps, dtype=np.float32).reshape(21, 2) if len(lkps) > 0 else []

    if len(lkps) == 0 and len(rkps) == 0:
        print(f"Skipping {img_idx}: One or more keypoints could not be loaded")
        return False

    l_img = visualize_2d_points(lkps, img, RED, GREEN, [0, 0])
    r_img = visualize_2d_points(rkps, img, BLUE, YELLOW, [0, 0])

    window_name = f"{camera}_{img_idx}"

    cv2.imshow(window_name, img)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, 0, -750)
    cv2.resizeWindow(window_name, 1280, 720)
    
    # Save the image
    os.makedirs(f"check_input_data/{camera}", exist_ok=True)
    cv2.imwrite(f"check_input_data/{camera}/{img_idx}.jpg", img)
    
    return True

def find_next_valid_image(camera, start_idx, direction, min_idx, max_idx):
    """Find the next valid image in the given direction"""
    current_idx = start_idx
    
    while min_idx <= current_idx <= max_idx:
        img_idx = f"{current_idx:06d}"
        BASE_PATH_CAM = tmp.replace("camera05", camera)
        img_path = tmp.replace("results", "data/input")
        
        # Check if all required files exist
        json_left = f"{BASE_PATH_CAM}/left/{camera}/{img_idx}.json"
        json_right = f"{BASE_PATH_CAM}/right/{camera}/{img_idx}.json"
        img_file = f"{img_path}/{camera}/{img_idx}.jpg"
        
        if (os.path.exists(json_left) and 
            os.path.exists(json_right) and 
            os.path.exists(img_file)):
            return current_idx
        
        current_idx += direction
    
    return None

for i in range(1, 7):
    if i in [2, 3]:
        continue
    import argparse
    parser = argparse.ArgumentParser(description="Swap left and right hand keypoint files for specific frames")
    parser.add_argument("--frame", required=True, help="Frame number to swap (e.g., '017997' or '17997')")
    args = parser.parse_args()
    frame = int(args.frame)

    camera = f"camera0{i}"
    min_idx = 2465
    # min_idx = frame
    # max_idx = frame
    max_idx = 2765
    
    # Find the first valid image to start with
    current_idx = find_next_valid_image(camera, min_idx, 1, min_idx, max_idx)
    if current_idx is None:
        print(f"No valid images found for {camera}")
        continue
    
    print(f"\nViewing {camera}")
    print("Controls: Left Arrow = Previous, Right Arrow = Next, ESC = Exit camera, Any other key = Next")
    
    while current_idx is not None and current_idx <= max_idx:
        # Try to process and show the current image
        if process_and_show_image(camera, current_idx):
            # Wait for key press
            key = cv2.waitKey(0) & 0xFF
            
            # Handle key presses
            if key == 27:  # ESC key - exit current camera
                cv2.destroyAllWindows()
                break
            elif key == 81 or key == 2:  # Left arrow (different systems may use different codes)
                # Go back - find previous valid image
                next_idx = find_next_valid_image(camera, current_idx - 1, -1, min_idx, max_idx)
                if next_idx is not None:
                    current_idx = next_idx
                else:
                    print("No more images to go back to")
                cv2.destroyAllWindows()
            elif key == 83 or key == 3:  # Right arrow
                # Go forward - find next valid image
                next_idx = find_next_valid_image(camera, current_idx + 1, 1, min_idx, max_idx)
                if next_idx is not None:
                    current_idx = next_idx
                else:
                    print("No more images to go forward to")
                    break
                cv2.destroyAllWindows()
            else:
                # Any other key - go forward (original behavior)
                next_idx = find_next_valid_image(camera, current_idx + 1, 1, min_idx, max_idx)
                if next_idx is not None:
                    current_idx = next_idx
                else:
                    print("No more images available")
                    break
                cv2.destroyAllWindows()
        else:
            # This shouldn't happen with our new logic, but just in case
            next_idx = find_next_valid_image(camera, current_idx + 1, 1, min_idx, max_idx)
            if next_idx is not None:
                current_idx = next_idx
            else:
                break