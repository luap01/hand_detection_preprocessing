import cv2
import json
import numpy as np
import math
import time
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib
import torch
from skimage.measure import label
import os
import sys
from datetime import datetime
from pathlib import Path

# Handle imports for both module and direct script usage
try:
    from . import util
    from .model import handpose_model
except ImportError:
    # When running directly
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import util
    from model import handpose_model
    from body import Body

def _json_load(p):
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d

SCRIPT_DIR = Path(__file__).resolve().parent

class Hand(object):
    def __init__(self, model_path):
        self.model = handpose_model()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        model_dict = util.transfer(self.model, torch.load(model_path))
        self.model.load_state_dict(model_dict)
        self.model.eval()

    def __call__(self, oriImg):
        scale_search = [0.5, 1.0, 1.5, 2.0]
        # scale_search = [0.5]
        boxsize = 368
        stride = 8
        padValue = 128
        thre = 0.05
        multiplier = [x * boxsize / oriImg.shape[0] for x in scale_search]
        heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 22))
        # paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

        for m in range(len(multiplier)):
            scale = multiplier[m]
            imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            imageToTest_padded, pad = util.padRightDownCorner(imageToTest, stride, padValue)
            im = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5
            im = np.ascontiguousarray(im)

            data = torch.from_numpy(im).float()
            if torch.cuda.is_available():
                data = data.cuda()
            # data = data.permute([2, 0, 1]).unsqueeze(0).float()
            with torch.no_grad():
                output = self.model(data).cpu().numpy()
                # output = self.model(data).numpy()q

            # extract outputs, resize, and remove padding
            heatmap = np.transpose(np.squeeze(output), (1, 2, 0))  # output 1 is heatmaps
            heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
            heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
            heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

            heatmap_avg += heatmap / len(multiplier)

        all_peaks = []
        confidences = []
        for part in range(21):
            map_ori = heatmap_avg[:, :, part]
            one_heatmap = gaussian_filter(map_ori, sigma=3)
            binary = np.ascontiguousarray(one_heatmap > thre, dtype=np.uint8)
            # 全部小于阈值
            if np.sum(binary) == 0:
                all_peaks.append([0, 0])
                confidences.append(0.0)  # Add zero confidence for no detection
                continue
            label_img, label_numbers = label(binary, return_num=True, connectivity=binary.ndim)
            max_index = np.argmax([np.sum(map_ori[label_img == i]) for i in range(1, label_numbers + 1)]) + 1
            label_img[label_img != max_index] = 0
            map_ori[label_img == 0] = 0

            y, x = util.npmax(map_ori)
            confidence = map_ori[y, x].item() if map_ori[y, x] > 0 else 0.0
            all_peaks.append([x, y])
            confidences.append(confidence)

        return np.array(all_peaks), np.array(confidences)
    
def assemble_img_and_kps(hand_estimation, oriImg, hands_list):
    all_peaks = []
    max_confs = 0
    avg_confs = 0
    for hand_idx, (x, y, w, is_left) in enumerate(hands_list):            
        # Extract and process hand region
        hand_roi = oriImg[y:y+w, x:x+w, :]
        peaks, conf = hand_estimation(hand_roi)
        all_peaks.append(peaks)
        
        max_conf = np.max(conf)
        avg_conf = np.mean(conf[conf > 0]) if any(conf > 0) else 0.0
        max_confs += max_conf
        avg_confs += avg_conf

        # Adjust peaks to image space
        peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
        peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)

        h, w = oriImg.shape[:2]
        normalized_peaks = peaks.astype(np.float64)  # Ensure float type
        normalized_peaks[:, 0] /= w
        normalized_peaks[:, 1] /= h
        
        hand_keypoints = []
        for i, keypoint in enumerate(peaks):
            x_coord, y_coord = keypoint
            hand_keypoints.extend([float(x_coord), float(y_coord), conf[i]])

        # Store keypoints based on handedness
        if is_left:
            data["people"][0]["hand_left_keypoints_2d"] = hand_keypoints
        else:
            data["people"][0]["hand_right_keypoints_2d"] = hand_keypoints
    
    print(max_confs, avg_confs)
    if max_confs + avg_confs < 1.1:
        return data, all_peaks, False
    
    return data, all_peaks, True

if __name__ == "__main__":
    hand_estimation = Hand(f'{SCRIPT_DIR}/model/hand_pose_model.pth')
    body_estimation = Body(f'{SCRIPT_DIR}/model/body_pose_model.pth')
    # test_image = '../images/hand.jpg'
    BASE_PATH = SCRIPT_DIR / "final_images_cropped"
    folder = datetime.now().strftime('%Y-%m-%d_%H:%M')
    folder = "2025-07-07_11:58"
    OUTPUT_PATH = f'{SCRIPT_DIR}/output/{folder}'
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    hands = list(os.listdir(BASE_PATH))

    cropped_input = True if "cropped" in str(BASE_PATH) else False
    DIRECT_CONF_THRESHOLD = 0.5
    NORMAL_CONF_THRESHOLD = 0.5
    for hand in hands:
        hand_pth = f'{BASE_PATH}/{hand}'
        cams = list(os.listdir(hand_pth))

        if hand == "right":
            continue
        for cam in cams:
            cam_pth = f'{hand_pth}/{cam}'
            files = list(os.listdir(cam_pth))
            for file in files:
                test_image_path = f'{cam_pth}/{file}'

                if cam in ['camera01', 'camera04', 'camera06']:
                    continue
                kps_pth = test_image_path.replace('.jpg', '.json').replace('or_new_images', 'or_new_kps').replace('left', '').replace('right', '')
                kps = _json_load(kps_pth) if os.path.exists(kps_pth) else None
                l_shift = kps['people'][0][f'hand_left_shift'] if kps else [0, 0]
                r_shift = kps['people'][0][f'hand_right_shift'] if kps else [0, 0]
                print(f"Processing image: {test_image_path}")
                oriImg = cv2.imread(test_image_path)  # B,G,R order
                
                alpha = 1.5 if cam in ['camera05', 'camera06'] else 0.5
                beta = 25 if cam not in ['camera05', 'camera06'] else -35
                enhanced = np.clip(oriImg.copy().astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
                # First detect body keypoints
                candidate, subset = body_estimation(enhanced)
                hands_list = util.handDetect(candidate, subset, enhanced)
                print(hands_list)

                # Initialize data structure for keypoints
                data = {
                    "people": [{
                        "hand_left_keypoints_2d": [],
                        "hand_right_keypoints_2d": [],
                        "hand_left_shift": r_shift,
                        "hand_right_shift": l_shift
                    }]
                }
                os.makedirs(f'{OUTPUT_PATH}/images/{hand}/{cam}', exist_ok=True)
                os.makedirs(f'{OUTPUT_PATH}/json/{hand}/{cam}', exist_ok=True)
                all_peaks = []
                found = False

                if cropped_input:
                    alpha = 1.5 if cam in ['camera05', 'camera06'] else 0.5
                    beta = 25 if cam not in ['camera05', 'camera06'] else -35

                    enhanced = np.clip(oriImg.copy().astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
                    peaks, conf = hand_estimation(enhanced)
                    # print(conf)
                    valid_confidences = conf[conf > 0]
                    avg_confidence = max(valid_confidences) if len(valid_confidences) > 0 else 0.0
                    print(avg_confidence)
                    if avg_confidence > DIRECT_CONF_THRESHOLD:
                        canvas = util.draw_handpose(oriImg, [peaks], False)
                        # Convert peaks to keypoints
                        hand_keypoints = []
                        for i, keypoint in enumerate(peaks):
                            x_coord, y_coord = keypoint
                            hand_keypoints.extend([float(x_coord), float(y_coord), conf[i]])

                        data["people"][0][f"hand_{hand}_keypoints_2d"] = hand_keypoints
                        cv2.imwrite(f'{OUTPUT_PATH}/images/{hand}/{cam}/{file}', canvas)

                        with open(f'{OUTPUT_PATH}/json/{hand}/{cam}/{file.replace("jpg", "json")}', 'w') as f:
                                json.dump(data, f, indent=4)
                    
                    else:
                        found = False
                        for beta in range(-35, 36, 10):
                            for alpha in np.arange(0.0, 2.1, 0.3):
                                enhanced = np.clip(oriImg.copy().astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
                                img_yuv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2YUV)
                                # Equalize only the Y channel (luminance)
                                img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
                                # Convert back to RGB
                                enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
                                peaks, conf = hand_estimation(enhanced)

                                max_conf = np.max(conf)
                                avg_conf = np.mean(conf[conf > 0]) if any(conf > 0) else 0.0

                                print(max_conf, avg_conf)
                                if max_conf + avg_conf > NORMAL_CONF_THRESHOLD:
                                    found = True
                                    print(f"Found hand with alpha={alpha:.2f}, beta={beta}")

                                canvas = util.draw_handpose(oriImg, [peaks], False)
                                # Convert peaks to keypoints
                                hand_keypoints = []
                                for i, keypoint in enumerate(peaks):
                                    x_coord, y_coord = keypoint
                                    hand_keypoints.extend([float(x_coord), float(y_coord), conf[i]])

                                data["people"][0][f"hand_{hand}_keypoints_2d"] = hand_keypoints
                                cv2.imwrite(f'{OUTPUT_PATH}/images/{hand}/{cam}/{file}', canvas)

                                with open(f'{OUTPUT_PATH}/json/{hand}/{cam}/{file.replace("jpg", "json")}', 'w') as f:
                                        json.dump(data, f, indent=4)
                                
                                if found:
                                    break
                            
                            if found:
                                break

                else:
                    if len(hands_list) > 0:
                        data, all_peaks, found = assemble_img_and_kps(hand_estimation, oriImg, hands_list)
                    if found:
                        canvas = util.draw_handpose(oriImg, all_peaks, False)
                        cv2.imwrite(f'{OUTPUT_PATH}/images/{hand}/{cam}/{file}', canvas)
                        with open(f'{OUTPUT_PATH}/json/{hand}/{cam}/{file.replace("jpg", "json")}', 'w') as f:
                            json.dump(data, f, indent=4)
                    else:
                        check = False
                        for alpha in np.arange(0.0, 2.1, 0.3):
                            for beta in range(-35, 36, 10):
                                enhanced = np.clip(oriImg.copy().astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
                                img_yuv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2YUV)
                                # Equalize only the Y channel (luminance)
                                img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
                                # Convert back to RGB
                                enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

                                # First detect body keypoints
                                candidate, subset = body_estimation(enhanced)
                                hands_list = util.handDetect(candidate, subset, enhanced)
                                data, all_peaks, found = assemble_img_and_kps(hand_estimation, oriImg, hands_list)
                                if found and len(hands_list) > 0:
                                    canvas = util.draw_handpose(oriImg, all_peaks, False)
                                    cv2.imwrite(f'{OUTPUT_PATH}/images/{hand}/{cam}/{file}', canvas)

                                    with open(f'{OUTPUT_PATH}/json/{hand}/{cam}/{file.replace("jpg", "json")}', 'w') as f:
                                        json.dump(data, f, indent=4)
                                        check = True
                                    
                                if check:
                                    break
                            if check:
                                break
                    # if cam in ['camera05', 'camera06']:

