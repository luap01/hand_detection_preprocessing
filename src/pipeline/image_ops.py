import cv2
import numpy as np
from typing import Optional, Tuple, List
from pathlib import Path

from utils.camera import load_cam_infos
from utils.image import undistort_image
from models.hand_detector import HandDetection


class ImageOperations:
    """Image loading, enhancement, lighting normalization, and rotation operations"""

    def __init__(self, config):
        self.config = config
        self.cam_params = None
        self.load_camera_params()

        # ---- normalization config (safe defaults if not present on config) ----
        self.normalize_to_first = getattr(self.config, 'normalize_to_first_lighting', False)
        # "reinhard" (recommended) or "wb_clahe"
        self.normalization_method = getattr(self.config, 'normalization_method', 'reinhard')
        # ignore rim so it doesn't skew statistics
        self.mask_margin = float(getattr(self.config, 'normalization_mask_margin', 0.02))
        # CLAHE / sharpening knobs
        self.clahe_clip = float(getattr(self.config, 'clahe_clip', 2.0))
        self.clahe_grid = int(getattr(self.config, 'clahe_grid', 8))
        self.unsharp_amount = float(getattr(self.config, 'unsharp_amount', 0.0))
        self.unsharp_sigma = float(getattr(self.config, 'unsharp_sigma', 1.0))

        # reference stats (filled by fit_reference_lighting)
        self._ref_lab_mean = None  # (L, a, b)
        self._ref_lab_std = None   # (L, a, b)
        self._ref_rgb_mean = None  # (R, G, B)

    # -------------------------------------------------------------------------
    # Camera
    # -------------------------------------------------------------------------
    def load_camera_params(self):
        """Load camera calibration parameters"""
        try:
            if self.config.camera_name in ["camera05", "camera06", "camera07"]:
                # For camera07, no calibration needed
                return
            cam_infos = load_cam_infos(Path(self.config.camera_path), orbbec=self.config.orbbec_cam)
            self.cam_params = cam_infos[self.config.camera_name]
        except Exception as e:
            raise Exception(f"Failed to load camera parameters: {repr(e)}")

    # -------------------------------------------------------------------------
    # I/O
    # -------------------------------------------------------------------------
    def load_and_undistort_image(self, img_path: str) -> Optional[np.ndarray]:
        """Load and undistort an image (returns BGR)"""
        img = cv2.imread(img_path)
        if img is None:
            return None

        if self.config.camera_name not in ["camera05", "camera06", "camera07"]:
            undistorted = undistort_image(img, self.cam_params, 'color', self.config.orbbec_cam)
        else:
            undistorted = img
        return undistorted

    # -------------------------------------------------------------------------
    # Lighting normalization: reference fitting + per-frame application
    # -------------------------------------------------------------------------
    def fit_reference_lighting_from_paths(self, ref_paths: List[str]) -> None:
        """Compute reference lighting statistics from file paths (expects BGR images)."""
        refs_bgr = []
        for p in ref_paths:
            im = cv2.imread(str(p))
            if im is not None:
                refs_bgr.append(im)
        if not refs_bgr:
            raise ValueError("No valid reference images were loaded.")
        self.fit_reference_lighting(refs_bgr)

    def fit_reference_lighting(self, ref_images_bgr: List[np.ndarray]) -> None:
        """Compute reference lighting statistics from BGR images (first recording)."""
        rgbs = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in ref_images_bgr]
        masks = [self._inscribed_circle_mask(im.shape[:2], self.mask_margin) for im in rgbs]
        self._ref_lab_mean, self._ref_lab_std = self._compute_lab_stats(rgbs, masks)
        self._ref_rgb_mean = self._compute_rgb_mean(rgbs, masks)

    def normalize_lighting(self, image_rgb: np.ndarray) -> np.ndarray:
        """Transform image_rgb to match reference lighting (or apply WB+CLAHE)."""
        mask = self._inscribed_circle_mask(image_rgb.shape[:2], self.mask_margin)

        # Preferred: reference-guided Reinhard transfer
        if self.normalization_method.lower() == 'reinhard' and \
           self._ref_lab_mean is not None and self._ref_lab_std is not None:
            out = self._reinhard_color_transfer(image_rgb, self._ref_lab_mean, self._ref_lab_std, mask)
            if self.unsharp_amount > 0:
                out = self._unsharp(out, amount=self.unsharp_amount, sigma=self.unsharp_sigma)
            return out

        # Alternative: WB to ref mean (if present) + CLAHE on L channel
        if self.normalization_method.lower() in ('wb_clahe', 'wb+clahe', 'wb'):
            ref_means = self._ref_rgb_mean
            # fallback to gray-world if no reference provided
            if ref_means is None:
                ref_means = self._compute_rgb_mean([image_rgb], [mask])
            out = self._von_kries_match_to_reference(image_rgb, ref_means, mask)
            out = self._clahe_L(out, clip=self.clahe_clip, grid=self.clahe_grid)
            if self.unsharp_amount > 0:
                out = self._unsharp(out, amount=self.unsharp_amount, sigma=self.unsharp_sigma)
            return out

        # If disabled or misconfigured, return original
        return image_rgb

    # ---- helpers for normalization ------------------------------------------------
    @staticmethod
    def _inscribed_circle_mask(hw: Tuple[int, int], margin: float = 0.02) -> np.ndarray:
        """Boolean mask for the largest circle (minus margin) within the frame."""
        h, w = hw
        cx, cy = w / 2.0, h / 2.0
        r = min(h, w) * 0.5 * (1.0 - margin)
        yy, xx = np.ogrid[:h, :w]
        return ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r ** 2)

    @staticmethod
    def _compute_rgb_mean(rgbs: List[np.ndarray], masks: List[np.ndarray]) -> np.ndarray:
        vals = []
        for img, m in zip(rgbs, masks):
            vals.append(img[m].reshape(-1, 3))
        return np.concatenate(vals, axis=0).mean(axis=0)  # (R,G,B)

    @staticmethod
    def _compute_lab_stats(rgbs: List[np.ndarray], masks: List[np.ndarray]) -> Tuple[Tuple[float, float, float],
                                                                                    Tuple[float, float, float]]:
        Ls, As, Bs = [], [], []
        for img, m in zip(rgbs, masks):
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
            l, a, b = cv2.split(lab)
            Ls.append(l[m]); As.append(a[m]); Bs.append(b[m])
        L = np.concatenate(Ls); A = np.concatenate(As); B = np.concatenate(Bs)
        mean = (float(L.mean()), float(A.mean()), float(B.mean()))
        std = (float(L.std() + 1e-6), float(A.std() + 1e-6), float(B.std() + 1e-6))
        return mean, std

    @staticmethod
    def _clahe_L(img_rgb: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))
        l2 = clahe.apply(l)
        lab2 = cv2.merge([l2, a, b])
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

    @staticmethod
    def _von_kries_match_to_reference(img_rgb: np.ndarray, ref_means_rgb: np.ndarray,
                                      mask: Optional[np.ndarray] = None) -> np.ndarray:
        img = img_rgb.astype(np.float32)
        if mask is None:
            mask = np.ones(img.shape[:2], dtype=bool)
        # current means under mask
        mean_cur = img[mask].reshape(-1, 3).mean(axis=0) + 1e-6
        gains = ref_means_rgb / mean_cur  # per-channel gains
        out = np.clip(img * gains, 0, 255).astype(np.uint8)
        return out

    @staticmethod
    def _reinhard_color_transfer(img_rgb: np.ndarray,
                                 ref_mean: Tuple[float, float, float],
                                 ref_std: Tuple[float, float, float],
                                 mask: Optional[np.ndarray] = None) -> np.ndarray:
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)
        if mask is None:
            mask = np.ones(l.shape, dtype=bool)
        # source stats
        l_m, a_m, b_m = float(l[mask].mean()), float(a[mask].mean()), float(b[mask].mean())
        l_s, a_s, b_s = float(l[mask].std() + 1e-6), float(a[mask].std() + 1e-6), float(b[mask].std() + 1e-6)
        # standardize then scale/shift to reference
        l[mask] = (l[mask] - l_m) / l_s * ref_std[0] + ref_mean[0]
        a[mask] = (a[mask] - a_m) / a_s * ref_std[1] + ref_mean[1]
        b[mask] = (b[mask] - b_m) / b_s * ref_std[2] + ref_mean[2]
        lab2 = cv2.merge([l, a, b])
        lab2 = np.clip(lab2, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

    @staticmethod
    def _unsharp(img_rgb: np.ndarray, amount: float = 1.0, sigma: float = 1.0) -> np.ndarray:
        blur = cv2.GaussianBlur(img_rgb, (0, 0), float(sigma))
        out = img_rgb.astype(np.float32) + float(amount) * (img_rgb.astype(np.float32) - blur.astype(np.float32))
        return np.clip(out, 0, 255).astype(np.uint8)

    # -------------------------------------------------------------------------
    # Rotation search (unchanged)
    # -------------------------------------------------------------------------
    def try_rotation_angles(self, image: np.ndarray, detector) -> Tuple[List, int, np.ndarray]:
        """Try different image rotations for hand detection"""
        angles = [0, 90, 180, 270]
        for angle in angles:
            if angle == 0:
                img_rotated = image
            else:
                if angle == 90:
                    img_rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    img_rotated = cv2.rotate(image, cv2.ROTATE_180)
                elif angle == 270:
                    img_rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

            results_rotated = detector.detect_hands(img_rotated)
            if len(results_rotated) > 0:
                return results_rotated, angle, img_rotated
        return [], 0, image

    def transform_landmarks_back(self, detection: HandDetection, angle: int,
                                 rotated_image_shape: Tuple[int, int],
                                 original_image_shape: Tuple[int, int]):
        """Transform landmarks from rotated image back to original coordinate system"""
        h_rot, w_rot = rotated_image_shape[:2]
        h_orig, w_orig = original_image_shape[:2]

        landmarks = detection.landmarks
        for idx in range(0, landmarks.shape[0]):
            x, y, z = landmarks[idx]
            # normalized -> pixel (rotated)
            x = x * w_rot
            y = y * h_rot

            # inverse rotation
            if angle == 90:
                x_new, y_new = y, w_rot - x
            elif angle == 180:
                x_new, y_new = w_rot - x, h_rot - y
            elif angle == 270:
                x_new, y_new = h_rot - y, x
            else:  # angle == 0
                x_new, y_new = x, y

            # pixel -> normalized (original)
            x = x_new / w_orig
            y = y_new / h_orig
            landmarks[idx] = [x, y, z]
        detection.landmarks = landmarks

    # -------------------------------------------------------------------------
    # Detection (now calls normalization first if enabled)
    # -------------------------------------------------------------------------
    def detect_hands_with_enhancement(self, image: np.ndarray, detector) -> Tuple[List, int]:
        """
        Detect hands with optional lighting normalization, then brightness/contrast
        enhancement and rotation if needed. Expects a BGR image as input.
        """
        # BGR -> RGB for the detector
        rgb_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

        # ---- NEW: normalize lighting before anything else ----
        if self.normalize_to_first:
            rgb_image = self.normalize_lighting(rgb_image)

        # Try normalized/original image first
        results = detector.detect_hands(rgb_image)
        if len(results) > 0:
            return results, 0

        # Try with brightness/contrast adjustments (apply to the same base image we just used)
        max_alpha = getattr(self.config, 'max_alpha', 1.6)
        alpha_step = getattr(self.config, 'alpha_step', 0.15)
        min_beta = getattr(self.config, 'min_beta', -30)
        max_beta = getattr(self.config, 'max_beta', 31)
        beta_step = getattr(self.config, 'beta_step', 10)

        for alpha in np.arange(0.8, max_alpha + 1e-6, alpha_step):
            for beta in range(min_beta, max_beta, beta_step):
                enhanced = np.clip(rgb_image.astype(np.float32) * float(alpha) + float(beta), 0, 255).astype(np.uint8)

                # (keep CLAHE only if not already applied in normalization)
                if not self.normalize_to_first or self.normalization_method.lower() != 'wb_clahe':
                    # small, safe local contrast boost
                    enhanced = self._clahe_L(enhanced, clip=self.clahe_clip, grid=self.clahe_grid)

                results = detector.detect_hands(enhanced)
                if len(results) > 0:
                    return results, 0

                # If enhancement alone doesn't work, try with rotation
                results_rotated, angle, rotated_img = self.try_rotation_angles(enhanced, detector)
                if len(results_rotated) > 0:
                    for detection in results_rotated:
                        self.transform_landmarks_back(
                            detection, angle, rotated_img.shape, image.shape
                        )
                    return results_rotated, angle

        # If enhancement fails, try rotation on the base normalized/original image
        results_rotated, angle, rotated_img = self.try_rotation_angles(rgb_image, detector)
        if len(results_rotated) > 0:
            for detection in results_rotated:
                self.transform_landmarks_back(
                    detection, angle, rotated_img.shape, image.shape
                )
            return results_rotated, angle

        return [], 0
