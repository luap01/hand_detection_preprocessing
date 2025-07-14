import cv2
import numpy as np


def undistort_image(img, cam_params, modality=str, orbbec=True):
    """
    Undistorts an image using the provided camera parameters.
    Parameters:
    img (numpy.ndarray): The input image to be undistorted.
    cam_params (dict): A dictionary containing camera parameters with the following keys:
        - 'intrinsics' (numpy.ndarray): The camera intrinsic matrix.
        - 'radial_params' (list): Radial distortion coefficients.
        - 'tangential_params' (list): Tangential distortion coefficients.
        - 'width' (int): The width of the image.
        - 'height' (int): The height of the image.
    modality (str): The modality of the image, either 'depth' or 'color'. Determines the interpolation and border mode.
    Returns:
    numpy.ndarray: The undistorted image.
    Raises:
    ValueError: If the modality is not 'depth' or 'color'.
    """
    if orbbec:
        if modality == 'depth':
            interpolation = cv2.INTER_NEAREST
            borderMode = cv2.BORDER_CONSTANT
        elif modality == 'color':
            interpolation = cv2.INTER_LINEAR
            borderMode = cv2.BORDER_TRANSPARENT
        else:
            raise ValueError("Invalid modality. Must be 'depth' or 'color'")
        
        K = cam_params['intrinsics']
        distortion_coeffs = np.array(
            cam_params['radial_params'][:2] + cam_params['tangential_params'] +
            cam_params['radial_params'][2:])
        
        # Adjusted alpha parameter to 1
        newCameraMatrix, _ = cv2.getOptimalNewCameraMatrix(
            K, distortion_coeffs, (cam_params['width'], cam_params['height']), 1)
        
        map1, map2 = cv2.initUndistortRectifyMap(
            K, distortion_coeffs, None, newCameraMatrix,
            (cam_params['width'], cam_params['height']), cv2.CV_32FC1)
        
        undistorted_image = cv2.remap(
            img, map1, map2, interpolation=interpolation, borderMode=borderMode)
        
        return undistorted_image
    else:
        return cv2.undistort(img, cam_params['intrinsics'], np.array([cam_params['radial_params'][0]] + [cam_params['radial_params'][1]] + list(cam_params['tangential_params'][:2]) + [cam_params['radial_params'][2]] + [0, 0, 0]))
