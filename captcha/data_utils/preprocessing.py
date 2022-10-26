import cv2
import numpy as np


def image_preprocess(image: np.ndarray) -> np.ndarray:
    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, im_inv = cv2.threshold(im_gray, 220, 255, cv2.THRESH_BINARY_INV)
    kernel = 1 / 16 * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    im_blur = cv2.filter2D(im_inv, -1, kernel)
    ret, im_res = cv2.threshold(im_blur, 150, 255, cv2.THRESH_BINARY)
    preprocessed_image = cv2.cvtColor(im_res,cv2.COLOR_GRAY2RGB)

    return preprocessed_image
