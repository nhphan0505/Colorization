import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(original, generated):
    original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    generated_gray = cv2.cvtColor(generated, cv2.COLOR_RGB2GRAY)

    ssim_value, _ = ssim(original_gray, generated_gray, full=True, data_range=255.0)
    return ssim_value