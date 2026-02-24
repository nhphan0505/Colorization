import numpy as np
import cv2

def calculate_colorfulness(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
    
    L, a, b = cv2.split(lab_image)
    
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    std_a = np.std(a)
    std_b = np.std(b)
    
    colorfulness = np.sqrt(mean_a**2 + mean_b**2 + 0.3 * (std_a**2 + std_b**2))
    
    return colorfulness