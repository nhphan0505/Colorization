import numpy as np

def calculate_psnr(original, generated):
    original = original.astype(np.float64)
    generated = generated.astype(np.float64)

    mse = np.mean((original - generated) ** 2)

    if mse == 0:
        return float('inf')

    max_pixel = 255.0

    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr