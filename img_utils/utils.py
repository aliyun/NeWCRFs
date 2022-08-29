import numpy as np
import cv2
def auto_gamma_simple(img):
    if img.dtype != np.uint8:
        img_uint = (255.0 * img).clip(0, 255).astype(np.uint8)
    else:
        img_uint = img
    # convert img to HSV
    hsv = cv2.cvtColor(img_uint, cv2.COLOR_RGB2HSV)
    hue, sat, val = cv2.split(hsv)

    # compute gamma = log(mid*255)/log(mean)
    mid = 0.5
    mean = np.mean(val)
    gamma = np.log(mid * 255) / np.log(mean)

    # do gamma correction on value channel
    val_gamma = np.power(val, gamma).clip(0, 255).astype(np.uint8)

    # combine new value channel with original hue and sat channels
    hsv_gamma = cv2.merge([hue, sat, val_gamma])
    img_gamma = cv2.cvtColor(hsv_gamma, cv2.COLOR_HSV2RGB)

    return img_gamma, gamma
def DoG(img, sig0=1, sig1=2):
    blur1 = cv2.GaussianBlur(img, (0, 0), sig0, borderType=cv2.BORDER_REPLICATE)
    blur2 = cv2.GaussianBlur(img, (0, 0), sig1, borderType=cv2.BORDER_REPLICATE)

    img_dog = blur1 - blur2

    img_dog = img_dog / np.amax(np.abs(img_dog))
    return img_dog
def dog_contrast_equalize(img_dog, alpha=0.1, tau=10., mask=None):
    mask = mask > 0 if mask is not None else 1.

    def norm(img, alpha, tau=None):
        img_den = np.abs(img).clip(0, tau) if tau is not None else np.abs(img)
        img_contrast_den = np.power(img_den, alpha)
        img_contrast_den = np.mean(img_contrast_den, axis=(0, 1))
        img_contrast_den = np.power(img_contrast_den, 1.0 / alpha)
        img_contrast = img / img_contrast_den
        return img_contrast

    # contrast equalization equation 1
    img_contrast1 = norm(img_dog, alpha)

    # contrast equalization equation 2
    img_contrast2 = norm(img_contrast1, alpha, tau)
    # tanh stretching
    img_contrast = tau * np.tanh(img_contrast2 / tau)

    img_masked_max = (img_contrast * mask).max()
    img_masked_min = (img_contrast * mask).min()

    # img_contrast_uint = (255.0 * ((0.5 * img_contrast / tau) + 0.5)).clip(0, 255).astype(np.uint8)
    img_contrast_255 = 255 * (img_contrast - img_masked_min) / (img_masked_max - img_masked_min)

    img_contrast_uint = (img_contrast_255.clip(0, 255) * mask).astype(np.uint8)

    return img_contrast_uint
def TT_preprocess_multi_lvl(img: np.ndarray, init_gaus_var=.0, sig0s=(0.5, 2, 4, 8, 16), sig1s=(2, 6, 12, 24, 96),
                            alpha=5., tau=10., mask=None) -> np.ndarray:
    if img.dtype == np.uint8:
        img = img / 255.
    if init_gaus_var > 0.:
        img_blur = cv2.GaussianBlur(img, (0, 0), init_gaus_var)
    else:
        img_blur = img
    img_gamma_corrected, gamma = auto_gamma_simple(img_blur)
    img_gamma_gray = cv2.cvtColor(img_gamma_corrected, cv2.COLOR_RGB2GRAY)
    img_dog = 0.
    for sig0, sig1 in zip(sig0s, sig1s):
        img_dog += DoG(img_gamma_gray / 255., sig0, sig1)

    img_dog /= len(sig0s)

    # if mask is not None:
    #     img_dog = img_dog * mask + (1 - mask) * 128

    img_eq = dog_contrast_equalize(img_dog, alpha, tau, mask=mask)

    return img_eq
def TT_preprocess_multi_lvl_rgb(img: np.ndarray, scaling_sigs=1, init_gaus_var=.0, sig0s=(0.5, 2, 4, 8, 16), sig1s=(2, 6, 12, 24, 96),
                                alpha=5., tau=10., mask=None) -> np.ndarray:

    if img.dtype == np.uint8:
        img = img / 255.
    if init_gaus_var > 0.:
        img_blur = cv2.GaussianBlur(img, (0, 0), init_gaus_var)
    else:
        img_blur = img
    img_gamma_corrected, gamma = auto_gamma_simple(img_blur)
    img_gamma_gray = cv2.cvtColor(img_gamma_corrected, cv2.COLOR_RGB2GRAY)
    img_dog = 0.
    for sig0, sig1 in zip(sig0s, sig1s):

        img_dog += DoG(img_gamma_gray / 255., sig0/scaling_sigs, sig1/scaling_sigs)

    img_dog /= len(sig0s)

    img_eq = dog_contrast_equalize(img_dog, alpha, tau, mask=mask)

    hue, sat, val = cv2.split(cv2.cvtColor((255.0 * img).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV))
    hsv_tt = cv2.merge([hue, sat, img_eq])
    img_tt_rgb = cv2.cvtColor(hsv_tt, cv2.COLOR_HSV2RGB)

    return img_tt_rgb