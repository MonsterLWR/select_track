import numpy as np
import cv2


def gauss_response(height, width, sigma=2.0):
    g = np.zeros((height, width), np.float32)
    g[height // 2, width // 2] = 1
    g = cv2.GaussianBlur(g, (-1, -1), sigma)
    g /= g.max()
    return g


def correlation_filter(f, G):
    f = pre_process(f)
    # cv2.imshow('pro_fi', fi)
    # cv2.waitKey()
    F = np.fft.fft2(f)
    A = G * np.conjugate(F)
    B = F * np.conjugate(F)
    return A / B


def pre_process(img):
    # get the size of the img...
    height, width = img.shape
    img = np.log(np.float32(img) + 1.0)
    img = (img - img.mean()) / (img.std() + 1e-5)
    # use the hanning window...
    # window = window_func_2d(height, width)
    window = cv2.createHanningWindow((width, height), cv2.CV_32F)
    img = img * window

    return img


def correlation(f, H):
    f = pre_process(f)
    G = H * np.fft.fft2(f)
    g = np.absolute(np.fft.ifft2(G))
    # cv2.imshow('g', g)
    # print(type(gi[0, 0]))
    return _compute_psr(g)


def _compute_psr(resp):
    _, mval, _, (mx, my) = cv2.minMaxLoc(resp)
    side_resp = resp.copy()
    cv2.rectangle(side_resp, (mx - 5, my - 5), (mx + 5, my + 5), 0, -1)
    smean, sstd = side_resp.mean(), side_resp.std()
    psr = (mval - smean) / (sstd + 1e-5)
    return psr


def linear_mapping(images):
    # [a,b]->[0,1]
    max_value = images.max()
    min_value = images.min()

    parameter_a = 1 / (max_value - min_value)
    parameter_b = 1 - max_value * parameter_a

    image_after_mapping = parameter_a * images + parameter_b

    return image_after_mapping
