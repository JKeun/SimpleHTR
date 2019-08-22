import numpy as np
import cv2


def preprocess(img, imgsize):
    # there are damaged files in iam dataset - just use black image instead
    if img is None:
        img = np.zeros([1, 1])

    # create target image and copy sample image into it
    (wt, ht) = imgsize
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    newsize = (min(wt, int(w / f)), min(ht, int(h / f)))
    img = cv2.resize(img, newsize)
    target = np.ones([ht, wt]) * 255
    target[0:newsize[1], 0:newsize[0]] = img

    # transpose for tf
    img = cv2.transpose(target)

    # normalize
    (m, s) = cv2.meanStdDev(img)
    m=m[0][0]
    s=s[0][0]
    img = img - m
    img = img / s if s>0 else img
    return img
