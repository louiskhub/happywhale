import os.path

import cv2
import numpy as np
import tensorflow as tf


def find_box(edges):
    # contour masking
    co, hi = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    con = max(co, key=cv2.contourArea, default=1)

    conv_hull = cv2.convexHull(con)

    top = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
    bottom = tuple(conv_hull[conv_hull[:, :, 1].argmax()][0])
    left = tuple(conv_hull[conv_hull[:, :, 0].argmin()][0])
    right = tuple(conv_hull[conv_hull[:, :, 0].argmax()][0])

    return top, bottom, left, right

def foreground_extraction(img, rec):
    mask= np.zeros(img.shape[:2], np.uint8)
    bgmodel= np.zeros((1, 65), np.float64)
    fgmodel= np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, rec, bgmodel, fgmodel, 3, cv2.GC_INIT_WITH_RECT)
    mask2= np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    img= img*mask2[:,:,np.newaxis]
    img[np.where((img == [0,0,0]).all(axis = 2))] = [255.0, 255.0, 255.0]
    return img


def extract_foreground(img_path):

    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3).numpy()
    org = img.copy()

    img_gray = cv2.cvtColor(np.uint8(img*255), cv2.COLOR_RGB2GRAY)
    img_gray = cv2.medianBlur(img_gray, 7)

    try:
        edges = cv2.Canny(img_gray, 100, 200)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        top, bottom, left, right = find_box(edges)
        rec = (left[0], top[1], right[0]-left[0], bottom[1]-top[1])

        return foreground_extraction(org, rec)

    except:
        return img_gray
