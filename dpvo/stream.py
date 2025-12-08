import os
import cv2
import numpy as np
from multiprocessing import Process, Queue
from pathlib import Path
from itertools import chain

def image_stream_tum(queue, imagedir, calib, stride, edge=0, skip=0):
    """ image generator; calib can be None/empty -> no intrinsics """

    intrinsics = None
    K = None
    if calib and os.path.isfile(calib):
        try:
            calib_arr = np.loadtxt(calib, delimiter=" ")
        except ValueError:
            calib_arr = np.loadtxt(calib, delimiter=None)
        fx, fy, cx, cy = calib_arr[:4]
        intrinsics = np.array([fx, fy, cx, cy])
        K = np.eye(3)
        K[0,0] = fx
        K[0,2] = cx
        K[1,1] = fy
        K[1,2] = cy

    image_list = sorted(Path(imagedir).glob("*.png"))[skip::stride]
    assert os.path.exists(imagedir), imagedir

    if edge>0 and intrinsics is not None:
        intrinsics[2] -= edge
        intrinsics[3] -= edge

    for t, imfile in enumerate(image_list):
        image = cv2.imread(str(imfile))
        if K is not None and calib_arr.size > 4:
            image = cv2.undistort(image, K, calib_arr[4:])
        
        if edge>0:
            image = image[edge:-edge, edge:-edge]

        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]


        queue.put((float(imfile.stem), image, intrinsics))

    queue.put((-1, image, intrinsics))


def image_stream(queue, imagedir, calib, stride, edge=0, skip=0):
    """ image generator; calib can be None/empty -> no intrinsics """
    intrinsics = None
    K = None
    if calib and os.path.isfile(calib):
        try:
            calib_arr = np.loadtxt(calib, delimiter=" ")
        except ValueError:
            calib_arr = np.loadtxt(calib, delimiter=None)
        fx, fy, cx, cy = calib_arr[:4]
        intrinsics = np.array([fx, fy, cx, cy])
        K = np.eye(3)
        K[0,0] = fx
        K[0,2] = cx
        K[1,1] = fy
        K[1,2] = cy

    img_exts = ["*.png", "*.jpeg", "*.jpg"]
    image_list = sorted(chain.from_iterable(Path(imagedir).glob(e) for e in img_exts))[skip::stride]
    assert os.path.exists(imagedir), imagedir

    if edge>0 and intrinsics is not None:
        intrinsics[2] -= edge
        intrinsics[3] -= edge

    for t, imfile in enumerate(image_list):
        image = cv2.imread(str(imfile))
        if K is not None and calib_arr.size > 4:
            image = cv2.undistort(image, K, calib_arr[4:])

        if edge:
            image = image[edge:-edge, edge:-edge]

        # keep intrinsics as computed above (or None)
            
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]

        queue.put((t, image, intrinsics))

    queue.put((-1, image, intrinsics))


def video_stream(queue, imagedir, calib, stride, edge, skip=0):
    """ video generator; calib can be None/empty -> no intrinsics """
    intrinsics = None
    K = None
    if calib and os.path.isfile(calib):
        try:
            calib_arr = np.loadtxt(calib, delimiter=" ")
        except ValueError:
            calib_arr = np.loadtxt(calib, delimiter=None)
        fx, fy, cx, cy = calib_arr[:4]
        intrinsics = np.array([fx, fy, cx, cy])
        K = np.eye(3)
        K[0,0] = fx
        K[0,2] = cx
        K[1,1] = fy
        K[1,2] = cy

    assert os.path.exists(imagedir), imagedir
    cap = cv2.VideoCapture(imagedir)

    t = 0

    for _ in range(skip):
        ret, image = cap.read()

    while True:
        # Capture frame-by-frame
        for _ in range(stride):
            ret, image = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                break

        if not ret:
            break

        if K is not None and calib_arr.size > 4:
            image = cv2.undistort(image, K, calib_arr[4:])

        image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]

        # We do not scale intrinsics when calib is None; DPVO will estimate
        intr_out = None
        if intrinsics is not None:
            intr_out = intrinsics.copy()
            # account for resize by 0.5
            intr_out[:2] *= 0.5
            intr_out[2:] *= 0.5
        queue.put((t, image, intr_out))

        t += 1

    queue.put((-1, None, None))
    cap.release()

