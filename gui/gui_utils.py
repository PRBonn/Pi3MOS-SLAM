import queue

import cv2
import numpy as np
import open3d as o3d
import torch

import matplotlib.pyplot as plt


cv_gl = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

class Frustum:
    def __init__(self, line_set, view_dir=None, view_dir_behind=None, size=None):
        self.line_set = line_set
        self.view_dir = view_dir
        self.view_dir_behind = view_dir_behind
        self.size = size

    def update_pose(self, pose):
        points = np.asarray(self.line_set.points)
        points_hmg = np.hstack([points, np.ones((points.shape[0], 1))])
        points = (pose @ points_hmg.transpose())[0:3, :].transpose()

        base = np.array([[0.0, 0.0, 0.0]]) * self.size
        base_hmg = np.hstack([base, np.ones((base.shape[0], 1))])
        cameraeye = pose @ base_hmg.transpose()
        cameraeye = cameraeye[0:3, :].transpose()
        eye = cameraeye[0, :]

        base_behind = np.array([[0.0, -2.5, -30.0]]) * self.size
        base_behind_hmg = np.hstack([base_behind, np.ones((base_behind.shape[0], 1))])
        cameraeye_behind = pose @ base_behind_hmg.transpose()
        cameraeye_behind = cameraeye_behind[0:3, :].transpose()
        eye_behind = cameraeye_behind[0, :]

        center = np.mean(points[1:, :], axis=0)
        up = points[2] - points[4]

        self.view_dir = (center, eye, up, pose)
        self.view_dir_behind = (center, eye_behind, up, pose)

        self.center = center
        self.eye = eye
        self.up = up

def create_color_lines(points):
    points_np = np.array([p.reshape(3) for p in points])
    N = len(points_np)
    
    sampled_points = []
    sampled_colors = []

    cmap = plt.get_cmap('jet')

    for i in range(N - 1):
        p1 = points_np[i]
        p2 = points_np[i + 1]
        
        for t in np.linspace(0, 1, 15):
            pt = (1 - t) * p1 + t * p2
            sampled_points.append(pt)

        # Gradient from early (blue) to late (red) using jet colormap
        color_idx = i / (N - 1) if N > 1 else 0.5
        color = cmap(color_idx)[:3] 
        sampled_colors.extend([color] * 15)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(sampled_points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(sampled_colors))

    return pcd

def create_frustum(pose, frusutum_color=[0, 1, 0], size=0.1):
    points = (
        np.array(
            [
                [0.0, 0.0, 0],
                [1.0, -0.5, 2],
                [-1.0, -0.5, 2],
                [1.0, 0.5, 2],
                [-1.0, 0.5, 2],
            ]
        )
        * size
    )

    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [2, 4], [3, 4]]
    colors = [frusutum_color for i in range(len(lines))]

    canonical_line_set = o3d.geometry.LineSet()
    canonical_line_set.points = o3d.utility.Vector3dVector(points)
    canonical_line_set.lines = o3d.utility.Vector2iVector(lines)
    canonical_line_set.colors = o3d.utility.Vector3dVector(colors)
    frustum = Frustum(canonical_line_set, size=size)
    frustum.update_pose(pose)
    return frustum


class DatePacket:
    def __init__(
        self,
        points=None,
        point_colors=None,
        pred_points=None,
        pred_point_colors=None,
        current_pose=None,
        gtcolor=None,
        dynamicmask=None,
        keyframes=None,
        gtframes=None,
        finish=False,
    ):
        self.points = points
        self.point_colors = point_colors
        self.pred_points = pred_points
        self.pred_point_colors = pred_point_colors
        self.current_pose = current_pose
        self.gtcolor = self.resize_img(gtcolor, 320)
        self.dynamic_mask = self.resize_img(dynamicmask, 320)
        self.keyframes = keyframes
        self.gtframes = gtframes
        self.finish = finish

    def resize_img(self, img, width):
        if img is None:
            return None
        # check if img is numpy
        if isinstance(img, np.ndarray):
            height = int(width * img.shape[0] / img.shape[1])
            return cv2.resize(img, (width, height))
        if isinstance(img, torch.Tensor):
            if img.dim() == 2:
                img = img.unsqueeze(0)
            if img.dim() == 3 and img.shape[0] in (1, 3):
                height = int(width * img.shape[1] / img.shape[2])
                img = torch.nn.functional.interpolate(
                    img.unsqueeze(0),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
                return img.squeeze(0)
        return None

def get_latest_queue(q):
    message = None
    while True:
        try:
            message_latest = q.get_nowait()
            if message is not None:
                del message
            message = message_latest
        except queue.Empty:
            if q.qsize() < 1:
                break
    return message


class Packet_vis2main:
    flag_pause = None


class ParamsGUI:
    def __init__(
        self,
        pipe=None,
        background=None,
        q_main2vis=None,
        q_vis2main=None,
    ):
        self.pipe = pipe
        self.background = background
        self.q_main2vis = q_main2vis
        self.q_vis2main = q_vis2main
