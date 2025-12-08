import os
import sys
import time
import signal
from argparse import ArgumentParser
from multiprocessing import Process, Queue

import numpy as np
import torch
import torch.multiprocessing as mp
from gui import gui_utils, gui
import os.path as osp
from scipy.spatial.transform import Rotation as R

# DPVO imports
from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.stream import image_stream, image_stream_tum, video_stream
from utils.eval_traj import run_eval_tum
from dpvo.lietorch import SE3
from pi3.utils.geometry import depth_edge

class Keyframe:
    """Simple keyframe class for GUI visualization"""
    def __init__(self, pose_matrix, uid):
        self.pose_matrix = pose_matrix
        self.uid = uid
        self._camera_center = torch.from_numpy(pose_matrix[:3, 3]).cuda()
    
    @property
    def get_inv_RT(self):
        return None, self._camera_center

class Pi_SAM:
    def __init__(self, config):
        self.device = 'cuda'
        
        # DPVO data parameters
        self.imagedir = config.get("imagedir", "")
        self.calib = config.get("calib", "")
        self.stride = config.get("stride", 1)
        self.viz = config.get("viz", False)
        self.tum = config.get("tum", False)
        self.edge = config.get("edge", 0)
        self.gt = config.get("gt", None)

        
        # Initialize parameters
        self.slam_config = cfg
        self.dpvo_network_path = "checkpoints/dpvo.pth"
        self.pi3_network_path = "checkpoints/model.safetensors"
        
        self.slam = None
        
        # For visualizer
        self.historical_poses = []
        self.frame_counter = 0
        self.background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
        signal.signal(signal.SIGINT, self.signal_handler)

        self.q_main2vis = mp.Queue()
        self.q_vis2main = mp.Queue()
        self.params_gui = gui_utils.ParamsGUI(
            background=self.background,
            q_main2vis=self.q_main2vis,
            q_vis2main=self.q_vis2main,
        )

        if self.viz:
            # Start GUI process
            self.gui_process = mp.Process(target=gui.run, args=(self.params_gui,))
            self.gui_process.start()

            # Pause/Resume state management
            self.is_paused = False

            time.sleep(1)  # Wait for GUI initialization

        
    def shutdown_gui(self):
        if self.gui_process and self.gui_process.is_alive():
            print("showdown GUI...")
            self.gui_process.terminate()
            self.gui_process.join(timeout=5)
            if self.gui_process.is_alive():
                self.gui_process.kill()
            self.gui_process.close()

    def signal_handler(self, signum, frame):
        self.shutdown_gui()
        sys.exit(0)
    
    def check_gui_commands(self):
        """Check for pause/resume commands from GUI"""
        try:
            while not self.q_vis2main.empty():
                packet = self.q_vis2main.get_nowait()
                if hasattr(packet, 'flag_pause') and packet.flag_pause is not None:
                    self.is_paused = packet.flag_pause
                    if self.is_paused:
                        print("SLAM paused by user")
                    else:
                        print("SLAM resumed by user")
        except:
            pass  # Queue is empty or other error, continue normally

    @torch.no_grad()
    def run(self):
        queue = Queue(maxsize=8)
        # Start image reader process
        if os.path.isdir(self.imagedir):
            if self.tum:
                stream = image_stream_tum
            else:
                stream = image_stream
            reader = Process(target=stream, args=(queue, self.imagedir, self.calib, self.stride, self.edge))
        else:
            reader = Process(target=video_stream, args=(queue, self.imagedir, self.calib, self.stride))
        
        reader.start()

        while True:
            # Check for GUI commands (pause/resume)
            self.check_gui_commands()
            
            # If paused, skip processing but still check for commands
            if self.viz and self.is_paused:
                time.sleep(0.1)
                continue
            
            (t, image, intrinsics) = queue.get()
            if t < 0:
                break
    
            img = torch.from_numpy(image).permute(2,0,1).cuda()
            intrinsics_tensor = None if intrinsics is None else torch.from_numpy(intrinsics).cuda()
            
            _, H, W = img.shape
            
            if self.slam is None:
                self.slam = DPVO(self.slam_config, self.dpvo_network_path, self.pi3_network_path, ht=H, wd=W)
            
            # When intrinsics are provided, pass them; otherwise estimates K internally
            if intrinsics_tensor is not None:
                predict_points, dynamic_mask, confidences = self.slam(t, img, intrinsics_tensor)
            else:
                predict_points, dynamic_mask, confidences = self.slam(t, img)

            if self.viz:
                points = self.slam.pg.points_.cpu().numpy()[:self.slam.m]
                colors = self.slam.pg.colors_.view(-1, 3).cpu().numpy()[:self.slam.m]
                colors = colors.astype(np.float32) / 255.0
                
                # Filter out points with high variance if available
                if hasattr(self.slam.pg, 'var_') and self.slam.pg.var_ is not None:
                    var_flat = self.slam.pg.var_.view(-1, 1).cpu().numpy()[:self.slam.m]
                    low_var_mask = var_flat.flatten() <= 1.0
                    
                    points = points[low_var_mask]
                    colors = colors[low_var_mask]

                frame_idx = self.slam.n - 1
                if frame_idx >= 0:
                    pose_matrix = SE3(self.slam.poses[0, frame_idx]).matrix().cpu().numpy()
                    estimated_pose = np.linalg.inv(pose_matrix)

                    if self.frame_counter % 3 == 0:
                        keyframe = Keyframe(estimated_pose, self.frame_counter)
                        self.historical_poses.append(keyframe)
                    self.frame_counter += 1
                else:
                    estimated_pose = np.eye(4, dtype=np.float32)

                img_for_gui = img.float() / 255.0
                img_for_gui = img_for_gui[[2, 1, 0], :, :]

                pred_points_np = pred_colors_np = dynamic_mask_for_gui = None
                if predict_points is not None:
                    dynamic_mask_for_gui = dynamic_mask.cpu()
                    
                    Hpi, Wpi = predict_points.shape[0], predict_points.shape[1]
                    device = predict_points.device
                    
                    # Resize image and get all colors
                    img_resized = torch.nn.functional.interpolate(
                        img_for_gui.unsqueeze(0), (Hpi, Wpi), mode="bilinear", align_corners=False
                    ).squeeze(0)
                    all_colors = img_resized.permute(1, 2, 0).reshape(-1, 3)  # [Hpi*Wpi, 3]
                    
                    # Resize dynamic_mask and confidences to match predict_points shape if needed
                    static_mask_thresh = self.slam_config.STATIC_MASK_THRESH
                    if dynamic_mask.shape[:2] != (Hpi, Wpi):
                        dynamic_mask = torch.nn.functional.interpolate(
                            dynamic_mask.unsqueeze(0).unsqueeze(0).to(device), 
                            (Hpi, Wpi), mode="bilinear", align_corners=False
                        ).squeeze(0).squeeze(0)
                    else:
                        dynamic_mask = dynamic_mask.to(device)
                    
                    if confidences.shape[:2] != (Hpi, Wpi):
                        confidences = torch.nn.functional.interpolate(
                            confidences.unsqueeze(0).unsqueeze(0).to(device), 
                            (Hpi, Wpi), mode="bilinear", align_corners=False
                        ).squeeze(0).squeeze(0)
                    else:
                        confidences = confidences.to(device)
                    
                    # Mark dynamic points in red before filtering
                    is_dynamic = dynamic_mask.view(-1) >= static_mask_thresh
                    if torch.any(is_dynamic):
                        red_color = torch.tensor([1.0, 0.0, 0.0], device=device)
                        all_colors[is_dynamic] = red_color * 0.7 + all_colors[is_dynamic] * 0.3
      
                    depth_map = predict_points[:, :, 2] 
                    edge_mask = depth_edge(depth_map, atol=0.5, rtol=0.1, kernel_size=3)
                    stable_mask = ~edge_mask.view(-1)
                    
                    valid_mask = (confidences.view(-1) >= 0.01) & stable_mask
                    # Apply subsampling only on valid points
                    sel = (torch.rand(Hpi * Wpi, device=device) < 0.3) & valid_mask
                    T = torch.from_numpy(estimated_pose).to(device)
                    pts_w = (predict_points.view(-1, 3)[sel] @ T[:3, :3].T) + T[:3, 3]
                    
                    pred_points_np = pts_w.cpu().numpy()
                    pred_colors_np = all_colors[sel].cpu().numpy()

                self.q_main2vis.put(
                    gui_utils.DatePacket(
                        points = points,
                        point_colors = colors,
                        pred_points = pred_points_np,
                        pred_point_colors = pred_colors_np,
                        current_pose = estimated_pose,
                        keyframes = self.historical_poses,
                        gtframes = None,
                        gtcolor = img_for_gui,
                        dynamicmask = dynamic_mask_for_gui
                    )
                )
                
                time.sleep(0.01)
        
        reader.join()
        poses, tstamps = self.slam.terminate()

        if self.gt is not None:
            ate = run_eval_tum(poses, tstamps, self.gt)
            print(f"ATE RMSE: {ate:.4f} m")

        self.shutdown_gui()



if __name__ == "__main__":
    parser = ArgumentParser(description="PI-SAM with DPVO data reading")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--imagedir", type=str, help="Path to image directory or video file")
    parser.add_argument("--calib", type=str, help="Path to calibration file")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    parser.add_argument('--opts', nargs='+', default=[])
    parser.add_argument("--viz", action="store_true", help="Enable GUI visualization")
    parser.add_argument("--tum", action="store_true", help="is TUM-format datasets?")
    parser.add_argument("--edge", type=int, default=0, help="The edge need to cut in raw image")
    parser.add_argument("--gt", type=str, default=None, help="TUM-format ground truth file (timestamp tx ty tz qx qy qz qw)")

    args = parser.parse_args(sys.argv[1:])

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    
    config = {
        "imagedir": args.imagedir,
        "calib": args.calib,
        "stride": args.stride,
        "viz": args.viz,
        "tum": args.tum,
        "edge": args.edge,
        "gt": args.gt
    }

    pisam = Pi_SAM(config)
    pisam.run()
