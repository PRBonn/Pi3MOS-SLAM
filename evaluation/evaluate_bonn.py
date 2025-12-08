import os
import os.path as osp
from pathlib import Path
import sys
import time
import numpy as np
import torch

# add project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dpvo.config import cfg
from dpvo.lietorch import SE3
from dpvo.stream import image_stream_tum
from utils.eval_traj import run_eval_tum


def _run_scene_tum(imagedir, calib_path, stride=1):
    from multiprocessing import Process, Queue
    from dpvo.dpvo import DPVO

    q = Queue(maxsize=8)
    edge = 6
    r = Process(target=image_stream_tum, args=(q, imagedir, calib_path, stride, edge))
    r.start()

    slam = None
    frame_count = 0
    start_time = None
    
    while True:
        t, image, intr = q.get()
        if t < 0:
            break

        # Record start time on first frame
        if start_time is None:
            start_time = time.time()

        img_t = torch.from_numpy(image).permute(2, 0, 1).cuda()
        intr_t = None if intr is None else torch.from_numpy(intr).cuda()

        if slam is None:
            slam = DPVO(cfg, osp.join(project_root, 'checkpoints/dpvo.pth'), osp.join(project_root, 'checkpoints/model.safetensors'), ht=img_t.shape[1], wd=img_t.shape[2])

        if intr_t is not None:
            slam(t, img_t, intr_t)
        else:
            slam(t, img_t)
        
        frame_count += 1

    end_time = time.time()
    r.join()
    
    poses, tstamps = slam.terminate()
    
    # Calculate frame rate
    total_time = end_time - start_time if start_time is not None else 0
    fps = frame_count / total_time if total_time > 0 else 0
    
    return poses, tstamps, fps


@torch.no_grad()
def evaluate_tum(dataset_root, scenes=None, stride=1, calib_file='calib/bonn.txt', out_dir='eval_results/bonn'):
    if scenes is None:
        scenes = [
            'rgbd_bonn_balloon',
            'rgbd_bonn_balloon2',
            #'rgbd_bonn_balloon_tracking',
            #'rgbd_bonn_balloon_tracking2',
            'rgbd_bonn_crowd',
            'rgbd_bonn_crowd2',
            # 'rgbd_bonn_crowd3',
            #'rgbd_bonn_kidnapping_box',
            #'rgbd_bonn_kidnapping_box2',
            #'rgbd_bonn_moving_nonobstructing_box',
            #'rgbd_bonn_moving_nonobstructing_box2',
            'rgbd_bonn_person_tracking',
            'rgbd_bonn_person_tracking2',
            #'rgbd_bonn_placing_nonobstructing_box',
            #'rgbd_bonn_placing_nonobstructing_box2',
            #'rgbd_bonn_placing_nonobstructing_box3',
            'rgbd_bonn_moving_nonobstructing_box',
            'rgbd_bonn_moving_nonobstructing_box2',
            # 'rgbd_bonn_removing_nonobstructing_box',
            # 'rgbd_bonn_removing_nonobstructing_box2',
            #'rgbd_bonn_synchronous',
            #'rgbd_bonn_synchronous2',
        ]

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    results = {}
    fps_results = {}
    for scene in scenes:
        print(f"Evaluating: {scene}")
        scene_dir = osp.join(dataset_root, scene)
        imagedir = osp.join(scene_dir, 'rgb')
        gt_path = osp.join(scene_dir, 'groundtruth.txt')
        if not osp.exists(imagedir) or not osp.exists(gt_path):
            print(f"  skip (missing rgb or groundtruth): {scene}")
            continue

        poses, tstamps, fps = _run_scene_tum(imagedir, calib_file, stride=stride)

        out_root = osp.join(out_dir, scene)
        os.makedirs(out_root, exist_ok=True)

        ate = run_eval_tum(poses, tstamps, gt_path, plot_file_name=scene, save_dir=out_dir, plot=True)
        print(f"ATE RMSE: {ate:.4f} m, FPS: {fps:.2f}")
        results[scene] = float(ate)
        fps_results[scene] = float(fps)

    # Print summary
    if results:
        mean_ate = float(np.mean(list(results.values())))
        mean_fps = float(np.mean(list(fps_results.values())))
        print("\nResults (ATE RMSE m):")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")
        print(f"Average ATE: {mean_ate:.4f}")
        print("\nFrame Rates (FPS):")
        for k, v in fps_results.items():
            print(f"  {k}: {v:.2f}")
        print(f"Average FPS: {mean_fps:.2f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate DPVO on Bonn Dynamic dataset')
    parser.add_argument('--config', default=osp.join(project_root, 'config/default.yaml'))
    parser.add_argument('--dataset_root', type=str, required=True, help='Root directory containing scene subfolders')
    parser.add_argument('--scenes', nargs='+', default=None, help='Scene names under dataset_root')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--calib', type=str, default=osp.join(project_root, 'calib/bonn.txt'))
    parser.add_argument('--opts', nargs='+', default=[])
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)

    print("\nRunning with config...")
    print(cfg, "\n")

    # Answer to the Ultimate Question of Life, the Universe, and Everything
    torch.manual_seed(42)

    evaluate_tum(args.dataset_root, args.scenes, args.stride, args.calib)
