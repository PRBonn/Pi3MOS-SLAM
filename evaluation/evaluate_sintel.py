import os
import os.path as osp
from pathlib import Path
import sys
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# add project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from evaluate_rpe import evaluate_trajectory

from dpvo.lietorch import SE3

from dpvo.config import cfg
from dpvo.utils import Timer
from dpvo.stream import image_stream


def rotmat2qvec(R):
  """Rotation matrix to quaternion."""
  Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
  K = (
      np.array([
          [Rxx - Ryy - Rzz, 0, 0, 0],
          [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
          [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
          [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
      ])
      / 3.0
  )
  eigvals, eigvecs = np.linalg.eigh(K)
  qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
  if qvec[0] < 0:
    qvec *= -1
  return qvec


def align_trajectories(model, data):
  """Align two trajectories using the method of Horn (closed-form).

  Args:
    model: first trajectory (3xn)
    data: second trajectory (3xn)

  Returns:
    rot: rotation matrix (3x3)
    trans: translation vector (3x1)
    trans_error: translational error per point (1xn)
  """
  np.set_printoptions(precision=3, suppress=True)
  model_mean = [[model.mean(1)[0]], [model.mean(1)[1]], [model.mean(1)[2]]]
  data_mean = [[data.mean(1)[0]], [data.mean(1)[1]], [data.mean(1)[2]]]
  model_zerocentered = model - model_mean
  data_zerocentered = data - data_mean

  W = np.zeros((3, 3))
  for column in range(model.shape[1]):
    W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
  U, _, Vh = np.linalg.linalg.svd(W.transpose())
  S = np.matrix(np.identity(3))
  if np.linalg.det(U) * np.linalg.det(Vh) < 0:
    S[2, 2] = -1
  rot = U * S * Vh  # pylint: disable=redefined-outer-name

  rotmodel = rot * model_zerocentered
  dots = 0.0
  norms = 0.0

  for column in range(data_zerocentered.shape[1]):
    dots += np.dot(
        data_zerocentered[:, column].transpose(), rotmodel[:, column]
    )
    normi = np.linalg.norm(model_zerocentered[:, column])
    norms += normi * normi

  s = float(dots / norms)

  # print ("scale: %f " % s)
  trans = data_mean - s * rot * model_mean  # pylint: disable=redefined-outer-name

  model_aligned = s * rot * model + trans
  alignment_error = model_aligned - data

  trans_error = np.sqrt(  # pylint: disable=redefined-outer-name
      np.sum(np.multiply(alignment_error, alignment_error), 0)
  ).A[0]

  return rot, trans, trans_error, s, model_aligned


def _run_scene(scene_dir, network, pi3_ckpt, stride=1, show=False, use_calib=True):
    from dpvo.dpvo import DPVO
    imagedir = osp.join(scene_dir, "rgb")
    calib = osp.join(scene_dir, "calibration.txt") if use_calib else ""

    from multiprocessing import Process, Queue
    q = Queue(maxsize=8)
    r = Process(target=image_stream, args=(q, imagedir, calib, stride, 0))
    r.start()

    slam = None
    while True:
        t, image, intr = q.get()
        if t < 0:
            break
        img_t = torch.from_numpy(image).permute(2, 0, 1).cuda()
        intr_t = None if intr is None else torch.from_numpy(intr).cuda()
        if slam is None:
            slam = DPVO(cfg, network, pi3_ckpt, ht=img_t.shape[1], wd=img_t.shape[2])
        with Timer("SLAM", enabled=False):
            if intr_t is not None:
                slam(t, img_t, intr_t)
            else:
                slam(t, img_t)
        if show:
            cv2.imshow("img", image)
            cv2.waitKey(1)

    r.join()
    return slam.terminate()


@torch.no_grad()
def evaluate_sintel(config, network, pi3_ckpt, sintel_dir, scenes=None, stride=1, trials=1,
                    plot=False, save_trajectory=False, show_img=False, out_dir="eval_results",
                    use_calib=True):
    if scenes is None:
        scenes = ["alley_2", "ambush_4", "ambush_5", "ambush_6", "cave_2", "cave_4", "market_2", "market_5", "market_6", "shaman_3", "sleeping_1", "sleeping_2", "temple_2", "temple_3"]
    
    ate = []
    rte = []
    rre = []
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for scene in scenes:
        print(f"Evaluating: {scene}")
        scene_dir = osp.join(sintel_dir, scene)
        gt_path = osp.join(scene_dir, "extrinsics.npy")
        if not osp.exists(gt_path):
            print(f"  skip (missing extrinsics.npy)")
            continue

        poses, tstamps = _run_scene(scene_dir, network, pi3_ckpt, stride=stride, show=False, use_calib=use_calib)
        gt_cam2w = np.load(gt_path)
        cam_c2w = SE3(torch.as_tensor(poses, device="cpu"))#.inv()
        est_cam2w = cam_c2w.matrix().numpy()
        num_cams = gt_cam2w.shape[0]

        assert gt_cam2w.shape[0] == est_cam2w.shape[0]

        full_t = np.dot(np.linalg.inv(gt_cam2w[-1]), gt_cam2w[0])
        normalize_scale = np.linalg.norm(full_t[:3, 3]) + 1e-8
        gt_cam2w[:, :3, 3] /= normalize_scale
        print(f'normalize_scale : {normalize_scale}')

        rot, trans, trans_error, scale, align_tj = align_trajectories(
            est_cam2w[:, :3, 3].transpose(1, 0), gt_cam2w[:, :3, 3].transpose(1, 0)
        )

        est_cam2w[:, :3, 3] = (
            scale * rot * est_cam2w[:, :3, 3].transpose(1, 0) + trans
        ).transpose(1, 0)

        for k in range(num_cams):
            est_cam2w[k, :3, :3] = rot @ est_cam2w[k, :3, :3]

        traj_est_dict = [est_cam2w[i, ...] for i in range(est_cam2w.shape[0])]
        traj_gt_dict = [gt_cam2w[i, ...] for i in range(gt_cam2w.shape[0])]
        rpe_result = evaluate_trajectory(
            traj_gt_dict, traj_est_dict, param_fixed_delta=True, param_delta=1
        )

        rte_error = np.array(rpe_result)[:, 2]
        rre_error = np.array(rpe_result)[:, 3]

        trans_error_mean = np.sqrt(np.mean(rte_error**2))
        rot_error_mean = np.sqrt(np.mean(rre_error**2))

        print(
            "absolute_translational_error.rmse %f m"
            % np.sqrt(np.dot(trans_error, trans_error) / len(trans_error))
        )
        print("relative translational_error %f m" % trans_error_mean)
        print("relative rotational_error %f deg" % np.rad2deg(rot_error_mean))

        ate.append(np.sqrt(np.dot(trans_error, trans_error) / len(trans_error)))
        rte.append(trans_error_mean)
        rre.append(np.rad2deg(rot_error_mean))

        if plot:
            gt_xyz = gt_cam2w[:, :3, 3]
            est_xyz = est_cam2w[:, :3, 3]
            # 3D trajectory (XYZ)
            fig3 = plt.figure(figsize=(8, 6))
            ax3 = fig3.add_subplot(111, projection='3d')
            ax3.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], 'k-', linewidth=2, label='GT (cam2w)')
            ax3.plot(est_xyz[:, 0], est_xyz[:, 1], est_xyz[:, 2], 'r--', linewidth=2, label='Aligned Est')
            ax3.scatter(gt_xyz[0, 0], gt_xyz[0, 1], gt_xyz[0, 2], c='g', s=40, label='Start')
            ax3.scatter(gt_xyz[-1, 0], gt_xyz[-1, 1], gt_xyz[-1, 2], c='b', s=40, label='End')
            ax3.set_xlabel('X (m)')
            ax3.set_ylabel('Y (m)')
            ax3.set_zlabel('Z (m)')
            ax3.set_title(f'{scene}: Trajectory (3D)')
            # Set equal aspect for 3D
            xs = np.concatenate([gt_xyz[:, 0], est_xyz[:, 0]])
            ys = np.concatenate([gt_xyz[:, 1], est_xyz[:, 1]])
            zs = np.concatenate([gt_xyz[:, 2], est_xyz[:, 2]])
            x_range = xs.max() - xs.min()
            y_range = ys.max() - ys.min()
            z_range = zs.max() - zs.min()
            max_range = max(x_range, y_range, z_range) + 1e-8
            x_mid = 0.5 * (xs.max() + xs.min())
            y_mid = 0.5 * (ys.max() + ys.min())
            z_mid = 0.5 * (zs.max() + zs.min())
            half = 0.5 * max_range
            ax3.set_xlim(x_mid - half, x_mid + half)
            ax3.set_ylim(y_mid - half, y_mid + half)
            ax3.set_zlim(z_mid - half, z_mid + half)
            ax3.legend(loc='best')
            fig3_path = osp.join(out_dir, f'{scene}_traj_3d.png')
            plt.tight_layout()
            plt.savefig(fig3_path, dpi=150)
            plt.close(fig3)

    print("Average ATE: ", np.mean(ate))
    print("Average RTE: ", np.mean(rte))
    print("Average RRE: ", np.mean(rre))



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate DPVO on Sintel dataset')
    parser.add_argument('--network', type=str, default=osp.join(project_root, 'checkpoints/dpvo.pth'))
    parser.add_argument('--pi3_ckpt', type=str, default=osp.join(project_root, 'checkpoints/model.safetensors'))
    parser.add_argument('--config', default=osp.join(project_root, 'config/default.yaml'))
    parser.add_argument('--dataset_root', default="/home/starry/Data/Sintel")
    parser.add_argument('--scenes', nargs='+', default=None, help='Specific scenes to evaluate')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--show_img', action="store_true")
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--save_trajectory', action="store_true")
    parser.add_argument('--backend_thresh', type=float, default=32.0)
    parser.add_argument('--opts', nargs='+', default=[])
    parser.add_argument('--no_calib', action='store_true', help='Ignore provided calibration, estimate intrinsics')
    args = parser.parse_args()
    
    cfg.merge_from_file(args.config)
    cfg.BACKEND_THRESH = args.backend_thresh
    cfg.merge_from_list(args.opts)
    
    print("\nRunning with config...")
    print(cfg, "\n")
    
    # Answer to the Ultimate Question of Life, the Universe, and Everything
    torch.manual_seed(42)
    
    evaluate_sintel(cfg, args.network, args.pi3_ckpt, args.sintel_dir, args.scenes,
                    args.stride, args.trials, args.plot, args.save_trajectory,
                    args.show_img, use_calib=not args.no_calib)
