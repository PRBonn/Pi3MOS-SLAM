from pathlib import Path

import evo.main_ape as main_ape
from evo.core import sync
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface#, plot
#from .plot_utils import plot_trajectory


def run_eval_tum(traj_est, timestamps, gt_file, plot_file_name=None, save_dir=None, plot=False):

    traj_ref = file_interface.read_tum_trajectory_file(gt_file)
    traj_est = PoseTrajectory3D(
                positions_xyz=traj_est[:,:3],
                orientations_quat_wxyz=traj_est[:, [6, 3, 4, 5]],
                timestamps=timestamps)

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
    ate_score = result.stats["rmse"]

    #if plot and plot_file_name is not None and save_dir is not None:
    #    plot_trajectory(traj_est, traj_ref, f"{plot_file_name} (ATE: {ate_score:.03f})",
    #                    f"{save_dir}/{plot_file_name}.pdf", align=True, correct_scale=True)

    return ate_score
