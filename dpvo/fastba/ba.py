import torch
import cuda_ba

neighbors = cuda_ba.neighbors
reproject = cuda_ba.reproject

def BA(poses, patches, intrinsics, target, weight, prior_depth, prior_weights, lmbda, ii, jj, kk, t0, t1, M, iterations, eff_impl=False, use_cov_adaptive=True, cov_steepness=10.0, cov_threshold=2.0):
    return cuda_ba.forward(poses.data, patches, intrinsics, target, weight, prior_depth, prior_weights, lmbda, ii, jj, kk, M, t0, t1, iterations, eff_impl, use_cov_adaptive, cov_steepness, cov_threshold)
