# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Computes relative pose error.

This script computes the relative pose error from the ground truth trajectory
and the estimated trajectory.
"""

import random
import numpy as np


def ominus(a, b):
  """Compute the relative 3D transformation between a and b.

  Args:
    a: first pose (homogeneous 4x4 matrix)
    b: second pose (homogeneous 4x4 matrix)

  Returns:
    Relative 3D transformation from a to b.
  """
  return np.dot(np.linalg.inv(a), b)


def compute_distance(transform):
  """Compute the distance of the translational component of a 4x4 homogeneous matrix."""
  return np.linalg.norm(transform[0:3, 3])


def compute_angle(transform):
  """Compute the rotation angle from a 4x4 homogeneous matrix."""
  # an invitation to 3-d vision, p 27
  return np.arccos(min(1, max(-1, (np.trace(transform[0:3, 0:3]) - 1) / 2)))


def distances_along_trajectory(traj):
  """Compute the translational distances along a trajectory."""
  motion = [ominus(traj[i + 1], traj[i]) for i in range(len(traj) - 1)]
  distances = [0]
  s = 0
  for t in motion:
    s += compute_distance(t)
    distances.append(s)
  return distances


def evaluate_trajectory(
    traj_gt,
    traj_est,
    param_max_pairs=10000,
    param_fixed_delta=False,
    param_delta=1.00,
):
  """Compute the relative pose error between two trajectories.

  Args:
    traj_gt: the first trajectory (ground truth)
    traj_est: the second trajectory (estimated trajectory)
    param_max_pairs: number of relative poses to be evaluated
    param_fixed_delta: false- evaluate over all possible pairs
                       true- only evaluate over pairs with a given
                         distance (delta)
    param_delta: distance between the evaluated pairs

  Returns:
    list of compared poses and the resulting translation and rotation error

  Raises:
    Exception: if no pairs can be found between the trajectories
  """

  if not param_fixed_delta:
    if param_max_pairs == 0 or len(traj_est) < np.sqrt(param_max_pairs):
      pairs = [
          (i, j) for i in range(len(traj_est)) for j in range(len(traj_est))  # pylint: disable=g-complex-comprehension
      ]
    else:
      pairs = [
          (
              random.randint(0, len(traj_est) - 1),
              random.randint(0, len(traj_est) - 1),
          )
          for _ in range(param_max_pairs)
      ]
  else:
    pairs = []
    for i in range(len(traj_est)):
      j = i + param_delta
      if j < len(traj_est):
        pairs.append((i, j))
    if param_max_pairs != 0 and len(pairs) > param_max_pairs:
      pairs = random.sample(pairs, param_max_pairs)

  result = []
  for i, j in pairs:
    # print(i, j)
    error44 = ominus(
        ominus(traj_est[j], traj_est[i]), ominus(traj_gt[j], traj_gt[i])
    )

    # breakpoint()

    trans = compute_distance(error44)
    rot = compute_angle(error44)

    result.append([i, j, trans, rot])

  if len(result) < 2:
    raise Exception(   # pylint: disable=broad-exception-raised
        "Couldn't find pairs between groundtruth and estimated trajectory!"
    )

  return result
