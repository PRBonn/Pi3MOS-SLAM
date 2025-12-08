import time
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from . import altcorr, fastba, lietorch
from . import projective_ops as pops
from .lietorch import SE3
from .net import VONet
from .patchgraph import PatchGraph
from .utils import *

from safetensors.torch import load_file
from pi3.utils.basic import preprocess_tensor_for_pi3
from pi3.utils.geometry import se3_inverse
from pi3.models.pi3 import Pi3

mp.set_start_method('spawn', True)


autocast = torch.cuda.amp.autocast
Id = SE3.Identity(1, device="cuda")


class DPVO:

    def __init__(self, cfg, network, pi3_ckpt, ht=480, wd=640):
        self.cfg = cfg
        self.load_weights(network)
        self.load_weights_pi3(pi3_ckpt)

        self.is_initialized = False
        self.enable_timing = False
        torch.set_num_threads(2)

        self.M = self.cfg.PATCHES_PER_FRAME
        self.N = self.cfg.BUFFER_SIZE

        self.ht = ht    # image height
        self.wd = wd    # image width

        DIM = self.DIM
        RES = self.RES

        self.dpvo_init_window = []
        self.intrinsic_init_window = []

        ### state attributes ###
        self.tlist = []
        self.counter = 0

        # keep track of global-BA calls
        self.ran_global_ba = np.zeros(100000, dtype=bool)

        self.frame_for_init = 12# should be smaller than self.pmem (frame memory size)
 
        self.omega = 0.0 # weight of depth prior normalization

        self.intrinsics_errors = []

        self.fmap_ht = ht // RES
        self.fmap_wd = wd // RES

        ### network attributes ###
        if self.cfg.MIXED_PRECISION:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.half}
        else:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.float}

        ### frame memory size ###
        self.pmem = self.mem = self.pi3_mem = 36 # 32 was too small given default settings
        if self.cfg.LOOP_CLOSURE:
            self.last_global_ba = -1000 # keep track of time since last global opt
            self.pmem = self.cfg.MAX_EDGE_AGE # patch memory

        self.pi3_images_ = None
        self.pi3_dynamic_masks_ = None

        self.imap_ = torch.zeros(self.pmem, self.M, DIM, **kwargs)
        self.gmap_ = torch.zeros(self.pmem, self.M, 128, self.P, self.P, **kwargs)

        self.pg = PatchGraph(self.cfg, self.P, self.DIM, self.pmem, **kwargs)

        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.load_long_term_loop_closure()

        self.fmap1_ = torch.zeros(1, self.mem, 128, self.fmap_ht // 1, self.fmap_wd // 1, **kwargs)
        self.fmap2_ = torch.zeros(1, self.mem, 128, self.fmap_ht // 4, self.fmap_wd // 4, **kwargs)

        # feature pyramid
        self.pyramid = (self.fmap1_, self.fmap2_)
    
    def convert_matrix_to_quat(self, pi3_pose):
        """
        Convert PI3 4x4 matrix pose to DPVO SE3 format
        Args:
            pi3_pose: [4, 4] transformation matrix from PI3 (camera-to-world)
            
        Returns:
            DPVO pose format [7] (translation + quaternion)
        """
        # DPVO uses SE3 format: [tx, ty, tz, qx, qy, qz, qw]
        device = pi3_pose.device
        translation = pi3_pose[:3, 3].to(device=device, dtype=torch.float32)  # [3]
        rotation_matrix = pi3_pose[:3, :3].to(device=device, dtype=torch.float32)
        # ensure orthonormality before converting to quaternion
        try:
            U, _, Vh = torch.linalg.svd(rotation_matrix)
            R_ortho = (U @ Vh).to(device='cpu')
        except Exception:
            R_ortho = rotation_matrix.to(device='cpu')
        r = Rotation.from_matrix(R_ortho.numpy())
        quat_np = r.as_quat()  # [x, y, z, w]
        quat = torch.tensor(quat_np, dtype=torch.float32, device=device)
        # normalize and ensure consistent sign
        quat = quat / (quat.norm() + 1e-8)
        if quat[3] < 0:
            quat = -quat
        dpvo_pose = torch.zeros(7, device=device, dtype=torch.float32)
        dpvo_pose[:3] = translation
        dpvo_pose[3:7] = quat  # qx, qy, qz, qw
        
        return dpvo_pose

    def load_long_term_loop_closure(self):
        try:
            from .loop_closure.long_term import LongTermLoopClosure
            self.long_term_lc = LongTermLoopClosure(self.cfg, self.pg)
        except ModuleNotFoundError as e:
            self.cfg.CLASSIC_LOOP_CLOSURE = False
            print(f"WARNING: {e}")

    def load_weights(self, network):
        # load network from checkpoint file
        if isinstance(network, str):
            from collections import OrderedDict
            state_dict = torch.load(network)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if "update.lmbda" not in k:
                    new_state_dict[k.replace('module.', '')] = v
            
            self.network = VONet()
            print(f'Loading dpvo net...')
            self.network.load_state_dict(new_state_dict)

        else:
            self.network = network

        # steal network attributes
        self.DIM = self.network.DIM
        self.RES = self.network.RES
        self.P = self.network.P

        self.network.cuda()
        self.network.eval()

    def load_weights_pi3(self, ckpt):
        self.pi3 = Pi3().to('cuda').eval()
        if ckpt.endswith('.safetensors'):
            weight = load_file(ckpt)
        else:
            weight = torch.load(ckpt, map_location='cuda', weights_only=False)
        print(f'Loading pi3...')
        self.pi3.load_state_dict(weight)

    @property
    def poses(self):
        return self.pg.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        return self.pg.patches_.view(1, self.N*self.M, 3, 3, 3)

    @property
    def init_inv_depth(self):
        return self.pg.prior_inv_depth_.view(1, self.N*self.M, 1)

    @property
    def confidences(self):
        return self.pg.confidences_.view(1, self.N*self.M, 1)

    @property
    def intrinsics(self):
        return self.pg.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.pg.index_.view(-1)

    @property
    def imap(self):
        return self.imap_.view(1, self.pmem * self.M, self.DIM)

    @property
    def gmap(self):
        return self.gmap_.view(1, self.pmem * self.M, 128, 3, 3)

    @property
    def n(self):
        return self.pg.n

    @n.setter
    def n(self, val):
        self.pg.n = val

    @property
    def m(self):
        return self.pg.m

    @m.setter
    def m(self, val):
        self.pg.m = val

    def get_pose(self, t):
        if t in self.traj:
            return SE3(self.traj[t])

        t0, dP = self.pg.delta[t]
        return dP * self.get_pose(t0)


    def terminate(self):

        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc.terminate(self.n)

        if self.cfg.LOOP_CLOSURE:
            self.append_factors(*self.pg.edges_loop())

        for _ in range(12):
            self.ran_global_ba[self.n] = False
            self.update()

        """ interpolate missing poses """
        self.traj = {}
        for i in range(self.n):
            self.traj[self.pg.tstamps_[i]] = self.pg.poses_[i]

        poses = [self.get_pose(t) for t in range(self.counter)]
        poses = lietorch.stack(poses, dim=0)
        poses = poses.inv().data.cpu().numpy()
        tstamps = np.array(self.tlist, dtype=np.float64)

        return poses, tstamps

    def corr(self, coords, indicies=None):
        """ local correlation volume """
        ii, jj = indicies if indicies is not None else (self.pg.kk, self.pg.jj)
        ii1 = ii % (self.M * self.pmem)
        jj1 = jj % (self.mem)
        corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
        corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3)
        return torch.stack([corr1, corr2], -1).view(1, len(ii), -1)

    def reproject(self, indicies=None):
        """ reproject patch k from i -> j """
        (ii, jj, kk) = indicies if indicies is not None else (self.pg.ii, self.pg.jj, self.pg.kk)
        coords = pops.transform(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk)
        return coords.permute(0, 1, 4, 2, 3).contiguous()

    def append_factors(self, ii, jj):
        self.pg.jj = torch.cat([self.pg.jj, jj])
        self.pg.kk = torch.cat([self.pg.kk, ii])
        self.pg.ii = torch.cat([self.pg.ii, self.ix[ii]])

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        self.pg.net = torch.cat([self.pg.net, net], dim=1)

    def remove_factors(self, m, store: bool):
        assert self.pg.ii.numel() == self.pg.weight.shape[1]
        if store:
            self.pg.ii_inac = torch.cat((self.pg.ii_inac, self.pg.ii[m]))
            self.pg.jj_inac = torch.cat((self.pg.jj_inac, self.pg.jj[m]))
            self.pg.kk_inac = torch.cat((self.pg.kk_inac, self.pg.kk[m]))
            self.pg.weight_inac = torch.cat((self.pg.weight_inac, self.pg.weight[:,m]), dim=1)
            self.pg.target_inac = torch.cat((self.pg.target_inac, self.pg.target[:,m]), dim=1)
        self.pg.weight = self.pg.weight[:,~m]
        self.pg.target = self.pg.target[:,~m]

        self.pg.ii = self.pg.ii[~m]
        self.pg.jj = self.pg.jj[~m]
        self.pg.kk = self.pg.kk[~m]
        self.pg.net = self.pg.net[:,~m]
        assert self.pg.ii.numel() == self.pg.weight.shape[1]

    def motion_probe(self):
        """ kinda hacky way to ensure enough motion for initialization """
        kk = torch.arange(self.m-self.M, self.m, device="cuda")
        jj = self.n * torch.ones_like(kk)
        ii = self.ix[kk]

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        coords = self.reproject(indicies=(ii, jj, kk))

        with autocast(enabled=self.cfg.MIXED_PRECISION):
            corr = self.corr(coords, indicies=(kk, jj))
            ctx = self.imap[:,kk % (self.M * self.pmem)]
            net, (delta, weight, _) = \
                self.network.update(net, ctx, corr, None, ii, jj, kk)

        return torch.quantile(delta.norm(dim=-1).float(), 0.5)


    def motionmag(self, i, j):
        k = (self.pg.ii == i) & (self.pg.jj == j)
        ii = self.pg.ii[k]
        jj = self.pg.jj[k]
        kk = self.pg.kk[k]

        flow, _ = pops.flow_mag(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5)
        return flow.mean().item()

    def keyframe(self):

        i = self.n - self.cfg.KEYFRAME_INDEX - 1
        j = self.n - self.cfg.KEYFRAME_INDEX + 1
        m = self.motionmag(i, j) + self.motionmag(j, i)
 
        if m / 2 < self.cfg.KEYFRAME_THRESH:
            k = self.n - self.cfg.KEYFRAME_INDEX
            t0 = self.pg.tstamps_[k-1]
            t1 = self.pg.tstamps_[k]

            dP = SE3(self.pg.poses_[k]) * SE3(self.pg.poses_[k-1]).inv()
            self.pg.delta[t1] = (t0, dP)

            to_remove = (self.pg.ii == k) | (self.pg.jj == k)
            self.remove_factors(to_remove, store=False)

            self.pg.kk[self.pg.ii > k] -= self.M
            self.pg.ii[self.pg.ii > k] -= 1
            self.pg.jj[self.pg.jj > k] -= 1

            for i in range(k, self.n-1):
                self.pg.tstamps_[i] = self.pg.tstamps_[i+1]
                self.pg.colors_[i] = self.pg.colors_[i+1]
                self.pg.poses_[i] = self.pg.poses_[i+1]
                self.pg.patches_[i] = self.pg.patches_[i+1]
                self.pg.intrinsics_[i] = self.pg.intrinsics_[i+1]
                self.pg.prior_inv_depth_[i] = self.pg.prior_inv_depth_[i+1]
                self.pg.confidences_[i] = self.pg.confidences_[i+1]
                self.pg.var_[i] = self.pg.var_[i+1]

                self.imap_[i % self.pmem] = self.imap_[(i+1) % self.pmem]
                self.gmap_[i % self.pmem] = self.gmap_[(i+1) % self.pmem]
                self.fmap1_[0,i%self.mem] = self.fmap1_[0,(i+1)%self.mem]
                self.fmap2_[0,i%self.mem] = self.fmap2_[0,(i+1)%self.mem]
                self.pi3_images_[i % self.pi3_mem] = self.pi3_images_[(i+1) % self.pi3_mem]

            self.n -= 1
            self.m -= self.M

            if self.cfg.CLASSIC_LOOP_CLOSURE:
                self.long_term_lc.keyframe(k)

        to_remove = self.ix[self.pg.kk] < self.n - self.cfg.REMOVAL_WINDOW # Remove edges falling outside the optimization window
        if self.cfg.LOOP_CLOSURE:
            # ...unless they are being used for loop closure
            lc_edges = ((self.pg.jj - self.pg.ii) > 30) & (self.pg.jj > (self.n - self.cfg.OPTIMIZATION_WINDOW))
            to_remove = to_remove & ~lc_edges
        self.remove_factors(to_remove, store=True)

    def __run_global_BA(self):
        """ Global bundle adjustment
         Includes both active and inactive edges """
        full_target = torch.cat((self.pg.target_inac, self.pg.target), dim=1)
        full_weight = torch.cat((self.pg.weight_inac, self.pg.weight), dim=1)
        full_ii = torch.cat((self.pg.ii_inac, self.pg.ii))
        full_jj = torch.cat((self.pg.jj_inac, self.pg.jj))
        full_kk = torch.cat((self.pg.kk_inac, self.pg.kk))

        self.pg.normalize()
        lmbda = torch.as_tensor([1e-4], device="cuda")
        t0 = self.pg.ii.min().item()
        confidences = self.confidences if self.cfg.USE_DEPTH_PRIOR else self.confidences * 0
        fastba.BA(self.poses, self.patches, self.intrinsics,
            full_target, full_weight, self.init_inv_depth, confidences, lmbda, full_ii, full_jj, full_kk, t0, self.n, M=self.M, iterations=2, eff_impl=True, use_cov_adaptive=False,
            cov_steepness=self.cfg.COV_STEEPNESS, cov_threshold=self.cfg.COV_THRESHOLD)
        self.ran_global_ba[self.n] = True

    def update(self, cov=True):
        with Timer("other", enabled=self.enable_timing):
            coords = self.reproject()

            with autocast(enabled=True):
                corr = self.corr(coords)
                ctx = self.imap[:, self.pg.kk % (self.M * self.pmem)]
                self.pg.net, (delta, weight, _) = \
                    self.network.update(self.pg.net, ctx, corr, None, self.pg.ii, self.pg.jj, self.pg.kk)

            lmbda = torch.as_tensor([1e-4], device="cuda")
            weight = weight.float()
            target = coords[...,self.P//2,self.P//2] + delta.float()

        self.pg.target = target
        self.pg.weight = weight

        with Timer("BA", enabled=self.enable_timing):
            try:
                # run global bundle adjustment if there exist long-range edges
                if (self.pg.ii < self.n - self.cfg.REMOVAL_WINDOW - 1).any() and not self.ran_global_ba[self.n]:
                    self.__run_global_BA()
                else:
                    t0 = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
                    t0 = max(t0, 1)
                    confidences = self.confidences if self.cfg.USE_DEPTH_PRIOR else self.confidences * 0
                    var_depth = fastba.BA(self.poses, self.patches, self.intrinsics, 
                            target, weight, self.init_inv_depth, confidences, lmbda, 
                            self.pg.ii, self.pg.jj, self.pg.kk, 
                            t0, self.n, M=self.M, iterations=2, eff_impl=False, use_cov_adaptive=cov,
                            cov_steepness=self.cfg.COV_STEEPNESS, cov_threshold=self.cfg.COV_THRESHOLD)
            except:
                print("Warning BA failed...")

            if cov:
                self.pg.var_[:var_depth[0].shape[0]] = var_depth[0].unsqueeze(-1)

            points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
            points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
            
            self.pg.points_[:len(points)] = points[:]
    
    
    def pi3_inference(self, id_list):
        imgs = [self.pi3_images_[i % self.pi3_mem] for i in id_list]
        images_tensor = torch.stack(imgs, dim=0).unsqueeze(0)  # [1, N, 3, H, W]

        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                pi3_output = self.pi3(images_tensor)

        pi3_points = pi3_output['local_points'][0]
        pi3_confidences = pi3_output['conf'][0].squeeze(-1)  # [N, H, W]

        if self.is_initialized:
            s = self.estimate_scale(pi3_points, pi3_confidences, id_list)
        else:
            s = 1.0

        # pi3_points: [N, H, W, 3]
        s_safe = max(1e-6, float(s))
        pi3_points = pi3_points / s_safe
        pi3_depths = pi3_points[..., 2]

        dynamic_prob = pi3_output['dynamic_masks'][0].squeeze(-1)
        pi3_depths_resized = self._resize_to_fmap(pi3_depths)
        pi3_confidences_resized = self._resize_to_fmap(pi3_confidences)
        dynamic_prob_resized = self._resize_to_fmap(dynamic_prob)

        if self.pi3_dynamic_masks_ == None:
            self.pi3_dynamic_masks_ = dynamic_prob_resized.new_zeros((self.pi3_mem, *dynamic_prob_resized.shape[1:]))

        for n, fid in enumerate(id_list):
            self.pi3_dynamic_masks_[fid % self.pi3_mem] = dynamic_prob_resized[n]

        # Align PI3 camera-to-world poses to DPVO frame so that the first frame matches DPVO's Tcw at id0
        Twc_pi3 = pi3_output['camera_poses'][0]             # [N, 4, 4]
        
        id0 = int(id_list[0])
        Tcw_dpvo_0 = SE3(self.pg.poses_[id0]).matrix()      # [4, 4]
        Twc_origin = se3_inverse(Twc_pi3[0]) @ Twc_pi3
        Twc_origin[:, :3, 3] = Twc_origin[:, :3, 3] / s_safe
        Twc_aligned = se3_inverse(Tcw_dpvo_0) @ Twc_origin
        Tcw_aligned = se3_inverse(Twc_aligned)

        return Tcw_aligned, pi3_points, pi3_depths_resized, pi3_confidences_resized, dynamic_prob_resized

    def estimate_scale(self, pi3_points, pi3_confidences, id_list):
        t_start = time.time()
        device = 'cuda'
        inv_u_list = []
        inv_v_list = []
        w_list = []

        for n, fid in enumerate(id_list[:-1]):
            if fid < 0 or fid >= self.n:
                continue

            patches_f = self.pg.patches_[int(fid)]  # [M, 3, 3, 3]
            inv_depths_f = patches_f[:, 2, 1, 1]
            valid_patch = torch.isfinite(inv_depths_f) & (inv_depths_f > 1e-6)
            if valid_patch.sum() == 0:
                continue

            depth_map = pi3_points[n, ..., 2]
            sampled_depths = self._sample_at_patch_centers(patches_f, depth_map)
            inv_u = (1.0 / sampled_depths.clamp(min=1e-6))
            inv_v = inv_depths_f.clamp(min=1e-6, max=1e6)

            conf_map = pi3_confidences[n]
            w = self._sample_at_patch_centers(patches_f, conf_map).clamp(min=1e-6)
            
            # Filter out patches with high variance
            if fid < self.n and hasattr(self.pg, 'var_'):
                var_f = self.pg.var_[int(fid)].squeeze(-1)  # [M]
                low_var_mask = var_f <= self.cfg.VAR_THRESHOLD
                valid_patch = valid_patch & low_var_mask

            valid = valid_patch & torch.isfinite(inv_u) & torch.isfinite(inv_v) & torch.isfinite(w)
            if valid.sum() == 0:
                continue

            inv_u_list.append(inv_u[valid].float())
            inv_v_list.append(inv_v[valid].float())
            w_list.append(w[valid].float())

        if len(inv_u_list) == 0:
            s_fallback = 1.0
            dt_ms = (time.time() - t_start) * 1000.0
            return s_fallback

        inv_u = torch.cat(inv_u_list, dim=0).to(device=device, dtype=torch.float32)
        inv_v = torch.cat(inv_v_list, dim=0).to(device=device, dtype=torch.float32)
        w0 = torch.cat(w_list, dim=0).to(device=device, dtype=torch.float32)

        valid = (inv_u > 1e-6) & (inv_v > 1e-6) & (w0 > 0) & torch.isfinite(inv_u) & torch.isfinite(inv_v) & torch.isfinite(w0)
        if not valid.any():
            s_fallback = 1.0
            dt_ms = (time.time() - t_start) * 1000.0
            return s_fallback
        inv_u = inv_u[valid]
        inv_v = inv_v[valid]
        w0 = w0[valid]

        eps = 1e-8
        # init s by weighted median of ratios
        r = (inv_v / (inv_u + eps)).clamp(min=1e-6, max=1e6)
        sort_idx = torch.argsort(r)
        r_sorted = r[sort_idx]
        w_sorted = w0[sort_idx]
        cumsum = torch.cumsum(w_sorted, dim=0)
        mid = 0.5 * cumsum[-1]
        k = torch.searchsorted(cumsum, mid)
        s = r_sorted[min(int(k.item()), r_sorted.numel()-1)]

        # set Huber delta from initial residuals (robust scale)
        res0 = (inv_v - s * inv_u).abs()
        try:
            delta = torch.quantile(res0, 0.7).clamp(min=1e-6)
        except Exception:
            delta = res0.median().clamp(min=1e-6)

        # IRLS iterations
        max_iters = 5
        for _ in range(max_iters):
            res = inv_v - s * inv_u
            abs_res = res.abs() + eps
            huber_w = torch.where(abs_res <= delta, torch.ones_like(abs_res), (delta / abs_res))
            w = (w0 * huber_w).clamp(min=1e-8)

            num = torch.sum(w * inv_u * inv_v)
            den = torch.sum(w * inv_u * inv_u) + eps
            s_new = num / den

            if not torch.isfinite(s_new):
                break
            if torch.abs(s_new - s) <= 1e-4 * (torch.abs(s) + 1e-6):
                s = s_new
                break
            s = s_new

        s = float(torch.clamp(s, 0.5, 2.0).item())
        return s

    def _sample_at_patch_centers(self, patches, value_map):
        """Sample a HxW map at patch centers, returning [M] values.

        Args:
            patches (Tensor): [M, 3, 3, 3]
            value_map (Tensor): [H, W]
        Returns:
            Tensor: [M] sampled values
        """
        patches = patches.squeeze(0)
        x = patches[:, 0, 1, 1]
        y = patches[:, 1, 1, 1]
        x_norm = 2.0 * (x + 0.5) / self.fmap_wd - 1.0
        y_norm = 2.0 * (y + 0.5) / self.fmap_ht - 1.0
        coords = torch.stack([x_norm, y_norm], dim=-1)
        sampled = F.grid_sample(
            value_map.unsqueeze(0).unsqueeze(0),
            coords.unsqueeze(0).unsqueeze(1),
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        ).squeeze()
        return sampled

    def _resize_to_fmap(self, tensor_batch: torch.Tensor) -> torch.Tensor:
        """Resize PI3 output maps to match DPVO feature-map resolution."""
        if tensor_batch.dim() == 3:
            tensor_batch = tensor_batch.unsqueeze(1)
        elif tensor_batch.dim() == 4 and tensor_batch.shape[1] != 1:
            raise ValueError(f"Unexpected tensor shape for resize: {tensor_batch.shape}")

        resized = F.interpolate(
            tensor_batch,
            size=(self.fmap_ht, self.fmap_wd),
            mode='bilinear',
            align_corners=False
        )
        return resized.squeeze(1)

    def __edges_forw(self):
        r=self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - r), 0)
        t1 = self.M * max((self.n - 1), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(self.n-1, self.n, device="cuda"), indexing='ij')

    def __edges_back(self):
        r=self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - 1), 0)
        t1 = self.M * max((self.n - 0), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(max(self.n-r, 0), self.n, device="cuda"), indexing='ij')

    def initialization(self):
        pi3_poses, pi3_points, pi3_depths, pi3_confidences, pi3_dynamics = self.pi3_inference(range(self.frame_for_init))

        for i, dpvo_image in enumerate(self.dpvo_init_window):

            with autocast(enabled=self.cfg.MIXED_PRECISION):
                fmap, gmap, imap, patches, _, clr = \
                    self.network.patchify(dpvo_image, pi3_dynamics[i], pi3_confidences[i],
                        patches_per_image=self.cfg.PATCHES_PER_FRAME, 
                        centroid_sel_strat=self.cfg.CENTROID_SEL_STRAT, 
                        return_color=True,
                        static_mask_thresh=self.cfg.STATIC_MASK_THRESH)
            
            pi3_pose = pi3_poses[i]  # [4, 4]
            self.pg.poses_[i] = self.convert_matrix_to_quat(pi3_pose)

            sampled_depths = self._sample_at_patch_centers(patches, pi3_depths[i])
            inv_depths = (1.0 / sampled_depths.clamp(min=1e-6))
            self.pg.patches_[i] = patches.squeeze(0)
            self.pg.patches_[i][:, 2] = inv_depths.view(-1, 1, 1)
            self.pg.prior_inv_depth_[i] = inv_depths.view(-1, 1)

            conf = self._sample_at_patch_centers(patches, pi3_confidences[i])
            self.pg.confidences_[i] = conf.view(-1, 1)

            ### update network attributes ###
            self.imap_[i % self.pmem] = imap.squeeze()
            self.gmap_[i % self.pmem] = gmap.squeeze()
            self.fmap1_[:, i % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
            self.fmap2_[:, i % self.mem] = F.avg_pool2d(fmap[0], 4, 4)

            k = self.intrinsic_init_window[i]
            if k is not None:
                self.pg.intrinsics_[i] = k / self.RES
            else:
                self.pg.intrinsics_[i] = self.estimate_K(pi3_points[i])

            # color info for visualization
            clr = (clr[0,:,[2,1,0]] + 0.5) * (255.0 / 2)
            self.pg.colors_[i] = clr.to(torch.uint8)
            
        for itr in range(12):
            self.update(cov=self.cfg.USE_COV_ADAPTIVE)

    def estimate_K(self, pre_points):
        H, W = int(pre_points.shape[0]), int(pre_points.shape[1])
        pp = torch.tensor([[W/2.0, H/2.0]], device=pre_points.device, dtype=pre_points.dtype)
        f = estimate_focal_knowing_depth(pre_points.unsqueeze(0), pp)
        f = float(f.item())
        # scale to feature-map coordinates
        sx = float(self.fmap_wd) / float(W)
        sy = float(self.fmap_ht) / float(H)
        fx = f * sx
        fy = f * sy
        cx = (W * 0.5) * sx
        cy = (H * 0.5) * sy
        intrinsics_feat = torch.tensor([fx, fy, cx, cy], device='cuda', dtype=torch.float32)
        return intrinsics_feat

    def __call__(self, tstamp, image, intrinsics=None):
        """ track new frame """

        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc(image, self.n)

        if (self.n+1) >= self.N:
            raise Exception(f'The buffer size is too small. You can increase it using "--opts BUFFER_SIZE={self.N*2}"')

        predict_points_vis = None
        dynamic_mask_vis = None
        confidence_vis = None

        # Preprocess image for PI3 model
        pi3_image = preprocess_tensor_for_pi3(image).cuda() #[3, 364, 686]

        if self.pi3_images_ == None:
            self.pi3_images_ = pi3_image.new_zeros((self.pi3_mem, *pi3_image.shape))
        
        self.pi3_images_[self.n % self.pi3_mem] = pi3_image
        dpvo_image = 2 * (image[None,None] / 255.0) - 0.5
      
        if not self.is_initialized:
            self.dpvo_init_window.append(dpvo_image)
            self.intrinsic_init_window.append(intrinsics)
        else:
            ids = [self.n - 5*k for k in range(self.cfg.PI3_FRAME_RANGE) if (self.n - 5*k) >= 0]
            pi3_frame_list = sorted(ids)
            pi3_poses, pi3_points, pi3_depths, pi3_confidences, pi3_dynamics = self.pi3_inference(pi3_frame_list)

            # last frame aligned local points for visualization
            predict_points_vis = pi3_points[-1]
            dynamic_mask_vis = pi3_dynamics[-1]
            confidence_vis = pi3_confidences[-1]

            with autocast(enabled=self.cfg.MIXED_PRECISION):
                fmap, gmap, imap, patches, _, clr = \
                    self.network.patchify(dpvo_image, pi3_dynamics[-1], pi3_confidences[-1],
                        patches_per_image=self.cfg.PATCHES_PER_FRAME, 
                        centroid_sel_strat=self.cfg.CENTROID_SEL_STRAT,
                        return_color=True,
                        static_mask_thresh=self.cfg.STATIC_MASK_THRESH)

            sampled_depths_cur = self._sample_at_patch_centers(patches, pi3_depths[-1])
            inv_depths_cur = (1.0 / sampled_depths_cur.clamp(min=1e-6))
            self.pg.patches_[self.n] = patches.squeeze(0)
            self.pg.patches_[self.n][:, 2] = inv_depths_cur.view(-1, 1, 1)
            self.pg.prior_inv_depth_[self.n] = inv_depths_cur.view(-1, 1)
            conf = self._sample_at_patch_centers(patches, pi3_confidences[-1])
            self.pg.confidences_[self.n] = conf.view(-1, 1)

            est_K = self.estimate_K(predict_points_vis)
            if intrinsics is not None:
                self.pg.intrinsics_[self.n] = intrinsics / self.RES
                # if self.is_initialized:
                #     est_fx = est_K[0]
                #     gt_fx = self.pg.intrinsics_[self.n][0]
                #     if abs((est_fx - gt_fx)/gt_fx) > 0.075:
                #         self.pg.confidences_[self.n] = 0.0
            else:
                self.pg.intrinsics_[self.n] = est_K

            # print(f'estimate_k :{self.estimate_K(predict_points_vis)}')
            # print(f'real_k :{intrinsics / self.RES}')

            for idx_in_list, fid in enumerate(pi3_frame_list[:-1]):
                depth_map_f = pi3_depths[idx_in_list]
                sampled_depths_f = self._sample_at_patch_centers(self.pg.patches_[fid].unsqueeze(0), depth_map_f)
                inv_depths_f = (1.0 / sampled_depths_f.clamp(min=1e-6))
                self.pg.prior_inv_depth_[fid] = inv_depths_f.view(-1, 1)

            if self.n > 1:
                if self.cfg.MOTION_MODEL == 'DAMPED_LINEAR':
                    P1 = SE3(self.pg.poses_[self.n-1])
                    P2 = SE3(self.pg.poses_[self.n-2])

                    # To deal with varying camera hz
                    *_, a,b,c = [1]*3 + self.tlist
                    if abs(b-a) < 1e-6:
                        fac = 1.0
                    else:
                        fac = (c-b) / (b-a)

                    xi = self.cfg.MOTION_DAMPING * fac * (P1 * P2.inv()).log()
                    tvec_qvec = (SE3.exp(xi) * P1).data
                    self.pg.poses_[self.n] = tvec_qvec
                else:
                    tvec_qvec = self.poses[self.n-1]
                    self.pg.poses_[self.n] = tvec_qvec
            
            ### update network attributes ###
            self.imap_[self.n % self.pmem] = imap.squeeze()
            self.gmap_[self.n % self.pmem] = gmap.squeeze()
            self.fmap1_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
            self.fmap2_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 4, 4)
            
            # color info for visualization
            clr = (clr[0,:,[2,1,0]] + 0.5) * (255.0 / 2)
            self.pg.colors_[self.n] = clr.to(torch.uint8)

        ### update state attributes ###
        self.tlist.append(tstamp)
        self.pg.tstamps_[self.n] = self.counter

        self.pg.index_[self.n + 1] = self.n + 1
        self.pg.index_map_[self.n + 1] = self.m + self.M

        self.counter += 1
        self.n += 1
        self.m += self.M

        if self.cfg.LOOP_CLOSURE:
            if self.n - self.last_global_ba >= self.cfg.GLOBAL_OPT_FREQ:
                """ Add loop closure factors """
                lii, ljj = self.pg.edges_loop()
                if lii.numel() > 0:
                    self.last_global_ba = self.n
                    self.append_factors(lii, ljj)

        self.append_factors(*self.__edges_forw())
        self.append_factors(*self.__edges_back())

        if self.is_initialized:
            self.update(cov=self.cfg.USE_COV_ADAPTIVE)
            self.keyframe()
        elif len(self.dpvo_init_window) == self.frame_for_init:
            self.initialization()
            self.is_initialized = True

        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc.attempt_loop_closure(self.n)
            self.long_term_lc.lc_callback()

        return predict_points_vis, dynamic_mask_vis, confidence_vis