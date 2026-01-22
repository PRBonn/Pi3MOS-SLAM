import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import click
from pi3.models.pi3 import Pi3
import os
import glob
import math
from PIL import Image
import numpy as np
import matplotlib.cm as cm
import csv

davis_2016_full = [
    "bear",
    "blackswan",
    "bmx-bumps",
    "bmx-trees",
    "boat",
    "breakdance",
    "breakdance-flare",
    "bus",
    "camel",
    "car-roundabout",
    "car-shadow",
    "car-turn",
    "cows",
    "dance-jump",
    "dance-twirl",
    "dog",
    "dog-agility",
    "drift-chicane",
    "drift-straight",
    "drift-turn",
    "elephant",
    "flamingo",
    "goat",
    "hike",
    "hockey",
    "horsejump-high",
    "horsejump-low",
    "kite-surf",
    "kite-walk",
    "libby",
    "lucia",
    "mallard-fly",
    "mallard-water",
    "motocross-bumps",
    "motocross-jump",
    "motorbike",
    "paragliding",
    "paragliding-launch",
    "parkour",
    "rhino",
    "rollerblade",
    "scooter-black",
    "scooter-gray",
    "soapbox",
    "soccerball",
    "stroller",
    "surf",
    "swing",
    "tennis",
    "train",
]

davis_2016 = [
    "blackswan",
    "bmx-trees",
    "breakdance",
    "camel",
    "car-roundabout",
    "car-shadow",
    "cows",
    "dance-twirl",
    "dog",
    "drift-chicane",
    "drift-straight",
    "goat",
    "horsejump-high",
    "kite-surf",
    "libby",
    "motocross-jump",
    "paragliding-launch",
    "parkour",
    "scooter-black",
    "soapbox",
]

davis_2017 = [
    "bike-packing",
    "blackswan",
    "bmx-trees",
    "breakdance",
    "camel",
    "car-roundabout",
    "car-shadow",
    "cows",
    "dance-twirl",
    "dog",
    "dogs-jump",
    "drift-chicane",
    "drift-straight",
    "goat",
    "gold-fish",
    "horsejump-high",
    "india",
    "judo",
    "kite-surf",
    "lab-coat",
    "libby",
    "loading",
    "mbike-trick",
    "motocross-jump",
    "paragliding-launch",
    "parkour",
    "pigs",
    "scooter-black",
    "shooting",
    "soapbox",
]


class DavisData:
    def __init__(self, data_dir, scene, resolution=480, device=None):
        self.scene = scene
        assert os.path.exists(data_dir), f"Data dir {data_dir} not found"
        gt_folder = os.path.join(data_dir, "Annotations", f"{resolution}p", scene)
        rgb_folder = os.path.join(data_dir, "JPEGImages", f"{resolution}p", scene)

        assert os.path.exists(gt_folder), f"GT folder not found: {gt_folder}"
        assert os.path.exists(rgb_folder), f"RGB folder not found: {rgb_folder}"

        self.gt_files = sorted(glob.glob(os.path.join(gt_folder, "*.png")))
        self.rgb_files = sorted(glob.glob(os.path.join(rgb_folder, "*.jpg")))
        assert len(self.gt_files) == len(self.rgb_files), "RGB/GT length mismatch"
        self.seq_len = len(self.gt_files)

        with Image.open(self.rgb_files[0]) as im0:
            W0, H0 = im0.size
        self.target_w, self.target_h = self._compute_target_size(W0, H0)

        rgb_list, gt_list = [], []
        for rgb_path, gt_path in zip(self.rgb_files, self.gt_files):
            # Read RGB -> tensor [3,H,W] in [0,1]
            rgb = Image.open(rgb_path).convert("RGB")
            rgb_t = TF.to_tensor(rgb)  # float32 [0,1], [3,H,W]

            # Read GT -> binary [H,W] (bool)
            gt = Image.open(gt_path).convert("L")
            gt_t = TF.to_tensor(gt)[0]  # [H,W] float in [0,1]
            gt_t = gt_t > 0.0  # -> bool

            # Resize pair to the fixed target size
            rgb_r, gt_r = self.preprocess_input_for_pi3(
                (self.target_w, self.target_h),
                rgb_t,
                gt_t,
            )

            rgb_list.append(rgb_r)
            gt_list.append(gt_r)
        # Stack: RGB [N,3,Ht,Wt], GT [N,Ht,Wt] (bool)
        self.rgb = torch.stack(rgb_list, dim=0)
        self.gt = torch.stack(gt_list, dim=0)

        if device is not None:
            self.rgb = self.rgb.to(device)
            self.gt = self.gt.to(device)

    def get_data(self, id):
        return self.rgb[id], self.gt[id]

    def _compute_target_size(self, W, H, PIXEL_LIMIT=255_000):
        # scale so that W*H <= PIXEL_LIMIT, then snap to multiples of 14
        scale = math.sqrt(PIXEL_LIMIT / (W * H)) if W * H > 0 else 1.0
        Wt, Ht = W * scale, H * scale
        k = max(1, round(Wt / 14))
        m = max(1, round(Ht / 14))
        # make sure snapped size still fits the pixel budget
        while (k * 14) * (m * 14) > PIXEL_LIMIT:
            if (k / m) > (Wt / Ht):
                k -= 1
            else:
                m -= 1
            k, m = max(k, 1), max(m, 1)
        return k * 14, m * 14  # (W, H)

    def preprocess_input_for_pi3(
        self,
        target_size,  # (W,H) override to keep all frames same size
        rgb=None,  # [3,H,W] or [H,W,3]
        gt=None,  # [H,W] binary (bool/0-1/0-255)
    ):
        TARGET_W, TARGET_H = target_size
        rgb_resized = None
        gt_resized = None
        # --- normalize shapes to CHW ---
        if rgb is not None:
            if rgb.ndim != 3:
                raise ValueError(f"rgb must be 3D, got {rgb.shape}")
            if rgb.shape[0] != 3 and rgb.shape[-1] == 3:
                rgb = rgb.permute(2, 0, 1)
            if rgb.shape[0] != 3:
                raise ValueError(f"rgb channel dim must be 3, got {rgb.shape}")

            rgb_f = rgb.float()
            if rgb_f.max() > 1.0:  # (in case caller passed 0..255)
                rgb_f = rgb_f / 255.0
            rgb_resized = TF.resize(
                rgb_f,
                [TARGET_H, TARGET_W],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
        if gt is not None:
            if gt.ndim != 2:
                raise ValueError(f"gt must be [H,W], got {gt.shape}")
            gt_f = gt.float().unsqueeze(0)  # [1,H,W]
            gt_resized_f = TF.resize(
                gt_f, [TARGET_H, TARGET_W], interpolation=InterpolationMode.NEAREST
            )[
                0
            ]  # -> [H,W]
            gt_resized = gt_resized_f >= 0.5  # bool

        return rgb_resized, gt_resized


class MOSEvaluator:
    def __init__(self, ckpt_path, threshold):
        self.load_weights_pi3(ckpt_path)
        self.threshold = threshold

    def load_weights_pi3(self, ckpt):
        self.pi3 = Pi3().to("cuda").eval()
        if ckpt.endswith(".safetensors"):
            from safetensors.torch import load_file

            weight = load_file(ckpt)
        else:
            weight = torch.load(ckpt, map_location="cuda", weights_only=False)
        print(f"Loading pi3...")
        self.pi3.load_state_dict(weight)

    def pi3_inference(self, imgs):
        images_tensor = imgs.unsqueeze(0)  # [1, N, 3, H, W]
        dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype):
                pi3_output = self.pi3(images_tensor)

        dynamic_prob = pi3_output["dynamic_masks"][0].squeeze(-1)
        dynamic_mask = dynamic_prob > self.threshold

        # Estimate single scale with confidence as weights (inverse-depth domain)
        pi3_points = pi3_output["local_points"][0]
        s = 1.0
        s_safe = max(1e-6, float(s))
        pi3_points = pi3_points / s_safe
        pi3_depths = pi3_points[..., 2]

        return dynamic_mask.squeeze(1), pi3_depths

    def cal_j_score(self, gt, pred):
        gt = gt.bool()
        pred = pred.bool()

        intersection = (gt & pred).float().sum(dim=(-2, -1))
        union = (gt | pred).float().sum(dim=(-2, -1))
        j = torch.where(union > 0, intersection / union, torch.ones_like(union))
        jm = torch.nanmean(j)
        jr = torch.nanmean((j > 0.5).float())
        return jm, jr  # mean over batch if batch dimension exists

    def cal_f_score(self, gt, pred, tolerance=0.008):
        tolerance = (
            tolerance
            if tolerance >= 1
            else int(np.ceil(tolerance * np.linalg.norm(gt[0].shape)))
        )

        if gt.ndim == 2:
            gt = gt.unsqueeze(0)
            pred = pred.unsqueeze(0)

        gt = gt.float()
        pred = pred.float()

        kernel = torch.ones((1, 1, 3, 3), device=gt.device)
        gt_edge = F.conv2d(gt.unsqueeze(1), kernel, padding=1).clamp(
            max=1
        ) - gt.unsqueeze(1)
        pred_edge = F.conv2d(pred.unsqueeze(1), kernel, padding=1).clamp(
            max=1
        ) - pred.unsqueeze(1)

        gt_edge = (gt_edge > 0).float()
        pred_edge = (pred_edge > 0).float()

        if tolerance > 0:
            dilation_kernel = torch.ones(
                (1, 1, 2 * tolerance + 1, 2 * tolerance + 1), device=gt.device
            )
            gt_dil = (F.conv2d(gt_edge, dilation_kernel, padding=tolerance) > 0).float()
            pred_dil = (
                F.conv2d(pred_edge, dilation_kernel, padding=tolerance) > 0
            ).float()
        else:
            gt_dil, pred_dil = gt_edge, pred_edge

        # True positives, false positives, false negatives
        tp = (pred_edge * gt_dil).sum(dim=(-2, -1))
        fp = (pred_edge * (1 - gt_dil)).sum(dim=(-2, -1))
        fn = (gt_edge * (1 - pred_dil)).sum(dim=(-2, -1))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f = 2 * precision * recall / (precision + recall + 1e-8)
        return precision.mean(), recall.mean(), f.mean()  # mean over batch

    def eval_offline(self, dataset, vis, output, max_len=20):
        rgb_list = []
        pred_list = []
        gt_list = []
        depth_list = []

        if dataset.seq_len > max_len:
            id_batches = [
                list(range(i, min(i + max_len, dataset.seq_len)))
                for i in range(0, dataset.seq_len, max_len)
            ]

        else:
            id_batches = [list(range(dataset.seq_len))]

        for i in id_batches:
            rgbs, gts = dataset.get_data(i)
            rgb_list.append(rgbs)
            gt_list.append(gts)

            preds, depths = self.pi3_inference(rgbs)
            pred_list.append(preds)
            depth_list.append(depths)

        rgb_list = torch.cat(rgb_list, dim=0)
        gt_list = torch.cat(gt_list, dim=0)
        pred_list = torch.cat(pred_list, dim=0)
        depth_list = torch.cat(depth_list, dim=0)

        jm_score, jr_score = self.cal_j_score(gt_list, pred_list)
        p_score, r_score, f_score = self.cal_f_score(gt_list, pred_list)

        if vis:
            save_path = f"{output}/{dataset.scene}/offline"
            self.save_output(rgb_list, gt_list, pred_list, depth_list, save_path)

        return [jm_score.item(), jr_score.item()], [
            p_score.item(),
            r_score.item(),
            f_score.item(),
        ]

    def eval_online(self, dataset, vis, output, window_len=8):
        rgb_list = []
        pred_list = []
        gt_list = []
        depth_list = []
        target_id = window_len
        while target_id <= dataset.seq_len:
            rgbs, gts = dataset.get_data(range(target_id - window_len, target_id))
            rgb_list.append(rgbs[-1])
            gt_list.append(gts[-1])

            preds, depths = self.pi3_inference(rgbs)
            pred_list.append(preds)
            depth_list.append(depths)
            target_id += 1

        rgb_list = torch.stack(rgb_list, dim=0)
        gt_list = torch.stack(gt_list, dim=0)
        pred_list = torch.stack(pred_list, dim=0)
        depth_list = torch.stack(depth_list, dim=0)

        jm_score, jr_score = self.cal_j_score(gt_list, pred_list)
        p_score, r_score, f_score = self.cal_f_score(gt_list, pred_list)

        if vis:
            save_path = f"{output}/{dataset.scene}/online"
            self.save_output(rgb_list, gt_list, pred_list, depth_list, save_path)

        return [jm_score.item(), jr_score.item()], [
            p_score.item(),
            r_score.item(),
            f_score.item(),
        ]

    def save_output(self, rgb, gt, pred, depth, save_path):
        os.makedirs(save_path, exist_ok=True)

        rgb = rgb.permute(0, 2, 3, 1).cpu().numpy()
        gt = gt.cpu().numpy()
        pred = pred.cpu().numpy()
        depth = depth.cpu().numpy()

        n = len(gt)
        for i in range(n):
            rgb_img = (rgb[i] * 255).astype(np.uint8)
            gt_img = gt[i].astype(np.uint8) * 255
            pred_img = pred[i].astype(np.uint8) * 255

            Image.fromarray(rgb_img).save(os.path.join(save_path, f"frame_{i:04d}.png"))
            Image.fromarray(gt_img).save(os.path.join(save_path, f"gt_{i:04d}.png"))
            Image.fromarray(pred_img).save(os.path.join(save_path, f"pred_{i:04d}.png"))

            depth_img = depth[i]
            depth_min, depth_max = np.min(depth_img), np.max(depth_img)
            if depth_max > depth_min:  # avoid divide by zero
                depth_norm = (depth_img - depth_min) / (depth_max - depth_min)
            else:
                depth_norm = np.zeros_like(depth_img)
            depth_colored = cm.viridis(depth_norm)[:, :, :3]  # RGBA -> RGB
            depth_colored = (depth_colored * 255).astype(np.uint8)

            Image.fromarray(depth_colored).save(
                os.path.join(save_path, f"depth_{i:04d}.png")
            )


scene_lists = {
    "davis_2016_full": davis_2016_full,
    "davis_2016": davis_2016,
    "davis_2017": davis_2017,
    "custom": [
        # "scene_name",
    ],
}


@click.command()
@click.option("--ckpt", required=True)
@click.option("--dataset_root", required=True)
@click.option("--scene_list", required=True)
@click.option("--threshold", default=0.25)
@click.option("--output", default="output")
@click.option("--vis", is_flag=True)
def main(ckpt, data_dir, scene_list, threshold, output, vis):
    # load evaluator
    mos_evaluator = MOSEvaluator(ckpt, threshold)

    # load dataset
    assert scene_list in scene_lists.keys(), "invalid scene list"

    selected_scenes = scene_lists[scene_list]

    os.makedirs(output, exist_ok=True)
    csv_path = os.path.join(output, f"{scene_list}({threshold}).csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scene",
                "JM_offline",
                "JR_offline",
                "F_offline",
                "JM_online",
                "JR_online",
                "F_online",
            ]
        )

        for scene in selected_scenes:
            davis_data = DavisData(data_dir, scene, device="cuda")

            ## run eval_offline
            j_off, f_off = mos_evaluator.eval_offline(davis_data, vis, output)
            print(
                f"{scene} (offline): jm_score = {j_off[0]}, jr_score = {j_off[1]}, f1 = {f_off[2]} \n"
            )

            ## run eval_online
            j_on, f_on = mos_evaluator.eval_online(davis_data, vis, output)
            print(
                f"{scene} (online): jm_score = {j_on[0]}, jr_score = {j_on[1]}, f1 = {f_on[2]} \n"
            )

            writer.writerow(
                [scene, j_off[0], j_off[1], f_off[2], j_on[0], j_on[1], f_on[2]]
            )
            f.flush()


if __name__ == "__main__":
    main()
