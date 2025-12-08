import os
import pathlib
import threading
import time
import math
from datetime import datetime

import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import torch
import rich
import matplotlib.pyplot as plt

from gui.gui_utils import (
    DatePacket,
    Packet_vis2main,
    create_frustum,
    create_color_lines,
    cv_gl,
    get_latest_queue,
)

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

# Simple logging for GUI
def Log(*args, tag="GUI"):
    rich.print(f"[bold magenta]{tag}:[/bold magenta]", *args)

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

class SLAM_GUI:
    def __init__(self, params_gui=None):
        self.step = 0
        self.saved_view_id = 0
        self.last_traj_name = None
        self.process_finished = False
        self.device = "cuda"

        self.frustum_dict = {}

        self.init_widget()

        self.q_main2vis = None
        self.background = None

        self.point_cloud = None
        self.point_color = None
        self.point_cloud_name = "pc"
        self.pred_point_cloud = None
        self.pred_point_color = None
        self.pred_point_cloud_name = "pred_pc"

        self.init = False
        self.kf_window = None
        self.reference_cam_id = None
        self.render_img = None
        self.did_initial_follow = False

        if params_gui is not None:
            self.background = params_gui.background
            self.init = True
            self.q_main2vis = params_gui.q_main2vis
            self.q_vis2main = params_gui.q_vis2main

        self.save_path = "."
        self.save_path = pathlib.Path(self.save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        threading.Thread(target=self._update_thread).start()

    def init_widget(self):
        self.window_w, self.window_h = 1600, 900

        self.window = gui.Application.instance.create_window(
            "PI-SAM", self.window_w, self.window_h
        )
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)
        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)

        cg_settings = rendering.ColorGrading(
            rendering.ColorGrading.Quality.ULTRA,
            rendering.ColorGrading.ToneMapping.LINEAR,
        )
        self.widget3d.scene.view.set_color_grading(cg_settings)

        self.window.add_child(self.widget3d)

        self.lit = rendering.MaterialRecord()
        self.lit.shader = "unlitLine"
        self.lit.line_width  = 3.0

        self.lit_geo = rendering.MaterialRecord()
        self.lit_geo.shader = "defaultUnlit"
        self.lit_geo.point_size = 3.0

        self.specular_geo = rendering.MaterialRecord()
        self.specular_geo.shader = "defaultLit"

        self.point_cloud_render = rendering.MaterialRecord()
        self.point_cloud_render.shader = "defaultUnlit"
        self.point_cloud_render.point_size = 3.0
        
        # Separate render material for points (larger size)
        self.points_render = rendering.MaterialRecord()
        self.points_render.shader = "defaultUnlit"
        self.points_render.point_size = 4.0  # Larger point size for points

        self.axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0]
        )

        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(60.0, bounds, bounds.get_center())
        em = self.window.theme.font_size
        margin = 0.5 * em
        self.panel = gui.Vert(0.5 * em, gui.Margins(margin))
        self.button = gui.ToggleSwitch("Resume/Pause")
        self.button.is_on = True
        self.button.set_on_clicked(self._on_button)
        self.panel.add_child(self.button)

        self.panel.add_child(gui.Label("Viewpoint Options"))

        viewpoint_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        vp_subtile1 = gui.Vert(0.5 * em, gui.Margins(margin))
        vp_subtile2 = gui.Vert(0.5 * em, gui.Margins(margin))

        ##Check boxes
        vp_subtile1.add_child(gui.Label("Camera follow options"))
        chbox_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        self.followcam_chbox = gui.Checkbox("Follow Camera")
        self.followcam_chbox.checked = False
        chbox_tile.add_child(self.followcam_chbox)

        self.staybehind_chbox = gui.Checkbox("From Behind")
        self.staybehind_chbox.checked = True
        chbox_tile.add_child(self.staybehind_chbox)
        vp_subtile1.add_child(chbox_tile)

        ##Combo panels
        combo_tile = gui.Vert(0.5 * em, gui.Margins(margin))

        ## Jump to the camera viewpoint
        self.combo_kf = gui.Combobox()
        self.combo_kf.set_on_selection_changed(self._on_combo_kf)
        combo_tile.add_child(gui.Label("Viewpoint list"))
        combo_tile.add_child(self.combo_kf)
        vp_subtile2.add_child(combo_tile)

        viewpoint_tile.add_child(vp_subtile1)
        viewpoint_tile.add_child(vp_subtile2)
        self.panel.add_child(viewpoint_tile)

        self.panel.add_child(gui.Label("3D Objects"))
        chbox_tile_3dobj = gui.Horiz(0.5 * em, gui.Margins(margin))
        self.cameras_chbox = gui.Checkbox("Cameras")
        self.cameras_chbox.checked = True
        self.cameras_chbox.set_on_checked(self._on_cameras_chbox)
        chbox_tile_3dobj.add_child(self.cameras_chbox)

        self.traj_chbox = gui.Checkbox("Trajectory")
        self.traj_chbox.checked = False
        self.traj_chbox.set_on_checked(self._on_traj_chbox)
        chbox_tile_3dobj.add_child(self.traj_chbox)

        # Points toggle
        self.points_chbox = gui.Checkbox("Patches")
        self.points_chbox.checked = True
        self.points_chbox.set_on_checked(self._on_points_chbox)
        chbox_tile_3dobj.add_child(self.points_chbox)
        
        # Predicted points toggle
        self.pred_points_chbox = gui.Checkbox("Points")
        self.pred_points_chbox.checked = True
        self.pred_points_chbox.set_on_checked(self._on_pred_points_chbox)
        chbox_tile_3dobj.add_child(self.pred_points_chbox)

        self.panel.add_child(chbox_tile_3dobj)

        # screenshot buttom
        self.screenshot_btn = gui.Button("Screenshot")
        self.screenshot_btn.set_on_clicked(
            self._on_screenshot_btn
        )  # set the callback function
        self.panel.add_child(self.screenshot_btn)

        self.saveview_btn = gui.Button("Save View")
        self.saveview_btn.set_on_clicked(
            self._on_saveview_btn
        )  # set the callback function
        self.panel.add_child(self.saveview_btn)

        ## Rendering Tab
        tab_margins = gui.Margins(0, int(np.round(0.5 * em)), 0, 0)
        tabs = gui.TabControl()

        tab_info = gui.Vert(0, tab_margins)

        self.in_rgb_widget = gui.ImageWidget()
        tab_info.add_child(gui.Label("Input Color"))
        tab_info.add_child(self.in_rgb_widget)

        self.dynamic_mask_widget = gui.ImageWidget()
        tab_info.add_child(gui.Label("Dynamic Mask"))
        tab_info.add_child(self.dynamic_mask_widget)

        tabs.add_tab("Info", tab_info)
        self.panel.add_child(tabs)
        self.window.add_child(self.panel)

    def add_trajectory(self, points, name):
        lines = create_color_lines(points)

        if self.last_traj_name is not None:
            self.widget3d.scene.remove_geometry(self.last_traj_name)

        self.widget3d.scene.add_geometry(name, lines, self.lit_geo)

        self.widget3d.scene.show_geometry(name, self.traj_chbox.checked)
        self.last_traj_name = name


    def add_camera(self, pose, name, color=[0, 1, 0], gt=False, size=0.02, update_color=False):
        # Convert to numpy array if needed
        if not isinstance(pose, np.ndarray):
            C2W = pose.cpu().numpy()
        else:
            C2W = pose
        if name not in self.frustum_dict.keys():
            frustum = create_frustum(C2W, color, size=size)
            self.combo_kf.add_item(name)
            self.frustum_dict[name] = frustum
            self.widget3d.scene.add_geometry(name, frustum.line_set, self.lit)
        else:
            # If camera exists and we need to update color, remove and recreate
            if update_color:
                self.widget3d.scene.remove_geometry(name)
                frustum = create_frustum(C2W, color, size=size)
                self.frustum_dict[name] = frustum
                self.widget3d.scene.add_geometry(name, frustum.line_set, self.lit)
            else:
                frustum = self.frustum_dict[name]
                frustum.update_pose(C2W)
        self.widget3d.scene.set_geometry_transform(name, C2W.astype(np.float64))
        self.widget3d.scene.show_geometry(name, self.cameras_chbox.checked)
        return frustum


    def _on_layout(self, layout_context):
        contentRect = self.window.content_rect
        self.widget3d_width_ratio = 0.75
        self.widget3d_width = int(
            self.window.size.width * self.widget3d_width_ratio
        )  # 15 ems wide
        self.widget3d.frame = gui.Rect(
            contentRect.x, contentRect.y, self.widget3d_width, contentRect.height
        )
        self.panel.frame = gui.Rect(
            self.widget3d.frame.get_right(),
            contentRect.y,
            contentRect.width - self.widget3d_width,
            contentRect.height,
        )

    def _on_close(self):
        self.is_done = True
        return True  # False would cancel the close


    def _on_combo_kf(self, new_val, new_idx):
        frustum = self.frustum_dict[new_val]
        viewpoint = frustum.view_dir

        self.widget3d.look_at(viewpoint[0], viewpoint[1], viewpoint[2])

    def _on_cameras_chbox(self, is_checked, name=None):
        names = self.frustum_dict.keys() if name is None else [name]
        for name in names:
            self.widget3d.scene.show_geometry(name, is_checked)

    def _on_traj_chbox(self, is_checked):
        if self.last_traj_name is not None:
            self.widget3d.scene.show_geometry(self.last_traj_name, is_checked)

    def _on_points_chbox(self, is_checked):
        self.widget3d.scene.show_geometry(self.point_cloud_name, is_checked)

    def _on_pred_points_chbox(self, is_checked):
        self.widget3d.scene.show_geometry(self.pred_point_cloud_name, is_checked)

    def _on_button(self, is_on):
        packet = Packet_vis2main()
        packet.flag_pause = not self.button.is_on
        self.q_vis2main.put(packet)


    def _on_saveview_btn(self):
        if self.render_img is None:
            return
        w2c = cv_gl @ self.widget3d.scene.camera.get_view_matrix()
        
        C2W = np.linalg.inv(w2c)
        color=[0, 1, 0]
        size=0.005
        name='view' + str(self.saved_view_id)
        print(f'saving {name}')
        
        file_path = name + '.txt'
        with open(file_path, 'w') as f:
            pose = w2c.reshape(-1)
            pose_str = ' '.join(map(str, pose))
            f.write(f"{0} {pose_str}\n")

        frustum = create_frustum(C2W, color, size=size)
        if name not in self.frustum_dict.keys():
            frustum = create_frustum(C2W, color)
            self.combo_kf.add_item(name)
            self.frustum_dict[name] = frustum
            # self.widget3d.scene.add_geometry(name, frustum.line_set, self.lit)
        frustum = self.frustum_dict[name]
        frustum.update_pose(C2W)
        # self.widget3d.scene.set_geometry_transform(name, C2W.astype(np.float64))
        # self.widget3d.scene.show_geometry(name, self.cameras_chbox.checked)

        self.saved_view_id += 1

    def _on_screenshot_btn(self):
        dt = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        save_dir = self.save_path / "screenshots" / dt
        save_dir.mkdir(parents=True, exist_ok=True)
        # create the filename
        filename = save_dir / "screenshot"
        height = self.window.size.height
        width = self.widget3d_width
        app = o3d.visualization.gui.Application.instance
        
        # Always save GUI window screenshot
        img = np.asarray(app.render_to_image(self.widget3d.scene, width, height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gui_path = f"{filename}-gui.png"
        cv2.imwrite(gui_path, img)
        print(f"Screenshot saved to:")
        print(f"  GUI: {os.path.abspath(gui_path)}")
        
        # Save render_img if available
        if self.render_img is not None:
            img = np.asarray(self.render_img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            render_path = f"{filename}.png"
            cv2.imwrite(render_path, img)
            print(f"  Render: {os.path.abspath(render_path)}")
        else:
            print("  (Render image not available)")

    @staticmethod
    def resize_img(img, width):
        height = int(width * img.shape[0] / img.shape[1])
        return cv2.resize(img, (width, height))

    def receive_data(self, q):
        if q is None:
            return

        data_packet = get_latest_queue(q)
        if data_packet is None:
            return
        
        if data_packet.points is not None:
            self.point_cloud = data_packet.points
            if data_packet.point_colors is not None:
                self.point_color = data_packet.point_colors
            self.init = True

        if data_packet.pred_points is not None:
            self.pred_point_cloud = data_packet.pred_points
            if data_packet.pred_point_colors is not None:
                self.pred_point_color = data_packet.pred_point_colors


        # Show both current camera and keyframes cameras
        if data_packet.current_pose is not None:
            frustum = self.add_camera(
                data_packet.current_pose, name="current", color=[0, 0, 0.8]
            )
            # One-time initial follow to set a good starting view, then keep follow off
            if self.followcam_chbox.checked or not self.did_initial_follow:
                viewpoint = (
                    frustum.view_dir_behind
                    if self.staybehind_chbox.checked
                    else frustum.view_dir
                )
                self.widget3d.look_at(viewpoint[0], viewpoint[1], viewpoint[2])
                if not self.did_initial_follow:
                    self.did_initial_follow = True

        if data_packet.keyframes is not None:
            points_list = []
            last_id = 0
            
            for idx, keyframe in enumerate(data_packet.keyframes): 
                _ , ct = keyframe.get_inv_RT
                np_t = ct.cpu().numpy()
                points_list.append(np_t)
                last_id = keyframe.uid
            
            if len(points_list) > 2:
                traj_name = 'traj' + str(last_id)
                self.add_trajectory(points_list, traj_name)

        if data_packet.gtframes is not None:
            for keyframe in data_packet.gtframes:
                name = "keyframe_gt_{}".format(keyframe.uid)
                frustum = self.add_camera(keyframe, name=name, color=[1, 0, 0])

        if data_packet.gtcolor is not None:
            rgb = torch.clamp(data_packet.gtcolor, min=0, max=1.0) * 255
            rgb = rgb.byte().permute(1, 2, 0).contiguous().cpu().numpy()
            rgb = o3d.geometry.Image(rgb)
            self.in_rgb_widget.update_image(rgb)

        if data_packet.dynamic_mask is not None:
            mask = torch.clamp(data_packet.dynamic_mask, min=0, max=1.0)
            mask = (mask * 255).byte().squeeze(0).contiguous().cpu().numpy()
            heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            self.dynamic_mask_widget.update_image(o3d.geometry.Image(heatmap))


        if data_packet.finish:
            Log("Received terminate signal", tag="GUI")
            # clean up the pipe
            while not self.q_main2vis.empty():
                self.q_main2vis.get()
            while not self.q_vis2main.empty():
                self.q_vis2main.get()
            self.q_vis2main = None
            self.q_main2vis = None
            self.process_finished = True


    
    def visualize_pointcloud(self):
        if self.point_cloud is None:
            return
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.point_cloud)
        if self.point_color is not None:
            pcd.colors = o3d.utility.Vector3dVector(self.point_color)
        
        self.widget3d.scene.remove_geometry(self.point_cloud_name)
        self.widget3d.scene.add_geometry(self.point_cloud_name, pcd, self.points_render)
        self.widget3d.scene.show_geometry(self.point_cloud_name, self.points_chbox.checked)

        # Predicted points visualization
        if self.pred_point_cloud is not None and self.pred_points_chbox.checked:
            pcd_pred = o3d.geometry.PointCloud()
            pcd_pred.points = o3d.utility.Vector3dVector(self.pred_point_cloud)
            if self.pred_point_color is not None:
                pcd_pred.colors = o3d.utility.Vector3dVector(self.pred_point_color)
            self.widget3d.scene.remove_geometry(self.pred_point_cloud_name)
            self.widget3d.scene.add_geometry(self.pred_point_cloud_name, pcd_pred, self.point_cloud_render)
            self.widget3d.scene.show_geometry(self.pred_point_cloud_name, True)
        else:
            self.widget3d.scene.show_geometry(self.pred_point_cloud_name, False)


    def scene_update(self):
        self.receive_data(self.q_main2vis)
        self.visualize_pointcloud()
        

    def _update_thread(self):
        while True:
            time.sleep(0.01)
            self.step += 1
            if self.process_finished:
                o3d.visualization.gui.Application.instance.quit()
                Log("Closing Visualization", tag="GUI")
                break

            def update():
                if self.step % 5 == 0:
                    self.scene_update()

                if self.step >= 1e9:
                    self.step = 0

            gui.Application.instance.post_to_main_thread(self.window, update)


def run(params_gui=None):
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    win = SLAM_GUI(params_gui)
    app.run()


def main():
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    win = SLAM_GUI()
    app.run()


if __name__ == "__main__":
    main()
