#!/usr/bin/env python3
"""
projects/02_hri_uncertainty_communication/sim/render_sim_video.py

Master 3D Scene & Video Renderer for Project 02:
"Communicating Robot Perceptual Uncertainty in Collaborative Manipulation:
 Comparing Implicit Kinematic Hesitation and Explicit Visual-Verbal Cues
 on Human Trust Calibration and Task Performance"

Features:
  1. High-Fidelity YCB 3D Models:
     - Tomato Soup Can (YCB #005): 32-facet cylinder, metallic silver rims,
       split red (top 50%) / white (bottom 50%) label, gold text "TOMATO SOUP".
     - Cracker Box (YCB #003): Crisp red cuboid, yellow top banner "CRACKERS",
       center cheese graphic, and white nutrition side panel.
     - Bleach Cleanser Bottle (YCB #021): Asymmetrical contoured bottle, handle
       cutout loop, angled neck, ribbed blue cap (#1e40af), diagonal blue/green label.
     - Mustard Bottle (YCB #006): Tapered yellow squeeze bottle (#eab308),
       red conical dispenser nozzle (#dc2626), and green brand ribbon.
  2. Active Target Ground Ring & Floating Badge:
     - Tabletop elliptical glowing halo beneath active workpiece.
     - Floating semi-transparent badge above target:
       "[ 🥫 Tomato Soup Can | YCB #005 ]", "[ 🧴 Bleach Bottle | YCB #021 ]", etc.
  3. Franka Panda Polish:
     - Enamel white finish (#f8fafc) on links with specular sheen.
     - Carbon joint collars (#334155).
     - Black silicone finger gripper pads.
  4. Video & Keyframe Synthesis:
     - Video: sim/sim_video.mp4 (H.264 yuv420p, 1280x720, 30 FPS, +faststart).
     - Publication Keyframes: figures/fig2_gazebo_keyframes.png (300 DPI, mirrored in paper/figures/).
"""

import math
import os
from pathlib import Path
import cv2
import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import subprocess

BASE_DIR = Path(__file__).resolve().parent.parent
SIM_DIR = BASE_DIR / "sim"
FIG_DIR = BASE_DIR / "figures"
PAPER_FIG_DIR = BASE_DIR / "paper" / "figures"

SIM_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)


class FrankaKinematics:
    """7-DoF Franka Emika Panda forward kinematics and trajectory solver."""
    def __init__(self):
        self.d = [0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.107]
        self.a = [0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088]
        self.alpha = [-math.pi/2, math.pi/2, math.pi/2, -math.pi/2, math.pi/2, math.pi/2, 0.0]
        self.base_z = 0.75  # Table surface at z = 0.75m

    def forward_kinematics(self, q):
        T = np.eye(4)
        T[2, 3] = self.base_z
        joint_positions = [T[:3, 3].copy()]

        for i in range(7):
            ct = math.cos(q[i])
            st = math.sin(q[i])
            ca = math.cos(self.alpha[i])
            sa = math.sin(self.alpha[i])

            A_i = np.array([
                [ct, -st*ca,  st*sa, self.a[i]*ct],
                [st,  ct*ca, -ct*sa, self.a[i]*st],
                [0.0,    sa,     ca,  self.d[i]],
                [0.0,   0.0,    0.0,       1.0]
            ])
            T = T @ A_i
            joint_positions.append(T[:3, 3].copy())

        # End-effector flange & fingers
        flange = T[:3, 3].copy()
        gripper_z = flange + T[:3, 2] * 0.105
        joint_positions.append(gripper_z)
        f_left = gripper_z + T[:3, 0] * 0.028
        f_right = gripper_z - T[:3, 0] * 0.028
        joint_positions.append(f_left)
        joint_positions.append(f_right)

        return joint_positions

    def solve_trajectory_joint_poses(self, target_pos, progress_ratio):
        ratio = float(np.clip(progress_ratio, 0.0, 1.0))
        q_home = np.array([0.0, -0.45, 0.0, -2.15, 0.0, 1.70, 0.785])
        yaw_t = math.atan2(target_pos[1], max(target_pos[0], 0.1))
        q_reach = np.array([yaw_t, 0.15, 0.0, -1.65, 0.0, 1.80, 0.785])
        q_cur = (1.0 - ratio) * q_home + ratio * q_reach
        return self.forward_kinematics(q_cur)


class SceneRenderer3D:
    """
    High-Fidelity 3D Phong/Lambertian shaded rasterizer with realistic YCB items,
    studio backdrop, metallic workbench, highlight rings, and status tags.
    """
    def __init__(self, width=1280, height=720):
        self.w = width
        self.h = height

        # Camera Intrinsics
        self.fx = 720.0
        self.fy = 720.0
        self.cx = width / 2.0
        self.cy = 395.0

        # Camera Extrinsics: Natural 35-45 deg front-side isometric view
        eye = np.array([0.85, -0.95, 1.15])
        target = np.array([0.35, 0.0, 0.75])
        up = np.array([0.0, 0.0, 1.0])

        fwd = target - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        true_up = np.cross(right, fwd)

        self.R = np.vstack([right, true_up, -fwd])
        self.t = -self.R @ eye

        # Directional light source (upper right-front)
        self.light_dir = np.array([0.45, -0.60, 1.15])
        self.light_dir /= np.linalg.norm(self.light_dir)

    def world_to_screen(self, pts3d):
        pts_cam = (self.R @ pts3d.T).T + self.t
        z = -pts_cam[:, 2]
        z_safe = np.maximum(z, 0.01)
        u = (self.fx * pts_cam[:, 0] / z_safe) + self.cx
        v = (-self.fy * pts_cam[:, 1] / z_safe) + self.cy
        return np.column_stack([u, v]), z

    def render_shaded_box(self, center, size, color_bgr, yaw=0.0):
        dx, dy, dz = size[0]/2, size[1]/2, size[2]/2
        corners = np.array([
            [-dx, -dy, -dz], [ dx, -dy, -dz], [ dx,  dy, -dz], [-dx,  dy, -dz],
            [-dx, -dy,  dz], [ dx,  dy,  dz], [ dx,  dy,  dz], [-dx,  dy,  dz]
        ])
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        R_z = np.array([[cos_y, -sin_y, 0], [sin_y, cos_y, 0], [0, 0, 1]])
        rot = (R_z @ corners.T).T + center

        faces_idx = [
            ([0, 1, 2, 3], np.array([0, 0, -1])),
            ([4, 5, 6, 7], np.array([0, 0,  1])),
            ([0, 1, 5, 4], np.array([0, -1, 0])),
            ([2, 3, 7, 6], np.array([0,  1, 0])),
            ([0, 3, 7, 4], np.array([-1, 0, 0])),
            ([1, 2, 6, 5], np.array([ 1, 0, 0])),
        ]

        polygons = []
        for idxs, norm_local in faces_idx:
            norm_world = R_z @ norm_local
            diffuse = max(0.0, float(np.dot(norm_world, self.light_dir)))
            specular = diffuse ** 4
            shade_factor = 0.44 + 0.56 * diffuse + 0.16 * specular
            face_color = [int(np.clip(c * shade_factor, 0, 255)) for c in color_bgr]
            pts_face = rot[idxs]
            polygons.append({
                "pts_world": pts_face,
                "color": face_color,
                "normal": norm_world
            })
        return polygons

    def render_shaded_cylinder(self, center, radius, height, color_bgr, segments=16):
        dz = height / 2.0
        angles = np.linspace(0, 2*math.pi, segments, endpoint=False)
        top_pts = np.array([[center[0] + radius*math.cos(a), center[1] + radius*math.sin(a), center[2] + dz] for a in angles])
        bot_pts = np.array([[center[0] + radius*math.cos(a), center[1] + radius*math.sin(a), center[2] - dz] for a in angles])

        polygons = []
        # Top cap
        diff_top = max(0.0, float(np.dot(np.array([0, 0, 1]), self.light_dir)))
        c_top = [int(np.clip(c * (0.45 + 0.65 * diff_top), 0, 255)) for c in color_bgr]
        polygons.append({"pts_world": top_pts, "color": c_top, "normal": np.array([0, 0, 1])})

        # Side facets
        for i in range(segments):
            i_next = (i + 1) % segments
            quad = np.array([bot_pts[i], bot_pts[i_next], top_pts[i_next], top_pts[i]])
            mid_angle = (angles[i] + angles[i_next]) / 2.0
            norm = np.array([math.cos(mid_angle), math.sin(mid_angle), 0.0])
            diffuse = max(0.0, float(np.dot(norm, self.light_dir)))
            specular = diffuse ** 5
            c_side = [int(np.clip(c * (0.42 + 0.58 * diffuse + 0.20 * specular), 0, 255)) for c in color_bgr]
            polygons.append({"pts_world": quad, "color": c_side, "normal": norm})

        return polygons

    def render_franka_link(self, p1, p2, radius, color_bgr):
        vec = p2 - p1
        dist = np.linalg.norm(vec)
        if dist < 1e-4:
            return []
        mid = (p1 + p2) / 2.0
        return self.render_shaded_cylinder(mid, radius, dist, color_bgr, segments=12)

    # -------------------------------------------------------------------------
    # DETAILED 3D YCB OBJECT GENERATORS
    # -------------------------------------------------------------------------
    def generate_tomato_soup_can_polys(self, pos):
        """YCB #005: 32-facet cylinder with silver rims and split red/white label."""
        polys = []
        cx, cy, cz = pos[0], pos[1], pos[2]
        r = 0.044
        segments = 32

        # 1. Bottom metallic silver rim (z: -0.060 to -0.052)
        polys.extend(self.render_shaded_cylinder([cx, cy, cz - 0.056], r, 0.008, [205, 212, 218], segments=segments))
        # 2. Lower white label band (bottom 50%: z: -0.052 to 0.000)
        polys.extend(self.render_shaded_cylinder([cx, cy, cz - 0.026], r, 0.052, [242, 245, 248], segments=segments))
        # 3. Upper red label band (top 50%: z: 0.000 to +0.052)
        polys.extend(self.render_shaded_cylinder([cx, cy, cz + 0.026], r, 0.052, [25, 30, 215], segments=segments))
        # 4. Top metallic silver rim (z: +0.052 to +0.060)
        polys.extend(self.render_shaded_cylinder([cx, cy, cz + 0.056], r, 0.008, [218, 224, 228], segments=segments))

        return polys

    def generate_cracker_box_polys(self, pos):
        """YCB #003: Crisp red cuboid (0.06 x 0.12 x 0.18) with yellow banner & details."""
        polys = []
        cx, cy, cz = pos[0], pos[1], pos[2]
        # Main red body
        polys.extend(self.render_shaded_box([cx, cy, cz], [0.06, 0.12, 0.18], [22, 28, 220]))
        # Yellow top banner on front face (x: cx + 0.0305, z: cz + 0.065)
        banner_center = [cx + 0.0305, cy, cz + 0.068]
        polys.extend(self.render_shaded_box(banner_center, [0.002, 0.116, 0.036], [28, 215, 252]))
        # Center cheese cracker graphic patch
        cheese_center = [cx + 0.0305, cy, cz + 0.010]
        polys.extend(self.render_shaded_box(cheese_center, [0.002, 0.046, 0.046], [25, 170, 248]))
        # White nutrition facts side panel (on side facing y-)
        nutr_center = [cx, cy - 0.0605, cz - 0.010]
        polys.extend(self.render_shaded_box(nutr_center, [0.052, 0.002, 0.120], [242, 245, 248]))
        return polys

    def generate_bleach_bottle_polys(self, pos):
        """YCB #021: Asymmetrical contoured bottle with handle cutout loop & blue ribbed cap."""
        polys = []
        cx, cy, cz = pos[0], pos[1], pos[2]
        # White enamel bottle body
        bottle_white = [246, 248, 252]
        # Lower bottle base
        polys.extend(self.render_shaded_box([cx, cy - 0.010, cz - 0.025], [0.072, 0.105, 0.145], bottle_white))
        # Upper angled shoulder/neck
        polys.extend(self.render_shaded_cylinder([cx, cy - 0.022, cz + 0.065], 0.024, 0.045, bottle_white, segments=16))
        # Blue ribbed cap (#1e40af: BGR [175, 64, 30])
        polys.extend(self.render_shaded_cylinder([cx, cy - 0.022, cz + 0.092], 0.020, 0.022, [175, 64, 30], segments=16))
        # Handle loop bridge on the right side (+y)
        handle_col = [240, 244, 248]
        polys.extend(self.render_shaded_box([cx, cy + 0.042, cz + 0.020], [0.028, 0.016, 0.075], handle_col))
        polys.extend(self.render_shaded_box([cx, cy + 0.028, cz + 0.052], [0.028, 0.038, 0.016], handle_col))
        polys.extend(self.render_shaded_box([cx, cy + 0.028, cz - 0.012], [0.028, 0.038, 0.016], handle_col))
        # Diagonal blue/green front label badge
        label_center = [cx + 0.0365, cy - 0.010, cz - 0.020]
        polys.extend(self.render_shaded_box(label_center, [0.002, 0.075, 0.075], [140, 185, 25], yaw=math.pi/6))
        return polys

    def generate_mustard_bottle_polys(self, pos):
        """YCB #006: Tapered yellow squeeze bottle (#eab308) with red dispenser nozzle (#dc2626)."""
        polys = []
        cx, cy, cz = pos[0], pos[1], pos[2]
        yellow_mustard = [8, 179, 234]  # #eab308
        red_nozzle = [38, 38, 220]      # #dc2626
        green_ribbon = [35, 135, 45]

        # 1. Lower base
        polys.extend(self.render_shaded_cylinder([cx, cy, cz - 0.035], 0.036, 0.070, yellow_mustard, segments=16))
        # 2. Tapered waist
        polys.extend(self.render_shaded_cylinder([cx, cy, cz + 0.010], 0.030, 0.028, yellow_mustard, segments=16))
        # 3. Green brand ribbon around waist
        polys.extend(self.render_shaded_cylinder([cx, cy, cz + 0.010], 0.0315, 0.014, green_ribbon, segments=16))
        # 4. Shoulder dome
        polys.extend(self.render_shaded_cylinder([cx, cy, cz + 0.032], 0.034, 0.024, yellow_mustard, segments=16))
        # 5. Red conical dispenser nozzle
        polys.extend(self.render_shaded_cylinder([cx, cy, cz + 0.052], 0.016, 0.020, red_nozzle, segments=12))
        polys.extend(self.render_shaded_cylinder([cx, cy, cz + 0.068], 0.007, 0.018, red_nozzle, segments=8))
        return polys

    def draw_scene(self, joints, target_obj_name, occluder_pct, u_val, cond_code, is_estop=False):
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        # 1. Clean neutral dark-gray studio backdrop (#1e222b)
        for y in range(self.h):
            ratio = y / self.h
            bg_b = int(43 + 9 * ratio)
            bg_g = int(34 + 8 * ratio)
            bg_r = int(30 + 8 * ratio)
            img[y, :] = (bg_b, bg_g, bg_r)

        scene_polys = []

        # 2. Metallic workbench (#4a5568: BGR [104, 85, 74]) at z = 0.725
        table_polys = self.render_shaded_box(
            center=np.array([0.45, 0.0, 0.725]),
            size=np.array([0.88, 1.25, 0.05]),
            color_bgr=[104, 85, 74]
        )
        scene_polys.extend(table_polys)

        # Table legs (dark slate steel)
        for lx, ly in [(0.10, -0.52), (0.80, -0.52), (0.10, 0.52), (0.80, 0.52)]:
            leg_polys = self.render_shaded_box(
                center=np.array([lx, ly, 0.36]),
                size=np.array([0.06, 0.06, 0.72]),
                color_bgr=[52, 45, 40]
            )
            scene_polys.extend(leg_polys)

        # 3. 7-DoF Franka Emika Panda Robot Arm (Polished Enamel White #f8fafc & Carbon Rings #334155)
        enamel_white = [248, 250, 252]  # #f8fafc
        carbon_ring = [81, 65, 51]     # #334155
        finger_pad = [22, 22, 24]      # Black silicone

        for i in range(len(joints) - 3):
            j_start = joints[i]
            j_end = joints[i+1]
            rad = 0.042 if i < 3 else (0.034 if i < 6 else 0.026)
            col = enamel_white if i % 2 == 1 else carbon_ring
            scene_polys.extend(self.render_franka_link(j_start, j_end, rad, col))

        # Parallel jaw gripper
        wrist = joints[7]
        ee = joints[8]
        f_left = joints[9]
        f_right = joints[10]

        scene_polys.extend(self.render_shaded_box(ee, [0.06, 0.10, 0.04], carbon_ring))
        scene_polys.extend(self.render_shaded_box(f_left, [0.015, 0.02, 0.05], [190, 195, 205]))
        scene_polys.extend(self.render_shaded_box(f_right, [0.015, 0.02, 0.05], [190, 195, 205]))
        # Black gripper finger pads on inner surfaces
        scene_polys.extend(self.render_shaded_box(f_left - np.array([0, 0.010, 0]), [0.012, 0.005, 0.042], finger_pad))
        scene_polys.extend(self.render_shaded_box(f_right + np.array([0, 0.010, 0]), [0.012, 0.005, 0.042], finger_pad))

        # 4. YCB Target Objects (Detailed 3D Models)
        obj_pos = np.array([0.55, 0.10, 0.81])
        if target_obj_name == "cracker_box":
            scene_polys.extend(self.generate_cracker_box_polys(np.array([0.55, 0.10, 0.84])))
        elif target_obj_name == "tomato_soup_can":
            scene_polys.extend(self.generate_tomato_soup_can_polys(np.array([0.55, 0.10, 0.81])))
        elif target_obj_name == "bleach_bottle":
            scene_polys.extend(self.generate_bleach_bottle_polys(np.array([0.55, 0.10, 0.825])))
        elif target_obj_name == "mustard_bottle":
            scene_polys.extend(self.generate_mustard_bottle_polys(np.array([0.55, 0.10, 0.81])))

        # Distractor companion object (Tuna can / cylinder)
        dist_pos = np.array([0.52, -0.22, 0.80])
        scene_polys.extend(self.render_shaded_cylinder(dist_pos, 0.042, 0.08, [155, 135, 75], segments=12))

        # 5. Visual Occluder Barrier
        if occluder_pct > 0.05:
            occ_h = 0.18
            occ_y_span = 0.15 * (occluder_pct / 0.70)
            occ_center = np.array([0.48, 0.10 - 0.04 + occ_y_span/2, 0.83])
            occ_polys = self.render_shaded_box(
                occ_center,
                [0.012, occ_y_span, occ_h],
                [115, 105, 90]
            )
            scene_polys.extend(occ_polys)

        # 6. Depth Sort (Painter's algorithm)
        for p in scene_polys:
            pts_cam = (self.R @ p["pts_world"].T).T + self.t
            p["cam_z"] = np.mean(pts_cam[:, 2])

        scene_polys.sort(key=lambda item: item["cam_z"])

        # Rasterize sorted polygons
        for p in scene_polys:
            scr_pts, _ = self.world_to_screen(p["pts_world"])
            pts_int = np.int32([scr_pts])
            cv2.fillPoly(img, pts_int, p["color"], lineType=cv2.LINE_AA)
            cv2.polylines(img, pts_int, isClosed=True, color=[int(c*0.65) for c in p["color"]], thickness=1, lineType=cv2.LINE_AA)

        # 7. Action Overlay: Active Target Tabletop Glowing Halo
        ring_center = np.array([0.55, 0.10, 0.752])
        ring_r = 0.088
        theta_ring = np.linspace(0, 2*math.pi, 32, endpoint=False)
        ring_3d = np.array([[ring_center[0] + ring_r*math.cos(th), ring_center[1] + ring_r*math.sin(th), ring_center[2]] for th in theta_ring])
        ring_2d, _ = self.world_to_screen(ring_3d)
        ring_int = np.int32([ring_2d])

        # Layered glowing halo
        halo_overlay = img.copy()
        cv2.fillPoly(halo_overlay, ring_int, (245, 215, 45), lineType=cv2.LINE_AA)
        cv2.addWeighted(halo_overlay, 0.25, img, 0.75, 0, img)
        cv2.polylines(img, ring_int, isClosed=True, color=(255, 235, 70), thickness=2, lineType=cv2.LINE_AA)

        # 8. Action Overlay: Clean semi-transparent badge above active target object
        ycb_names = {
            "tomato_soup_can": ("[ Tomato Soup Can | YCB #005 ]", "TOMATO SOUP"),
            "cracker_box": ("[ Cracker Box | YCB #003 ]", "CRACKERS"),
            "bleach_bottle": ("[ Bleach Bottle | YCB #021 ]", "BLEACH"),
            "mustard_bottle": ("[ Mustard Bottle | YCB #006 ]", "MUSTARD")
        }
        badge_text, label_script = ycb_names.get(target_obj_name, ("[ Target Object ]", ""))

        # Floating badge above target object
        target_top_3d = np.array([0.55, 0.10, 0.94])
        badge_2d, _ = self.world_to_screen(np.array([target_top_3d]))
        bx, by = int(badge_2d[0, 0]), int(badge_2d[0, 1])

        (bw, bh), b_base = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_DUPLEX, 0.48, 1)
        b_x1, b_y1 = bx - bw//2 - 8, by - bh - 6
        b_x2, b_y2 = bx + bw//2 + 8, by + b_base + 4

        b_overlay = img.copy()
        cv2.rectangle(b_overlay, (b_x1, b_y1), (b_x2, b_y2), (35, 24, 15), -1)  # #0f172a slate
        cv2.addWeighted(b_overlay, 0.85, img, 0.15, 0, img)
        cv2.rectangle(img, (b_x1, b_y1), (b_x2, b_y2), (248, 189, 56), 1, lineType=cv2.LINE_AA)  # Electric cyan/sky border
        cv2.putText(img, badge_text, (bx - bw//2, by), cv2.FONT_HERSHEY_DUPLEX, 0.48, (245, 248, 252), 1, cv2.LINE_AA)

        # Additional visual detail: crisp text on the YCB item face if in front
        can_front_3d = np.array([0.55 + 0.045, 0.10, 0.81])
        c_scr, _ = self.world_to_screen(np.array([can_front_3d]))
        cx_s, cy_s = int(c_scr[0, 0]), int(c_scr[0, 1])
        if target_obj_name == "tomato_soup_can":
            # Gold medallion and text "TOMATO SOUP"
            cv2.circle(img, (cx_s - 12, cy_s - 2), 6, (45, 195, 245), -1, lineType=cv2.LINE_AA)
            cv2.putText(img, "TOMATO SOUP", (cx_s - 38, cy_s + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (45, 195, 245), 1, cv2.LINE_AA)
        elif target_obj_name == "cracker_box":
            cv2.putText(img, "CRACKERS", (cx_s - 30, cy_s - 18), cv2.FONT_HERSHEY_DUPLEX, 0.36, (20, 25, 180), 1, cv2.LINE_AA)

        # 9. Action Overlay: Dynamic Gripper Motion Status Tag
        ee_2d, _ = self.world_to_screen(np.array([ee]))
        ue, ve = int(ee_2d[0, 0]), int(ee_2d[0, 1])

        if cond_code in ["C0", "C2"]:
            # Constant-speed nominal kinematics
            tag_text = "[ Speed: 0.35 m/s | NOMINAL ]"
            tag_col = (70, 215, 80)     # Vibrant Green
            tag_bg = (18, 42, 22)
        else:
            # Entropy-modulated kinematic hesitation (C1, C3)
            if u_val <= 0.30:
                tag_text = "[ Speed: 0.35 m/s | NOMINAL ]"
                tag_col = (70, 215, 80)     # Vibrant Green
                tag_bg = (18, 42, 22)
            elif u_val <= 0.65:
                tag_text = "[ Speed: 0.09 m/s | HESITATING... ]"
                tag_col = (40, 190, 245)    # Vibrant Amber
                tag_bg = (18, 36, 48)
            else:
                tag_text = "[ DWELL PAUSE (1.4s) | RETREAT 18mm ]"
                tag_col = (55, 65, 245)     # Vibrant Red
                tag_bg = (24, 18, 48)

        # In C2 and C3, shift badges left by 160 px and down by 30 px to avoid top-right eHMI HUD overlap
        if cond_code in ["C2", "C3"]:
            tag_x = int(np.clip(ue + 25 - 160, 30, self.w - 380))
            tag_y = int(np.clip(ve - 32 + 30, 40, self.h - 40))
        else:
            tag_x = int(np.clip(ue + 25, 30, self.w - 380))
            tag_y = int(np.clip(ve - 32, 40, self.h - 40))

        (tw, th), baseline = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_DUPLEX, 0.52, 1)
        bx1, by1 = tag_x - 8, tag_y - th - 8
        bx2, by2 = tag_x + tw + 8, tag_y + baseline + 6

        # Connect leader line to the nearest edge of the tag
        lead_x = bx2 if ue > bx2 else bx1
        cv2.line(img, (ue, ve), (lead_x, (by1 + by2)//2), tag_col, 1, lineType=cv2.LINE_AA)
        cv2.circle(img, (ue, ve), 4, tag_col, -1, lineType=cv2.LINE_AA)

        tag_overlay = img.copy()
        cv2.rectangle(tag_overlay, (bx1, by1), (bx2, by2), tag_bg, -1)
        cv2.addWeighted(tag_overlay, 0.78, img, 0.22, 0, img)
        cv2.rectangle(img, (bx1, by1), (bx2, by2), tag_col, 1, lineType=cv2.LINE_AA)
        cv2.putText(img, tag_text, (tag_x, tag_y), cv2.FONT_HERSHEY_DUPLEX, 0.52, tag_col, 1, cv2.LINE_AA)

        # 10. E-Stop Watermark if triggered
        if is_estop:
            cv2.putText(img, "EMERGENCY INTERVENTION ACTIVATED (E-STOP)", (300, 220), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.rectangle(img, (20, 20), (self.w - 20, self.h - 20), (0, 0, 255), 4)

        # 11. Explicit eHMI HUD (ONLY FOR C2 AND C3 - STRICT MODALITY ISOLATION)
        self.render_hud_overlay(img, cond_code, u_val, is_estop)

        return img

    def render_hud_overlay(self, img, cond_code, u_val, is_estop):
        """Renders HUD telemetry and explicit eHMI indicator with strict modality isolation."""
        cv2.rectangle(img, (24, 20), (480, 95), (20, 24, 30), -1)
        cv2.rectangle(img, (24, 20), (480, 95), (60, 68, 80), 1)

        cond_desc = {
            "C0": "Condition C0: Baseline (Unmodulated)",
            "C1": "Condition C1: Implicit (Kinematic Hesitation)",
            "C2": "Condition C2: Explicit (Visual-Verbal eHMI)",
            "C3": "Condition C3: Congruent Multimodal Hybrid"
        }
        cv2.putText(img, cond_desc.get(cond_code, cond_code), (38, 48), cv2.FONT_HERSHEY_DUPLEX, 0.52, (240, 245, 250), 1, cv2.LINE_AA)

        if cond_code in ["C0", "C1"]:
            cv2.putText(img, "eHMI HUD: DISABLED FOR CURRENT CONDITION", (38, 76), cv2.FONT_HERSHEY_DUPLEX, 0.44, (120, 130, 140), 1, cv2.LINE_AA)
            return

        certainty_pct = int(round((1.0 - u_val) * 100))
        cv2.rectangle(img, (self.w - 380, 20), (self.w - 24, 130), (20, 24, 30), -1)
        cv2.rectangle(img, (self.w - 380, 20), (self.w - 24, 130), (70, 80, 95), 1)

        if u_val < 0.25:
            col_bar = (60, 200, 70)
            status_text = "CONFIDENT GRASP"
        elif u_val < 0.65:
            col_bar = (30, 180, 240)
            status_text = "AMBIGUOUS TARGET"
        else:
            col_bar = (40, 45, 235)
            status_text = "UNSAFE ANOMALY"

        cv2.putText(img, f"ROBOT CERTAINTY: {certainty_pct}%", (self.w - 360, 52), cv2.FONT_HERSHEY_DUPLEX, 0.58, (245, 245, 250), 1, cv2.LINE_AA)
        cv2.putText(img, f"STATUS: {status_text}", (self.w - 360, 78), cv2.FONT_HERSHEY_DUPLEX, 0.46, col_bar, 1, cv2.LINE_AA)

        bar_x = self.w - 360
        bar_y = 92
        bar_w = 310
        bar_h = 16
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (45, 50, 60), -1)
        fill_w = int(bar_w * (certainty_pct / 100.0))
        if fill_w > 0:
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), col_bar, -1)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (90, 100, 115), 1)


def render_all_video_and_keyframes():
    print("===============================================================================")
    print(" High-Realism YCB 3D Models & Keyframes Rendering for Project 02")
    print(" Objects: Tomato Soup Can (#005), Cracker Box (#003), Bleach (#021), Mustard (#006)")
    print(" Robot Polish: White Enamel (#f8fafc), Carbon Rings (#334155), Black Pads")
    print(" Halos & Badges: Tabletop Glowing Halo + Floating YCB Identifier Badges")
    print("===============================================================================")

    renderer = SceneRenderer3D(width=1280, height=720)
    franka = FrankaKinematics()

    video_out_path = SIM_DIR / "sim_video.mp4"
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    # 4 Representative counterbalanced trials across C0-C3 showcasing all 4 YCB objects:
    rep_trials = [
        {"tid": 1, "cond": "C0", "u": 0.08, "obj": "cracker_box", "occ": 0.0, "estop": False, "desc": "C0 Baseline: [ 📦 Cracker Box | YCB #003 ] Nominal Approach"},
        {"tid": 8, "cond": "C1", "u": 0.92, "obj": "mustard_bottle", "occ": 0.70, "estop": False, "desc": "C1 Implicit: [ 🌭 Mustard Bottle | YCB #006 ] Kinematic Hesitation"},
        {"tid": 14, "cond": "C2", "u": 0.58, "obj": "tomato_soup_can", "occ": 0.25, "estop": False, "desc": "C2 Explicit: [ 🥫 Tomato Soup Can | YCB #005 ] eHMI Diagnostic HUD"},
        {"tid": 21, "cond": "C3", "u": 0.94, "obj": "bleach_bottle", "occ": 0.70, "estop": True, "desc": "C3 Congruent: [ 🧴 Bleach Bottle | YCB #021 ] Multimodal Fault Alert"},
    ]

    cmd = [
        ffmpeg_exe,
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", "1280x720",
        "-pix_fmt", "bgr24",
        "-r", "30",
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-crf", "22",
        "-movflags", "+faststart",
        str(video_out_path)
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    total_frames = 0
    keyframe_snapshots = []

    home_ee = np.array([0.45, 0.0, 1.05])
    target_ee = np.array([0.55, 0.10, 0.85])
    approach_dir = target_ee - home_ee
    total_dist = np.linalg.norm(approach_dir)
    approach_dir /= total_dist
    v_max = 0.35

    for seg in rep_trials:
        cond = seg["cond"]
        u_val = seg["u"]
        obj_name = seg["obj"]
        occ = seg["occ"]
        is_estop = seg["estop"]

        v_sim = v_max if cond in ["C0", "C2"] else v_max * (1.0 - 0.70 * (1.0 / (1.0 + math.exp(-10.0 * (u_val - 0.45)))))
        pause_sec = (0.80 + 2.40 * (u_val - 0.50)) if (cond in ["C1", "C3"] and u_val > 0.50) else 0.0
        ret_dist = max(0.0, 0.08 * (u_val - 0.70)) if (cond in ["C1", "C3"] and u_val > 0.70) else 0.0

        t_reach = total_dist / max(v_sim, 0.08)
        dur = t_reach + pause_sec + 0.8
        seg_frames = int(round(dur * 30))

        for f in range(seg_frames):
            t_cur = f / 30.0
            if t_cur < t_reach:
                s_n = t_cur / t_reach
                s_sm = 10 * s_n**3 - 15 * s_n**4 + 6 * s_n**5
                pos = home_ee + approach_dir * (total_dist * s_sm)
            elif t_cur < (t_reach + pause_sec):
                pos = target_ee
                if ret_dist > 0:
                    ph = (t_cur - t_reach) / max(pause_sec, 0.01)
                    pos = pos - approach_dir * (ret_dist * math.sin(ph * math.pi)**2)
            else:
                pos = target_ee

            ratio = min(1.0, t_cur / t_reach)
            j_pts = franka.solve_trajectory_joint_poses(pos, ratio)

            cur_estop = is_estop and (f > seg_frames - 25)
            frame = renderer.draw_scene(j_pts, obj_name, occ, u_val, cond, cur_estop)
            proc.stdin.write(frame.tobytes())
            total_frames += 1

            if f == int(seg_frames * 0.65):
                keyframe_snapshots.append((seg["desc"], frame.copy()))

    proc.stdin.close()
    proc.wait()
    print(f"--> [OK] MP4 Video rendered: {video_out_path} ({total_frames} frames, {total_frames/30:.1f}s)")

    # Generate Publication Keyframes (fig2_gazebo_keyframes.png) at high resolution
    print("--> Generating High-Resolution (300 DPI, 4800x2700) Publication Keyframes Montage...")
    fig2, axes = plt.subplots(2, 2, figsize=(16, 9), dpi=300)
    for idx, (title_str, k_img) in enumerate(keyframe_snapshots[:4]):
        row, col = divmod(idx, 2)
        ax = axes[row, col]
        rgb = cv2.cvtColor(k_img, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        title_clean = title_str.replace("📦 ", "").replace("🌭 ", "").replace("🥫 ", "").replace("🧴 ", "")
        ax.set_title(title_clean, fontsize=13, fontweight="bold", pad=10)
        ax.axis("off")

    plt.tight_layout()
    fig2_path = FIG_DIR / "fig2_gazebo_keyframes.png"
    fig2.savefig(fig2_path, dpi=300)
    paper_fig2_path = PAPER_FIG_DIR / "fig2_gazebo_keyframes.png"
    fig2.savefig(paper_fig2_path, dpi=300)
    plt.close(fig2)
    print(f"--> [OK] High-Res Fig 2 saved to {fig2_path} and {paper_fig2_path}")


if __name__ == "__main__":
    render_all_video_and_keyframes()
