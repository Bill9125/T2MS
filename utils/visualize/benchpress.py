import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import math

class RearV_BenchpressAnimator():
    def __init__(self, config: dict):
        # rear view
        self.frames = [i for i in range(len(config["left_shoulder"]))]
        self.left_shoulder_angles  = np.asarray(config["left_shoulder"], dtype=float)
        self.left_elbow_angles     = np.asarray(config["left_elbow"], dtype=float)
        self.left_shoulder_y    = self._normalize(np.asarray(config["left_shoulder_y"], dtype=float))
        self.right_shoulder_angles = np.asarray(config["right_shoulder"], dtype=float)
        self.right_elbow_angles    = np.asarray(config["right_elbow"], dtype=float)
        self.right_shoulder_y    = self._normalize(np.asarray(config["right_shoulder_y"], dtype=float))
        
        self.theta_R_list = np.asarray(config["right_torso-arm"], dtype=float)
        self.theta_L_list = np.asarray(config["left_torso-arm"], dtype=float)

        # 幾何與視覺參數
        self.L_upper = float(config.get("L_upper", 1.0))
        self.L_fore  = float(config.get("L_fore", 1.0))

        self.interval    = int(config.get("interval", 50))
        self.fps         = int(config.get("fps", 30))
        self.figsize     = tuple(config.get("figsize", (7, 7)))
        self.xlim        = tuple(config.get("xlim", (-3, 3)))
        self.ylim        = tuple(config.get("ylim", (-3, 1)))
        self.invert_y    = bool(config.get("invert_y", True))

        # Matplotlib figure/axes 與 artist
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.ax.set_aspect('equal')
        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.ylim)
        if self.invert_y:
            self.ax.invert_yaxis()

        # 鎖骨
        self.clavicle_line, = self.ax.plot([], [], '-', color='black', lw=2, animated=True)

        # 骨架線條
        self.line_left_upper,  = self.ax.plot([], [], 'o-', lw=3, color='tab:blue')
        self.line_left_fore,   = self.ax.plot([], [], 'o-', lw=3, color='tab:orange')
        self.line_right_upper, = self.ax.plot([], [], 'o-', lw=3, color='tab:blue')
        self.line_right_fore,  = self.ax.plot([], [], 'o-', lw=3, color='tab:orange')
        self.wrist_bridge, = self.ax.plot([], [], '-', color='tab:gray', lw=3, alpha=0.9, animated=True)

        # 角度文字
        self.text_left_shoulder  = self.ax.text(-1.2, 0.3, "", fontsize=8, color="red")
        self.text_right_shoulder = self.ax.text( 0.8, 0.3, "", fontsize=8, color="red")
        self.text_left_elbow     = self.ax.text(-1.8,-0.7, "", fontsize=8, color="green")
        self.text_right_elbow    = self.ax.text( 1.2,-0.7, "", fontsize=8, color="green")

        # 建立動畫物件佔位
        self.ani = None
        
    @staticmethod
    def _normalize(arr, a=0.0, b=640.0, c=-3.0, d=1.0):
        return (arr - a) * (d - c) / (b - a) + c

    def _get_arm_coords(self, shoulder_angle_deg, elbow_angle_deg, torso_arm_angle, origin, side="left"):
        """
        shoulder_angle_deg, elbow_angle_deg 皆為度數。
        left: 以 base=0，肩角度順時針為正（畫面 y 軸已 invert）。  
        right: 以 base=pi，肩角度順時針為正的對稱處理。
        """
        L_upper, L_fore = self.L_upper, self.L_fore
        phi_top = np.deg2rad(torso_arm_angle)
        L_rear_L = self.L_upper * abs(np.sin(phi_top))

        if side == "left":
            base = 0.0
            upper_dir = base - np.deg2rad(shoulder_angle_deg)
            bend = np.pi - np.deg2rad(elbow_angle_deg)
            forearm_dir = upper_dir + bend
        else:
            base = np.pi
            upper_dir = base + np.deg2rad(shoulder_angle_deg)
            bend = np.pi - np.deg2rad(elbow_angle_deg)
            forearm_dir = upper_dir - bend

        shoulder = np.array(origin, dtype=float)
        elbow = shoulder + L_rear_L * np.array([np.cos(upper_dir), np.sin(upper_dir)])
        wrist = elbow   + L_fore  * np.array([np.cos(forearm_dir), np.sin(forearm_dir)])
        return shoulder, elbow, wrist

    def _init_artists(self):
        for ln in [self.line_left_upper, self.line_left_fore,
                self.line_right_upper, self.line_right_fore,
                self.clavicle_line]:
            ln.set_data([], [])
        self.text_left_shoulder.set_text("")
        self.text_right_shoulder.set_text("")
        self.text_left_elbow.set_text("")
        self.text_right_elbow.set_text("")
        self.wrist_bridge.set_data([], [])
        # 回傳所有 animated 的 artists（tuple/list 皆可）
        return (self.line_left_upper, self.line_left_fore,
                self.line_right_upper, self.line_right_fore,
                self.clavicle_line,
                self.text_left_shoulder, self.text_right_shoulder,
                self.text_left_elbow, self.text_right_elbow, self.wrist_bridge)

    def _update_frame(self, i):
        # 左手
        shoulder, elbow, wrist = self._get_arm_coords(
            self.left_shoulder_angles[i], self.left_elbow_angles[i], self.theta_L_list[i], origin=(-1, self.left_shoulder_y[i]), side="left"
        )
        self.line_left_upper.set_data([shoulder[0], elbow[0]], [shoulder[1], elbow[1]])
        self.line_left_fore.set_data([elbow[0], wrist[0]], [elbow[1], wrist[1]])

        # 右手
        shoulder_r, elbow_r, wrist_r = self._get_arm_coords(
            self.right_shoulder_angles[i], self.right_elbow_angles[i], self.theta_R_list[i], origin=(1, self.right_shoulder_y[i]), side="right"
        )
        self.line_right_upper.set_data([shoulder_r[0], elbow_r[0]], [shoulder_r[1], elbow_r[1]])
        self.line_right_fore.set_data([elbow_r[0], wrist_r[0]], [elbow_r[1], wrist_r[1]])

        # 更新鎖骨：端點 y 改為兩側肩膀 y（可取平均或各自連線）
        y_left  = self.left_shoulder_y[i]
        y_right = self.right_shoulder_y[i]
        self.clavicle_line.set_data([-1, 1], [y_left, y_right])
        
        # 手腕連線並外插
        wR = np.array(wrist_r, dtype=float)  # 右腕
        wL = np.array(wrist, dtype=float)  # 左腕
        seg = wL - wR
        seg_norm = np.hypot(seg[0], seg[1])
        if seg_norm > 1e-9:
            u = seg / seg_norm
        else:
            u = np.array([1.0, 0.0])  # 避免零長度，給個水平方向
        # 外插長度（可依畫面比例調整，例：肩寬 1 的 5%）
        t = 0.15
        p_start = wR - t * u
        p_end   = wL + t * u
        self.wrist_bridge.set_data([p_start[0], p_end[0]], [p_start[1], p_end[1]])

        # 角度文字
        self.text_left_shoulder.set_text(f"L_shoulder: {self.left_shoulder_angles[i]:.1f}°")
        self.text_right_shoulder.set_text(f"R_shoulder: {self.right_shoulder_angles[i]:.1f}°")
        self.text_left_elbow.set_text(f"L_elbow: {self.left_elbow_angles[i]:.1f}°")
        self.text_right_elbow.set_text(f"R_elbow: {self.right_elbow_angles[i]:.1f}°")

        # 注意：標題在 blit 下常見不更新，若一定要更新標題，建議 blit=False
        self.ax.set_title(f"Frame {self.frames[i]}")

        return (self.line_left_upper, self.line_left_fore,
                self.line_right_upper, self.line_right_fore,
                self.clavicle_line,
                self.text_left_shoulder, self.text_right_shoulder,
                self.text_left_elbow, self.text_right_elbow)

    def animate(self, output_file):
        """建立動畫並輸出 mp4 檔案。"""
        self.ani = animation.FuncAnimation(
            self.fig, self._update_frame,
            frames=len(self.frames),
            init_func=self._init_artists,
            blit=True, interval=self.interval
        )
        # 使用 ffmpeg writer
        self.ani.save(output_file, writer="ffmpeg", fps=self.fps)


class TopV_BenchpressAnimator():
    def __init__(self, config: dict):
        # top view
        self.frames = [i for i in range(len(config["left_dist"]))]
        self.dist_L_list = self._normalize(np.asarray(config["left_dist"], dtype=float))
        self.theta_L_list = np.asarray(config["left_torso-arm"], dtype=float)
        self.dist_R_list = self._normalize(np.asarray(config["right_dist"], dtype=float))
        self.theta_R_list = np.asarray(config["right_torso-arm"], dtype=float)
        self.left_shoulder_angles  = np.asarray(config["left_shoulder"], dtype=float)
        self.right_shoulder_angles = np.asarray(config["right_shoulder"], dtype=float)
        
        # 視覺與動畫設定
        self.figsize = tuple(config.get("figsize", (7,7)))
        self.xlim = tuple(config.get("xlim", (-2, 2)))
        self.ylim = tuple(config.get("ylim", (-2, 2)))
        self.fps = int(config.get("fps", 30))
        self.interval_ms = int(config.get("interval_ms", 33))

        # 幾何參數
        self.center = (0, 1)
        self.shoulder_width = 1
        self.hip_width = 0.8
        self.torso_len = 2
        self.upper_arm = 0.8
        self.forearm = 1
        
        # 關節連線（與你提供版本一致）
        self.connections = [(0,1),(0,4),(4,6),(1,5),(5,7),(0,2),(1,3),(2,3)]
        
        # 建立 Figure/Axes 與 artists
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self._setup_axes()
        self._build_artists()

        # 動畫物件
        self.ani = None
    
    @staticmethod
    def _normalize(arr, a=0.0, b=480.0, c=0.0, d=4.0):
        return (arr - a) * (d - c) / (b - a) + c
    
    @staticmethod
    def _unit(v):
        x, y = v
        n = math.hypot(x, y)
        return (x/n, y/n) if n > 0 else (0.0, -1.0)

    @staticmethod
    def _rot(v, ang_rad):
        x, y = v
        c, s = math.cos(ang_rad), math.sin(ang_rad)
        return (x*c - y*s, x*s + y*c)

    def _synth_pose_four_inputs(self, theta_L, theta_R, dist_L, dist_R, R_L_shoulder, R_R_shoulder):
        """
        回傳 8 點：
          0: 右肩, 1: 左肩, 2: 右髖, 3: 左髖, 4: 右肘, 5: 左肘, 6: 右腕, 7: 左腕
        幾何定義沿用你現有程式。
        """
        cx, cy = self.center
        L_top_L = self.upper_arm*abs(np.cos(np.deg2rad(R_L_shoulder)))
        L_top_R = self.upper_arm*abs(np.cos(np.deg2rad(R_R_shoulder)))
        

        p0 = (cx - self.shoulder_width/2, cy)            # 右肩
        p1 = (cx + self.shoulder_width/2, cy)            # 左肩
        p2 = (cx - self.hip_width/2, cy - self.torso_len)# 右髖
        p3 = (cx + self.hip_width/2, cy - self.torso_len)# 左髖
        
        v_shoulder = (p1[0]-p0[0], p1[1]-p0[1])
        trunk_dir = self._unit((v_shoulder[1], -v_shoulder[0]))

        arm_dir_R = self._unit(self._rot(trunk_dir, -math.radians(theta_R)))
        arm_dir_L = self._unit(self._rot(trunk_dir,  math.radians(theta_L)))

        p4 = (p0[0] + arm_dir_R[0]*L_top_R, p0[1] + arm_dir_R[1]*L_top_R)  # 右肘
        p5 = (p1[0] + arm_dir_L[0]*L_top_L, p1[1] + arm_dir_L[1]*L_top_L)  # 左肘

        wy_R = p0[1] - dist_R
        p6 = (p4[0], wy_R)

        wy_L = p1[1] - dist_L
        p7 = (p5[0], wy_L)

        return [p0,p1,p2,p3,p4,p5,p6,p7]

    # ---------- 視覺建置 ----------
    def _setup_axes(self):
        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.ylim)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(True)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')

    def _build_artists(self):
        # 線段（骨架）
        self.lines = [self.ax.plot([], [], color='orange', lw=2, animated=True)[0] for _ in self.connections]
        # 點（關節）
        self.points = self.ax.scatter([], [], color='blue', s=60)
        self.points.set_animated(True)

        # 輔助線：右/左腕垂線 + 肩線中線
        self.vline_R, = self.ax.plot([], [], '--', color='purple', lw=2, animated=True)
        self.vline_L, = self.ax.plot([], [], '--', color='green',  lw=2, animated=True)
        self.hline,   = self.ax.plot([], [], '--', color='gray',   lw=1.5, animated=True)
        self.wrist_bridge, = self.ax.plot([], [], '-', color='tab:gray', lw=3, alpha=0.9, animated=True)

        # 文字
        self.text_title   = self.ax.text(self.xlim[0],  self.ylim[1], "", fontsize=9, color='darkred', animated=True)

    # ---------- 動畫介面 ----------
    def _init(self):
        for ln in self.lines:
            ln.set_data([], [])
        self.points.set_offsets(np.empty((0, 2)))
        self.vline_R.set_data([], [])
        self.vline_L.set_data([], [])
        self.hline.set_data([], [])
        self.wrist_bridge.set_data([], [])
        self.text_title.set_text("")
        return self._all_artists()

    def _all_artists(self):
        return tuple(self.lines + [self.points, self.vline_R, self.vline_L, self.hline,
                               self.text_title, self.wrist_bridge])

    def _update(self, frame):
        thL = self.theta_L_list[frame]
        thR = self.theta_R_list[frame]
        dL  = self.dist_L_list[frame]
        dR  = self.dist_R_list[frame]
        R_L_shoulder = self.left_shoulder_angles[frame]
        R_R_shoulder = self.right_shoulder_angles[frame]
        
        pts = self._synth_pose_four_inputs(thL, thR, dL, dR, R_L_shoulder, R_R_shoulder)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        for i, (a, b) in enumerate(self.connections):
            self.lines[i].set_data([xs[a], xs[b]], [ys[a], ys[b]])

        self.points.set_offsets(list(zip(xs, ys)))

        # 垂直距離線（腕到各自肩的垂直線）
        self.vline_R.set_data([pts[6][0], pts[6][0]], [min(pts[6][1], pts[0][1]), max(pts[6][1], pts[0][1])])
        self.vline_L.set_data([pts[7][0], pts[7][0]], [min(pts[7][1], pts[1][1]), max(pts[7][1], pts[1][1])])

        # 肩線中線（僅示意）
        y_mid = 0.5 * (pts[0][1] + pts[1][1])
        self.hline.set_data([self.xlim[0], self.xlim[1]], [y_mid, y_mid])
        
        # 手腕連線並外插
        wR = np.array(pts[6], dtype=float)  # 右腕
        wL = np.array(pts[7], dtype=float)  # 左腕
        seg = wL - wR
        seg_norm = np.hypot(seg[0], seg[1])
        if seg_norm > 1e-9:
            u = seg / seg_norm
        else:
            u = np.array([1.0, 0.0])  # 避免零長度，給個水平方向
        # 外插長度（可依畫面比例調整，例：肩寬 1 的 5%）
        t = 0.15
        p_start = wR - t * u
        p_end   = wL + t * u
        self.wrist_bridge.set_data([p_start[0], p_end[0]], [p_start[1], p_end[1]])
        
        self.text_title.set_text(f"Frame {frame+1}: θR={thR:.3f}°, θL={thL:.3f}°, dR={dR:.3f}, dL={dL:.3f}")
        return self._all_artists()

    def animate(self, output_file=None):
        # blit 模式＋正確回傳被修改 artists 可顯著降低重繪成本
        self.ani = animation.FuncAnimation(
            self.fig,
            self._update,
            frames=self.frames,
            init_func=self._init,
            blit=False,
            interval=self.interval_ms
        )
        if output_file:
            # 需要系統已安裝 ffmpeg，Matplotlib 會呼叫對應 writer 輸出 mp4
            self.ani.save(output_file, writer="ffmpeg", fps=self.fps)
            
class LateralV_BenchpressAnimator():
    def __init__(self, config: dict):
        # lateral view
        self.frames = [i for i in range(len(config["bar_x"]))]
        self.bar_x  = np.asarray(config["bar_x"], dtype=float)
        self.bar_y  = np.asarray(config["bar_y"], dtype=float)
        
        # 視覺與動畫設定
        self.figsize = tuple(config.get("figsize", (6, 8)))
        self.xlim = tuple(config.get("xlim", (0, 640)))
        self.ylim = tuple(config.get("ylim", (0, 480)))
        self.fps = int(config.get("fps", 30))
        self.interval_ms = int(config.get("interval_ms", 33))
        
        # 建立 Figure/Axes 與 artists
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self._setup_axes()
        self._build_artists()

        # 動畫物件
        self.ani = None

    def _setup_axes(self):
        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.ylim)
        self.ax.invert_yaxis()  # 原點(0,0)在左上角
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(True, linestyle=':', alpha=0.6)
        self.ax.set_xlabel('px (x)')
        self.ax.set_ylabel('px (y)')
        self.ax.set_title("Lateral View - Bar Path")

    def _build_artists(self):
        # 槓鈴路徑 (線)
        self.line_path, = self.ax.plot([], [], '-', color='tab:blue', alpha=0.5, lw=2, animated=True)
        # 槓鈴 (點)
        self.point_bar, = self.ax.plot([], [], 'o', color='red', markersize=12, markeredgecolor='black', animated=True)
        
        # 文字標籤 (座標改為像素單位下的固定位置)
        self.text_title = self.ax.text(20, 30, "", fontsize=10, color='darkred', animated=True)
        self.text_info = self.ax.text(20, 70, "", fontsize=9, animated=True)

    def _all_artists(self):
        return (self.line_path, self.point_bar, self.text_title, self.text_info)

    def _init(self):
        self.line_path.set_data([], [])
        self.point_bar.set_data([], [])
        self.text_title.set_text("")
        self.text_info.set_text("")
        return self._all_artists()

    def _update(self, i):
        bx = self.bar_x[i]
        by = self.bar_y[i]
        
        # 更新目前槓鈴位置
        self.point_bar.set_data([bx], [by])
        # 更新軌跡
        self.line_path.set_data(self.bar_x[:i+1], self.bar_y[:i+1])
        
        # 更新文字
        self.text_title.set_text(f"Frame {i}")
        self.text_info.set_text(f"X: {bx:.3f}\nY: {by:.3f}")
        
        # 如果 blit=False，這裡可以直接更新 ax.set_title
        # self.ax.set_title(f"Lateral View - Frame {i}")
        
        return self._all_artists()

    def animate(self, output_file):
        """建立動畫並輸出 mp4 或 gif。"""
        self.ani = animation.FuncAnimation(
            self.fig,
            self._update,
            frames=len(self.frames),
            init_func=self._init,
            blit=True,
            interval=self.interval_ms
        )
        if output_file:
            # 需要系統已安裝 ffmpeg
            self.ani.save(output_file, writer="ffmpeg", fps=self.fps)
            print(f"動畫已輸出完成：{os.path.abspath(output_file)}")
            
class FullViewBenchpressAnimator():
    def __init__(self, config: dict):
        self.config = config
        self.frames = range(len(config["bar_x"]))
        self.fps = int(config.get("fps", 30))
        self.interval = int(config.get("interval_ms", 50))
        
        # 建立 1x3 的圖表
        self.fig, (self.ax_rear, self.ax_top, self.ax_lateral) = plt.subplots(1, 3, figsize=(18, 6))
        self.fig.subplots_adjust(wspace=0.3)
        
        # --- 初始化各個視角的參數 ---
        # Rear View parameters
        self.L_upper = float(config.get("L_upper", 1.0))
        self.L_fore  = float(config.get("L_fore", 1.0))
        self.rear_ls_angles = np.asarray(config["left_shoulder"], dtype=float)
        self.rear_le_angles = np.asarray(config["left_elbow"], dtype=float)
        self.rear_ls_y = self._normalize_rear(np.asarray(config["left_shoulder_y"], dtype=float))
        self.rear_rs_angles = np.asarray(config["right_shoulder"], dtype=float)
        self.rear_re_angles = np.asarray(config["right_elbow"], dtype=float)
        self.rear_rs_y = self._normalize_rear(np.asarray(config["right_shoulder_y"], dtype=float))
        self.theta_R_list = np.asarray(config["right_torso-arm"], dtype=float)
        self.theta_L_list = np.asarray(config["left_torso-arm"], dtype=float)
        
        # Top View parameters
        self.top_dist_L = self._normalize_top(np.asarray(config["left_dist"], dtype=float))
        self.top_dist_R = self._normalize_top(np.asarray(config["right_dist"], dtype=float))
        self.top_center = (0, 1)
        self.top_shoulder_width = 1
        self.top_hip_width = 0.8
        self.top_torso_len = 2
        self.top_upper_arm = 0.8
        self.top_forearm = 1
        self.top_connections = [(0,1),(0,4),(4,6),(1,5),(5,7),(0,2),(1,3),(2,3)]
        
        # Lateral View parameters
        self.lat_bar_x = np.asarray(config["bar_x"], dtype=float)
        self.lat_bar_y = np.asarray(config["bar_y"], dtype=float)
        
        # --- 設置 Axes ---
        self._setup_rear_ax()
        self._setup_top_ax()
        self._setup_lat_ax()
        
        # --- 建立 Artists ---
        self._build_rear_artists()
        self._build_top_artists()
        self._build_lat_artists()
        
        self.ani = None

    @staticmethod
    def _normalize_rear(arr, a=0.0, b=640.0, c=-3.0, d=1.0):
        return (arr - a) * (d - c) / (b - a) + c

    @staticmethod
    def _normalize_top(arr, a=0.0, b=480.0, c=0.0, d=4.0):
        return (arr - a) * (d - c) / (b - a) + c

    # --- Rear View Logic ---
    def _setup_rear_ax(self):
        ax = self.ax_rear
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 1)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title("Rear View")
        ax.grid(True, linestyle=':', alpha=0.5)

    def _build_rear_artists(self):
        ax = self.ax_rear
        self.rear_clavicle, = ax.plot([], [], '-', color='black', lw=2)
        self.rear_l_upper,  = ax.plot([], [], 'o-', lw=3, color='tab:blue')
        self.rear_l_fore,   = ax.plot([], [], 'o-', lw=3, color='tab:orange')
        self.rear_r_upper,  = ax.plot([], [], 'o-', lw=3, color='tab:blue')
        self.rear_r_fore,   = ax.plot([], [], 'o-', lw=3, color='tab:orange')
        self.rear_bridge,   = ax.plot([], [], '-', color='tab:gray', lw=3, alpha=0.9)
        self.rear_text_l    = ax.text(-2.8, 0.8, "", fontsize=8, color="blue")
        self.rear_text_r    = ax.text(0.5, 0.8, "", fontsize=8, color="blue")

    def _get_rear_arm_coords(self, s_ang, e_ang, t_ang, origin, side="left"):
        phi_top = np.deg2rad(t_ang)
        L_rear_L = self.L_upper * abs(np.sin(phi_top))
        if side == "left":
            base = 0.0
            u_dir = base - np.deg2rad(s_ang)
            bend = np.pi - np.deg2rad(e_ang)
            f_dir = u_dir + bend
        else:
            base = np.pi
            u_dir = base + np.deg2rad(s_ang)
            bend = np.pi - np.deg2rad(e_ang)
            f_dir = u_dir - bend
        sh = np.array(origin, dtype=float)
        el = sh + L_rear_L * np.array([np.cos(u_dir), np.sin(u_dir)])
        wr = el + self.L_fore * np.array([np.cos(f_dir), np.sin(f_dir)])
        return sh, el, wr

    # --- Top View Logic ---
    def _setup_top_ax(self):
        ax = self.ax_top
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.set_title("Top View")
        ax.grid(True, linestyle=':', alpha=0.5)

    def _build_top_artists(self):
        ax = self.ax_top
        self.top_lines = [ax.plot([], [], color='orange', lw=2)[0] for _ in self.top_connections]
        self.top_points = ax.scatter([], [], color='blue', s=40)
        self.top_bridge, = ax.plot([], [], '-', color='tab:gray', lw=3, alpha=0.9)
        self.top_text = ax.text(-1.8, 1.8, "", fontsize=8)

    def _synth_top_pose(self, thL, thR, dL, dR, R_L_s, R_R_s):
        cx, cy = self.top_center
        ltl = self.top_upper_arm * abs(np.cos(np.deg2rad(R_L_s)))
        ltr = self.top_upper_arm * abs(np.cos(np.deg2rad(R_R_s)))
        p0 = (cx - self.top_shoulder_width/2, cy)
        p1 = (cx + self.top_shoulder_width/2, cy)
        p2 = (cx - self.top_hip_width/2, cy - self.top_torso_len)
        p3 = (cx + self.top_hip_width/2, cy - self.top_torso_len)
        def unit(v): n = math.hypot(*v); return (v[0]/n, v[1]/n) if n>0 else (0,-1)
        def rot(v, a): c,s = math.cos(a), math.sin(a); return (v[0]*c-v[1]*s, v[0]*s+v[1]*c)
        trunk_dir = unit((p1[0]-p0[0], p1[1]-p0[1]))
        trunk_dir = unit((trunk_dir[1], -trunk_dir[0]))
        adR = unit(rot(trunk_dir, -math.radians(thR)))
        adL = unit(rot(trunk_dir,  math.radians(thL)))
        p4 = (p0[0] + adR[0]*ltr, p0[1] + adR[1]*ltr)
        p5 = (p1[0] + adL[0]*ltl, p1[1] + adL[1]*ltl)
        p6 = (p4[0], p0[1] - dR)
        p7 = (p5[0], p1[1] - dL)
        return [p0,p1,p2,p3,p4,p5,p6,p7]

    # --- Lateral View Logic ---
    def _setup_lat_ax(self):
        ax = self.ax_lateral
        ax.set_xlim(0, 640)
        ax.set_ylim(0, 480)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_title("Lateral View")
        ax.grid(True, linestyle=':', alpha=0.5)

    def _build_lat_artists(self):
        ax = self.ax_lateral
        self.lat_line, = ax.plot([], [], '-', color='tab:blue', alpha=0.5, lw=2)
        self.lat_point, = ax.plot([], [], 'o', color='red', markersize=10, markeredgecolor='black')
        self.lat_text = ax.text(30, 50, "", fontsize=8)

    # --- Animation Callbacks ---
    def _init(self):
        # Rear
        self.rear_clavicle.set_data([], [])
        self.rear_l_upper.set_data([], [])
        self.rear_l_fore.set_data([], [])
        self.rear_r_upper.set_data([], [])
        self.rear_r_fore.set_data([], [])
        self.rear_bridge.set_data([], [])
        self.rear_text_l.set_text("")
        self.rear_text_r.set_text("")
        # Top
        for ln in self.top_lines: ln.set_data([], [])
        self.top_points.set_offsets(np.empty((0, 2)))
        self.top_bridge.set_data([], [])
        self.top_text.set_text("")
        # Lateral
        self.lat_line.set_data([], [])
        self.lat_point.set_data([], [])
        self.lat_text.set_text("")
        return []

    def _update(self, i):
        # 1. Update Rear
        sh_l, el_l, wr_l = self._get_rear_arm_coords(self.rear_ls_angles[i], self.rear_le_angles[i], self.theta_L_list[i], (-1, self.rear_ls_y[i]), "left")
        sh_r, el_r, wr_r = self._get_rear_arm_coords(self.rear_rs_angles[i], self.rear_re_angles[i], self.theta_R_list[i], (1, self.rear_rs_y[i]), "right")
        self.rear_l_upper.set_data([sh_l[0], el_l[0]], [sh_l[1], el_l[1]])
        self.rear_l_fore.set_data([el_l[0], wr_l[0]], [el_l[1], wr_l[1]])
        self.rear_r_upper.set_data([sh_r[0], el_r[0]], [sh_r[1], el_r[1]])
        self.rear_r_fore.set_data([el_r[0], wr_r[0]], [el_r[1], wr_r[1]])
        self.rear_clavicle.set_data([-1, 1], [self.rear_ls_y[i], self.rear_rs_y[i]])
        # Bridge
        seg = wr_l - wr_r
        n = np.hypot(*seg)
        u = seg/n if n>1e-9 else np.array([1,0])
        p_s, p_e = wr_r - 0.15*u, wr_l + 0.15*u
        self.rear_bridge.set_data([p_s[0], p_e[0]], [p_s[1], p_e[1]])
        self.rear_text_l.set_text(f"L: S={self.rear_ls_angles[i]:.0f} E={self.rear_le_angles[i]:.0f}")
        self.rear_text_r.set_text(f"R: S={self.rear_rs_angles[i]:.0f} E={self.rear_re_angles[i]:.0f}")

        # 2. Update Top
        pts = self._synth_top_pose(self.theta_L_list[i], self.theta_R_list[i], self.top_dist_L[i], self.top_dist_R[i], self.rear_ls_angles[i], self.rear_rs_angles[i])
        for idx, (a, b) in enumerate(self.top_connections):
            self.top_lines[idx].set_data([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]])
        self.top_points.set_offsets(pts)
        wR, wL = np.array(pts[6]), np.array(pts[7])
        seg_t = wL - wR
        nt = np.hypot(*seg_t)
        ut = seg_t/nt if nt>1e-9 else np.array([1,0])
        self.top_bridge.set_data([wR[0]-0.15*ut[0], wL[0]+0.15*ut[0]], [wR[1]-0.15*ut[1], wL[1]+0.15*ut[1]])
        self.top_text.set_text(f"dL={self.top_dist_L[i]:.2f} dR={self.top_dist_R[i]:.2f}")

        # 3. Update Lateral
        self.lat_point.set_data([self.lat_bar_x[i]], [self.lat_bar_y[i]])
        self.lat_line.set_data(self.lat_bar_x[:i+1], self.lat_bar_y[:i+1])
        self.lat_text.set_text(f"Frame: {i}\nX={self.lat_bar_x[i]:.0f} Y={self.lat_bar_y[i]:.0f}")

        self.fig.suptitle(f"Bench Press Analysis - Frame {i}", fontsize=14)
        return []

    def animate(self, output_file):
        self.ani = animation.FuncAnimation(self.fig, self._update, frames=self.frames, init_func=self._init, blit=False, interval=self.interval)
        if output_file:
            self.ani.save(output_file, writer="ffmpeg", fps=self.fps)
            print(f"Combined animation saved to {output_file}")