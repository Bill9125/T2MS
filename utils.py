import os
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import yaml
import matplotlib.animation as animation

def plot_loss_curve(loss_list, save_path, filename='loss_curve.png'):
    if len(loss_list) == 0:
        print("loss_list is empty, skipping plotting.")
        return
    plt.figure(figsize=(10, 6))
    x = [i for i in range(len(loss_list))]
    plt.plot(loss_list, label='Training Loss')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, filename)
    plt.savefig(full_path)
    plt.close()
    print(f"Loss curve saved to {full_path}")
    
def seed_everything(seed, cudnn_deterministic=False):
    if seed is not None:
        print(f"Global seed set to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False

    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        
def get_cfg(args):
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        args.dataset_root = config.get('dataset_root', './Data')
        args.general_seed = config.get('general_seed', 2025)
        
        cfg = config[args.dataset_name]
        args.features = [feature[0]["name"] for feature in cfg["features"].values()]
        args.flow_dim = cfg.get('flow_dim', 128)
        args.input_dim = cfg.get('input_dim', 10)
        args.split_base_num = cfg['dataset'].get('split_base_num', 36)
        args.caption = cfg['dataset'].get('caption', 'Caption_explain_no_barbell')

        args.embedding_dim = cfg['vae'].get('embedding_dim', 64)
        args.block_hidden_size = cfg['vae'].get('block_hidden_size', 128)
        args.num_residual_layers = cfg['vae'].get('num_residual_layers', 3)
        args.res_hidden_size = cfg['vae'].get('res_hidden_size', 256)
        args.pretrained_epc = cfg['vae'].get('epoch', 80000)
        
        args.denoiser = cfg['diffusion'].get('denoiser', 'DiT')
        args.backbone = cfg['diffusion'].get('backbone', 'flowmatching')
    return args

class BenchpressAnimator:
    def __init__(self, config: dict):
        # 讀取必要資料
        required = [
            "left_shoulder", "left_elbow",
            "right_shoulder", "right_elbow",
            "left_shoulder_y", "right_shoulder_y"
        ]
        missing = [k for k in required if k not in config]
        if missing:
            raise ValueError(f"Missing required keys: {missing}")

        # 基本參數
        self.frames = [i for i in range(len(config["left_shoulder"]))]
        self.left_shoulder_angles  = np.asarray(config["left_shoulder"], dtype=float)
        self.left_elbow_angles     = np.asarray(config["left_elbow"], dtype=float)
        self.left_shoulder_y    = self.minmax_01(np.asarray(config["left_shoulder_y"], dtype=float))
        self.right_shoulder_angles = np.asarray(config["right_shoulder"], dtype=float)
        self.right_elbow_angles    = np.asarray(config["right_elbow"], dtype=float)
        self.right_shoulder_y    = self.minmax_01(np.asarray(config["right_shoulder_y"], dtype=float))

        # 幾何與視覺參數
        self.L_upper = float(config.get("L_upper", 1.0))
        self.L_fore  = float(config.get("L_fore", 1.0))

        self.interval    = int(config.get("interval", 50))
        self.fps         = int(config.get("fps", 30))
        self.figsize     = tuple(config.get("figsize", (6, 4)))
        self.xlim        = tuple(config.get("xlim", (-3, 3)))
        self.ylim        = tuple(config.get("ylim", (-2.5, 2.0)))
        self.invert_y    = bool(config.get("invert_y", True))

        # Matplotlib figure/axes 與 artist
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.ax.set_aspect('equal')
        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.ylim)
        if self.invert_y:
            self.ax.invert_yaxis()

        # 鎖骨
        self.ax.plot([-1, 1], [0, 0], '-', color='black', lw=2)

        # 骨架線條
        self.line_left_upper,  = self.ax.plot([], [], 'o-', lw=3, color='tab:blue')
        self.line_left_fore,   = self.ax.plot([], [], 'o-', lw=3, color='tab:orange')
        self.line_right_upper, = self.ax.plot([], [], 'o-', lw=3, color='tab:blue')
        self.line_right_fore,  = self.ax.plot([], [], 'o-', lw=3, color='tab:orange')

        # 角度文字
        self.text_left_shoulder  = self.ax.text(-1.2, 0.3, "", fontsize=8, color="red")
        self.text_right_shoulder = self.ax.text( 0.8, 0.3, "", fontsize=8, color="red")
        self.text_left_elbow     = self.ax.text(-1.8,-0.7, "", fontsize=8, color="green")
        self.text_right_elbow    = self.ax.text( 1.2,-0.7, "", fontsize=8, color="green")

        # 建立動畫物件佔位
        self.ani = None
        
    def minmax_01(self, x):
        xmin = np.min(x)
        xmax = np.max(x)
        denom = xmax - xmin
        return (x - xmin) / denom

    def _get_arm_coords(self, shoulder_angle_deg, elbow_angle_deg, origin, side="left"):
        """
        shoulder_angle_deg, elbow_angle_deg 皆為度數。
        left: 以 base=0，肩角度順時針為正（畫面 y 軸已 invert）。  
        right: 以 base=pi，肩角度順時針為正的對稱處理。
        """
        L_upper, L_fore = self.L_upper, self.L_fore

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
        elbow = shoulder + L_upper * np.array([np.cos(upper_dir), np.sin(upper_dir)])
        wrist = elbow   + L_fore  * np.array([np.cos(forearm_dir), np.sin(forearm_dir)])
        return shoulder, elbow, wrist

    def _init_artists(self):
        for ln in [self.line_left_upper, self.line_left_fore,
                   self.line_right_upper, self.line_right_fore]:
            ln.set_data([], [])
        self.text_left_shoulder.set_text("")
        self.text_right_shoulder.set_text("")
        self.text_left_elbow.set_text("")
        self.text_right_elbow.set_text("")
        return (self.line_left_upper, self.line_left_fore,
                self.line_right_upper, self.line_right_fore,
                self.text_left_shoulder, self.text_right_shoulder,
                self.text_left_elbow, self.text_right_elbow)

    def _update_frame(self, i):
        # 左手
        shoulder, elbow, wrist = self._get_arm_coords(
            self.left_shoulder_angles[i], self.left_elbow_angles[i],
            origin=(-1, self.left_shoulder_y[i]), side="left"
        )
        self.line_left_upper.set_data([shoulder[0], elbow[0]], [shoulder[1], elbow[1]])
        self.line_left_fore.set_data([elbow[0], wrist[0]], [elbow[1], wrist[1]])

        # 右手
        shoulder_r, elbow_r, wrist_r = self._get_arm_coords(
            self.right_shoulder_angles[i], self.right_elbow_angles[i],
            origin=(1, self.right_shoulder_y[i]), side="right"
        )
        self.line_right_upper.set_data([shoulder_r[0], elbow_r[0]], [shoulder_r[1], elbow_r[1]])
        self.line_right_fore.set_data([elbow_r[0], wrist_r[0]], [elbow_r[1], wrist_r[1]])

        # 角度文字與標題
        self.text_left_shoulder.set_text(f"L_shoulder: {self.left_shoulder_angles[i]:.1f}°")
        self.text_right_shoulder.set_text(f"R_shoulder: {self.right_shoulder_angles[i]:.1f}°")
        self.text_left_elbow.set_text(f"L_elbow: {self.left_elbow_angles[i]:.1f}°")
        self.text_right_elbow.set_text(f"R_elbow: {self.right_elbow_angles[i]:.1f}°")

        self.ax.set_title(f"Frame {self.frames[i]}")
        return (self.line_left_upper, self.line_left_fore,
                self.line_right_upper, self.line_right_fore,
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
        print(f"✅ 動畫已輸出完成：{os.path.abspath(output_file)}")
