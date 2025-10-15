import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import os, json
# === 手臂計算函式 ===
L_upper, L_fore = 1.0, 1.0


def get_arm_coords(shoulder_angle, elbow_angle, origin, side="left"):
    if side == "left":
        base = 0.0
        upper_dir = base - np.deg2rad(shoulder_angle)
        bend = np.pi - np.deg2rad(elbow_angle)
        forearm_dir = upper_dir + bend
    else:
        base = np.pi
        upper_dir = base + np.deg2rad(shoulder_angle)
        bend = np.pi - np.deg2rad(elbow_angle)
        forearm_dir = upper_dir - bend
    shoulder = np.array(origin)
    elbow = shoulder + L_upper * np.array([np.cos(upper_dir), np.sin(upper_dir)])
    wrist = elbow + L_fore  * np.array([np.cos(forearm_dir), np.sin(forearm_dir)])
    return shoulder, elbow, wrist

# === 畫布設定 ===
fig, ax = plt.subplots(figsize=(6,4))
ax.set_aspect('equal')
ax.set_xlim(-3, 3)
ax.set_ylim(-2.5, 2.0)
ax.invert_yaxis()   # ← 上下翻轉畫面


# 手臂骨架線條
line_left_upper, = ax.plot([], [], 'o-', lw=3, color='tab:blue')
line_left_fore,  = ax.plot([], [], 'o-', lw=3, color='tab:orange')
line_right_upper, = ax.plot([], [], 'o-', lw=3, color='tab:blue')
line_right_fore,  = ax.plot([], [], 'o-', lw=3, color='tab:orange')

# 角度文字
text_left_shoulder  = ax.text(-1.2, 0.3, "", fontsize=8, color="red")
text_right_shoulder = ax.text( 0.8, 0.3, "", fontsize=8, color="red")
text_left_elbow     = ax.text(-1.8,-0.7, "", fontsize=8, color="green")
text_right_elbow    = ax.text( 1.2,-0.7, "", fontsize=8, color="green")

# 鎖骨
ax.plot([-1, 1], [0, 0], '-', color='black', lw=2)

# === 初始化 ===
def init():
    for ln in [line_left_upper, line_left_fore, line_right_upper, line_right_fore]:
        ln.set_data([], [])
    text_left_shoulder.set_text("")
    text_right_shoulder.set_text("")
    text_left_elbow.set_text("")
    text_right_elbow.set_text("")
    return line_left_upper, line_left_fore, line_right_upper, line_right_fore, \
           text_left_shoulder, text_right_shoulder, text_left_elbow, text_right_elbow

# === 更新每一幀 ===
def update(i):
    # 左手
    shoulder, elbow, wrist = get_arm_coords(left_shoulder_angles[i], left_elbow_angles[i], origin=(-1,0), side="left")
    line_left_upper.set_data([shoulder[0], elbow[0]], [shoulder[1], elbow[1]])
    line_left_fore.set_data([elbow[0], wrist[0]], [elbow[1], wrist[1]])

    # 右手
    shoulder_r, elbow_r, wrist_r = get_arm_coords(right_shoulder_angles[i], right_elbow_angles[i], origin=(1,0), side="right")
    line_right_upper.set_data([shoulder_r[0], elbow_r[0]], [shoulder_r[1], elbow_r[1]])
    line_right_fore.set_data([elbow_r[0], wrist_r[0]], [elbow_r[1], wrist_r[1]])

    # 更新角度文字
    text_left_shoulder.set_text(f"L_shoulder: {left_shoulder_angles[i]:.1f}°")
    text_right_shoulder.set_text(f"R_shoulder: {right_shoulder_angles[i]:.1f}°")
    text_left_elbow.set_text(f"L_elbow: {left_elbow_angles[i]:.1f}°")
    text_right_elbow.set_text(f"R_elbow: {right_elbow_angles[i]:.1f}°")

    ax.set_title(f"Frame {frames[i]}")
    return line_left_upper, line_left_fore, line_right_upper, line_right_fore, \
           text_left_shoulder, text_right_shoulder, text_left_elbow, text_right_elbow

# === 建立動畫 ===
ani = animation.FuncAnimation(fig, update, frames=len(frames),
                              init_func=init, blit=True, interval=50)

# === 輸出影片 ===
output_file = "/home/jeter/frame/arms_motion.mp4"
ani.save(output_file, writer="ffmpeg", fps=30)
print(f"✅ 動畫已輸出完成：{os.path.abspath(output_file)}")