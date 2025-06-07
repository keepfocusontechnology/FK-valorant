import os
import json
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import time
import win32api
import win32con
from mss import mss
from filterpy.kalman import KalmanFilter
from driver import LGDriver

def load_config(config_file):
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
            for key, value in DEFAULT_CONFIG.items():
                if key not in config:
                    config[key] = value
            return config
        except Exception as e:
            print("加载配置失败，使用默认配置。", e)
    return DEFAULT_CONFIG.copy()

def save_config(config_file, config):
    try:
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print("保存配置失败：", e)

CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "sleep_time": 0.1,
    "click_time": 0.12,
    "display": False,
    "threshold": 0.3,
    "scale": 0.5,
    "size": 60,
    "recoil_comp": 8
}

def create_tk_window(root, scale, capture_x=640, capture_y=480):
    width = int(capture_x * scale.get())
    height = int(capture_y * scale.get())
    tk_window = tk.Toplevel(root)
    tk_window.overrideredirect(True)
    tk_window.attributes("-topmost", True)
    tk_window.geometry(f"{width}x{height}+10+280")
    tk_window.attributes("-alpha", 0.98)
    tk_window.withdraw()
    tk_window.img_label = tk.Label(tk_window)
    tk_window.img_label.pack(fill="both", expand=True)
    tk_window.fps_label = tk.Label(tk_window, text="FPS: 0", fg="white", bg="black")
    tk_window.fps_label.place(relx=0.1, rely=0.1, anchor=tk.CENTER)
    tk_window._last_size = (width, height)
    tk_window._capture_x = capture_x
    tk_window._capture_y = capture_y
    return tk_window

def create_control_panel(root, sleep_time_var, click_time, display_var, threshold, scale, size, tk_window, recoil_comp_var):
    root.geometry("+10+25")
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.attributes("-alpha", 0.96)
    def on_close():
        print("[GUI] 用户点击关闭按钮，程序退出。")
        root.quit()
        root.destroy()
        os._exit(0)
    close_btn = tk.Button(root, text="✕", command=on_close, bg="black", fg="red", borderwidth=0, font=("Arial", 14, "bold"), highlightthickness=0, relief="flat", cursor="hand2")
    close_btn.place(relx=1.0, rely=0.0, anchor="ne", x=0, y=0)
    close_btn.lift()
    frame = tk.Frame(root, bg="black")
    frame.pack(fill="both", expand=True, pady=(18, 12))
    label_config = {"fg": "white", "bg": "black", "font": ("Arial", 12)}
    button_config = {"bg": "black", "fg": "white", "font": ("Arial", 12)}
    param_settings = [
        ("弹道恢复:", sleep_time_var, 0.01, 0.25, lambda v: 0.01 if v < 0.05 else 0.02, 2),
        ("射击时间:", click_time, 0.01, 0.25, lambda v: 0.01 if v < 0.05 else 0.02, 2),
        ("识别精度:", threshold, 0.1, 1.0, lambda v: 0.05 if v < 0.3 else 0.1, 2),
        ("瞄准范围:", size, 10, 200, lambda v: 2 if v < 40 else 10, 0),
        ("窗口大小:", scale, 0.1, 1.0, lambda v: 0.05 if v < 0.3 else 0.1, 2),
        ("后坐力补偿:", recoil_comp_var, 0, 30, lambda v: 1, 0),
    ]
    for i, (text, var, min_val, max_val, step_func, round_digits) in enumerate(param_settings):
        tk.Label(frame, text=text, **label_config).grid(row=i, column=1, padx=5, pady=5)
        def make_format_var(v, digits):
            fmt = f"%.{digits}f" if digits > 0 else "%d"
            return tk.Label(frame, textvariable=v, **label_config, anchor="w", width=6, justify="left")
        make_format_var(var, round_digits).grid(row=i, column=2, padx=5, pady=5)
        def make_inc_dec(var, min_val, max_val, step_func, round_digits):
            def inc():
                val = var.get()
                step = step_func(val)
                new_val = min(round(val + step, round_digits), max_val)
                var.set(new_val)
            def dec():
                val = var.get()
                step = step_func(val)
                new_val = max(round(val - step, round_digits), min_val)
                var.set(new_val)
            return inc, dec
        inc, dec = make_inc_dec(var, min_val, max_val, step_func, round_digits)
        tk.Button(frame, text=" + ", command=inc, **button_config).grid(row=i, column=3, padx=5, pady=5)
        tk.Button(frame, text=" - ", command=dec, **button_config).grid(row=i, column=0, padx=5, pady=5)
    def toggle_display():
        display_var.set(not display_var.get())
        if not display_var.get():
            tk_window.withdraw()
        else:
            root.update_idletasks()
            root_x = root.winfo_x()
            root_y = root.winfo_y()
            root_h = root.winfo_height()
            win_w = tk_window.winfo_width() if tk_window.winfo_width() > 1 else tk_window._last_size[0]
            win_h = tk_window.winfo_height() if tk_window.winfo_height() > 1 else tk_window._last_size[1]
            x = root_x
            y = root_y + root_h + 20
            tk_window.geometry(f"{win_w}x{win_h}+{x}+{y}")
            tk_window.deiconify()
            tk_window.lift()
    tk.Button(frame, text="显示/隐藏", command=toggle_display, **button_config).grid(row=len(param_settings), column=0, columnspan=4, padx=5, pady=(10, 5))

class ControlPanelApp:
    """
    UI主类，负责参数、窗口、控件、配置、回调等，逻辑与UI分离
    """
    def __init__(self):
        self.root = tk.Tk()
        self.config = load_config(CONFIG_FILE)
        # 参数变量
        self.sleep_time_var = tk.DoubleVar(value=self.config.get("sleep_time", DEFAULT_CONFIG["sleep_time"]))
        self.click_time = tk.DoubleVar(value=self.config.get("click_time", DEFAULT_CONFIG["click_time"]))
        self.display_var = tk.BooleanVar(value=self.config.get("display", DEFAULT_CONFIG["display"]))
        self.threshold = tk.DoubleVar(value=self.config.get("threshold", DEFAULT_CONFIG["threshold"]))
        self.scale = tk.DoubleVar(value=self.config.get("scale", DEFAULT_CONFIG["scale"]))
        self.size = tk.DoubleVar(value=self.config.get("size", DEFAULT_CONFIG["size"]))
        self.recoil_comp_var = tk.IntVar(value=self.config.get("recoil_comp", DEFAULT_CONFIG["recoil_comp"]))
        # 绑定trace自动保存
        self._bind_traces()
        # UI窗口
        self.tk_window = create_tk_window(self.root, self.scale, 480, 360)
        self.control_panel_visible = True
        create_control_panel(self.root, self.sleep_time_var, self.click_time, self.display_var, self.threshold, self.scale, self.size, self.tk_window, self.recoil_comp_var)
        # 入口参数
        self.capture_x = 480
        self.capture_y = 360
        # 可选：自动启动检测主循环
        self.root.after(100, self.start_detection_loop)

    def _bind_traces(self):
        def update_config(*args):
            self.config["sleep_time"] = self.sleep_time_var.get()
            self.config["click_time"] = self.click_time.get()
            self.config["display"] = self.display_var.get()
            self.config["threshold"] = self.threshold.get()
            self.config["scale"] = self.scale.get()
            self.config["size"] = self.size.get()
            self.config["recoil_comp"] = self.recoil_comp_var.get()
            save_config(CONFIG_FILE, self.config)
        self.sleep_time_var.trace_add("write", update_config)
        self.click_time.trace_add("write", update_config)
        self.display_var.trace_add("write", update_config)
        self.threshold.trace_add("write", update_config)
        self.scale.trace_add("write", update_config)
        self.size.trace_add("write", update_config)
        self.recoil_comp_var.trace_add("write", update_config)

    def run(self):
        self.root.mainloop()

    def start_detection_loop(self):
        # import time, win32api, win32con, numpy as np
        # from mss import mss
        # from filterpy.kalman import KalmanFilter
        # from driver import LGDriver
        # from wwqy import initialize_model_and_driver, detect_enemy, perform_action, perform_action_body, display_image_with_detections, create_kalman_filter
        # 初始化模型和驱动
        model, driver = initialize_model_and_driver(self.click_time)
        capture_x = self.capture_x
        capture_y = self.capture_y
        sct = mss()
        monitor = sct.monitors[1]
        screen_cx = monitor['width'] // 2
        screen_cy = monitor['height'] // 2
        def get_capture_area():
            x = int(screen_cx - capture_x // 2)
            y = int(screen_cy - capture_y // 2)
            return {"top": y, "left": x, "width": capture_x, "height": capture_y}
        capture_area = get_capture_area()
        previous_scale = self.scale.get()
        fps_state = {'last_time': time.time(), 'count': 0}
        fps_update_interval = 1.0
        frame_count = 0
        last_print_time = time.time()
        kalman = None
        last_head = None
        last_head_xyxy = None
        lost_count = 0
        max_lost = 5
        def loop():
            nonlocal capture_area, previous_scale, frame_count, last_print_time, kalman, last_head, last_head_xyxy, lost_count
            loop_start = time.time()
            fps_state['count'] += 1
            frame_count += 1
            if loop_start - fps_state['last_time'] >= fps_update_interval:
                elapsed_time = loop_start - fps_state['last_time']
                current_fps = fps_state['count'] / elapsed_time if elapsed_time > 0 else 0
                if self.tk_window and hasattr(self.tk_window, 'fps_label'):
                    self.tk_window.fps_label.config(text=f"FPS: {current_fps:.1f}")
                fps_state['count'] = 0
                fps_state['last_time'] = loop_start
            current_scale = self.scale.get()
            if current_scale != previous_scale:
                width = int(capture_x * current_scale)
                height = int(capture_y * current_scale)
                self.tk_window.geometry(f"{width}x{height}+10+280")
                self.tk_window._last_size = (width, height)
                previous_scale = current_scale
            img_start = time.time()
            if win32api.GetAsyncKeyState(0xC0) < 0:
                img = np.array(sct.grab(capture_area))
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                img_end = time.time()
                det_start = time.time()
                closest_enemy_head, closest_enemy = detect_enemy(model, img, capture_x, capture_y, self.threshold.get())
                det_end = time.time()
                fire_time = None
                if closest_enemy_head and len(closest_enemy_head) > 2:
                    head_x, head_y, head_xyxy, conf = closest_enemy_head[:4]
                    is_same = False
                    if last_head_xyxy is not None:
                        x1, y1, x2, y2 = head_xyxy
                        lx1, ly1, lx2, ly2 = last_head_xyxy
                        iou = (max(0, min(x2, lx2) - max(x1, lx1)) * max(0, min(y2, ly2) - max(y1, ly1))) / (
                            (x2 - x1) * (y2 - y1) + (lx2 - lx1) * (ly2 - ly1) - max(0, min(x2, lx2) - max(x1, lx1)) * max(0, min(y2, ly2) - max(y1, ly1)) + 1e-6)
                        dist = np.linalg.norm([head_x - last_head[0], head_y - last_head[1]])
                        if iou > 0.2 or dist < 30:
                            is_same = True
                    velocity = 0
                    if last_head is not None:
                        velocity = np.linalg.norm([head_x - last_head[0], head_y - last_head[1]])
                    Q_base, R_base = 0.2, 0.05
                    Q = Q_base + min(velocity / 30, 2.0)
                    R = R_base + min(velocity / 60, 0.2)
                    base_radius = 28
                    radius = base_radius + min(velocity * 0.8, 40)
                    if kalman is None or not is_same:
                        kalman = create_kalman_filter(head_x, head_y)
                        lost_count = 0
                    else:
                        kalman.Q = np.eye(4) * Q
                        kalman.R = np.eye(2) * R
                        kalman.predict()
                        kalman.update([head_x, head_y])
                    kalman.predict()
                    pred_x, pred_y = kalman.x[0], kalman.x[1]
                    pred_dist = np.linalg.norm([pred_x - head_x, pred_y - head_y])
                    if pred_dist < radius:
                        aim_x, aim_y = pred_x, pred_y
                        absorb_state = '吸附'
                    else:
                        alpha = min(0.25 + velocity/60, 0.7)
                        aim_x = (1-alpha)*head_x + alpha*pred_x
                        aim_y = (1-alpha)*head_y + alpha*pred_y
                        absorb_state = '插值靠近'
                    if time.time() - last_print_time > 0.5:
                        print(f"[KF] v={velocity:.1f} Q={Q:.2f} R={R:.2f} 吸附区r={radius:.1f} 状态:{absorb_state} 原:({head_x:.1f},{head_y:.1f}) 预测:({pred_x:.1f},{pred_y:.1f}) 瞄准:({aim_x:.1f},{aim_y:.1f}) 距离:{pred_dist:.1f}")
                    fire_time = perform_action(driver, aim_x, aim_y, self.sleep_time_var.get(), self.size.get(), head_xyxy, detect_end_time=det_end, recoil_comp=self.recoil_comp_var.get())
                    last_head = (head_x, head_y)
                    last_head_xyxy = head_xyxy
                    lost_count = 0
                    if time.time() - last_print_time > 0.5:
                        delay_ms = (fire_time - det_end) * 1000 if fire_time else None
                        delay_str = f"{delay_ms:.1f}ms" if delay_ms is not None else "--"
                        print(f"[右键] 抓图: {img_end-img_start:.3f}s, 检测: {det_end-det_start:.3f}s, 检测到开火: {delay_str}, 总: {det_end-loop_start:.3f}s")
                        last_print_time = time.time()
                else:
                    lost_count += 1
                    if lost_count > max_lost:
                        kalman = None
                        last_head = None
                        last_head_xyxy = None
                if closest_enemy and len(closest_enemy) > 2:
                    fire_time = perform_action_body(driver, *closest_enemy[:2], self.sleep_time_var.get(), self.size.get(), closest_enemy[2], detect_end_time=det_end)
                    if time.time() - last_print_time > 1.0:
                        delay_ms = (fire_time - det_end) * 1000 if fire_time else None
                        delay_str = f"{delay_ms:.1f}ms" if delay_ms is not None else "--"
                        print(f"[右键] 抓图: {img_end-img_start:.3f}s, 检测: {det_end-det_start:.3f}s, 检测到开火: {delay_str}, 总: {det_end-loop_start:.3f}s")
                        last_print_time = time.time()
            if self.display_var.get() and frame_count % 2 == 0:
                img = np.array(sct.grab(capture_area))
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                img_end = time.time()
                det_start = time.time()
                closest_enemy_head, closest_enemy = detect_enemy(model, img, capture_x, capture_y, self.threshold.get())
                det_end = time.time()
                disp_start = time.time()
                display_image_with_detections(img, closest_enemy_head, closest_enemy, self.scale.get(), self.tk_window)
                disp_end = time.time()
                if time.time() - last_print_time > 1.0:
                    print(f"[显示] 抓图: {img_end-img_start:.3f}s, 检测: {det_end-det_start:.3f}s, 显示: {disp_end-disp_start:.3f}s, 总: {disp_end-loop_start:.3f}s")
                    last_print_time = time.time()
            if (win32api.GetAsyncKeyState(win32con.VK_SHIFT) < 0 and win32api.GetAsyncKeyState(win32con.VK_ESCAPE) < 0):
                print("退出程序中...")
                return
            self.root.after(1, loop)
        loop()
