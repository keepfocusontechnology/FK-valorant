import math
import os
import sys
import time
import json
import cv2
import torch
import random
import numpy as np
import win32con
from mss import mss
import tkinter as tk
from PIL import Image, ImageTk
import ctypes
import win32api
from filterpy.kalman import KalmanFilter

# 鼠标按键对应的虚拟键码
VK_LBUTTON = 0x01  # 左键
VK_RBUTTON = 0x02  # 右键
VK_MBUTTON = 0x04  # 中键

# 配置文件路径
CONFIG_FILE = "config.json"

# 默认配置参数
DEFAULT_CONFIG = {
    "sleep_time": 0.1,
    "click_time": 0.12,
    "display": False,
    "threshold": 0.3,
    "scale": 0.5,
    "size": 60,
    "recoil_comp": 8  # 新增：后坐力补偿像素
}


def load_config(config_file):
    """加载配置文件，如果不存在则返回默认配置"""
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
            # 如果配置文件缺少某个参数，补上默认值
            for key, value in DEFAULT_CONFIG.items():
                if key not in config:
                    config[key] = value
            return config
        except Exception as e:
            print("加载配置失败，使用默认配置。", e)
    return DEFAULT_CONFIG.copy()


def save_config(config_file, config):
    """保存配置到文件"""
    try:
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print("保存配置失败：", e)


class LGDriver:
    def __init__(self, dll_path, click_time):
        self.click_time = click_time
        self.lg_driver = ctypes.CDLL(dll_path)
        self.ok = self.lg_driver.device_open() == 1
        if not self.ok:
            raise Exception("驱动加载失败!")

    def press(self, code):
        """
        按下指定的鼠标按钮
        code: 1: 左键, 2: 中键, 3: 右键
        """
        if not self.ok:
            return
        self.lg_driver.mouse_down(code)

    def release(self, code):
        """
        释放指定的鼠标按钮
        code: 1: 左键, 2: 中键, 3: 右键
        """
        if not self.ok:
            return
        self.lg_driver.mouse_up(code)

    def click(self, code=1):
        """
        点击指定的鼠标按钮
        code: 1: 左键, 2: 中键, 3: 右键
        """
        if not self.ok:
            return
        self.lg_driver.mouse_down(code)
        # 兼容 tk.DoubleVar 和 float
        click_time = self.click_time.get() if hasattr(self.click_time, 'get') else float(self.click_time)
        # 在click_time基础上随机加减微小随机量
        time_variation = random.uniform(-0.02, 0.03)
        adjusted_click_time = max(click_time + time_variation, 0)
        self.microsecond_sleep(adjusted_click_time * 1000)
        self.lg_driver.mouse_up(code)

    def scroll(self, a):
        """
        滚动鼠标滚轮
        a: 滚动的距离
        """
        if not self.ok:
            return
        self.lg_driver.scroll(a)

    def move(self, x, y):
        """
        相对移动鼠标位置
        x: 水平移动的方向和距离, 正数向右, 负数向左
        y: 垂直移动的方向和距离
        """
        if not self.ok:
            return
        if x == 0 and y == 0:
            return
        self.lg_driver.moveR(int(x), int(y), True)


    @staticmethod
    def microsecond_sleep(sleep_time):
        """
        微秒级睡眠，使用 Windows 高精度计时器实现
        :param sleep_time: int, 微秒 (1e-6 秒)
        """
        kernel32 = ctypes.WinDLL('kernel32')
        freq = ctypes.c_int64()
        start = ctypes.c_int64()
        end = ctypes.c_int64()

        kernel32.QueryPerformanceFrequency(ctypes.byref(freq))
        kernel32.QueryPerformanceCounter(ctypes.byref(start))

        target_ticks = int(freq.value * sleep_time / 1e6)
        while True:
            kernel32.QueryPerformanceCounter(ctypes.byref(end))
            if end.value - start.value >= target_ticks:
                break

    def smooth_move(self, x, y, min_steps=5, max_steps=20, scale_factor=5):
        """
        平滑相对移动鼠标位置，步数根据移动距离动态计算，并避免移动偏差
        参数:
            x: 水平移动的总距离（正数向右，负数向左）
            y: 垂直移动的总距离（正数向下，负数向上）
            min_steps: 最小步数，默认 5
            max_steps: 最大步数，默认 20
            scale_factor: 距离缩放因子，用于调整步数与距离的敏感度，默认 5
            delay: 每步之间的延迟时间（秒），默认 0.01 秒
        """
        if not self.ok:
            return
        if x == 0 and y == 0:
            return

        # 计算总移动距离
        distance = math.sqrt(x ** 2 + y ** 2)

        # 动态计算步数
        steps = int(min(max(distance / scale_factor, min_steps), max_steps))

        # 初始化累积误差
        error_x = 0.0
        error_y = 0.0

        # 计算每步的理论移动量
        step_x_float = x / steps
        step_y_float = y / steps

        # 分步执行移动
        for _ in range(steps):
            # 累积浮点移动量
            error_x += step_x_float
            error_y += step_y_float

            # 计算当前步的整数移动量
            move_x = int(round(error_x))
            move_y = int(round(error_y))

            # 更新累积误差
            error_x -= move_x
            error_y -= move_y

            # 执行移动
            self.lg_driver.moveR(move_x, move_y, True)
            self.microsecond_sleep(2000)

def initialize_model_and_driver(click_time, retries=3, delay=5):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(__file__)
    repo_path = os.path.join(base_path, './yolov5-master')
    model_path = os.path.join(base_path, 'runs/train/exp3/weights/best.pt')
    driver_path = os.path.join(base_path, 'driver/logitech.driver.dll')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for attempt in range(retries):
        try:
            model = torch.hub.load(repo_or_dir=repo_path,
                                   model='custom',
                                   path=model_path,
                                   source='local').to(device)
            driver = LGDriver(driver_path, click_time)
            return model, driver
        except Exception as e:
            print(f"模型或驱动加载失败 (尝试 {attempt + 1}/{retries}): {e}")
            if "logitech.driver.dll" in str(e):
                print("提示：请确保安装了 driver/lghub 目录中的驱动程序。")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                return None, None


def create_control_panel(root, sleep_time_var, click_time, display_var, threshold, scale, size, tk_window, recoil_comp_var):
    # 只设置窗口位置，不强制宽高，让内容自适应
    root.geometry("+10+25")
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.attributes("-alpha", 0.96)  # 提高不透明度，颜色更深

    # 右上角关闭按钮，绝对定位
    def on_close():
        print("[GUI] 用户点击关闭按钮，程序退出。")
        root.quit()
        root.destroy()
        os._exit(0)
    close_btn = tk.Button(root, text="✕", command=on_close, bg="black", fg="red", borderwidth=0, font=("Arial", 14, "bold"), highlightthickness=0, relief="flat", cursor="hand2")
    close_btn.place(relx=1.0, rely=0.0, anchor="ne", x=0, y=0)
    close_btn.lift()

    frame = tk.Frame(root, bg="black")
    frame.pack(fill="both", expand=True, pady=(18, 12))  # 顶部留18，底部留12，保证内容完整

    label_config = {"fg": "white", "bg": "black", "font": ("Arial", 12)}
    button_config = {"bg": "black", "fg": "white", "font": ("Arial", 12)}

    labels = [
        ("弹道恢复:", sleep_time_var),
        ("射击时间:", click_time),
        ("识别精度:", threshold),
        ("瞄准范围:", size),
        ("窗口大小:", scale),
        ("后坐力补偿:", recoil_comp_var)
    ]

    # 参数标签及变量、步进、显示格式配置
    param_settings = [
        ("弹道恢复:", sleep_time_var, 0.01, 0.25, lambda v: 0.01 if v < 0.05 else 0.02, 2),
        ("射击时间:", click_time, 0.01, 0.25, lambda v: 0.01 if v < 0.05 else 0.02, 2),
        ("识别精度:", threshold, 0.1, 1.0, lambda v: 0.05 if v < 0.3 else 0.1, 2),
        ("瞄准范围:", size, 10, 200, lambda v: 2 if v < 40 else 10, 0),
        ("窗口大小:", scale, 0.1, 1.0, lambda v: 0.05 if v < 0.3 else 0.1, 2),
        ("后坐力补偿:", recoil_comp_var, 0, 30, lambda v: 1, 0),  # 新增后坐力补偿
    ]

    for i, (text, var, min_val, max_val, step_func, round_digits) in enumerate(param_settings):
        tk.Label(frame, text=text, **label_config).grid(row=i, column=1, padx=5, pady=5)
        # 优化显示格式
        def make_format_var(v, digits):
            fmt = f"%.{digits}f" if digits > 0 else "%d"
            return tk.Label(frame, textvariable=v, **label_config, anchor="w", width=6, justify="left")
        make_format_var(var, round_digits).grid(row=i, column=2, padx=5, pady=5)

        # 自适应步进按钮
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

    # 显示/隐藏按钮功能
    def toggle_display():
        display_var.set(not display_var.get())
        if not display_var.get():
            tk_window.withdraw()
        else:
            # 获取主控面板(root)的位置和高度
            root.update_idletasks()
            root_x = root.winfo_x()
            root_y = root.winfo_y()
            root_h = root.winfo_height()
            win_w = tk_window.winfo_width() if tk_window.winfo_width() > 1 else tk_window._last_size[0]
            win_h = tk_window.winfo_height() if tk_window.winfo_height() > 1 else tk_window._last_size[1]
            # 弹窗左对齐主控面板，纵向紧贴主控面板下方
            x = root_x
            y = root_y + root_h + 20  # 下方留20像素间隔
            tk_window.geometry(f"{win_w}x{win_h}+{x}+{y}")
            tk_window.deiconify()
            tk_window.lift()

    # 调整显示/隐藏按钮位置，放在参数区下方单独一行
    tk.Button(frame, text="显示/隐藏", command=toggle_display, **button_config).grid(row=len(param_settings), column=0, columnspan=4, padx=5, pady=(10, 5))


def create_tk_window(root, scale, capture_x=640, capture_y=480):
    width = int(capture_x * scale.get())
    height = int(capture_y * scale.get())

    tk_window = tk.Toplevel(root)
    tk_window.overrideredirect(True)
    tk_window.attributes("-topmost", True)
    tk_window.geometry(f"{width}x{height}+10+280")
    tk_window.attributes("-alpha", 0.98)  # 提高不透明度
    tk_window.withdraw()

    tk_window.img_label = tk.Label(tk_window)
    tk_window.img_label.pack(fill="both", expand=True)

    tk_window.fps_label = tk.Label(tk_window, text="FPS: 0", fg="white", bg="black")
    tk_window.fps_label.place(relx=0.1, rely=0.1, anchor=tk.CENTER)

    # 保存当前宽高，便于主循环动态调整
    tk_window._last_size = (width, height)
    tk_window._capture_x = capture_x
    tk_window._capture_y = capture_y

    return tk_window


def get_screen_center(monitor):
    screen_width = monitor['width']
    screen_height = monitor['height']
    return screen_width // 2, screen_height // 2


def capture_screen(sct, capture_area):
    screen_img = np.array(sct.grab(capture_area))
    return cv2.cvtColor(screen_img, cv2.COLOR_BGRA2BGR)


def detect_enemy(model, img, capture_x, capture_y, confidence_threshold):
    # 降低推理分辨率，加速
    results = model(img, size=320)
    detections = results.xyxy[0].cpu().numpy()

    # --- 识别与目标筛选 ---
    enemy_head_results = []
    enemy_results = []

    for *xyxy, conf, cls in detections:
        if conf < confidence_threshold:
            continue
        center_x = (xyxy[0] + xyxy[2]) / 2
        center_y = (xyxy[1] + xyxy[3]) / 2
        relative_x = center_x - capture_x // 2
        relative_y = center_y - capture_y // 2
        distance_to_center = np.sqrt(relative_x ** 2 + relative_y ** 2)
        if model.names[int(cls)] == 'enemy_head':
            enemy_head_results.append((relative_x, relative_y + 4, xyxy, conf, distance_to_center))
        elif model.names[int(cls)] == 'enemy':
            enemy_results.append((relative_x, relative_y, xyxy, conf, distance_to_center))

    # --- 筛选距离中心最近的头部和身体 ---
    closest_enemy_head = min(enemy_head_results, key=lambda x: x[4])[:4] if enemy_head_results else []
    closest_enemy = min(enemy_results, key=lambda x: x[4])[:4] if enemy_results else []

    return closest_enemy_head, closest_enemy


def perform_action(driver, relative_x, relative_y, sleep_time, size, head_xyxy, detect_end_time=None, recoil_comp=8):
    abs_x = abs(relative_x)
    abs_y = abs(relative_y)
    xyxy = head_xyxy
    x1, y1, x2, y2 = xyxy
    xx = x2 - x1

    delta_size = size * (xx / 13)
    m_x = abs((x2 - x1) / 2)
    m_y = abs((y2 - y1) / 2)

    fire_time = None
    if abs_x < m_x and abs_y < m_y:
        fire_time = time.time()
        driver.click()
        driver.move(0, -recoil_comp)  # 新增后坐力补偿，负值向下
        time.sleep(sleep_time)
    else:
        if abs_x <= delta_size and abs_y <= delta_size:
            driver.move(relative_x, relative_y)
            fire_time = time.time()
            driver.click()
            driver.move(0, -recoil_comp)  # 新增后坐力补偿
            time.sleep(sleep_time)
    if detect_end_time is not None and fire_time is not None:
        return fire_time
    return None


def perform_action_body(driver, relative_x, relative_y, sleep_time, size, body_xyxy, detect_end_time=None):
    x1, y1, x2, y2 = body_xyxy
    delta_y = y2 - y1
    adjustment_factor = 0.34 + 0.1 * (1 - math.exp(-0.01 * (delta_y - 50)))
    relative_y -= delta_y * adjustment_factor

    abs_x = abs(relative_x)
    abs_y = abs(relative_y)
    xx = x2 - x1
    delta_size = size * (xx / 50)

    fire_time = None
    if abs_x <= delta_size and abs_y <= delta_size:
        driver.move(relative_x, relative_y)
        fire_time = time.time()
        driver.click()
        time.sleep(sleep_time)
    if detect_end_time is not None and fire_time is not None:
        return fire_time
    return None


def display_image_with_detections(img, closest_enemy_head, closest_enemy, scale, tk_window):
    if closest_enemy_head:
        label_head = "Enemy Head"
        _, _, xyxy, conf = closest_enemy_head
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, f"{label_head} : {conf:0.3}", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

    if closest_enemy:
        label_enemy = "Enemy"
        _, _, xyxy, conf = closest_enemy
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, f"{label_enemy} : {conf:0.3}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)

    height, width = img.shape[:2]
    resized_img = cv2.resize(img, (int(width * scale), int(height * scale)))
    image = Image.fromarray(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
    tk_img = ImageTk.PhotoImage(image)
    tk_window.img_label.config(image=tk_img)
    tk_window.img_label.image = tk_img


def create_kalman_filter(init_x, init_y):
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([init_x, init_y, 0, 0], dtype=float)
    kf.F = np.array([[1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]], dtype=float)
    kf.H = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0]], dtype=float)
    kf.P *= 10
    kf.R = np.eye(2) * 0.05  # 观测噪声
    kf.Q = np.eye(4) * 0.2   # 过程噪声
    return kf


def main():
    root = tk.Tk()
    # 加载配置文件
    config = load_config(CONFIG_FILE)
    # 根据配置初始化各个参数
    sleep_time_var = tk.DoubleVar(value=config.get("sleep_time", DEFAULT_CONFIG["sleep_time"]))
    click_time = tk.DoubleVar(value=config.get("click_time", DEFAULT_CONFIG["click_time"]))
    display_var = tk.BooleanVar(value=config.get("display", DEFAULT_CONFIG["display"]))
    threshold = tk.DoubleVar(value=config.get("threshold", DEFAULT_CONFIG["threshold"]))
    scale = tk.DoubleVar(value=config.get("scale", DEFAULT_CONFIG["scale"]))
    size = tk.DoubleVar(value=config.get("size", DEFAULT_CONFIG["size"]))
    recoil_comp_var = tk.IntVar(value=config.get("recoil_comp", DEFAULT_CONFIG["recoil_comp"]))  # 新增

    # 定义当参数发生变化时自动保存到配置文件的回调函数
    def update_config(*args):
        config["sleep_time"] = sleep_time_var.get()
        config["click_time"] = click_time.get()
        config["display"] = display_var.get()
        config["threshold"] = threshold.get()
        config["scale"] = scale.get()
        config["size"] = size.get()
        config["recoil_comp"] = recoil_comp_var.get()  # 新增
        save_config(CONFIG_FILE, config)

    # 为变量添加 trace，当值发生变化时触发保存
    sleep_time_var.trace_add("write", update_config)
    click_time.trace_add("write", update_config)
    display_var.trace_add("write", update_config)
    threshold.trace_add("write", update_config)
    scale.trace_add("write", update_config)
    size.trace_add("write", update_config)
    recoil_comp_var.trace_add("write", update_config)  # 新增

    capture_x = 480
    capture_y = 360

    tk_window = create_tk_window(root, scale, capture_x, capture_y)
    control_panel_visible = True
    create_control_panel(root, sleep_time_var, click_time, display_var, threshold, scale, size, tk_window, recoil_comp_var)

    model, driver = initialize_model_and_driver(click_time)

    previous_scale = scale.get()

    # 初始化 mss 截图对象
    sct = mss()
    # 获取主显示器信息
    monitor = sct.monitors[1]  # 1为主屏
    screen_cx, screen_cy = get_screen_center(monitor)
    # 计算初始截图区域
    def get_capture_area():
        x = int(screen_cx - capture_x // 2)
        y = int(screen_cy - capture_y // 2)
        return {"top": y, "left": x, "width": capture_x, "height": capture_y}
    capture_area = get_capture_area()

    # 当窗口缩放或截图分辨率变化时，动态更新截图区域
    def update_capture_area():
        nonlocal capture_area
        capture_area = get_capture_area()

    # 若后续有动态调整capture_x/capture_y/scale，可在对应位置调用update_capture_area()

    # 设定目标帧率为60fps
    target_fps = 60
    frame_interval = 1.0 / target_fps

    fps_state = {'last_time': time.time(), 'count': 0}
    fps_update_interval = 1.0
    frame_count = 0  # 新增帧计数
    last_print_time = time.time()  # 控制print频率

    kalman = None
    last_head = None
    last_head_xyxy = None
    lost_count = 0
    max_lost = 5  # 允许丢失帧数

    while True:
        loop_start = time.time()  # 循环开始计时

        fps_state['count'] += 1
        frame_count += 1
        if loop_start - fps_state['last_time'] >= fps_update_interval:
            elapsed_time = loop_start - fps_state['last_time']
            current_fps = fps_state['count'] / elapsed_time if elapsed_time > 0 else 0
            if tk_window and hasattr(tk_window, 'fps_label'):
                tk_window.fps_label.config(text=f"FPS: {current_fps:.1f}")
            fps_state['count'] = 0
            fps_state['last_time'] = loop_start

        current_scale = scale.get()
        if current_scale != previous_scale:
            width = int(capture_x * current_scale)
            height = int(capture_y * current_scale)
            tk_window.geometry(f"{width}x{height}+10+280")
            tk_window._last_size = (width, height)
            previous_scale = current_scale

        img_start = time.time()
        # 将右键检测改为~键（VK_OEM_3）
        if win32api.GetAsyncKeyState(0xC0) < 0:  # 0xC0为~键（美式键盘VK_OEM_3）
            img = capture_screen(sct, capture_area)
            img_end = time.time()
            det_start = time.time()
            closest_enemy_head, closest_enemy = detect_enemy(model, img, capture_x, capture_y, threshold.get())
            det_end = time.time()
            fire_time = None
            # 卡尔曼滤波优化 + 动态吸附区 + Q/R自适应 + 平滑插值
            if closest_enemy_head and len(closest_enemy_head) > 2:
                head_x, head_y, head_xyxy, conf = closest_enemy_head[:4]
                # 判断是否为同一目标（IOU或中心点距离）
                is_same = False
                if last_head_xyxy is not None:
                    x1, y1, x2, y2 = head_xyxy
                    lx1, ly1, lx2, ly2 = last_head_xyxy
                    iou = (max(0, min(x2, lx2) - max(x1, lx1)) * max(0, min(y2, ly2) - max(y1, ly1))) / (
                        (x2 - x1) * (y2 - y1) + (lx2 - lx1) * (ly2 - ly1) - max(0, min(x2, lx2) - max(x1, lx1)) * max(0, min(y2, ly2) - max(y1, ly1)) + 1e-6)
                    dist = np.linalg.norm([head_x - last_head[0], head_y - last_head[1]])
                    if iou > 0.2 or dist < 30:
                        is_same = True
                # 计算目标速度
                velocity = 0
                if last_head is not None:
                    velocity = np.linalg.norm([head_x - last_head[0], head_y - last_head[1]])
                # Q/R自适应调整
                Q_base, R_base = 0.2, 0.05
                Q = Q_base + min(velocity / 30, 2.0)  # 速度越大Q越大
                R = R_base + min(velocity / 60, 0.2)  # 速度越大R略增
                # 吸附区半径自适应
                base_radius = 28
                radius = base_radius + min(velocity * 0.8, 40)  # 速度越快吸附区越大
                if kalman is None or not is_same:
                    kalman = create_kalman_filter(head_x, head_y)
                    lost_count = 0
                else:
                    kalman.Q = np.eye(4) * Q
                    kalman.R = np.eye(2) * R
                    kalman.predict()
                    kalman.update([head_x, head_y])
                kalman.predict()  # 再次预测，获取下一时刻预测值
                pred_x, pred_y = kalman.x[0], kalman.x[1]
                # 计算预测点与头部中心距离
                pred_dist = np.linalg.norm([pred_x - head_x, pred_y - head_y])
                # 吸附区机制+平滑插值
                if pred_dist < radius:
                    aim_x, aim_y = pred_x, pred_y
                    absorb_state = '吸附'
                else:
                    # 区外平滑插值靠近预测点
                    alpha = min(0.25 + velocity/60, 0.7)  # 插值系数随速度增大
                    aim_x = (1-alpha)*head_x + alpha*pred_x
                    aim_y = (1-alpha)*head_y + alpha*pred_y
                    absorb_state = '插值靠近'
                # 日志输出
                if time.time() - last_print_time > 0.5:
                    print(f"[KF] v={velocity:.1f} Q={Q:.2f} R={R:.2f} 吸附区r={radius:.1f} 状态:{absorb_state} 原:({head_x:.1f},{head_y:.1f}) 预测:({pred_x:.1f},{pred_y:.1f}) 瞄准:({aim_x:.1f},{aim_y:.1f}) 距离:{pred_dist:.1f}")
                # 用aim_x, aim_y瞄准
                fire_time = perform_action(driver, aim_x, aim_y, sleep_time_var.get(), size.get(), head_xyxy, detect_end_time=det_end, recoil_comp=recoil_comp_var.get())
                last_head = (head_x, head_y)
                last_head_xyxy = head_xyxy
                lost_count = 0
                if time.time() - last_print_time > 0.5:
                    delay_ms = (fire_time - det_end) * 1000 if fire_time else None
                    delay_str = f"{delay_ms:.1f}ms" if delay_ms is not None else "--"
                    print(f"[右键] 抓图: {img_end-img_start:.3f}s, 检测: {det_end-det_start:.3f}s, 检测到开火: {delay_str}, 总: {det_end-loop_start:.3f}s")
                    last_print_time = time.time()
                continue
            else:
                lost_count += 1
                if lost_count > max_lost:
                    kalman = None
                    last_head = None
                    last_head_xyxy = None
            if closest_enemy and len(closest_enemy) > 2:
                fire_time = perform_action_body(driver, *closest_enemy[:2], sleep_time_var.get(), size.get(), closest_enemy[2], detect_end_time=det_end)
                if time.time() - last_print_time > 1.0:
                    delay_ms = (fire_time - det_end) * 1000 if fire_time else None
                    delay_str = f"{delay_ms:.1f}ms" if delay_ms is not None else "--"
                    print(f"[右键] 抓图: {img_end-img_start:.3f}s, 检测: {det_end-det_start:.3f}s, 检测到开火: {delay_str}, 总: {det_end-loop_start:.3f}s")
                    last_print_time = time.time()

        if win32api.GetAsyncKeyState(win32con.VK_F6) < 0:
            click_time.set(0.01)
            size.set(80)

        if win32api.GetAsyncKeyState(win32con.VK_F5) < 0:
            click_time.set(0.12)
            size.set(60)

        if win32api.GetAsyncKeyState(win32con.VK_HOME) < 0:
            if control_panel_visible:
                root.withdraw()
            else:
                root.deiconify()
            control_panel_visible = not control_panel_visible
            time.sleep(0.2)

        # Tk窗口每2帧刷新一次，减少PIL+Tk转换频率
        if display_var.get() and frame_count % 2 == 0:
            img = capture_screen(sct, capture_area)
            img_end = time.time()
            det_start = time.time()
            closest_enemy_head, closest_enemy = detect_enemy(model, img, capture_x, capture_y, threshold.get())
            det_end = time.time()
            disp_start = time.time()
            display_image_with_detections(img, closest_enemy_head, closest_enemy, scale.get(), tk_window)
            disp_end = time.time()
            if time.time() - last_print_time > 1.0:
                print(f"[显示] 抓图: {img_end-img_start:.3f}s, 检测: {det_end-det_start:.3f}s, 显示: {disp_end-disp_start:.3f}s, 总: {disp_end-loop_start:.3f}s")
                last_print_time = time.time()

        if (win32api.GetAsyncKeyState(win32con.VK_SHIFT) < 0 and
                win32api.GetAsyncKeyState(win32con.VK_ESCAPE) < 0):
            print("退出程序中...")
            break

        root.update_idletasks()
        root.update()

        # 注释掉帧率sleep，测试极限FPS
        # elapsed_time = time.time() - loop_start
        # remaining_time = frame_interval - elapsed_time
        # if remaining_time > 0:
        #     time.sleep(remaining_time)

if __name__ == "__main__":
    print("主进程PID:", os.getpid())
    main()
