import torch
import numpy as np
import math
import os
import sys
import time
import cv2
import random
import win32con
from mss import mss
import ctypes
import win32api
from filterpy.kalman import KalmanFilter
from driver import LGDriver
from control_panel import ControlPanelApp
# 只在需要的地方临时import Image, ImageTk
from PIL import Image, ImageTk

# 启用cudnn benchmark加速卷积
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# 鼠标按键对应的虚拟键码
VK_LBUTTON = 0x01  # 左键
VK_RBUTTON = 0x02  # 右键
VK_MBUTTON = 0x04  # 中键


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


def detect_enemy(model, img, capture_x, capture_y, confidence_threshold):
    # 保证推理在GPU上
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(img, np.ndarray):
        img_tensor = torch.from_numpy(img).to(device)
    else:
        img_tensor = img.to(device)
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

    # --- 只保留距离中心最近的头部和身体目标，避免多余排序 ---
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


# 移除原ControlPanelApp类定义
if __name__ == "__main__":
    print("主进程PID:", os.getpid())
    app = ControlPanelApp()
    def main_loop():
        # 主检测/推理/射击循环
        model, driver = initialize_model_and_driver(app.click_time)
        capture_x = app.capture_x
        capture_y = app.capture_y
        sct = mss()
        monitor = sct.monitors[1]
        screen_cx = monitor['width'] // 2
        screen_cy = monitor['height'] // 2
        def get_capture_area():
            x = int(screen_cx - capture_x // 2)
            y = int(screen_cy - capture_y // 2)
            return {"top": y, "left": x, "width": capture_x, "height": capture_y}
        capture_area = get_capture_area()
        previous_scale = app.scale.get()
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
                if app.tk_window and hasattr(app.tk_window, 'fps_label'):
                    app.tk_window.fps_label.config(text=f"FPS: {current_fps:.1f}")
                fps_state['count'] = 0
                fps_state['last_time'] = loop_start
            current_scale = app.scale.get()
            if current_scale != previous_scale:
                width = int(capture_x * current_scale)
                height = int(capture_y * current_scale)
                app.tk_window.geometry(f"{width}x{height}+10+280")
                app.tk_window._last_size = (width, height)
                previous_scale = current_scale
            img_start = time.time()
            if win32api.GetAsyncKeyState(0xC0) < 0:
                img = np.array(sct.grab(capture_area))
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                img_end = time.time()
                det_start = time.time()
                closest_enemy_head, closest_enemy = detect_enemy(model, img, capture_x, capture_y, app.threshold.get())
                det_end = time.time()
                fire_time = None
                shot_success = False
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
                    # 磁力吸附：距离越近吸附越强，远处更平滑
                    beta = np.exp(-pred_dist / (radius + 1e-6))
                    aim_x = (1 - beta) * head_x + beta * pred_x
                    aim_y = (1 - beta) * head_y + beta * pred_y
                    absorb_state = f'磁力吸附 β={beta:.2f}'
                    if time.time() - last_print_time > 0.5:
                        print(f"[KF] v={velocity:.1f} Q={Q:.2f} R={R:.2f} 吸附区r={radius:.1f} 状态:{absorb_state} 原:({head_x:.1f},{head_y:.1f}) 预测:({pred_x:.1f},{pred_y:.1f}) 瞄准:({aim_x:.1f},{aim_y:.1f}) 距离:{pred_dist:.1f}")
                    try:
                        fire_time = perform_action(driver, aim_x, aim_y, app.sleep_time_var.get(), app.size.get(), head_xyxy, detect_end_time=det_end, recoil_comp=0)
                        driver.smooth_move(0, -app.recoil_comp_var.get(), min_steps=2, max_steps=6, scale_factor=2)
                        if fire_time:
                            shot_success = True
                    except Exception as e:
                        print(f"[异常] 鼠标操作失败: {e}")
                        fire_time = None
                        shot_success = False
                    last_head = (head_x, head_y)
                    last_head_xyxy = head_xyxy
                    if shot_success:
                        lost_count = 0
                    else:
                        lost_count += 1
                        print(f"[未射击] 未满足吸附/判定条件或鼠标异常，lost_count={lost_count}")
                    if time.time() - last_print_time > 0.5:
                        delay_ms = (fire_time - det_end) * 1000 if fire_time else None
                        delay_str = f"{delay_ms:.1f}ms" if delay_ms is not None else "--"
                        print(f"[右键] 抓图: {img_end-img_start:.3f}s, 检测: {det_end-det_start:.3f}s, 检测到开火: {delay_str}, 总: {det_end-loop_start:.3f}s")
                        last_print_time = time.time()
                    if lost_count > max_lost:
                        print(f"[重置] 连续未射击，重置kalman/last_head/lost_count")
                        kalman = None
                        last_head = None
                        last_head_xyxy = None
                        lost_count = 0
                else:
                    lost_count += 1
                    if lost_count > max_lost:
                        kalman = None
                        last_head = None
                        last_head_xyxy = None
                    # 新增：头部丢失时自动尝试锁定身体目标
                    if closest_enemy and len(closest_enemy) > 2:
                        print("[INFO] 头部丢失，自动切换锁定身体目标")
                        fire_time = perform_action_body(driver, *closest_enemy[:2], app.sleep_time_var.get(), app.size.get(), closest_enemy[2], detect_end_time=det_end)
                        if fire_time:
                            print("[INFO] 身体目标已射击")
                            lost_count = 0
                        else:
                            print("[INFO] 身体目标未能射击")
            if app.display_var.get() and frame_count % 2 == 0:
                img = np.array(sct.grab(capture_area))
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                img_end = time.time()
                det_start = time.time()
                closest_enemy_head, closest_enemy = detect_enemy(model, img, capture_x, capture_y, app.threshold.get())
                det_end = time.time()
                disp_start = time.time()
                display_image_with_detections(img, closest_enemy_head, closest_enemy, app.scale.get(), app.tk_window)
                disp_end = time.time()
                if time.time() - last_print_time > 1.0:
                    print(f"[显示] 抓图: {img_end-img_start:.3f}s, 检测: {det_end-det_start:.3f}s, 显示: {disp_end-disp_start:.3f}s, 总: {disp_end-loop_start:.3f}s")
                    last_print_time = time.time()
            if (win32api.GetAsyncKeyState(win32con.VK_SHIFT) < 0 and win32api.GetAsyncKeyState(win32con.VK_ESCAPE) < 0):
                print("退出程序中...")
                return
            app.root.after(1, loop)
        loop()
    app.root.after(100, main_loop)
    app.run()
