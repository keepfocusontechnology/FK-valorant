import math
import ctypes
import random
import time

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
