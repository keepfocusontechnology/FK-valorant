import time
from wwqy import LGDriver

class ClickTest:
    """
    用于单独测试LGDriver点击事件的类。
    """
    def __init__(self, driver):
        self.driver = driver

    def test_click(self, n=3, interval=1.0):
        print(f"[ClickTest] 开始测试鼠标点击，共{n}次，每次间隔{interval}秒。")
        for i in range(n):
            print(f"[ClickTest] 第{i+1}次点击...")
            self.driver.click()
            time.sleep(interval)
        print("[ClickTest] 测试结束。")

    def test_move(self, n=3, interval=1.0, dx=100, dy=0):
        print(f"[ClickTest] 开始测试鼠标移动，共{n}次，每次间隔{interval}秒，每次移动({dx},{dy})。")
        for i in range(n):
            print(f"[ClickTest] 第{i+1}次移动...")
            self.driver.move(dx, dy)
            time.sleep(interval)
        print("[ClickTest] 移动测试结束。")

if __name__ == "__main__":
    # 这里直接用 float 类型，无需 tk.DoubleVar
    click_time = 0.12
    driver = LGDriver("driver/logitech.driver.dll", click_time)
    print("驱动加载状态 driver.ok =", driver.ok)
    tester = ClickTest(driver)
    tester.test_click(n=5, interval=1.0)
    tester.test_move(n=5, interval=1.0, dx=100, dy=0)
