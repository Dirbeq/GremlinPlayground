import win32gui
import win32ui
import win32con


class ScreenCapture:
    w = 0
    h = 0
    hwnd = None
    window_name = None

    def __init__(self, window_name):
        self.window_name = window_name
        self.hwnd = win32gui.FindWindow(None, window_name)
        if not self.hwnd:
            raise Exception('Window not found: {}'.format(window_name))
        self.w = win32gui.GetWindowRect(self.hwnd)[2]
        self.h = win32gui.GetWindowRect(self.hwnd)[3]

    def free_resources(self, dc, c):
        dc.DeleteDC()
        c.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, dc)
        win32gui.DeleteObject(c.GetHandle())
