import win32gui
import win32ui
import win32con

class ScreenCapture:
    x = 0  # target x
    y = 0  # target y
    w = 0  # target width
    h = 0  # target height
    hwnd = None  # handle to window
    window_name = None  # name of window

    def __init__(self, window_name, x, y, w, h):
        do_window_exist(window_name)
        self.window_name = window_name
        self.hwnd = win32gui.FindWindow(None, window_name)
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        print("Capture object created")

    def __del__(self):
        print("Capture object destroyed")

    def free_resources(self, dc, c):
        dc.DeleteDC()
        c.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, dc)
        win32gui.DeleteObject(c.GetHandle())

    def debug_draw_rectangle(self, dc, c):
        win32gui.Rectangle(dc, self.x, self.y, self.x + self.w, self.y + self.h)
        self.free_resources(dc, c)

    def capture(self):
        # create a device context
        dc = win32gui.GetWindowDC(self.hwnd)
        c = win32ui.CreateDCFromHandle(dc)
        c.CreateCompatibleDC()

        # create a bitmap object
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(c, self.w, self.h)
        c.SelectObject(dataBitMap)
        c.BitBlt((0, 0), (self.w, self.h), dc, (self.x, self.y), win32con.SRCCOPY)
        # self.debug_draw_rectangle(dc, c)
        bmpinfo = dataBitMap.GetInfo()
        bmpstr = dataBitMap.GetBitmapBits(True)
        self.free_resources(dc, c)
        return bmpstr, bmpinfo



def do_window_exist(window_name):
    try:
        win32gui.FindWindow(None, window_name)
    except win32gui.error:
        raise Exception("Window with name '" + window_name + "' does not exist")
    else:
        print("Window with name '" + window_name + "' exists")
        return True


