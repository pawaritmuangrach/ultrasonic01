"""อ่านภาพสี + ความลึกจากกล้อง Orbbec Astra Pro — แยกออกมาให้ตัวเก็บ dataset เรียกใช้

ยกมาจาก ../webcam/capture_pairs.py ที่พิสูจน์แล้วว่าใช้ได้กับกล้องตัวนี้:
ภาพสีของ Astra Pro เป็น UVC ธรรมดา (เปิดผ่าน cv2) ส่วน depth ต้องผ่าน OpenNI2
และ OpenNI2 warp depth ให้ทับเฟรมสีด้วย extrinsics จากโรงงาน (DEPTH_TO_COLOR)
"""
import os
# cv2 / numpy(MKL) / OpenNI native โหลด OpenMP ซ้อนกันแล้วล้มเงียบบน Windows
# (กับดักข้อ 4 ใน webcam/README) — ตั้งก่อน import อะไรที่ดึง OpenMP
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

# โฟลเดอร์ redist ของ OpenNI อยู่ในโปรเจกต์กล้อง ใช้ตัวเดียวกัน
_WEBCAM = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "webcam"))
ONI_REDIST_DIR = os.path.join(
    _WEBCAM,
    "Orbbec_OpenNI_v2.3.0.86-beta6_windows_release",
    "OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows",
    "OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows",
    "Win64-Release", "tools", "NiViewer")

RGB_INDEX = 1          # กล้องสีของ Astra (index 0 คือเว็บแคมโน้ตบุ๊ก)
WIDTH, HEIGHT = 640, 480
MAX_DEPTH_MM = 8000


class Astra:
    """เปิดกล้องครั้งเดียว อ่านซ้ำได้ ปิดด้วย close() หรือ with-block"""

    def __init__(self, rgb_index=RGB_INDEX, want_rgb=True, verbose=False,
                 depth_size=None):
        import cv2
        from primesense import openni2
        from primesense import _openni2 as c_api
        self._cv2 = cv2
        self._openni2 = openni2

        def step(msg):
            if verbose:
                print(f"  [astra] {msg}", flush=True)

        if not os.path.isdir(ONI_REDIST_DIR):
            raise RuntimeError(f"ไม่พบโฟลเดอร์ OpenNI: {ONI_REDIST_DIR}")
        step(f"initialize OpenNI ...")
        openni2.initialize(ONI_REDIST_DIR)
        step("initialize สำเร็จ · เปิดอุปกรณ์ (open_any) ...")
        try:
            self.dev = openni2.Device.open_any()
        except Exception as e:
            openni2.unload()
            raise RuntimeError(
                "เปิดกล้อง depth ไม่ได้ (open_any ล้ม) — มักเพราะกล้องไม่ถูกตรวจเจอ "
                "หรือมีโปรแกรมอื่นจับอยู่ ลองเสียบตรงเข้าคอม (ไม่ผ่านฮับ) แล้วปิด "
                f"NiViewer/โปรแกรมกล้องอื่น\n  รายละเอียด: {e}") from e
        step("เปิดอุปกรณ์สำเร็จ · สร้าง depth stream ...")
        self.depth = self.dev.create_depth_stream()
        # เลือกความละเอียด depth ก่อน start — กล้องตัวนี้รองรับ 160x120 / 320x240 /
        # 640x480 ที่ 30 fps เท่านั้น (**ไม่มีโหมด 15 fps**) ถ้าต้องการ 15 fps ให้ทำ
        # ด้วยซอฟต์แวร์ฝั่งผู้เรียก ไม่ใช่ตั้งที่กล้อง
        if depth_size is not None:
            want_w, want_h = depth_size
            pick = None
            for m in self.depth.get_sensor_info().videoModes:
                if (m.resolutionX == want_w and m.resolutionY == want_h
                        and m.pixelFormat ==
                        c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM):
                    if pick is None or m.fps > pick.fps:
                        pick = m
            if pick is None:
                raise RuntimeError(
                    f"กล้องไม่รองรับ depth {want_w}x{want_h} (หน่วย 1 mm) — "
                    f"รันไฟล์นี้ตรง ๆ เพื่อดูโหมดที่มี")
            self.depth.set_video_mode(pick)
            step(f"ตั้ง depth เป็น {pick.resolutionX}x{pick.resolutionY} @ {pick.fps} fps")
        self.depth.start()
        vm = self.depth.get_video_mode()
        self.depth_size = (vm.resolutionX, vm.resolutionY)
        self.depth_fps = vm.fps
        step("depth stream เริ่มแล้ว · ตั้ง registration ...")
        # ต้อง start ก่อนตั้ง registration ไม่งั้น BAD_PARAMETER
        self.dev.set_image_registration_mode(
            c_api.OniImageRegistrationMode.ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR)
        got = self.dev.get_image_registration_mode()
        want = c_api.OniImageRegistrationMode.ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR
        self.registered = (got == want)
        if not self.registered:
            print(f"เตือน: registration ไม่ติด (mode={got}) depth จะไม่ทับภาพสี")

        self.cap = None
        if want_rgb:
            step(f"เปิดกล้องสี index {rgb_index} (cv2/DSHOW) ...")
            self.cap = cv2.VideoCapture(rgb_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                print(f"เตือน: เปิดกล้องสี index {rgb_index} ไม่ได้ — เก็บเฉพาะ depth "
                      f"(depth ใช้เทรนได้อยู่แล้ว ลอง --rgb-index อื่นถ้าอยากได้ภาพสี)")
                self.cap = None
            else:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        step("กล้องพร้อม")

    def read_depth(self):
        """depth เป็น uint16 หน่วยมิลลิเมตร 0 = อ่านไม่ได้"""
        f = self.depth.read_frame()
        buf = np.frombuffer(f.get_buffer_as_uint16(), dtype=np.uint16)
        return buf.reshape(f.height, f.width).copy()

    def read_rgb(self):
        if self.cap is None:
            return None
        ok, rgb = self.cap.read()
        return rgb if ok else None

    def read(self):
        """คืน (rgb, depth) — rgb เป็น None ถ้าไม่มีกล้องสี"""
        rgb = self.read_rgb()
        depth = self.read_depth()
        if rgb is not None and rgb.shape[:2] != depth.shape[:2]:
            rgb = self._cv2.resize(rgb, (depth.shape[1], depth.shape[0]))
        return rgb, depth

    def valid_fraction(self, depth):
        return float(np.count_nonzero(depth)) / depth.size

    def close(self):
        try:
            self.depth.stop()
        except Exception:
            pass
        try:
            self._openni2.unload()
        except Exception:
            pass
        if self.cap is not None:
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def depth_to_heatmap(depth):
    """depth mm -> ภาพ JET สำหรับดูด้วยตา (0 = ดำ)"""
    import cv2
    norm = np.clip(depth, 0, MAX_DEPTH_MM).astype(np.float32) / MAX_DEPTH_MM
    vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    vis[depth == 0] = 0
    return vis


def _selftest():
    """python car/astra.py — เปิดกล้องทีละขั้นให้เห็นว่าตายตรงไหน แล้วอ่านหนึ่งเฟรม"""
    print("ทดสอบเปิดกล้อง Astra ทีละขั้น (ถ้าค้างหรือเด้งออก = ตายที่ขั้นล่าสุดที่พิมพ์)")
    cam = Astra(verbose=True)
    rgb, depth = cam.read()
    v = cam.valid_fraction(depth)
    lo = int(depth[depth > 0].min()) if (depth > 0).any() else 0
    print(f"depth {depth.shape}  ใช้ได้ {v:.0%}  ช่วง {lo}..{int(depth.max())} mm  "
          f"rgb {'มี' if rgb is not None else 'ไม่มี'}")
    print(f"โหมดที่ตั้งอยู่: {cam.depth_size[0]}x{cam.depth_size[1]} @ {cam.depth_fps} fps")
    print("โหมด depth ที่กล้องรองรับทั้งหมด:")
    seen = set()
    for m in cam.depth.get_sensor_info().videoModes:
        k = (m.resolutionX, m.resolutionY, m.fps)
        if k in seen:
            continue
        seen.add(k)
        print(f"   {m.resolutionX:4d} x {m.resolutionY:3d} @ {m.fps:2d} fps")
    print("อ่านเฟรมได้ — กล้องพร้อมใช้")
    cam.close()


if __name__ == "__main__":
    _selftest()
