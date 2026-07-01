import time

import matplotlib.pyplot as plt
import numpy as np

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

TARGET_FPS = 30
TARGET_WIDTH = 640
TARGET_HEIGHT = 480
TARGET_FOURCC = "MJPG"


def find_available_cameras() -> list[str]:
    """Return all discoverable OpenCV camera device IDs."""
    discovered = OpenCVCamera.find_cameras()
    return [str(cam["id"]) for cam in discovered]


def connect_camera(device: str) -> OpenCVCamera | None:
    """Connect one camera using LeRobot's OpenCVCamera wrapper."""
    config = OpenCVCameraConfig(
        index_or_path=device,
        color_mode=ColorMode.RGB,
        fps=TARGET_FPS,
        width=TARGET_WIDTH,
        height=TARGET_HEIGHT,
        fourcc=TARGET_FOURCC,
    )
    camera = OpenCVCamera(config)

    try:
        camera.connect(warmup=True)
        print(f"Opened {device}")
        return camera
    except Exception as error:
        print(f"Failed to open {device}: {error}")
        if camera.is_connected:
            camera.disconnect()
        return None


def main() -> None:
    cameras: dict[str, OpenCVCamera] = {}
    should_stop = False

    available_devices = find_available_cameras()
    if not available_devices:
        print("No OpenCV cameras were found by OpenCVCamera.find_cameras().")
        return

    print(f"Found OpenCV devices: {available_devices}")

    for device in available_devices:
        camera = connect_camera(device)
        if camera is not None:
            cameras[device] = camera

    if not cameras:
        print("No camera available.")
        return

    print("Press 'q' in the matplotlib window or close the window to quit.")

    plt.ion()
    fig, axes = plt.subplots(1, len(cameras), figsize=(6 * len(cameras), 4))
    fig.subplots_adjust(bottom=0.12)
    fig.text(0.5, 0.02, "Press 'q' to quit", ha="center", va="bottom", fontsize=11, color="dimgray")
    if len(cameras) == 1:
        axes = [axes]

    image_handles: dict[str, object] = {}
    fps_text_handles: dict[str, object] = {}
    last_fps_time_by_device: dict[str, float] = {}
    frame_counter_by_device: dict[str, int] = {}

    for axis, device in zip(axes, cameras, strict=True):
        # Initialize each subplot with a blank frame.
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        image = axis.imshow(blank)
        axis.set_title(device)
        axis.axis("off")
        image_handles[device] = image
        fps_text_handles[device] = axis.text(
            0.98,
            0.98,
            "FPS: 0.0",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=11,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.6, "pad": 4},
        )
        last_fps_time_by_device[device] = time.perf_counter()
        frame_counter_by_device[device] = 0

    def on_key(event: object) -> None:
        nonlocal should_stop
        key = getattr(event, "key", None)
        if key == "q":
            should_stop = True

    fig.canvas.mpl_connect("key_press_event", on_key)

    try:
        while plt.fignum_exists(fig.number) and not should_stop:
            for device, camera in cameras.items():
                try:
                    # Prefer non-blocking reads to avoid display loop stalls.
                    frame = camera.read_latest(max_age_ms=300)
                except TimeoutError:
                    # Fallback to blocking read when the latest frame is stale.
                    frame = camera.read()
                except Exception as error:
                    print(f"Failed to read frame from {device}: {error}")
                    continue
                image_handles[device].set_data(frame)
                frame_counter_by_device[device] += 1

                now = time.perf_counter()
                elapsed = now - last_fps_time_by_device[device]
                if elapsed >= 0.5:
                    fps = frame_counter_by_device[device] / elapsed
                    fps_text_handles[device].set_text(f"FPS: {fps:.1f}")
                    frame_counter_by_device[device] = 0
                    last_fps_time_by_device[device] = now

            fig.canvas.draw_idle()
            plt.pause(0.001)
    finally:
        for camera in cameras.values():
            if camera.is_connected:
                camera.disconnect()
        plt.close("all")


if __name__ == "__main__":
    main()
