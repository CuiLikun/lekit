import numpy as np

from .isaac_teleop import XRController, XRControllerConfig


def main():
    config = XRControllerConfig()
    teleop = XRController(config)
    teleop.connect()

    while True:
        xr_action = teleop.get_action()
        grip_pos = np.asarray(xr_action["grip_pos"], dtype=float)
        grip_quat = np.asarray(xr_action["grip_quat"], dtype=float)
        squeeze = float(xr_action["squeeze"])
        trigger = float(xr_action["trigger"])
        print(f"grip_pos={grip_pos}, grip_quat={grip_quat}, squeeze={squeeze}, trigger={trigger}")


if __name__ == "__main__":
    main()
