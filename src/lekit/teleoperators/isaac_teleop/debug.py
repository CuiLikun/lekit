"""Display the standalone Quest controller action stream in a terminal."""

from __future__ import annotations

import time

import numpy as np

from .config import IsaacTeleopConfig
from .xr_controller import IsaacXRController


def main() -> None:
    """Run a small hardware diagnostic without commanding any robot."""

    from rich.console import Console
    from rich.live import Live
    from rich.table import Table

    controller = IsaacXRController(IsaacTeleopConfig())
    console = Console()
    with controller, Live(console=console, refresh_per_second=15) as live:
        try:
            while True:
                action = controller.get_action()
                table = Table(title="Isaac XR controllers", expand=False)
                table.add_column("Field")
                table.add_column("Left")
                table.add_column("Right")
                values = {}
                for side in ("left", "right"):
                    values[side] = {
                        "tracking": f"grip={action[f'{side}.is_tracking']}, aim={action[f'{side}.is_aim_tracking']}",
                        "clutch": str(action[f"{side}.is_engaged"]),
                        "translation": np.array2string(
                            np.asarray(action[f"{side}.translation"], dtype=float) * 1000.0,
                            precision=1,
                        ),
                        "rotation": np.array2string(
                            np.asarray(action[f"{side}.rotation"], dtype=float), precision=3
                        ),
                        "analog": (
                            f"squeeze={action[f'{side}.squeeze']:.2f}, "
                            f"trigger={action[f'{side}.trigger']:.2f}"
                        ),
                        "thumbstick": np.array2string(np.asarray(action[f"{side}.thumbstick"]), precision=2),
                        "buttons": (
                            f"primary={action[f'{side}.primary_button']:.0f}, "
                            f"secondary={action[f'{side}.secondary_button']:.0f}, "
                            f"menu={action[f'{side}.menu_button']:.0f}"
                        ),
                    }
                table.add_row("Tracking", values["left"]["tracking"], values["right"]["tracking"])
                table.add_row("Squeeze clutch", values["left"]["clutch"], values["right"]["clutch"])
                table.add_row(
                    "Translation [mm] right, forward, up",
                    values["left"]["translation"],
                    values["right"]["translation"],
                )
                table.add_row(
                    "Relative rotation [xyzw]", values["left"]["rotation"], values["right"]["rotation"]
                )
                table.add_row("Analog", values["left"]["analog"], values["right"]["analog"])
                table.add_row("Thumbstick", values["left"]["thumbstick"], values["right"]["thumbstick"])
                table.add_row("Buttons", values["left"]["buttons"], values["right"]["buttons"])
                live.update(table)
                time.sleep(1.0 / 30.0)
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    main()
