from lerobot.robots import Robot, RobotConfig


def make_robot_from_config(config: RobotConfig) -> Robot:
    match config.type:
        case "mock_robot":
            from lekit.robots.mock_robot import MockRobot

            return MockRobot(config)
        case "sim_robot":
            from lekit.robots.sim_robot import SimRobot

            return SimRobot(config)

        case "jaka_robot":
            from lekit.robots.jaka_robot import JakaRobot

            return JakaRobot(config)
        case "agx_arm":
            from lekit.robots.agx_arm import AgxArm

            return AgxArm(config)

    from lerobot.robots import make_robot_from_config as lerobot_make_robot_from_config

    return lerobot_make_robot_from_config(config)
