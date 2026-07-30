from lerobot.robots import Robot, RobotConfig


def make_robot_from_config(config: RobotConfig) -> Robot:
    match config.type:
        case "mock_robot":
            from src.hardwares.mock_robot import MockRobot

            return MockRobot(config)
        case "sim_robot":
            from src.hardwares.sim_robot import SimRobot

            return SimRobot(config)

        case "jaka_robot":
            from src.hardwares.jaka_robot import JakaRobot

            return JakaRobot(config)

    from lerobot.robots import make_robot_from_config as lerobot_make_robot_from_config

    return lerobot_make_robot_from_config(config)
