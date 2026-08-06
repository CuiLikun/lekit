from lerobot.datasets import LeRobotDatasetMetadata
from robots.jaka_robot.dataset_features import build_dataset_features
from robots.jaka_robot.jaka_robot import JakaRobot, JakaRobotConfig


def test_dataset_features_are_lerobot_metadata_compatible(tmp_path):
    """The recorder must convert robot descriptors to dataset feature metadata."""
    robot = JakaRobot(JakaRobotConfig(ip="127.0.0.1"))

    features = build_dataset_features(robot, use_videos=True)

    assert all("dtype" in feature for feature in features.values())
    metadata = LeRobotDatasetMetadata.create(
        repo_id="test/jaka-schema",
        fps=30,
        robot_type=robot.name,
        features=features,
        root=tmp_path / "dataset",
        use_videos=True,
    )
    assert features.items() <= metadata.features.items()
