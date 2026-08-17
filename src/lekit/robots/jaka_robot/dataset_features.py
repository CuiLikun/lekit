"""LeRobot dataset schema helpers for the JAKA robot."""

from lerobot.datasets import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.processor import make_default_processors
from lerobot.utils.feature_utils import combine_feature_dicts

from .jaka_robot import JakaRobot


def build_dataset_features(robot: JakaRobot, *, use_videos: bool) -> dict[str, dict]:
    """Convert the JAKA hardware schema into LeRobot dataset feature metadata."""
    teleop_processor, _, observation_processor = make_default_processors()
    return combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=use_videos,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=use_videos,
        ),
    )
