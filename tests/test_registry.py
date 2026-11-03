import pytest

from conftest import CAMERAS, CALIBRATION, LABELS, make_features_df
from horizon_pipeline.raw.registry import Item, Registry


@pytest.fixture
def registry() -> Registry:
    return Registry(make_features_df())  # with the real presets.toml


def test_default_preset_is_labels_calibration_and_camera_timestamps(registry):
    items = registry.resolve(["default"])
    assert items == (
        [Item(f, f) for f in LABELS]
        + [Item(f) for f in CALIBRATION]
        + [Item(c, "frame_timestamps") for c in CAMERAS]
    )


def test_scene_adds_lidar(registry):
    assert registry.resolve(["scene"]) == registry.resolve(["default"]) + [
        Item("lidar_top_360fov", "pointclouds")
    ]


def test_alpamayo_preset(registry):
    items = registry.resolve(["alpamayo"])
    assert items[0] == Item("egomotion", "egomotion")
    assert {i.feature for i in items[1:]} == {
        "camera_cross_left_120fov",
        "camera_front_wide_120fov",
        "camera_cross_right_120fov",
        "camera_front_tele_30fov",
    }
    assert {i.member for i in items[1:]} == {"video"}


def test_bare_feature_means_all_members_and_duplicates_are_dropped(registry):
    items = registry.resolve(["camera_front_wide_120fov", "camera_front_wide_120fov:video,videos"])
    assert items[:3] == [
        Item("camera_front_wide_120fov", m) for m in ("video", "frame_timestamps", "blurred_boxes")
    ]
    assert len(items) == 3 + 6  # the other 6 cameras' videos
    assert len(set(items)) == len(items)


def test_groups(registry):
    assert registry.resolve(["radar"]) == [
        Item("radar_front_center_mrr_2", "scans"),
        Item("radar_side_left_srr_0", "scans"),
    ]


@pytest.mark.parametrize(
    "selection, expected",
    [
        ("no_such_feature", ["presets:", "default", "groups:", "camera", "features:", "egomotion.offline"]),
        ("camera_front_wide_120fov:audio", ["Valid members: video, frame_timestamps, blurred_boxes"]),
        ("camera:audio", ["Valid members: blurred_boxes, frame_timestamps, video"]),
        ("vehicle_dimensions:x", ["single parquet file"]),
        ("scenery", ["scene", "default", "alpamayo", "videos"]),
    ],
)
def test_unknown_names_list_the_valid_ones(registry, selection, expected):
    with pytest.raises(ValueError) as error:
        registry.resolve([selection])
    for text in expected:
        assert text in str(error.value)


def test_preset_cycle_is_an_error():
    registry = Registry(make_features_df(), presets={"a": ["b"], "b": ["a"]})
    with pytest.raises(ValueError, match="cycle"):
        registry.resolve(["a"])


def test_local_paths(registry):
    assert registry.local_path(Item("egomotion.offline", "egomotion.offline")) == "egomotion.offline.parquet"
    assert registry.local_path(Item("vehicle_dimensions")) == "vehicle_dimensions.parquet"
    assert registry.local_path(Item("camera_front_wide_120fov", "video")) == "camera_front_wide_120fov/video.mp4"
    assert (
        registry.local_path(Item("camera_front_wide_120fov", "frame_timestamps"))
        == "camera_front_wide_120fov/frame_timestamps.parquet"
    )
    assert registry.local_path(Item("lidar_top_360fov", "pointclouds")) == "lidar_top_360fov/pointclouds.parquet"


def test_keys(registry):
    assert Item("egomotion", "egomotion").key == "egomotion"
    assert Item("vehicle_dimensions").key == "vehicle_dimensions"
    assert Item("lidar_top_360fov", "pointclouds").key == "lidar_top_360fov:pointclouds"
