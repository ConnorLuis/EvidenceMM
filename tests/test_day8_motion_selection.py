from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from evidencemm.motion_selection import (
    MotionFrameScore,
    apply_camera_transform,
    fuse_motion_scores,
    mean_absolute_pixel_difference,
    preprocess_motion_image,
    select_motion_frame,
)


def test_ccw90_transform_rotates_geometry():
    image = Image.new("L", (4, 2))
    rotated = apply_camera_transform(image, "ccw90")
    assert rotated.size == (2, 4)


def test_preprocess_produces_frozen_grayscale_shape(
    tmp_path: Path,
):
    path = tmp_path / "sample.png"
    Image.new("RGB", (640, 480), color=(255, 0, 0)).save(path)
    array = preprocess_motion_image(path, transform="none")
    assert array.shape == (120, 160)
    assert array.dtype == np.float32


def test_mean_absolute_pixel_difference_is_exact():
    previous = np.array(
        [[0.0, 10.0], [20.0, 30.0]],
        dtype=np.float32,
    )
    current = np.array(
        [[10.0, 10.0], [10.0, 50.0]],
        dtype=np.float32,
    )
    assert mean_absolute_pixel_difference(previous, current) == 10.0


def test_motion_fusion_uses_max_camera_score():
    assert fuse_motion_scores(2.5, 7.0) == 7.0
    assert fuse_motion_scores(9.0, 1.0) == 9.0


def test_motion_selection_tie_breaks_to_lower_frame():
    scores = [
        MotionFrameScore(
            frame_index=11,
            timestamp_sec=0.7,
            front_motion_score=4.0,
            wrist_motion_score=8.0,
            fused_motion_score=8.0,
        ),
        MotionFrameScore(
            frame_index=10,
            timestamp_sec=0.6,
            front_motion_score=8.0,
            wrist_motion_score=1.0,
            fused_motion_score=8.0,
        ),
        MotionFrameScore(
            frame_index=12,
            timestamp_sec=0.8,
            front_motion_score=2.0,
            wrist_motion_score=3.0,
            fused_motion_score=3.0,
        ),
    ]
    assert select_motion_frame(scores).frame_index == 10
