import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "soft_visual_hull_refine.py"
spec = importlib.util.spec_from_file_location("soft_visual_hull_refine", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_sample_mask_bilinear_returns_zero_outside_image():
    mask = np.ones((4, 4), dtype=np.float32)
    uv = np.array([[-0.1, 0.5], [0.5, 1.2], [1.1, -0.2]], dtype=np.float32)

    sampled = module.sample_mask_bilinear(mask, uv)

    assert np.allclose(sampled, 0.0)


def test_sample_mask_bilinear_interpolates_inside_image():
    mask = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    uv = np.array([[0.5, 0.5]], dtype=np.float32)

    sampled = module.sample_mask_bilinear(mask, uv)

    assert np.allclose(sampled, 0.25)


def test_support_to_strength_only_moves_low_support_points():
    support = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)

    strength = module.support_to_strength(support, threshold=0.5, softness=0.25, max_strength=0.04)

    assert strength[0] > strength[1] > strength[2]
    assert strength[3] < 0.01
    assert strength[4] < 0.005
    assert np.all(strength >= 0.0)
    assert np.all(strength <= 0.04)
