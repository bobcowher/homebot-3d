import numpy as np
from homebot3d.teleop_input import keys_to_action


def test_empty_is_zero():
    np.testing.assert_array_equal(keys_to_action(set()), [0.0, 0.0])


def test_single_keys():
    np.testing.assert_array_equal(keys_to_action({"w"}), [1.0, 0.0])
    np.testing.assert_array_equal(keys_to_action({"s"}), [-1.0, 0.0])
    np.testing.assert_array_equal(keys_to_action({"a"}), [0.0, 1.0])   # left = +CCW
    np.testing.assert_array_equal(keys_to_action({"d"}), [0.0, -1.0])  # right = -CW


def test_opposing_keys_cancel():
    np.testing.assert_array_equal(keys_to_action({"w", "s"}), [0.0, 0.0])
    np.testing.assert_array_equal(keys_to_action({"a", "d"}), [0.0, 0.0])


def test_combined_drive_and_turn():
    np.testing.assert_array_equal(keys_to_action({"w", "a"}), [1.0, 1.0])


def test_returns_float32():
    assert keys_to_action({"w"}).dtype == np.float32
