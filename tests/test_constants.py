from homebot3d import constants


def test_robot_embodiment_constants_present():
    assert constants.CAMERA_HEIGHT == 0.6
    assert constants.ROBOT_BODY_HALF == 0.12
    assert constants.ROBOT_BODY_HALFHEIGHT == 0.09
    assert constants.WHEEL_RADIUS == 0.06
    # Camera must sit above the base cylinder top.
    assert constants.CAMERA_HEIGHT > 2 * constants.ROBOT_HALFHEIGHT
