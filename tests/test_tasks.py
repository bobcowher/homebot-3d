import numpy as np
from homebot3d.maps import DefaultHouseMap
from homebot3d.goals import goal_to_coordinates, GOAL_TARGETS
from homebot3d.tasks import TaskManager
from homebot3d.world import tile_center

class FakeRobot:
    def __init__(self, x, y): self.x, self.y = x, y

def test_goal_targets_mapping():
    assert GOAL_TARGETS == {"trash": "trash", "drink": "recliner", "package": "door"}

def test_goal_to_coordinates_maps_drink_to_recliner():
    m = DefaultHouseMap()
    assert goal_to_coordinates("drink", m) == tile_center(*m.fixtures["recliner"])

def test_reaching_recliner_rewards_once_and_completes():
    m = DefaultHouseMap()
    tm = TaskManager(goals=["drink"])
    tm.reset(m, n_trash=0, rng=np.random.default_rng(0))
    rx, ry = tile_center(*m.fixtures["recliner"])
    assert tm.step(FakeRobot(rx, ry)) == 1.0
    assert tm.step(FakeRobot(rx, ry)) == 0.0      # no double reward
    assert tm.is_done() is True

def test_far_away_gives_no_reward():
    m = DefaultHouseMap()
    tm = TaskManager(goals=["drink"])
    tm.reset(m, n_trash=0, rng=np.random.default_rng(0))
    assert tm.step(FakeRobot(-100.0, -100.0)) == 0.0
    assert tm.is_done() is False

def test_trash_targets_reached_individually():
    m = DefaultHouseMap()
    tm = TaskManager(goals=["trash"])
    tm.reset(m, n_trash=2, rng=np.random.default_rng(1))
    assert len(tm.trash_positions) == 2
    (c0, r0) = tm.trash_positions[0]
    x0, y0 = tile_center(c0, r0)
    assert tm.step(FakeRobot(x0, y0)) == 1.0
    assert len(tm.trash_positions) == 1
