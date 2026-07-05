import numpy as np
from homebot3d.maps import DefaultHouseMap
from homebot3d.goals import goal_to_coordinates
from homebot3d.tasks import TaskManager
from homebot3d.world import tile_center

class FakeRobot:
    def __init__(self, x, y): self.x, self.y = x, y

def test_goal_to_coordinates_maps_drink_to_recliner():
    m = DefaultHouseMap()
    assert goal_to_coordinates("drink", m) == tile_center(*m.dropoff_tiles["drink"])

def test_drink_pickup_then_deliver():
    m = DefaultHouseMap()
    tm = TaskManager(goals=["drink"])
    tm.reset(m, n_trash=0, rng=np.random.default_rng(0))
    rx, ry = tile_center(*m.dropoff_tiles["drink"])     # recliner (target)
    sx, sy = tile_center(*m.pickup_tiles["drink"])      # fridge (source)
    # Reaching the target FIRST, before pickup, does nothing.
    assert tm.step(FakeRobot(rx, ry)) == 0.0
    assert tm.is_done() is False
    # Reach the source: pickup, reward, now carrying.
    assert tm.step(FakeRobot(sx, sy)) == 1.0
    assert "drink" in tm.carrying
    assert tm.is_done() is False
    # Reach the target: deliver, reward, done, no longer carrying.
    assert tm.step(FakeRobot(rx, ry)) == 1.0
    assert "drink" not in tm.carrying
    assert tm.is_done() is True
    # No further reward once done.
    assert tm.step(FakeRobot(rx, ry)) == 0.0

def test_package_pickup_then_deliver():
    m = DefaultHouseMap()
    tm = TaskManager(goals=["package"])
    tm.reset(m, n_trash=0, rng=np.random.default_rng(0))
    sx, sy = tile_center(*m.pickup_tiles["package"])    # door (source)
    tx, ty = tile_center(*m.dropoff_tiles["package"])   # (16,4) kitchen table
    assert tm.step(FakeRobot(sx, sy)) == 1.0
    assert "package" in tm.carrying
    assert tm.step(FakeRobot(tx, ty)) == 1.0
    assert "package" not in tm.carrying
    assert tm.is_done() is True

def test_current_goal_xy_switches_at_pickup():
    m = DefaultHouseMap()
    tm = TaskManager(goals=["drink"])
    tm.reset(m, n_trash=0, rng=np.random.default_rng(0))
    assert tm.current_goal_xy("drink") == tile_center(*m.pickup_tiles["drink"])
    sx, sy = tile_center(*m.pickup_tiles["drink"])
    tm.step(FakeRobot(sx, sy))
    assert tm.current_goal_xy("drink") == tile_center(*m.dropoff_tiles["drink"])

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
