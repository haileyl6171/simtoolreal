"""Randomized cabinet clutter + ghost goal sampling (pure Python, no numpy/isaaclab).

Shared by the task generator and the viser viewer; the Isaac Sim env will get a
batched torch port of ``sample_layout``. All geometry is in the **cabinet
frame** (see ``dextoolbench/cabinet.py``) unless a name says ``_env``.

Two stages, mirroring what the sim can and cannot change at runtime:

* ``sample_clutter_pool``  (build time, per env): each of ``num_slots`` slots
  gets a pool of item specs (shape, size, mass, friction, color). Env i uses
  ``pool[k][i % pool_size]`` for slot k. A ``short_fraction`` of each pool is
  drawn no taller than ``h_front_max`` so items can go in the front corridor.
* ``sample_layout``  (reset time, per episode): place the task object as a
  ghost on a shelf, reserve its hand keep-out, put up to ``front_count_range``
  short items in the corridor between it and the cabinet opening, scatter the
  rest, then derive the carry height and the 4 goal waypoints.

Shape kinds are a registry (``SHAPE_SAMPLERS``) so procedural meshes can be
added later as another kind returning the same spec fields.
"""

from __future__ import annotations

import math
import random

from dextoolbench.cabinet import cabinet_spec, cabinet_to_env, shelf_surfaces

CLUTTER_DEFAULTS = dict(
    # --- items ---
    num_slots=6,
    pool_size=64,
    count_range=(3, 6),  # active items per episode (incl. front items)
    front_count_range=(0, 2),  # items deliberately placed in the front corridor
    short_fraction=0.4,  # share of each slot's pool no taller than h_front_max
    levels=(0, 1),  # support levels (0 = cabinet floor) that may hold clutter
    # Only clutter the ghost goal's level; other shelves stay empty (they never
    # interact with the hand, so they only cost sim time). Overrides ``levels``.
    goal_level_only=True,
    shapes=(
        dict(kind="box", weight=2.0, size_x=(0.03, 0.09), size_y=(0.03, 0.09), height=(0.03, 0.20)),
        dict(kind="cylinder", weight=1.5, radius=(0.015, 0.04), height=(0.04, 0.20)),
    ),
    mass_range=(0.05, 0.4),
    friction_range=(0.4, 1.0),
    min_gap=0.005,  # between item footprints
    wall_margin=0.005,  # between items and cabinet walls
    max_tries=60,
    seed=0,
    # --- ghost goal ---
    sample_goal=True,
    goal_levels=(0, 1),
    goal_yaw_range_deg=0.0,
    # Goal-footprint centre region as fractions of each shelf's interior
    # (x: 0 = left wall .. 1 = right wall, y: 0 = back wall .. 1 = front).
    goal_region=dict(x=(0.15, 0.85), y=(0.0, 0.6)),
    # --- hand leeway (object frame offsets, metres) ---
    hand_envelope=dict(up=0.12, down=0.0, side=0.05),
    clutter_margin=0.01,  # carried object/hand bottom above corridor clutter tops
    ceil_margin=0.01,  # hand top below the shelf above
    approach_offset=0.05,  # pre-insert: object's back edge this far in front of the opening
    insert_lift=0.02,  # carry at least this far above the place pose (empty corridor)
    min_lift=0.03,  # first waypoint at least this far above the start
)


def clutter_spec(clutter: dict | None) -> dict:
    """Fill a (possibly partial) JSON ``clutter`` block with defaults."""
    out = dict(CLUTTER_DEFAULTS)
    out.update(clutter or {})
    return out


# ----------------------------------------------------------------------------
# build-time item pool
# ----------------------------------------------------------------------------


def _u(rng, lo_hi):
    return rng.uniform(lo_hi[0], lo_hi[1])


def _capped(lo_hi, cap):
    """Range clipped to at most ``cap`` (kept non-empty)."""
    lo, hi = lo_hi
    return (min(lo, cap), max(min(hi, cap), min(lo, cap)))


def _box(shape, rng, cap):
    h = _u(rng, _capped(shape["height"], cap) if cap else shape["height"])
    sx, sy = _u(rng, shape["size_x"]), _u(rng, shape["size_y"])
    return dict(kind="box", size=(sx, sy, h), height=h, radius=0.5 * math.hypot(sx, sy))


def _cylinder(shape, rng, cap):
    h = _u(rng, _capped(shape["height"], cap) if cap else shape["height"])
    r = _u(rng, shape["radius"])
    return dict(kind="cylinder", radius=r, height=h)


def _sphere(shape, rng, cap):
    r_range = _capped(shape["radius"], cap / 2) if cap else shape["radius"]
    r = _u(rng, r_range)
    return dict(kind="sphere", radius=r, height=2 * r)


SHAPE_SAMPLERS = {"box": _box, "cylinder": _cylinder, "sphere": _sphere}


def sample_item(clutter: dict, rng: random.Random, height_cap: float | None = None) -> dict:
    """One item spec. ``radius`` is the conservative footprint circle used for spacing."""
    shapes = clutter["shapes"]
    shape = rng.choices(shapes, weights=[s.get("weight", 1.0) for s in shapes])[0]
    item = SHAPE_SAMPLERS[shape["kind"]](shape, rng, height_cap)
    item["mass"] = _u(rng, clutter["mass_range"])
    item["friction"] = _u(rng, clutter["friction_range"])
    item["color"] = tuple(rng.uniform(0.25, 0.9) for _ in range(3))
    return item


def sample_clutter_pool(clutter: dict, h_front_max: float) -> list[list[dict]]:
    """``pool[k][j]``: item spec of slot k for envs with ``i % pool_size == j``."""
    rng = random.Random(clutter["seed"])
    pool = []
    for _ in range(clutter["num_slots"]):
        n_short = round(clutter["short_fraction"] * clutter["pool_size"])
        items = [
            sample_item(clutter, rng, height_cap=h_front_max if j < n_short else None)
            for j in range(clutter["pool_size"])
        ]
        rng.shuffle(items)  # each slot gets its own short/tall order
        pool.append(items)
    return pool


def env_items(pool: list[list[dict]], env_index: int) -> list[dict]:
    return [slot[env_index % len(slot)] for slot in pool]


# ----------------------------------------------------------------------------
# geometry helpers (cabinet frame, xy footprints)
# ----------------------------------------------------------------------------


def _rect_overlap(a, b):
    """a, b = (x_lo, x_hi, y_lo, y_hi)."""
    return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]


def _item_rect(item, x, y, yaw):
    """Axis-aligned footprint of a placed item."""
    if item["kind"] == "box":
        sx, sy = item["size"][0] / 2, item["size"][1] / 2
        c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
        hx, hy = c * sx + s * sy, s * sx + c * sy
    else:
        hx = hy = item["radius"]
    return (x - hx, x + hx, y - hy, y + hy)


def _rotate_rect(rect, yaw):
    """AABB of a rect (about the origin) rotated by yaw."""
    c, s = math.cos(yaw), math.sin(yaw)
    xs, ys = [], []
    for px in rect[:2]:
        for py in rect[2:]:
            xs.append(c * px - s * py)
            ys.append(s * px + c * py)
    return (min(xs), max(xs), min(ys), max(ys))


def _quat_mul_xyzw(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def front_height_cap(shelf: dict, obj: dict, clutter: dict) -> float:
    """Tallest item allowed in the front corridor of ``shelf`` (<= 0: none, or infeasible)."""
    env = clutter["hand_envelope"]
    h_o = obj["aabb_z"][1] - obj["aabb_z"][0]
    return (
        shelf["clear_height"] - clutter["ceil_margin"] - env["up"] - h_o - env["down"]
        - clutter["clutter_margin"]
    )


# ----------------------------------------------------------------------------
# reset-time layout
# ----------------------------------------------------------------------------


def sample_layout(cab, clutter: dict, obj: dict, items: list[dict], start_pose, rng: random.Random) -> dict:
    """Sample one episode: ghost goal, clutter placement, carry height, waypoints.

    ``obj``: the task object's resting geometry (``"object"`` block written by the
    generator): ``rest_quat`` (xyzw), ``rest_dz`` (origin height above its
    support), ``aabb_x/y/z`` (origin-relative extents at rest_quat).
    ``items``: this env's slot specs (``env_items``). ``start_pose``: env-local
    object start [x, y, z, qx, qy, qz, qw].
    """
    cab = cabinet_spec(cab)
    surfaces = shelf_surfaces(cab)
    env_e = clutter["hand_envelope"]
    wm, gap = clutter["wall_margin"], clutter["min_gap"]

    # 1. Ghost goal: level, yaw, footprint centre inside goal_region.
    goal_levels = [lv for lv in clutter["goal_levels"] if lv < len(surfaces)
                   and front_height_cap(surfaces[lv], obj, clutter) >= 0.0]
    assert goal_levels, "no goal level leaves room for the hand (see front_height_cap)"
    level = rng.choice(goal_levels)
    shelf = surfaces[level]
    yaw = math.radians(rng.uniform(-1, 1) * clutter["goal_yaw_range_deg"])
    foot = _rotate_rect((*obj["aabb_x"], *obj["aabb_y"]), yaw)  # origin-relative footprint
    hx, hy = (foot[1] - foot[0]) / 2, (foot[3] - foot[2]) / 2
    (sx_lo, sx_hi), (sy_lo, sy_hi) = shelf["x_range"], shelf["y_range"]
    # Keep the hand envelope inside the side walls / back wall.
    cx_lo, cx_hi = sx_lo + hx + env_e["side"], sx_hi - hx - env_e["side"]
    cy_lo, cy_hi = sy_lo + hy + env_e["side"], sy_hi - hy
    assert cx_lo <= cx_hi and cy_lo <= cy_hi, "task object + hand envelope do not fit on the shelf"
    gr = clutter["goal_region"]
    cx = cx_lo + _u(rng, gr["x"]) * (cx_hi - cx_lo)
    cy = cy_lo + _u(rng, gr["y"]) * (cy_hi - cy_lo)
    goal_foot = (cx - hx, cx + hx, cy - hy, cy + hy)
    # Origin xy so the footprint centre lands on (cx, cy).
    ox = cx - (foot[0] + foot[1]) / 2
    oy = cy - (foot[2] + foot[3]) / 2

    # 2. Keep-outs: goal footprint (+ side envelope all round) is forbidden;
    #    the corridor to the opening is height-capped.
    s = env_e["side"]
    goal_keepout = (goal_foot[0] - s, goal_foot[1] + s, goal_foot[2] - s, goal_foot[3] + s)
    corridor = (goal_foot[0] - s, goal_foot[1] + s, goal_foot[3], sy_hi)
    h_cap = front_height_cap(shelf, obj, clutter)

    # 3. Place clutter. Front items first (short slots, centre inside the corridor).
    n_active = rng.randint(*clutter["count_range"])
    order = list(range(len(items)))
    rng.shuffle(order)
    active = order[:n_active]
    short = [k for k in active if items[k]["height"] <= h_cap]
    n_front = min(rng.randint(*clutter["front_count_range"]), len(short))
    front_slots = short[:n_front]
    placed = []  # dicts: slot, level, x, y, yaw, rect, front
    if clutter.get("goal_level_only", True):
        clutter_levels = [level]
    else:
        clutter_levels = [lv for lv in clutter["levels"] if lv < len(surfaces)]

    def fits(k, lv, x, y, iyaw):
        it = items[k]
        sf = surfaces[lv]
        rect = _item_rect(it, x, y, iyaw)
        if (rect[0] < sf["x_range"][0] + wm or rect[1] > sf["x_range"][1] - wm
                or rect[2] < sf["y_range"][0] + wm or rect[3] > sf["y_range"][1]):
            return None
        if it["height"] > sf["clear_height"] - clutter["ceil_margin"]:
            return None
        if lv == level:
            if _rect_overlap(rect, goal_keepout):
                return None
            if _rect_overlap(rect, corridor) and it["height"] > h_cap:
                return None
        for p in placed:
            if p["level"] == lv and math.hypot(p["x"] - x, p["y"] - y) < (
                items[p["slot"]]["radius"] + it["radius"] + gap
            ):
                return None
        return rect

    def place(k, front: bool):
        for _ in range(clutter["max_tries"]):
            if front:
                lv = level
                x = rng.uniform(corridor[0], corridor[1])
                y = rng.uniform(corridor[2], corridor[3])
            else:
                lv = rng.choice(clutter_levels)
                x = rng.uniform(*surfaces[lv]["x_range"])
                y = rng.uniform(*surfaces[lv]["y_range"])
            iyaw = rng.uniform(-math.pi, math.pi)
            rect = fits(k, lv, x, y, iyaw)
            if rect is not None:
                placed.append(dict(slot=k, level=lv, x=x, y=y, yaw=iyaw, rect=rect,
                                   front=lv == level and _rect_overlap(rect, corridor)))
                return True
        return False

    for k in front_slots:
        place(k, front=True)
    for k in active:
        if k not in front_slots:
            place(k, front=False)
    placed_slots = {p["slot"] for p in placed}
    parked = [k for k in range(len(items)) if k not in placed_slots]

    # 4. Carry height from the tallest corridor item (incl. ones that landed there by chance).
    front_tops = [surfaces[p["level"]]["z_top"] + items[p["slot"]]["height"] for p in placed if p["front"]]
    h_top = max(front_tops, default=shelf["z_top"])
    z_lo, z_hi = obj["aabb_z"]
    carry_z = h_top + clutter["clutter_margin"] + env_e["down"] - z_lo  # object origin, cabinet frame
    goal_z = shelf["z_top"] + obj["rest_dz"]
    carry_z = max(carry_z, goal_z + clutter["insert_lift"])
    hand_top = carry_z + z_hi + env_e["up"]
    ceiling = shelf["z_top"] + shelf["clear_height"]
    assert hand_top <= ceiling - clutter["ceil_margin"] + 1e-9, "carry height leaves no hand clearance"

    # 5. Waypoints (env-local, xyzw quats).
    world_yaw = yaw + math.radians(cab["yaw_deg"])
    quat = list(_quat_mul_xyzw(
        (0.0, 0.0, math.sin(world_yaw / 2), math.cos(world_yaw / 2)), tuple(obj["rest_quat"])
    ))
    gx, gy, gz = cabinet_to_env(cab, (ox, oy, goal_z))
    carry_env = cabinet_to_env(cab, (0.0, 0.0, carry_z))[2]
    pre_x, pre_y, _ = cabinet_to_env(cab, (ox, sy_hi + clutter["approach_offset"] - foot[2], carry_z))
    sx, sy, sz = start_pose[:3]
    start_quat = list(start_pose[3:7])
    waypoints = [
        [sx, sy, max(carry_env, sz + clutter["min_lift"])] + start_quat,  # lift
        [pre_x, pre_y, carry_env] + quat,  # pre-insert, outside the opening
        [gx, gy, carry_env] + quat,  # over the goal
        [gx, gy, gz] + quat,  # place
    ]

    return dict(
        level=level,
        goal_pose=waypoints[-1],
        waypoints=waypoints,
        goal_keepout=goal_keepout,
        corridor=corridor,
        h_front_max=h_cap,
        carry_z=carry_z,
        hand_top=hand_top,
        ceiling=ceiling,
        placed=placed,
        parked=parked,
        n_front=sum(p["front"] for p in placed),
    )


def item_pose_env(cab, item: dict, placement: dict, surfaces=None):
    """Env-local centre (x, y, z) and yaw (rad, world z) of a placed item."""
    cab = cabinet_spec(cab)
    surfaces = surfaces or shelf_surfaces(cab)
    z = surfaces[placement["level"]]["z_top"] + item["height"] / 2
    pos = cabinet_to_env(cab, (placement["x"], placement["y"], z))
    return pos, placement["yaw"] + math.radians(cab["yaw_deg"])


def layout_obstacles(cab, items: list[dict], layout: dict) -> list[dict]:
    """Placed items as ``ObstacleCfg`` field dicts (env-local, at the default table z)."""
    cab = cabinet_spec(cab)
    surfaces = shelf_surfaces(cab)
    out = []
    for p in layout["placed"]:
        it = items[p["slot"]]
        pos, yaw = item_pose_env(cab, it, p, surfaces)
        size = it["size"] if it["kind"] == "box" else (2 * it["radius"], 2 * it["radius"], it["height"])
        out.append(dict(
            shape=it["kind"], size=list(size), radius=0.0 if it["kind"] == "box" else it["radius"],
            pos=list(pos), yaw_deg=math.degrees(yaw), mass=it["mass"], friction=it["friction"],
            color=list(it["color"]),
        ))
    return out


def sample_task_layout(traj_data: dict, rng: random.Random, env_index: int | None = None):
    """Sample one episode for a cabinet task JSON: (layout, obstacle dicts).

    The item pool is built as in ``generate_cabinet_task.py``; ``env_index``
    (random if None) picks this env's items from it.
    """
    cab = cabinet_spec(traj_data["cabinet"])
    clutter = clutter_spec(traj_data["clutter"])
    obj = traj_data["object"]
    surfaces = shelf_surfaces(cab)
    caps = [front_height_cap(s, obj, clutter) for s in surfaces]
    goal_caps = [caps[lv] for lv in clutter["goal_levels"] if lv < len(surfaces) and caps[lv] >= 0]
    pool = sample_clutter_pool(clutter, min(goal_caps))
    if env_index is None:
        env_index = rng.randrange(clutter["pool_size"])
    items = env_items(pool, env_index)
    layout = sample_layout(cab, clutter, obj, items, traj_data["start_pose"], rng)
    return layout, layout_obstacles(cab, items, layout)
