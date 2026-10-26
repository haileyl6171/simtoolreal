"""Open-front cabinet geometry, shared by the Isaac Sim env, task generators and viewers.

Pure Python (no isaaclab import) so viser tools can use it. Every function takes
a cabinet spec as either a dict (the ``"cabinet"`` block of a trajectory JSON)
or any object with the same attributes (``CabinetCfg``).

Cabinet frame: origin at the cabinet's outer-box centre, x = width, y = depth,
z = up. The opening faces +y (toward the robot, which stands at y = +0.8);
the back wall is at -y. ``pos`` is that centre in env-local coordinates at the
default table height, ``yaw_deg`` rotates about z.
"""

from __future__ import annotations

import math
from pathlib import Path

CABINET_DEFAULTS = dict(
    pos=(0.10, -0.10, 0.71),
    yaw_deg=0.0,
    size=(0.26, 0.20, 0.36),  # outer (width x, depth y, height z)
    wall_thickness=0.015,
    # Interior shelves between bottom and top panels; heights are the shelf
    # *top* surfaces relative to the cabinet bottom (outer). Empty = evenly
    # spaced num_shelves.
    num_shelves=1,
    shelf_heights=(),
    color=(0.55, 0.40, 0.28),
    friction=0.5,
)


def _get(cab, name):
    if isinstance(cab, dict):
        return cab.get(name, CABINET_DEFAULTS[name])
    return getattr(cab, name)


def cabinet_spec(cab) -> dict:
    """Normalize a dict/cfg into a full dict with every field filled in."""
    return {k: _get(cab, k) for k in CABINET_DEFAULTS}


def shelf_top_heights(cab) -> list[float]:
    """Top-surface z of every support level (bottom panel first), cabinet frame."""
    w, d, h = _get(cab, "size")
    t = _get(cab, "wall_thickness")
    bottom_top = -h / 2 + t
    heights = list(_get(cab, "shelf_heights"))
    if heights:
        shelves = [-h / 2 + z for z in heights]
    else:
        n = int(_get(cab, "num_shelves"))
        # Split the inner height (bottom top .. top panel underside) into n+1
        # equal compartments, counting each shelf's thickness.
        inner = (h - 2 * t) - n * t
        gap = inner / (n + 1)
        shelves = [bottom_top + (i + 1) * gap + (i + 1) * t for i in range(n)]
    return [bottom_top] + shelves


def cabinet_panels(cab) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
    """(name, centre, size) of every box panel, in the cabinet frame."""
    w, d, h = _get(cab, "size")
    t = _get(cab, "wall_thickness")
    panels = [
        ("bottom", (0.0, 0.0, -h / 2 + t / 2), (w, d, t)),
        ("top", (0.0, 0.0, h / 2 - t / 2), (w, d, t)),
        ("left", (-w / 2 + t / 2, 0.0, 0.0), (t, d, h)),
        ("right", (w / 2 - t / 2, 0.0, 0.0), (t, d, h)),
        ("back", (0.0, -d / 2 + t / 2, 0.0), (w - 2 * t, t, h - 2 * t)),
    ]
    # Interior shelves span between the side walls, from the back wall to the front.
    for i, z_top in enumerate(shelf_top_heights(cab)[1:]):
        panels.append(
            (f"shelf_{i}", (0.0, t / 2, z_top - t / 2), (w - 2 * t, d - t, t))
        )
    return panels


def shelf_surfaces(cab) -> list[dict]:
    """Usable support surfaces, cabinet frame: z_top, x/y ranges, clear height above."""
    w, d, h = _get(cab, "size")
    t = _get(cab, "wall_thickness")
    tops = shelf_top_heights(cab)
    # Underside of whatever is above each level (next shelf, or the top panel).
    ceilings = [z - t for z in tops[1:]] + [h / 2 - t]
    return [
        dict(
            level=i,
            z_top=z,
            x_range=(-w / 2 + t, w / 2 - t),
            y_range=(-d / 2 + t, d / 2),
            clear_height=ceil - z,
        )
        for i, (z, ceil) in enumerate(zip(tops, ceilings))
    ]


def cabinet_to_env(cab, p) -> tuple[float, float, float]:
    """Cabinet-frame point -> env-local point (at the default table height)."""
    px, py, pz = _get(cab, "pos")
    yaw = math.radians(_get(cab, "yaw_deg"))
    c, s = math.cos(yaw), math.sin(yaw)
    return (px + c * p[0] - s * p[1], py + s * p[0] + c * p[1], pz + p[2])


def write_cabinet_urdf(cab, path: str | Path, name: str = "cabinet") -> Path:
    """Write the cabinet as one link with a box collision/visual per panel."""
    r, g, b = _get(cab, "color")
    blocks = []
    for pname, (cx, cy, cz), (sx, sy, sz) in cabinet_panels(cab):
        geom = (
            f'      <origin xyz="{cx:.6f} {cy:.6f} {cz:.6f}"/>\n'
            f'      <geometry>\n        <box size="{sx:.6f} {sy:.6f} {sz:.6f}"/>\n      </geometry>\n'
        )
        blocks.append(
            f"    <!-- {pname} -->\n"
            f'    <visual>\n      <material name="cabinet_wood">\n'
            f'        <color rgba="{r} {g} {b} 1.0"/>\n      </material>\n{geom}    </visual>\n'
            f"    <collision>\n{geom}    </collision>\n"
        )
    text = (
        '<?xml version="1.0"?>\n'
        f'<robot name="{name}">\n  <link name="box">\n'
        + "".join(blocks)
        + '    <inertial>\n      <mass value="500"/>\n'
        '      <inertia ixx="1000.0" ixy="0.0" ixz="0.0" iyy="1000.0" iyz="0.0" izz="1000.0"/>\n'
        "    </inertial>\n  </link>\n</robot>\n"
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path
