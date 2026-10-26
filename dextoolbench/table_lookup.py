"""Table URDF selection logic, shared between eval and visualize scripts."""

TABLE_URDF = "urdf/table_narrow.urdf"
TABLE_LONG_URDF = "urdf/table_narrow_long.urdf"  # +5 cm toward the robot (cabinet tasks)
TABLE_WHITEBOARD_URDF = "urdf/table_narrow_whiteboard.urdf"
TABLE_NAIL_URDF = "urdf/table_narrow_nail.urdf"
TABLE_BOWL_PLATE_URDF = "urdf/table_narrow_bowl_plate.urdf"

OBJECT_CATEGORY_TO_TABLE_URDF = {
    "hammer": TABLE_NAIL_URDF,
    "spatula": TABLE_BOWL_PLATE_URDF,
    "eraser": TABLE_WHITEBOARD_URDF,
    "screwdriver": TABLE_URDF,
    "marker": TABLE_WHITEBOARD_URDF,
    "brush": TABLE_URDF,
}

# Task-specific overrides (keyed by (object_category, task_name)).
# Pick-and-place uses the plain table; its obstacle is a separate rigid body.
OBJECT_CATEGORY_TASK_TO_TABLE_URDF = {
    ("eraser", "pick_place"): TABLE_URDF,
    ("eraser", "pick_place_obstacle"): TABLE_URDF,
    ("eraser", "pick_place_obstacle_2goal"): TABLE_URDF,
    ("eraser", "pick_place_obstacle_1goal"): TABLE_URDF,
    ("eraser", "pick_place_push"): TABLE_URDF,
    ("eraser", "pick_place_push_noobstacle"): TABLE_URDF,
    ("eraser", "cabinet_place"): TABLE_LONG_URDF,
}


def select_table_urdf(object_category: str, task_name: str, force_default: bool = False) -> str:
    """Return the relative table URDF path for the given task."""
    if force_default:
        return TABLE_URDF
    key = (object_category, task_name)
    return OBJECT_CATEGORY_TASK_TO_TABLE_URDF.get(
        key, OBJECT_CATEGORY_TO_TABLE_URDF.get(object_category, TABLE_URDF)
    )
