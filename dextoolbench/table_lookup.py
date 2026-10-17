"""Table URDF selection logic, shared between eval and visualize scripts."""

TABLE_URDF = "urdf/table_narrow.urdf"
TABLE_WHITEBOARD_URDF = "urdf/table_narrow_whiteboard.urdf"
TABLE_NAIL_URDF = "urdf/table_narrow_nail.urdf"
TABLE_BOWL_PLATE_URDF = "urdf/table_narrow_bowl_plate.urdf"
TABLE_CLUTTER_URDF = "urdf/table_narrow_clutter.urdf"

OBJECT_CATEGORY_TO_TABLE_URDF = {
    "hammer": TABLE_NAIL_URDF,
    "spatula": TABLE_BOWL_PLATE_URDF,
    "eraser": TABLE_WHITEBOARD_URDF,
    "screwdriver": TABLE_URDF,
    "marker": TABLE_WHITEBOARD_URDF,
    "brush": TABLE_URDF,
}

# Task-specific overrides (keyed by (object_category, task_name)).
OBJECT_CATEGORY_TASK_TO_TABLE_URDF = {
    ("eraser", "clutter"): TABLE_CLUTTER_URDF,
}


def select_table_urdf(object_category: str, task_name: str, force_default: bool = False) -> str:
    """Return the relative table URDF path for the given task."""
    if force_default:
        return TABLE_URDF
    key = (object_category, task_name)
    return OBJECT_CATEGORY_TASK_TO_TABLE_URDF.get(
        key, OBJECT_CATEGORY_TO_TABLE_URDF.get(object_category, TABLE_URDF)
    )
