import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tyro
import viser
from termcolor import colored
from viser.extras import ViserUrdf

from dextoolbench.table_lookup import select_table_urdf
from dextoolbench.objects import NAME_TO_OBJECT
from isaacgymenvs.utils.utils import get_repo_root_dir


def quat_xyzw_to_wxyz(q):
    return (q[3], q[0], q[1], q[2])


@dataclass
class VisualizeTaskArgs:
    object_category: str = "hammer"
    """Object category (e.g. hammer, marker, eraser, brush, spatula, screwdriver)."""

    object_name: str = "claw_hammer"
    """Object name (e.g. claw_hammer, mallet_hammer)."""

    task_name: str = "swing_down"
    """Task name (e.g. swing_down, swing_side)."""

    table_urdf: Optional[str] = None
    """Override the table URDF path (relative to assets/). Auto-selected from task if None."""

    @property
    def trajectory_path(self) -> Path:
        return (
            Path("dextoolbench/trajectories")
            / self.object_category
            / self.object_name
            / f"{self.task_name}.json"
        )

    @property
    def resolved_table_urdf(self) -> Path:
        if self.table_urdf is not None:
            return get_repo_root_dir() / "assets" / self.table_urdf
        rel = select_table_urdf(self.object_category, self.task_name)
        return get_repo_root_dir() / "assets" / rel


class TaskVisualizer:
    def __init__(self, trajectory_data, object_name, table_urdf: Path, port=8080):
        self.port = port
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.start_pose = np.array(trajectory_data["start_pose"])
        self.goals = [np.array(g) for g in trajectory_data["goals"]]
        self._setup_scene(object_name, table_urdf)

    def _setup_scene(self, object_name, table_urdf: Path):
        @self.server.on_client_connect
        def _(client):
            client.camera.position = (0.0, -1.5, 1.5)
            client.camera.look_at = (0.0, -0.5, 0.7)

        self.server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
        self.server.scene.add_frame(
            "/table", position=(0, 0, 0.38), wxyz=(1, 0, 0, 0), show_axes=False
        )
        ViserUrdf(
            self.server,
            table_urdf,
            root_node_name="/table",
            mesh_color_override=(0, 0, 0, 0.5),
        )

        object_urdf = NAME_TO_OBJECT[object_name].urdf_path
        self.start_frame = self.server.scene.add_frame(
            "/start",
            position=self.start_pose[:3],
            wxyz=quat_xyzw_to_wxyz(self.start_pose[3:7]),
            show_axes=True,
            axes_length=0.1,
            axes_radius=0.002,
        )
        ViserUrdf(self.server, object_urdf, root_node_name="/start")

        self.goal_frame = self.server.scene.add_frame(
            "/goal", show_axes=True, axes_length=0.1, axes_radius=0.002
        )
        ViserUrdf(
            self.server,
            object_urdf,
            root_node_name="/goal",
            mesh_color_override=(0, 255, 0, 0.5),
        )

        self.server.gui.add_markdown(f"**{object_name}** - {len(self.goals)} goals")
        self.slider = self.server.gui.add_slider(
            "Goal", min=0, max=len(self.goals) - 1, step=1, initial_value=0
        )
        self.slider.on_update(lambda _: self._update_goal())
        self._update_goal()

    def _update_goal(self):
        pose = self.goals[int(self.slider.value)]
        self.goal_frame.position = pose[:3]
        self.goal_frame.wxyz = quat_xyzw_to_wxyz(pose[3:7])

    def run(self):
        print(colored(f"Viser: http://localhost:{self.port}", "green"))
        while True:
            time.sleep(1.0)


def main():
    args: VisualizeTaskArgs = tyro.cli(VisualizeTaskArgs)
    trajectory_path = args.trajectory_path
    assert trajectory_path.exists(), (
        f"Trajectory file does not exist: {trajectory_path}"
    )
    print(colored(f"Table URDF: {args.resolved_table_urdf}", "cyan"))
    with open(trajectory_path) as f:
        TaskVisualizer(json.load(f), args.object_name, args.resolved_table_urdf).run()


if __name__ == "__main__":
    main()
