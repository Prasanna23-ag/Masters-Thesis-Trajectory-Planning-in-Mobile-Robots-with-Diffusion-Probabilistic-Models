import os
import numpy as np
import matplotlib.pyplot as plt
import imageio

# -----------------------------------------------------------------------------#
# ------------------------------ Gazebo Renderer ------------------------------#
# -----------------------------------------------------------------------------#

class GazeboRenderer:
    """
    Renderer for Gazebo environments.
    - Plots robot trajectories using odometry or rollout JSON files.
    - Optionally overlays a static map image (PGM/PNG).
    """

    def __init__(self, map_path=None, bounds=(0, 6, 0, 6)):
        self.bounds = bounds
        self._extent = (bounds[0], bounds[1], bounds[2], bounds[3])

        if map_path is not None and os.path.exists(map_path):
            self._background = imageio.imread(map_path)
        else:
            self._background = None

    def renders(self, observations, conditions=None, title=None, save_path=None, show=True):
        """
        Render a trajectory.
        - observations: np.array of shape (T, 2) or (T, N) where [:2] are x,y
        - conditions: optional waypoints/goals to overlay
        - title: optional plot title
        - save_path: optional path to save image
        - show: whether to display with matplotlib
        """
        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches(6, 6)

        # Draw background map if available
        if self._background is not None:
            plt.imshow(self._background, extent=self._extent, origin="upper", cmap="gray")

        traj = np.array(observations)
        colors = plt.cm.jet(np.linspace(0, 1, len(traj)))
        plt.plot(traj[:, 0], traj[:, 1], c="black", zorder=10)
        plt.scatter(traj[:, 0], traj[:, 1], c=colors, zorder=20)

        if conditions is not None:
            cond = np.array(conditions)
            plt.scatter(cond[:, 0], cond[:, 1], color="red", marker="x", label="conditions")

        plt.xlim(self.bounds[0], self.bounds[1])
        plt.ylim(self.bounds[2], self.bounds[3])
        plt.axis("off")
        if title:
            plt.title(title)

        if save_path is not None:
            plt.savefig(save_path, dpi=150)
            print(f"Saved trajectory to {save_path}")
        if show:
            plt.show()

        plt.close(fig)

    def composite(self, savepath, paths, ncol=5, **kwargs):
        """
        Save multiple trajectories in a grid.
        - paths: list of trajectories [n_paths x horizon x 2]
        """
        images = []
        for path in paths:
            img = self.renders(path, show=False, **kwargs)
            images.append(img)
        images = np.stack(images, axis=0)

        nrow = len(images) // ncol
        images = np.reshape(images, (nrow, ncol, *images.shape[1:]))
        imageio.imsave(savepath, images)
        print(f"Saved {len(paths)} samples to: {savepath}")

    # -------------------------------------------------------------------------#
    # Example: integrate with ROS odometry
    # -------------------------------------------------------------------------#
    def from_ros_odometry(self, odom_msgs):
        """
        Convert a list of ROS Odometry messages into trajectory observations.
        """
        traj = []
        for msg in odom_msgs:
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            traj.append([x, y])
        return np.array(traj)
