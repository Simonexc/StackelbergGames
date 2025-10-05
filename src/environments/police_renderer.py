import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch

from .police import PoliceEnv
from config import Player


class Renderer:
    def __init__(self, env: PoliceEnv, seed: int | None = None) -> None:
        self._seed = seed
        assert env.batch_size == torch.Size(()), f"Batch size should be empty, but got {env.batch_size}"

        self.env = env
        self._graph: nx.Graph | None = None
        self._initialize_graph()

        plt.ion()  # Interactive mode on
        self._fig, self._ax = plt.subplots(figsize=(10, 10))
        self._pos = nx.spring_layout(self._graph, k=6/np.sqrt(self.env.map.num_nodes), seed=self._seed)  # Consistent layout

    def _initialize_graph(self) -> None:
        self._graph = nx.from_edgelist(self.env.map.edge_index.t().numpy(), create_using=nx.Graph)

    def render(self):
        self._ax.clear()

        node_colors = []
        border_colors = []

        for node in self._graph.nodes:
            # Determine node color based on who's there
            if node == self.env.position[-1]:  # Attacker/Robber position
                if (self.env.position[:-1] == node).any():  # Both attacker and defender(s)
                    node_colors.append("purple")
                else:
                    node_colors.append("red")
            elif (self.env.position[:-1] == node).any():  # Defender(s) position
                node_colors.append("blue")
            else:
                node_colors.append("gray")  # Empty nodes

            # Determine border color based on node type
            if self.env.hideout_idx == node:
                border_colors.append("darkgreen")  # Hideout
            elif self.env.map.target_nodes[node]:
                if self.env.targets_attacked[node]:
                    border_colors.append("black")  # Attacked target
                else:
                    border_colors.append("gold")  # Unattacked target
            else:
                border_colors.append("gray")  # Regular node

        # Draw edges
        nx.draw_networkx_edges(self._graph, self._pos, ax=self._ax, edge_color='gray')

        # Draw nodes
        nx.draw_networkx_nodes(self._graph, self._pos, ax=self._ax, node_color=node_colors,
                               node_size=500, edgecolors=border_colors, linewidths=3)

        # Draw node labels
        nx.draw_networkx_labels(self._graph, self._pos, ax=self._ax, font_color='white', font_weight='bold')

        # Add labels for special nodes and rewards
        text_offset_y = 0.05  # Vertical offset for text below nodes

        for i in range(self.env.map.num_nodes):
            labels = []
            if self.env.map.target_nodes[i]:
                labels.append(f"{self.env.map.x[i, 0]:.2f}")  # Reward value

            if (self.env.position[:-1] == i).any():
                labels.append(f"P: {",".join(map(str, torch.where(self.env.position[:-1] == i)[0].numpy().tolist()))}")

            if not labels:
                continue

            x, y = self._pos[i]

            self._ax.text(x, y - text_offset_y, "; ".join(labels),
                          fontsize=8,  # Smaller font size for readability
                          ha='center',  # Horizontal alignment: center the text below the node
                          va='top',  # Vertical alignment: top of the text box aligns with y - offset
                          color='black',
                          # Optional: Add a background box for better readability, especially if edges overlap
                          bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6)
                          )

        # Add title with game info
        title = (
            f"Police - Step: {self.env.step_count[0].item()}/{self.env.num_steps}\n"
            # f"Scores: Def={self.env.cumulative_rewards[Player.defender.value]:.2f}, "
            # f"Att={self.env.cumulative_rewards[Player.attacker.value]:.2f}"
        )
        self._ax.set_title(title)
        self._ax.axis('off')  # Hide axes

        plt.pause(0.5 / 1)  # Pause for visibility
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()  # Process GUI events

    def close(self):
        """Closes the rendering window."""
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None
            plt.ioff()  # Turn interactive mode off