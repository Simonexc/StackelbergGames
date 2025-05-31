import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch

from .flipit_geometric import FlipItEnv
from config import Player
from torchrl.envs.libs.dm_control import DMControlEnv

class Renderer:
    def __init__(self, env: FlipItEnv, seed: int | None = None) -> None:
        self._seed = seed
        assert env.batch_size == torch.Size(()), f"Batch size should be empty, but got {env.batch_size}"

        self.env = env
        self._graph: nx.DiGraph | None = None
        self._initialize_graph()

        plt.ion()  # Interactive mode on
        self._fig, self._ax = plt.subplots(figsize=(8, 8))
        self._pos = nx.spring_layout(self._graph, k=4/np.sqrt(self.env.map.num_nodes), seed=self._seed)  # Consistent layout

    def _initialize_graph(self) -> None:
        self._graph = nx.from_edgelist(self.env.map.edge_index.t().numpy(), create_using=nx.DiGraph)

    def render(self):
        self._ax.clear()

        node_colors = []
        border_colors = []
        for node in self._graph.nodes:
            owner = self.env.node_owners[node]
            is_entry = self.env.map.entry_nodes[node]

            if owner == Player.defender.value:
                node_colors.append('blue')
            elif owner == Player.attacker.value:
                node_colors.append('red')
            else:
                raise RuntimeError(f"Invalid node owner: {owner}")

            if is_entry:
                border_colors.append('green')
            else:
                border_colors.append('black')

        nx.draw_networkx_edges(self._graph, self._pos, ax=self._ax, edge_color='gray', arrows=True, arrowstyle='->',
                               arrowsize=15)

        nx.draw_networkx_nodes(self._graph, self._pos, ax=self._ax, node_color=node_colors,
                               node_size=500, edgecolors=border_colors, linewidths=3)  # Use edgecolors for border
        nx.draw_networkx_labels(self._graph, self._pos, ax=self._ax, font_color='white', font_weight='bold')

        # --- Add this section to display rewards and costs ---
        # Determine a suitable vertical offset for the text below the node
        # This value might need slight adjustment depending on your graph size and layout
        text_offset_y = 0.05  # Example offset

        for i in range(self.env.map.num_nodes):
            # Get node position
            x, y = self._pos[i]

            # Get reward and cost (assuming node index 'i' corresponds to array index 'i')
            reward = self.env.map.x[i, 0]
            cost = self.env.map.x[i, 1]  # Note: costs are negative

            # Format the label string. Use f-strings for formatting decimal places.
            label = f"{reward:.2f} / {cost:.2f}"

            # Add text label to the axes
            self._ax.text(x, y - text_offset_y, label,
                          fontsize=8,  # Smaller font size for readability
                          ha='center',  # Horizontal alignment: center the text below the node
                          va='top',  # Vertical alignment: top of the text box aligns with y - offset
                          color='black',
                          # Optional: Add a background box for better readability, especially if edges overlap
                          bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6)
                          )

        # Add title with game info
        title = (
            f"FlipIt - Step: {self.env.step_count[0].item()}/{self.env.num_steps}\n"
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
