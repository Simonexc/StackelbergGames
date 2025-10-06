from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, final

import torch
from torch_geometric.data import Data
from .config import GraphType


if TYPE_CHECKING:
    from config import EnvConfig


class EnvMapBase(Data, ABC):
    """
    Base class for environment maps.

    This class is used to define the structure of the environment map.
    It inherits from torch_geometric.data.Data to leverage PyTorch Geometric's data handling capabilities.
    Maps are generated using Watts-Strogatz-like topology.
    """
    GRAPH_TYPE: GraphType

    MEAN_DEGREE: int = 4
    MAX_DEGREE: int | None = None
    REWIRING_PROBABILITY: float = 0.1

    @final
    def __init__(self, config: "EnvConfig", device: torch.device | str | None = None) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        generator = torch.Generator()
        generator.seed()
        if config.seed is not None:
            generator = generator.manual_seed(config.seed)

        edge_index = self._generate_graph(config.num_nodes, generator, device)
        x, kwargs = self._prepare_x_and_kwargs(config, generator, device)

        super().__init__(
            x=x,
            edge_index=edge_index,
            device=device,
            **kwargs,
        )

    @abstractmethod
    def _prepare_x_and_kwargs(self, config: "EnvConfig", generator: torch.Generator, device: torch.device | str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Prepare the node features tensor `x` and additional keyword arguments based on the environment configuration.
        This method should be implemented by subclasses to define how node features are initialized.
        """

    @classmethod
    @final
    def load(cls, path: str, device: torch.device) -> "EnvMapBase":
        obj = torch.load(path, weights_only=False).to(device)
        obj.device = device
        return obj

    @final
    def save(self, path: str) -> None:
        torch.save(self, path)

    @final
    def _generate_graph(
        self,
        num_nodes: int,
        generator: torch.Generator,
        device: torch.device,
    ) -> torch.Tensor:
        edge_list = []
        degrees = torch.zeros(num_nodes, dtype=torch.int32)
        for i in range(num_nodes):
            for j in range(1, self.MEAN_DEGREE // 2 + 1):
                neighbor = (i + j) % num_nodes
                # Check if rewiring
                if torch.rand(torch.Size(()), generator=generator) < self.REWIRING_PROBABILITY:
                    neighbor = torch.randint(
                        low=0,
                        high=num_nodes,
                        size=torch.Size(()),
                        generator=generator,
                    ).item()

                # Skip self-loops
                if neighbor == i:
                    continue

                edges = [(i, neighbor), (neighbor, i)]
                if self.GRAPH_TYPE == GraphType.directed:
                    edges = [edges[
                        torch.randint(
                            low=0,
                            high=2,
                            size=torch.Size(()),
                            generator=generator,
                        ).item()
                    ]]

                # Degree for directed graph is outdegree (number of outgoing edges)
                if self.MAX_DEGREE is not None and any([degrees[edge[0]] >= self.MAX_DEGREE for edge in edges]):
                    continue
                edge_list.extend(edges)
                for edge in edges:
                    degrees[edge[0]] += 1

        # Remove duplicates and self-loops
        edge_set = set((u, v) for u, v in edge_list if u != v)
        edge_index = torch.tensor(list(edge_set), dtype=torch.long, device=device).t().contiguous()  # convert to shape [2, num_edges]

        return edge_index

    @staticmethod
    @final
    def _create_binary_nodes(
        min_num: int,
        percentage: float,
        num_nodes: int,
        generator: torch.Generator,
        device: torch.device,
        exclude_nodes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Create a binary tensor indicating which nodes are selected based on the given percentage.
        Ensures at least `min_num` nodes are selected.
        """
        if exclude_nodes is None:
            exclude_nodes = torch.tensor([], dtype=torch.int32, device=device)

        num_selected = max(min_num, int(num_nodes * percentage))
        selected_nodes = torch.zeros(num_nodes, dtype=torch.bool, device=device)

        perm = torch.randperm(num_nodes, generator=generator).to(device)
        perm = perm[~torch.isin(perm, exclude_nodes)]  # Exclude specified nodes
        selected_nodes[perm[:num_selected]] = True

        return selected_nodes


class EnvMapUndirected(EnvMapBase, ABC):
    """
    Base class for undirected environment maps.
    Inherits from EnvMapBase.
    """
    GRAPH_TYPE = GraphType.undirected


class EnvMapDirected(EnvMapBase, ABC):
    """Base class for directed environment maps."""
    GRAPH_TYPE = GraphType.directed

    @final
    def get_successors_batched(self, node_indices: torch.Tensor) -> torch.Tensor:
        """
        Return a boolean (*node_indices_shape, num_nodes) mask indicating which nodes are successors of the given nodes.
        """
        mask = self.edge_index[0] == node_indices.unsqueeze(-1)
        non_zero = mask.nonzero(as_tuple=True)
        batch_node_indices = non_zero[:-1]
        edge_indices = non_zero[-1]
        successor_nodes = self.edge_index[1][edge_indices]

        successors_matrix = torch.zeros((*node_indices.shape, self.num_nodes), dtype=torch.bool, device=self.device)
        successors_matrix[*batch_node_indices, successor_nodes] = True
        return successors_matrix

    @final
    def get_successors(self, node_idx: int) -> torch.Tensor:
        """
        Return a boolean mask indicating which nodes are successors of the given node.
        """
        mask = self.edge_index[0] == node_idx
        is_successor = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)
        is_successor[self.edge_index[1][mask]] = True
        return is_successor

    @final
    def get_predecessors_batched(self, node_indices: torch.Tensor) -> torch.Tensor:
        """
        Return a boolean (*node_indices_shape, num_nodes) mask indicating which nodes are predecessors of the given nodes.
        """
        mask = self.edge_index[1] == node_indices.unsqueeze(-1)
        non_zero = mask.nonzero(as_tuple=True)
        batch_node_indices = non_zero[:-1]
        edge_indices = non_zero[-1]
        predecessor_nodes = self.edge_index[0][edge_indices]

        predecessor_matrix = torch.zeros((*node_indices.shape, self.num_nodes), dtype=torch.bool, device=self.device)
        predecessor_matrix[*batch_node_indices, predecessor_nodes] = True
        return predecessor_matrix

    @final
    def get_predecessors(self, node_idx: int) -> torch.Tensor:
        """
        Return a boolean mask indicating which nodes are predecessors of the given node.
        """
        mask = self.edge_index[1] == node_idx
        is_predecessor = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)
        is_predecessor[self.edge_index[0][mask]] = True
        return is_predecessor
