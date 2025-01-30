import networkx as nx
import math

def adjacency_matrix_to_graph(adj_matrix):
    """
    Converts an adjacency matrix into a weighted directed NetworkX graph.
    :param adj_matrix: 2D list or numpy array representing the adjacency matrix.
                       Use float('inf') for no connection between nodes.
    :return: A NetworkX directed graph.
    """
    n = len(adj_matrix)
    G = nx.DiGraph()

    for i in range(n):
        for j in range(n):
            if adj_matrix[i][j] != math.inf:  # Add edge if weight is not infinite
                G.add_edge(i, j, weight=adj_matrix[i][j])

    return G


def find_k_shortest_paths_from_matrix(adj_matrix, source, target, k):
    """
    Finds the k shortest paths in a weighted graph represented as an adjacency matrix.
    :param adj_matrix: 2D list or numpy array representing the adjacency matrix.
    :param source: Source node.
    :param target: Target node.
    :param k: Number of shortest paths to find.
    :return: List of tuples [(path, path_length)] containing paths and their weights.
    """
    # Convert the adjacency matrix to a NetworkX graph
    G = adjacency_matrix_to_graph(adj_matrix)

    # Use NetworkX's built-in k-shortest paths algorithm
    paths = nx.shortest_simple_paths(G, source, target, weight="weight")
    results = []

    # Get the k paths and their weights
    for path in paths[:k]:
        weight_sum = sum(G[path[i]][path[i + 1]]['weight'] for i in range(len(path) - 1))
        results.append((path, weight_sum))

    return results

def find_k_shortest_paths_under_weight_sum_threshold(adj_matrix, source, target, th):
    """
    Finds the k shortest paths in a weighted graph represented as an adjacency matrix.
    :param adj_matrix: 2D list or numpy array representing the adjacency matrix.
    :param source: Source node.
    :param target: Target node.
    :param k: Number of shortest paths to find.
    :return: List of tuples [(path, path_length)] containing paths and their weights.
    """
    # Convert the adjacency matrix to a NetworkX graph
    G = adjacency_matrix_to_graph(adj_matrix)

    # Use NetworkX's built-in k-shortest paths algorithm
    paths = nx.shortest_simple_paths(G, source, target, weight="weight")
    results = []

    # Get the k paths and their weights
    for path in paths:
        weight_sum = sum(G[path[i]][path[i + 1]]['weight'] for i in range(len(path) - 1))
        if weight_sum > th:
            break
        results.append((path, weight_sum))

        # print(f"Found ({path}, {weight_sum})")

    return results


# Example usage
if __name__ == "__main__":
    # Define the graph as an adjacency matrix
    # Use float('inf') to represent no direct edge
    adj_matrix = [
        [0, 2, 4, float('inf'), float('inf')],
        [float('inf'), 0, 1, 7, float('inf')],
        [float('inf'), float('inf'), 0, 3, float('inf')],
        [float('inf'), float('inf'), float('inf'), 0, 1],
        [float('inf'), float('inf'), float('inf'), float('inf'), 0]
    ]

    source = 0  # Start node
    target = 4  # End node
    k = 3       # Number of shortest paths to find

    # Find the k shortest paths
    k_shortest_paths = find_k_shortest_paths_from_matrix(adj_matrix, source, target, k)
    print(f"The {k} shortest paths from {source} to {target} are:")
    for i, (path, weight) in enumerate(k_shortest_paths, 1):
        print(f"{i}: Path: {path}, Total Weight: {weight}")
