import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from src.pipeline.layer_detection import get_axis_for_nuclei


def plot_image_with_points(image: np.ndarray, data_dict: dict[str,np.ndarray], ax=None) -> None:
    """Plot an image with overlaid nucleus centroid points.

    Args:
        image: Input image as a numpy array.
        data_dict: Dictionary containing nuclei data with a 'points' key for centroid
            coordinates.
        ax: Optional matplotlib axes object. If None, creates a new figure.
            Defaults to None.
    """
    if ax is None:
        plt.figure(figsize=(12, 12))
        show_plot = True
    else:
        show_plot = False
    
    points = data_dict["points"]
    plt.imshow(image)
    n_points = len(points)
    random_values = np.random.rand(n_points)
    plt.scatter(points[:, 1], points[:, 0], 
             c=random_values, cmap='tab20', s=15, alpha=1)
    plt.axis('off')
    plt.tight_layout()

    if show_plot:
        plt.show()


def visualize_graph_overlay(image: np.ndarray, filtered_graph: nx.Graph, 
                           node_size: int = 50, edge_width: float = 2.0,
                           node_color: str = 'blue', edge_color: str = 'cyan',
                           alpha: float = 0.7, ax=None) -> None:
    """Visualize nuclei graph overlay with connected components colored distinctly.

    Args:
        image: Input image as a numpy array.
        filtered_graph: NetworkX graph containing nuclei nodes and their connections.
        node_size: Size of node markers. Defaults to 50.
        edge_width: Width of edge lines. Defaults to 2.0.
        node_color: Color for nodes (overridden by component colors). Defaults to 'blue'.
        edge_color: Color for edges (overridden by component colors). Defaults to 'cyan'.
        alpha: Transparency level for nodes and edges (0.0 to 1.0). Defaults to 0.7.
        ax: Optional matplotlib axes object. If None, creates a new figure.
            Defaults to None.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))
        show_plot = True
    else:
        show_plot = False
    
    # Display image
    ax.imshow(image)
    ax.axis('off')

    components = list(nx.connected_components(filtered_graph))
    n_components = len(components)
    cmap = plt.cm.get_cmap('tab20')
    colors = [cmap(i / max(n_components - 1, 1)) for i in range(n_components)]

    node_to_color = {}
    for i, component in enumerate(components):
        for node in component:
            node_to_color[node] = colors[i]

    # Draw edges
    for (n1, n2) in filtered_graph.edges():
        pos1 = filtered_graph.nodes[n1]['pos']
        pos2 = filtered_graph.nodes[n2]['pos']
        edge_color = node_to_color[n1]
        ax.plot([pos1[1], pos2[1]], [pos1[0], pos2[0]], 
                color=edge_color, linewidth=edge_width, alpha=alpha)
    
    # Draw nodes
    for node in filtered_graph.nodes():
        pos = filtered_graph.nodes[node]['pos']
        node_color = node_to_color[node]
        ax.scatter(pos[1], pos[0], s=node_size, c=[node_color], 
                  linewidths=1, alpha=alpha, zorder=5)
    
    plt.tight_layout()

    if show_plot:
        plt.show()

    
def visualize_nodes_by_similarity(image: np.ndarray, filtered_graph: nx.Graph,
                                  node_size: int = 50, alpha: float = 0.7, ax=None) -> None:
    """Visualize nuclei nodes colored by their similarity scores.

    Args:
        image: Input image as a numpy array.
        filtered_graph: NetworkX graph containing nuclei nodes with 'best_similarity'
            attributes.
        node_size: Size of node markers. Defaults to 50.
        alpha: Transparency level for nodes (0.0 to 1.0). Defaults to 0.7.
        ax: Optional matplotlib axes object. If None, creates a new figure.
            Defaults to None.
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))
        show_plot = True
    else:
        show_plot = False
    
    ax.imshow(image)
    ax.axis('off')
    
    similarities = []
    positions = []
    
    for node in filtered_graph.nodes():
        similarity = filtered_graph.nodes[node].get('best_similarity', 0.0)
        pos = filtered_graph.nodes[node]['pos']
        similarities.append(similarity)
        positions.append([pos[0], pos[1]])
    
    if len(similarities) == 0:
        print("No nodes with best_similarity attribute found")
        plt.tight_layout()
        plt.show()
        return
    
    similarities = np.array(similarities)
    positions = np.array(positions)
    
    # Create blue-to-red colormap
    cmap = plt.cm.get_cmap('coolwarm')
    
    # Plot all nodes at once for better performance
    scatter = ax.scatter(positions[:, 1], positions[:, 0], 
                        s=node_size, c=similarities, 
                        cmap=cmap, alpha=alpha, zorder=5, vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Best Similarity', rotation=270, labelpad=15)
    
    plt.tight_layout()

    if show_plot:
        plt.show()


def visualize_nuclei_axes(image: np.ndarray, points: np.ndarray, axes: np.ndarray,
                          line_length: float = 25.0, point_size: int = 30,
                          line_color: str = 'green', point_color: str = 'blue',
                          alpha: float = 0.8, linewidth: float = 2.0, ax=None) -> None:
    """Visualize nuclei with their principal orientation axes.

    Args:
        image: Input image as a numpy array.
        points: Array of nucleus centroid coordinates.
        axes: Array of orientation axis vectors for each nucleus.
        line_length: Half-length of the axis line extending from centroid.
            Defaults to 25.0.
        point_size: Size of point markers for centroids. Defaults to 30.
        line_color: Color for axis lines. Defaults to 'green'.
        point_color: Color for centroid points. Defaults to 'blue'.
        alpha: Transparency level for visualization (0.0 to 1.0). Defaults to 0.8.
        linewidth: Width of axis lines. Defaults to 2.0.
        ax: Optional matplotlib axes object. If None, creates a new figure.
            Defaults to None.
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))
        show_plot = True
    else:
        show_plot = False
    
    ax.imshow(image)
    ax.axis('off')
    
    for point, axis in zip(points, axes):
        # Calculate line endpoints (extending in both directions)
        start_point = point - axis * line_length
        end_point = point + axis * line_length
        
        # Draw line through the point
        ax.plot([start_point[1], end_point[1]], 
               [start_point[0], end_point[0]],
               color=line_color, linewidth=linewidth, alpha=alpha, zorder=4)
    
    ax.scatter(points[:, 1], points[:, 0], s=point_size, c=point_color,
              alpha=0.5, zorder=5)
    
    plt.tight_layout()

    if show_plot:
        plt.show()


def visualize_graph_overlay_with_axes(image: np.ndarray, filtered_graph: nx.Graph, filtered_data_dict: dict[str, np.ndarray],
                           node_size: int = 50, edge_width: float = 2.0,
                           node_color: str = 'blue', edge_color: str = 'cyan',
                           alpha: float = 0.5) -> None:
    """Visualize nuclei graph with both connections and orientation axes.

    Args:
        image: Input image as a numpy array.
        filtered_graph: NetworkX graph containing nuclei nodes and their connections.
        filtered_data_dict: Dictionary containing nuclei data with 'points' and 'coord'
            keys for centroids and boundary coordinates.
        node_size: Size of node markers. Defaults to 50.
        edge_width: Width of edge lines. Defaults to 2.0.
        node_color: Color for nodes. Defaults to 'blue'.
        edge_color: Color for edges. Defaults to 'cyan'.
        alpha: Transparency level for visualization (0.0 to 1.0). Defaults to 0.5.
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Display image
    ax.imshow(image)
    ax.axis('off')

    points = filtered_data_dict["points"]
    boundary_points = filtered_data_dict["coord"]
    axes = get_axis_for_nuclei(boundary_points)
    
    # Draw edges
    for (n1, n2) in filtered_graph.edges():
        pos1 = filtered_graph.nodes[n1]['pos']
        pos2 = filtered_graph.nodes[n2]['pos']
        ax.plot([pos1[1], pos2[1]], [pos1[0], pos2[0]], 
                color=edge_color, linewidth=edge_width, alpha=alpha)
    
    # Draw nodes
    for node in filtered_graph.nodes():
        pos = filtered_graph.nodes[node]['pos']
        ax.scatter(pos[1], pos[0], s=node_size, c=node_color, 
                  edgecolors='black', linewidths=1, alpha=alpha, zorder=5)

    # Draw axes
    for point, axis in zip(points, axes):
        start_point = point - axis * 25
        end_point = point + axis * 25
        
        ax.plot([start_point[1], end_point[1]], 
               [start_point[0], end_point[0]],
               color="red", linewidth=2, alpha=alpha, zorder=4)
    
    plt.tight_layout()
    plt.show()


def visualize_classification(
    image: np.ndarray, 
    filtered_graph: nx.Graph, 
    classifications: dict[int, str],
    node_size: int = 50, 
    alpha: float = 0.7, 
    ax=None
) -> None:
    """Visualize nuclei classification as organized or unorganized.

    Args:
        image: Input image as a numpy array.
        filtered_graph: NetworkX graph containing nuclei nodes with position information.
        classifications: Dictionary mapping node indices to classification labels
            ("organized" or "unorganized").
        node_size: Size of node markers. Defaults to 50.
        alpha: Transparency level for nodes (0.0 to 1.0). Defaults to 0.7.
        ax: Optional matplotlib axes object. If None, creates a new figure.
            Defaults to None.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))
        show_plot = True
    else:
        show_plot = False
    
    ax.imshow(image)
    ax.axis('off')
    
    organized_pos = []
    unorganized_pos = []
    
    for node in filtered_graph.nodes():
        pos = filtered_graph.nodes[node]['pos']
        classification = classifications.get(node, 'unorganized')
        
        if classification == 'organized':
            organized_pos.append([pos[0], pos[1]])
        else:
            unorganized_pos.append([pos[0], pos[1]])
    
    # Plot organized nuclei in green
    if len(organized_pos) > 0:
        organized_pos = np.array(organized_pos)
        ax.scatter(organized_pos[:, 1], organized_pos[:, 0], 
                  s=node_size, c='green', alpha=alpha, 
                  label="Organized", zorder=5)
    
    # Plot unorganized nuclei in red
    if len(unorganized_pos) > 0:
        unorganized_pos = np.array(unorganized_pos)
        ax.scatter(unorganized_pos[:, 1], unorganized_pos[:, 0], 
                  s=node_size, c='red', alpha=alpha, 
                  label="Unorganized", zorder=5)
    
    ax.legend(loc='upper right')
    
    if show_plot:
        plt.tight_layout()
        plt.show()
