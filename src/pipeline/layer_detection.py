import numpy as np
import networkx as nx
from scipy.spatial import Delaunay
from sklearn.decomposition import PCA
from collections import defaultdict
from math import acos, degrees

class LayerDetectionModule:
    def __init__(self, image: np.ndarray, nuclei_data_dict: dict[str, np.ndarray], nuclei_mask: np.ndarray, tissue_mask: np.ndarray, area_th: float = 0.5) -> None:
        self.image = image
        self.nuclei_data_dict = nuclei_data_dict
        self.nuclei_mask = nuclei_mask
        self.tissue_mask = tissue_mask
        self.area_th = area_th
        self.filtered_data_dict = self._filter_data_dict(tissue_mask, nuclei_data_dict, area_th)

    
    def _filter_data_dict(self, mask: np.ndarray, data_dict: dict[str, any], area_th: float = 0.5) -> dict[str, np.ndarray]:
        """Filter nuclei detections based on tissue mask and area threshold.

        Args:
            mask: Binary tissue segmentation mask.
            data_dict: Dictionary containing nuclei detection results with keys
                'points', 'coord', and 'prob'.
            area_th: Area threshold as a fraction of median area. Nuclei with area
                below this threshold are filtered out. Defaults to 0.5.

        Returns:
            Filtered dictionary containing only nuclei that are within tissue regions
            and meet the area threshold criteria.
        """
        points = data_dict["points"]
        median_area = self._calculate_median_area(data_dict["coord"])
        filtered_points = []
        filtered_coords = []
        filtered_probs = []

        binary_mask = (mask > 0).astype(int)
        for i, (point, coord) in enumerate(zip(points, data_dict["coord"])):
            x, y = int(point[0]), int(point[1])
            area = self._poly_area(np.array(coord[0]), np.array(coord[1]))
            if binary_mask[x, y] == 1 and area > area_th * median_area:
                filtered_points.append([point[0], point[1]])
                filtered_coords.append(coord)
                filtered_probs.append(data_dict["prob"][i])

        filtered_data_dict = dict(data_dict)
        filtered_data_dict["points"] = np.array(filtered_points)
        filtered_data_dict["coord"] = np.array(filtered_coords)
        filtered_data_dict["prob"] = np.array(filtered_probs)
        
        return filtered_data_dict

    def _poly_area(self, x: int, y: int) -> float:
        """Calculate the area of a polygon using the shoelace formula.

        Args:
            x: Array of x-coordinates of the polygon vertices.
            y: Array of y-coordinates of the polygon vertices.

        Returns:
            Area of the polygon as a float.
        """
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

    
    def _calculate_median_area(self, coordinates: np.ndarray) -> float:
        """Calculate the median area of all given nuclei.

        Args:
            coordinates: Array of nuclei coordinates where each nucleus is represented
                as a tuple of (x_coords, y_coords).

        Returns:
            Median area across all polygons as a float.
        """
        areas = []
        for coord in coordinates:
            area = self._poly_area(np.array(coord[0]), np.array(coord[1]))
            areas.append(area)
        
        median_area = np.median(np.array(areas))
        return median_area


    def _get_delaunay_neighbors(self, points) -> tuple[dict[int, set[int]], np.ndarray]:
        """Compute Delaunay triangulation neighbors for a set of points.

        Args:
            points: Array of 2D point coordinates.

        Returns:
            Tuple containing:
                - Dictionary mapping each point index to its set of neighbor indices.
                - Numpy array of the input points.
        """
        points = np.array(points)
        tri = Delaunay(points)
        
        neighbors = defaultdict(set)
        
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(3):
                    if i != j:
                        neighbors[simplex[i]].add(simplex[j])
        
        return dict(neighbors), points

    
    def _get_nucleus_orientation(self, boundary_points) -> np.ndarray:
        """Determine the main orientation axis of a nucleus using PCA.

        Args:
            boundary_points: Array of boundary points defining the nucleus contour.
                Can be of shape (n, 2) or (2, n).

        Returns:
            Numpy array representing the main orientation axis (first principal component).
        """
        boundary_points = np.array(boundary_points)
        
        if boundary_points.shape[0] == 2:
            boundary_points = boundary_points.T

        # Center the points
        centroid = boundary_points.mean(axis=0)
        centered = boundary_points - centroid
        
        # PCA to find main axis
        pca = PCA(n_components=2)
        pca.fit(centered)
        
        # First principal component is the main axis
        main_axis = pca.components_[0]
        
        return main_axis

    
    def _calculate_alignment_similarity(self, main_axis: np.ndarray, compared_axis: np.ndarray) -> float:
        """Calculate alignment similarity between two orientation axes.

        Args:
            main_axis: Main orientation axis vector.
            compared_axis: Comparison orientation axis vector.

        Returns:
            Absolute cosine similarity between the two axes (0.0 to 1.0),
            where 1.0 indicates perfect alignment.
        """
        norm_main = np.linalg.norm(main_axis)
        norm_compared = np.linalg.norm(compared_axis)
        
        if norm_main == 0 or norm_compared == 0:
            return 0.0
        
        cosine_similarity = np.dot(main_axis, compared_axis) / (norm_main * norm_compared)
        return np.abs(cosine_similarity)

    
    def _calculate_alignment_angle(self, main_point: tuple[int, int], compared_point: tuple[int, int], main_axis: np.ndarray) -> int:
        """Calculate the alignment angle between a point-to-point vector and an orientation axis.

        Args:
            main_point: Coordinates of the main point (x, y).
            compared_point: Coordinates of the compared point (x, y).
            main_axis: Main orientation axis vector.

        Returns:
            Alignment angle in degrees (0 to 90), where 0 indicates perpendicular
            alignment and 90 indicates parallel alignment.
        """
        direction_vector = np.array([
            compared_point[0] - main_point[0],
            compared_point[1] - main_point[1]
        ])
        
        direction_norm = np.linalg.norm(direction_vector)
        if direction_norm == 0:
            return 0.0
        
        direction_vector = direction_vector / direction_norm
        main_axis = main_axis / np.linalg.norm(main_axis)
        
        cos_angle = np.clip(np.dot(direction_vector, main_axis), -1.0, 1.0)
        
        # Calculate angle in radians, then convert to degrees
        angle_rad = acos(cos_angle)
        angle_deg = degrees(angle_rad)
        
        if angle_deg > 90:
            angle_deg = 180 - angle_deg
        angle_deg = 90 - angle_deg
        
        return np.round(angle_deg)


    def _get_distance(self, main_point: np.ndarray, compared_point: np.ndarray) -> float:
        """Calculate Euclidean distance between two points.

        Args:
            main_point: Coordinates of the first point.
            compared_point: Coordinates of the second point.

        Returns:
            Euclidean distance as a float.
        """
        return np.linalg.norm(main_point - compared_point)


    def _calculate_similarity_score(self, alignment: float, angle: float, weights: np.ndarray = np.array([0.5, 0.5])) -> float:
        """Calculate a combined similarity score from alignment and angle metrics.

        Args:
            alignment: Alignment similarity value (0.0 to 1.0).
            angle: Alignment angle in degrees (0 to 90).
            weights: Array of weights for [alignment_score, angle_score].
                Defaults to [0.5, 0.5] for equal weighting.

        Returns:
            Combined similarity score as a weighted sum of alignment and angle scores.
        """
        alignment_score = alignment
        angle_score = 1.0 - (angle / 90.0)
        
        similarity = np.sum(np.array([alignment_score, angle_score]) * weights)
        
        return similarity


    def _get_median_distance(self, points: np.ndarray, neighbor_dict: dict[int, set[int]]) -> float:
        """Calculate median distance between neighboring points.

        Args:
            points: Array of point coordinates.
            neighbor_dict: Dictionary mapping point indices to their neighbor indices.

        Returns:
            Median distance between all neighboring pairs, or 0.0 if no neighbors exist.
        """
        distances = []
        for i, point in enumerate(points):
            for neighbor in neighbor_dict[i]:
                if i < neighbor:
                    neighbor_point = points[neighbor]
                    distances.append(self._get_distance(point, neighbor_point))
        
        if len(distances) == 0:
            return 0.0
        
        return np.median(distances)

    
    def build_neighbor_graph(self, points: np.ndarray, neighbor_dict: dict[int, set[int]], boundary_points: np.ndarray, distance_threshold: float) -> nx.Graph:
        """Build a graph of nuclei neighbors with edge attributes for alignment analysis.

        Args:
            points: Array of nucleus centroid coordinates.
            neighbor_dict: Dictionary mapping each nucleus index to its neighbor indices.
            boundary_points: Array of boundary coordinates for each nucleus.
            distance_threshold: Maximum distance for including edges in the graph.

        Returns:
            NetworkX graph where nodes represent nuclei and edges contain distance,
            alignment, and angle attributes. Each node has a 'best_similarity' attribute
            representing the average similarity to its top 2 neighbors.
        """
        G = nx.Graph()

        for i, (x, y) in enumerate(points):
            G.add_node(i, pos=(x, y))
        
        for i, point in enumerate(points):
            main_boundary = boundary_points[i]
            main_axis = self._get_nucleus_orientation(main_boundary)

            for neighbor in neighbor_dict[i]:
                neighbor_point = points[neighbor]
                distance = self._get_distance(point, neighbor_point)

                if distance <= distance_threshold:
                    neighbor_boundary = boundary_points[neighbor]
                    neighbor_axis = self._get_nucleus_orientation(neighbor_boundary)
                    
                    alignment = self._calculate_alignment_similarity(main_axis, neighbor_axis)
                    angle = self._calculate_alignment_angle(point, neighbor_point, main_axis)
                    

                    G.add_edge(i, neighbor, distance=distance, alignment=alignment, angle=angle)

        for i in G.nodes():
            neighbors = list(G.neighbors(i))
            if len(neighbors) == 0:
                G.nodes[i]['best_similarity'] = 0.0
                continue
            
            neighbor_scores = []
            for neighbor in neighbors:
                edge_data = G[i][neighbor]
                similarity = self._calculate_similarity_score(
                    edge_data['alignment'],
                    edge_data['angle']
                )
                neighbor_scores.append((neighbor, similarity))
            
            # Get top 2 neighbors
            neighbor_scores.sort(key=lambda x: x[1], reverse=True)
            top_two = neighbor_scores[:2]
            
            # Calculate combined similarity score
            if len(top_two) == 2:
                best_similarity = (top_two[0][1] + top_two[1][1]) / 2.0
            elif len(top_two) == 1:
                best_similarity = top_two[0][1]
            else:
                best_similarity = 0.0
            
            G.nodes[i]['best_similarity'] = best_similarity

        return G

    def _filter_neighbor_graph(self, G: nx.Graph, n1: int, n2: int, alignment_threshold: float, angle_threshold: float, distance_threshold: float | None = None) -> bool:
        """Determine if an edge between two nodes meets filtering criteria.

        Args:
            G: NetworkX graph containing nuclei and their relationships.
            n1: Index of the first node.
            n2: Index of the second node.
            alignment_threshold: Minimum alignment value for edge to pass filter.
            angle_threshold: Maximum angle value (in degrees) for edge to pass filter.
            distance_threshold: Maximum distance for edge to pass filter. If None,
                distance is not checked. Defaults to None.

        Returns:
            True if the edge meets all threshold criteria, False otherwise.
        """
        edge = G[n1][n2]

        alignment = edge["alignment"]
        angle = edge["angle"]
        distance = edge["distance"]

        alignment_ok = alignment >= alignment_threshold
        angle_ok = angle <= angle_threshold
        distance_ok = (distance_threshold is None) or (distance <= distance_threshold)

        return alignment_ok and angle_ok and distance_ok

    
    def _filter_graph_top_n(self, G: nx.Graph, n: int = 2) -> nx.Graph:
        """Filter graph to keep only the top N most similar neighbors for each node.

        Args:
            G: NetworkX graph containing nuclei and their relationships.
            n: Number of top neighbors to keep for each node. Defaults to 2.

        Returns:
            Filtered graph containing only edges to the top N neighbors for each node,
            based on similarity scores.
        """
        G_filtered = G.copy()

        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if len(neighbors) <= n:
                continue
            
            neighbor_scores = []
            for neighbor in neighbors:
                edge_data = G[node][neighbor]
                similarity = self._calculate_similarity_score(
                    edge_data['alignment'],
                    edge_data['angle']
                )
                neighbor_scores.append((neighbor, similarity))
            neighbor_scores.sort(key=lambda x: x[1], reverse=True)
            top_n_neighbors = {neighbor for neighbor, _ in neighbor_scores[:n]}

            for neighbor in neighbors:
                if neighbor not in top_n_neighbors and G_filtered.has_edge(node, neighbor):
                    G_filtered.remove_edge(node, neighbor)
            
        return G_filtered

    
    def _get_axis_for_nuclei(self, boundary_points: np.ndarray) -> np.ndarray:
        """Calculate orientation axes for multiple nuclei.

        Args:
            boundary_points: Array of boundary coordinates for multiple nuclei.

        Returns:
            Array of orientation axis vectors, one for each nucleus.
        """
        all_axises = []
        for boundary in boundary_points:
            axis = self._get_nucleus_orientation(boundary)
            all_axises.append(axis)

        return np.array(all_axises)


    def _get_median_similarity(self, filtered_graph: nx.Graph) -> float:
        """Calculate median similarity across all nodes in the graph.

        Args:
            filtered_graph: NetworkX graph with nodes containing 'best_similarity' attributes.

        Returns:
            Median of the best_similarity values across all nodes.
        """
        similarities = []
        for node in filtered_graph.nodes():
            similarity = filtered_graph.nodes[node].get("best_similarity", 0.0)
            similarities.append(similarity)
        return np.median(similarities)

    
    def _classify_nuclei(
            self,
            filtered_graph: nx.Graph,
            k_neighbors: int = 5,
            use_second_order: bool = True,
            threshold: float = None,
            distance_weight: bool = True
        ) -> dict[int, str]:
        """Classify nuclei as organized or unorganized based on neighbor similarity.

        Args:
            filtered_graph: NetworkX graph containing nuclei and their relationships.
            k_neighbors: Number of nearest neighbors to consider. Defaults to 5.
            use_second_order: Whether to include second-order neighbors in classification.
                Defaults to True.
            threshold: Similarity threshold for classification. If None, uses median
                similarity across all nodes. Defaults to None.
            distance_weight: Whether to weight neighbor contributions by inverse distance.
                Defaults to True.

        Returns:
            Dictionary mapping node indices to classification labels ("organized" or
            "unorganized").
        """
        if threshold is None:
            threshold = self._get_median_similarity(filtered_graph)
        
        classifications = {}
        
        for node in filtered_graph.nodes():
            neighbors_with_dist = []
            
            for neighbor in filtered_graph.neighbors(node):
                edge_data = filtered_graph[node][neighbor]
                distance = edge_data.get('distance', 1.0)
                neighbors_with_dist.append((neighbor, distance))
            
            neighbors_with_dist.sort(key=lambda x: x[1])
            k_nearest = neighbors_with_dist[:k_neighbors]
            
            weighted_similarities = []
            total_weight = 0.0
            
            for neighbor, distance in k_nearest:
                sim = filtered_graph.nodes[neighbor].get('best_similarity', 0.0)
                
                if distance_weight and distance > 0:
                    # Weight by inverse distance
                    weight = 1.0 / (distance)
                else:
                    weight = 1.0
                
                weighted_similarities.append(sim * weight)
                total_weight += weight

            if use_second_order and len(k_nearest) > 0:
                second_order_neighbors = {}
                
                for neighbor, _ in k_nearest:
                    for second_neighbor in filtered_graph.neighbors(neighbor):
                        if second_neighbor not in second_order_neighbors and second_neighbor != node:
                            
                            if second_neighbor not in [n for n, d in k_nearest]:
                                try:
                                    edge_data = filtered_graph[neighbor][second_neighbor]
                                    dist = edge_data.get('distance', 1.0)
                                    second_order_neighbors[second_neighbor] = dist
                                except KeyError:
                                    pass
                
                for neighbor, distance in second_order_neighbors.items():
                    sim = filtered_graph.nodes[neighbor].get('best_similarity', 0.0)
                    
                    # Reduce weight for second-order neighbors
                    if distance_weight and distance > 0:
                        weight = 0.25 / (distance + 1e-6)
                    else:
                        weight = 0.25
                    
                    weighted_similarities.append(sim * weight)
                    total_weight += weight
            
            if total_weight > 0:
                avg_weighted_similarity = sum(weighted_similarities) / total_weight
            else:
                avg_weighted_similarity = filtered_graph.nodes[node].get('best_similarity', 0.0)
            
            if avg_weighted_similarity > threshold:
                classifications[node] = "organized"
            else:
                classifications[node] = "unorganized"
        
        return classifications
