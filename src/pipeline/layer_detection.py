import numpy as np


class LayerDetectionModule:
    def __init__(self, image: np.ndarray, nuclei_data_dict: dict[str, np.ndarray], nuclei_mask: np.ndarray, tissue_mask: np.ndarray, area_th: float = 0.5) -> None:
        self.image = image
        self.nuclei_data_dict = nuclei_data_dict
        self.nuclei_mask = nuclei_mask
        self.tissue_mask = tissue_mask
        self.area_th = area_th
        self.filtered_data_dict = self._filter_data_dict(tissue_mask, nuclei_data_dict, area_th)

    
    def _filter_data_dict(self, mask: np.ndarray, data_dict: dict[str, any], area_th: float = 0.5) -> list:
        points = data_dict["points"]
        median_area = calculate_median_area(data_dict["coord"])
        filtered_points = []
        filtered_coords = []
        filtered_probs = []

        binary_mask = (mask > 0).astype(int)
        for i, (point, coord) in enumerate(zip(points, data_dict["coord"])):
            x, y = int(point[0]), int(point[1])
            area = poly_area(np.array(coord[0]), np.array(coord[1]))
            if binary_mask[x, y] == 1 and area > area_th * median_area:
                filtered_points.append([point[0], point[1]])
                filtered_coords.append(coord)
                filtered_probs.append(data_dict["prob"][i])

        filtered_data_dict = dict(data_dict)
        filtered_data_dict["points"] = np.array(filtered_points)
        filtered_data_dict["coord"] = np.array(filtered_coords)
        filtered_data_dict["prob"] = np.array(filtered_probs)
        
        return filtered_data_dict

    