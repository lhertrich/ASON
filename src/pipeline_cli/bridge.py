import argparse
import json
import cv2
import numpy as np
import traceback
import sys
from pathlib import Path
from skimage import measure

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.segment import SegmentationModule 

def mask_to_features(mask, class_name="Tissue"):
    """Converts a binary mask into QuPath Annotations."""
    # Ensure mask is binary 0/1 or 0/255
    mask = (mask > 0).astype(np.uint8)
    contours = measure.find_contours(mask, 0.5)
    features = []
    for contour in contours:
        # skimage (row, col) -> QuPath [x, y]
        coords = [[float(p[1]), float(p[0])] for p in contour]
        if len(coords) < 3: 
            continue # Skip tiny noise
        coords.append(coords[0]) # Close the polygon
        
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": {
                "objectType": "annotation", 
                "classification": {"name": class_name, "colorRGB": -16711681} # Cyan
            }
        })
    return features

def stardist_to_features(data_dict):
    """Converts StarDist results into QuPath Detections."""
    features = []
    # coord shape is (n_instances, 2, n_points)
    for i in range(len(data_dict['coord'])):
        y_coords = data_dict['coord'][i][0]
        x_coords = data_dict['coord'][i][1]
        coords = [[float(x), float(y)] for x, y in zip(x_coords, y_coords)]
        coords.append(coords[0]) # Close poly
        
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": {
                "objectType": "detection",
                "classification": {"name": "Nucleus", "colorRGB": -65536}, # Red
                "measurements": {"Probability": float(data_dict['prob'][i])}
            }
        })
    return features

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path")
    parser.add_argument("--output_json")
    parser.add_argument("--task")
    parser.add_argument("--prob_thresh", type=float)
    args = parser.parse_args()

    try:
        # 1. Load Image and Model
        img = cv2.imread(args.image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        module = SegmentationModule()
        
        all_features = []

        # 2. Run logic
        if args.task == 'tissue':
            mask = module.segment_tissue(img_rgb)
            all_features = mask_to_features(mask)
            print(f"Generated {len(all_features)} tissue annotations.")
            
        elif args.task == 'nuclei':
            data_dict = module.segment_nuclei(img_rgb, prob_thresh=args.prob_thresh)
            all_features = stardist_to_features(data_dict)
            print(f"Generated {len(all_features)} nuclei detections.")

        # 3. Write GeoJSON FeatureCollection
        output = {"type": "FeatureCollection", "features": all_features}
        with open(args.output_json, 'w') as f:
            json.dump(output, f)
            
    except Exception as e:
        print(f"PYTHON ERROR: {str(e)}")
        traceback.print_exc()