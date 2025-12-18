import sys
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="Guest")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    print(f"Hello from Python! Processing for: {args.name}")

    # Create a dummy QuPath square [x, y]
    # A small 100x100 pixel square at the top left
    geojson_output = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0,0], [100,0], [100,100], [0,100], [0,0]]]
            },
            "properties": {
                "objectType": "annotation",
                "classification": {
                    "name": "Python_Square",
                    "color": [255, 0, 0] # Optional: Red color
                },
                "measurements": {
                    "Status": 1.0
                }
            }
        }
    ]
}

    with open(args.output, 'w') as f:
        json.dump([geojson_output], f)
    
    print("Bridge test successful. JSON exported.")

if __name__ == "__main__":
    main()