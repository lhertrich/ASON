import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.pipeline.image_analysis_gui import main

if __name__ == "__main__":
    main()