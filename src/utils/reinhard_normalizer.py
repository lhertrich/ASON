import tifffile
import os
import sys
import numpy as np
from tiatoolbox.tools import stainnorm
from pathlib import Path

script_path = Path(__file__).resolve()
src_path = script_path.parent.parent
project_root = src_path.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

class ReinhardNormalizer:

    def __init__(self, reference_image_path=None):
        """Initializes the normalizer
        
        Args:
            reference_image_path: Path to reference image. If None, uses the default reference image.
        """
        if reference_image_path is None:
            reference_image_path = str(project_root) + "/reference_image/E2+P4+DHT_1_M7_3L_0013.tif"
            
            if not os.path.exists(reference_image_path):
                raise FileNotFoundError(
                    f"Reference image not found at: {reference_image_path}\n"
                    "Please ensure the reference_image folder contains E2+P4+DHT_1_M7_3L_0013.tif"
                )
        
        self.reference_image = tifffile.imread(reference_image_path)
        reinhard_norm = stainnorm.ReinhardNormalizer()
        reinhard_norm.fit(self.reference_image)
        self.norm = reinhard_norm

    
    def normalize(self, target_img: np.ndarray) -> np.ndarray:
        """Normalizes a given target image using the global reference image and reinhard normalization

        Args:
            target_img (np.ndarray): The target image to normalize  

        Raises:
            ValueError: If the given image is not a RGB numpy ndarray

        Returns:
            np.ndarray: The normalized target image
        """

        if not isinstance(target_img, np.ndarray) or target_img.shape[2] != 3:
            raise ValueError("Image has to be RGB numpy array with shape (x, y, 3)")

        target_img_normed = self.norm.transform(target_img)
        return target_img_normed

