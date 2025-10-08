import tifffile
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np

from histomicstk.preprocessing.color_normalization import deconvolution_based_normalization
from histomicstk.saliency.tissue_detection import (
    get_slide_thumbnail, get_tissue_mask)
from skimage.transform import resize

from utils.helpers import compare_two_images


def load_images(data_path="data/images/"):
    image_paths = sorted([data_path+path for path in os.listdir(data_path) if not path.startswith(".")])
    images = list(map(tifffile.imread, image_paths))
    reference_image = tifffile.imread("data/reference/E2+P4+DHT_4_M14_3L_0004.tif")

    return images, reference_image


def create_mask(img):
    mask_out, _ = get_tissue_mask(
    img, deconvolve_first=True,
    n_thresholding_steps=1, sigma=1.5, min_size=30)

    mask_out_fixed = resize(
    mask_out == 0, output_shape=img.shape[:2],
    order=0, preserve_range=True) == 1

    return ~mask_out_fixed


def color_normalization_macenko(target_img, reference_img, mask=None, stains=['hematoxylin', 'eosin'], stain_unmixing_method='macenko_pca'):
    stain_unmixing_routine_params = {
        'stains': stains,
        'stain_unmixing_method': stain_unmixing_method
    }

    normalized_target_img = deconvolution_based_normalization(
            target_img, im_target=reference_img,
            mask_out = mask,
            stain_unmixing_routine_params=stain_unmixing_routine_params,)

    return normalized_target_img


def main(save_path = "data/normalized_images/"):
    
    images, reference_image = load_images()
    print("loaded images")

    masks = [create_mask(img) for img in images]
    print("created masks")

    normalized_images = [color_normalization_macenko(image, reference_image, masks[i]) for i, image in enumerate(images)]
    print("normalized images")
    
    os.makedirs(save_path, exist_ok=True)
    
    image_paths = sorted([p for p in os.listdir("data/images/") if not p.startswith(".")])
    for img, filename in zip(normalized_images, image_paths):
        save_file = os.path.join(save_path, f"normalized_{filename}")
        tifffile.imwrite(save_file, img)
    print("Saved images successfully!")


if __name__ == "__main__":
    main()