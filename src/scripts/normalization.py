import tifffile
import os
from tiatoolbox.tools import stainnorm


def load_images(data_path="data/images/"):
    """Load all images from a specified directory and a fixed reference image.

    Args:
        data_path (str, optional): Path to the directory containing image files.
            Defaults to "data/images/".

    Returns:
        tuple: A tuple (images, reference_image) where images is a list of image arrays
            and reference_image is the reference image array.
    """
    image_paths = sorted(
        [data_path + path for path in os.listdir(data_path) if not path.startswith(".")]
    )
    images = list(map(tifffile.imread, image_paths))
    reference_image = tifffile.imread("data/images/E2+P4+DHT_1_M7_3L_0013.tif")

    return images, reference_image


def color_normalization_reinhard(target_img, reference_img):
    """Normalize the color of a target image using Reinhard normalization with a reference image.

    Args:
        target_img (np.ndarray): The input target image to be normalized.
        reference_img (np.ndarray): The reference image for normalization.

    Returns:
        np.ndarray: The color-normalized image.
    """
    reinhard_norm = stainnorm.ReinhardNormalizer()
    reinhard_norm.fit(reference_img)

    normalized_target_img = reinhard_norm.transform(target_img)
    return normalized_target_img


def main(save_path="data/normalized_images/"):
    """Main function to load images, normalize them, and save the results.

    Args:
        save_path (str, optional): Directory where the normalized images will be saved.
            Defaults to "data/normalized_images/".

    Returns:
        None
    """
    images, reference_image = load_images()
    normalized_images = [
        color_normalization_reinhard(image, reference_image) for image in images
    ]
    os.makedirs(save_path, exist_ok=True)

    image_paths = sorted(
        [p for p in os.listdir("data/images/") if not p.startswith(".")]
    )
    for img, filename in zip(normalized_images, image_paths):
        save_file = os.path.join(save_path, f"normalized_{filename}")
        tifffile.imwrite(save_file, img)
    print("Saved images successfully!")


if __name__ == "__main__":
    main()
