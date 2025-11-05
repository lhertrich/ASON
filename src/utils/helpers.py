import matplotlib.pyplot as plt

from skimage.morphology import remove_small_objects, remove_small_holes, binary_closing, disk
from skimage.transform import resize


def clean_mask(mask):
    mask = mask.astype(bool)
    mask = remove_small_objects(mask, min_size=200)
    mask = remove_small_holes(mask, area_threshold=10)
    rad = disk(radius=2)
    smoothed_mask = binary_closing(mask, rad)
    return smoothed_mask


def cut_out_image(image, mask):
    resized_mask = resize(mask, (image.shape[0], image.shape[1]), order=0, preserve_range=True)
    cut_image = image.copy()
    resized_mask = resized_mask.astype(bool)
    cut_image[~resized_mask] = 0
    return cut_image


def compare_two_images(image_1, image_2, title_1 = None, title_2 = None, size=(12, 6)):
    plt.figure(figsize=size)

    plt.subplot(1, 2, 1)
    plt.axis(False)
    plt.title(title_1)
    plt.imshow(image_1)

    plt.subplot(1, 2, 2)
    plt.axis(False)
    plt.title(title_2)
    plt.imshow(image_2)

    plt.show()