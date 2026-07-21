import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


def augment_image(img_ref, augmentation="rotate", angles=(0, 45, 90, 135, 180, 225, 270, 315)):
    """Simple image augmentation, currently just rotation."""
    imgs = []
    if augmentation == "rotate":
        for angle in angles:
            imgs.append(rotate_image(img_ref, angle))
    return imgs


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    return cv2.warpAffine(
        image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_DEFAULT
    )


def dists2map(dists, img_shape):
    # Resize and smooth the patch-distance grid into a pixel-space anomaly map.
    # cv2.resize takes (width, height), i.e. the reverse of numpy's (height, width).
    dists = cv2.resize(dists, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_LINEAR)
    return gaussian_filter(dists, sigma=4)


def resize_mask_img(mask, image_shape, grid_size1):
    mask = mask.reshape(grid_size1)
    imgd1 = image_shape[0] // grid_size1[0]
    imgd2 = image_shape[1] // grid_size1[1]
    mask = np.repeat(mask, imgd1, axis=0)
    return np.repeat(mask, imgd2, axis=1)


def plot_ref_images(img_list, mask_list, vis_background_list, grid_size, save_path, img_names=None):
    k = min(len(img_list), 32)  # cap the number of reference samples we plot

    n_aug = len(img_list) // len(img_names)

    _fig, axs = plt.subplots(k, 3, figsize=(10, 3 * k + 1.0))
    if k == 1:
        axs = axs.reshape(1, -1)
    for i in range(k):
        axs[i, 0].imshow(img_list[i])
        axs[i, 1].imshow(vis_background_list[i])
        axs[i, 2].imshow(img_list[i])
        axs[i, 2].imshow(resize_mask_img(mask_list[i], img_list[i].shape, grid_size), alpha=0.5)
        axs[i, 0].axis("off")
        axs[i, 1].axis("off")
        axs[i, 2].axis("off")
        if i % n_aug == 0:
            axs[i, 0].title.set_text(f"Image: {img_names[i // n_aug]}")
        else:
            axs[i, 0].title.set_text(f"Augmentation of Image {img_names[i // n_aug]}")
        axs[i, 1].title.set_text("PCA + Mask")
        axs[i, 2].title.set_text("Mask")
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path + "reference_samples.png")
    plt.close()
