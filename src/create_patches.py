import config

import os
import numpy as np

from tqdm import tqdm

def threshold(x, y, z, image, threshold_value):
    """True if sample co-ordinates are at least threshold value away."""
    min_threshold_output = x>threshold_value and y>threshold_value and z>threshold_value
    max_threshold_output = x<image.shape[0]-threshold_value and y<image.shape[1]-threshold_value and z<image.shape[2]-threshold_value
    return min_threshold_output and max_threshold_output

def create_patch(img, mask, coordinate, mask_patch_size, high_resolution_patch_size, low_resolution_crop_size, low_resolution_patch_size):
    """Outputs patches in order: mask, high_resolution, low_resolution resized"""
    x_curr_ind, y_curr_ind, z_curr_ind = coordinate

    patches = []
    # Find three types of patches
    for ind, size in enumerate((mask_patch_size, high_resolution_patch_size, low_resolution_crop_size)):
        
        # Find start and end indices
        x_start = x_curr_ind-(size//2)
        x_end = x_curr_ind+(size//2)

        y_start = y_curr_ind-(size//2)
        y_end = y_curr_ind+(size//2)

        z_start = z_curr_ind-(size//2)
        z_end = z_curr_ind+(size//2)

        # For mask patch use mask image otherwise use input image
        if ind==0:
            patches.append(mask[x_start:x_end, y_start:y_end, z_start:z_end])
        else:
            patches.append(img[x_start:x_end, y_start:y_end, z_start:z_end])

    # Resize low resolution patch
    patches[-1] = np.resize(patches[-1], (low_resolution_patch_size, low_resolution_patch_size, low_resolution_patch_size))

    return patches


def main(input_path, mask_path, mask_patch_size, high_resolution_patch_size, low_resolution_crop_size, low_resolution_patch_size):

    # List full path of the image files
    image_filenames = os.listdir(input_path)
    image_filenames = list(sorted(image_filenames, key=lambda val: val.split("_")[0]))
    image_filenames = list(map(lambda img_filename: "/".join([input_path, img_filename]), image_filenames))

    # List full path of the mask files
    mask_filenames = os.listdir(mask_path)
    mask_filenames = list(sorted(mask_filenames, key=lambda val: val.split("_")[0]))
    mask_filenames = list(map(lambda mask_filename: "/".join([mask_path, mask_filename]), mask_filenames))

    # Calculate how many number of positive and negative patches to sample per image
    num_positive_samples = int(config.SAMPLES_PER_IMAGE*config.BALANCE)
    num_negative_samples = config.SAMPLES_PER_IMAGE - num_positive_samples

    # Calculate minimum distance away from the boundary for a sampled patch to be valid
    min_boundary_width = max(
        mask_patch_size,
        high_resolution_patch_size,
        low_resolution_crop_size
    )

    # Iterative over image-mask pair
    patch_cnt = 0 # Patch count
    for (img_filename, mask_filename) in tqdm(zip(image_filenames, mask_filenames), total=len(image_filenames)):

        # Load input image and mask
        img, mask = np.load(img_filename), np.load(mask_filename)

        # List x, y, z indices where mask is 1
        xind_list, yind_list, zind_list = np.where(mask==1)
        # All co-ordinates where mask is 1
        rand_ind = list(zip(xind_list, yind_list, zind_list))
        # Filter out images to avoid corner or edge cases using previously calculated min_boundary_width
        filter_rand_ind = list(filter(lambda ind: threshold(ind[0], ind[1], ind[2], mask, min_boundary_width), rand_ind))
        # List positive patch (mask=1) co-ordinates
        rand_sample_positive_ind = [filter_rand_ind[ind] for ind in np.random.choice(range(len(filter_rand_ind)), num_positive_samples)]

        # List positive patch (mask=0) co-ordinates
        neg_cnt = 0
        rand_sample_negative_ind = []
        while neg_cnt<num_negative_samples:
            neg_coordinate = (np.random.randint(img.shape[0]), np.random.randint(img.shape[1]), np.random.randint(img.shape[2]))
            thresholding_output = threshold(neg_coordinate[0], neg_coordinate[1], neg_coordinate[2], img, min_boundary_width)
            not_positive_sample = neg_coordinate not in rand_sample_positive_ind
            if thresholding_output and not_positive_sample:
                rand_sample_negative_ind.append(neg_coordinate)
                neg_cnt+=1

        # Store three types of patches per positive and negative co-ordinate
        for coordinate_list in (rand_sample_positive_ind, rand_sample_negative_ind):
            for coordinate in coordinate_list: # positive, negative co-ordinate list
                # Get patches in a list in order: mask, high_resolution, low_resolution resized
                patches = create_patch(img, mask, coordinate, mask_patch_size, high_resolution_patch_size, low_resolution_crop_size, low_resolution_patch_size)

                # Save patches                
                for patch, fpath in zip(patches, (config.MASK_PATCH_PATH, config.HIGH_RESOLUTION_PATCH_PATH, config.LOW_RESOLUTION_PATCH_PATH)):
                    with open(f"{fpath}{patch_cnt}.npy", "wb") as file:
                        np.save(file, patch)
                
                patch_cnt+=1

    return None

if __name__ == "__main__":
    main(
        config.INPUT_PATH,
        config.MASK_PATH,
        config.MASK_PATCH_SIZE,
        config.HIGH_RESOLUTION_PATCH_SIZE,
        config.LOW_RESOLUTION_CROP_SIZE,
        config.LOW_RESOLUTION_PATCH_SIZE,
    )
