import config

import os
import shutil
import numpy as np

from skimage.transform import resize
from tqdm import tqdm

def threshold(x, y, z, image, threshold_value):
    """True if sample co-ordinates are at least threshold value away."""
    min_threshold_output = x>threshold_value and y>threshold_value and z>threshold_value
    max_threshold_output = x<image.shape[0]-threshold_value and y<image.shape[1]-threshold_value and z<image.shape[2]-threshold_value
    return min_threshold_output and max_threshold_output

def create_train_patch(img, mask, coordinate, mask_patch_size, high_resolution_patch_size, low_resolution_crop_size, low_resolution_patch_size):
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
    patches[-1] = resize(patches[-1], (low_resolution_patch_size, low_resolution_patch_size, low_resolution_patch_size))

    return patches

def create_filenames(filenames, path):
    # Returns a list of filenames
    filenames_list = list(sorted(filenames, key=lambda val: val.split("_")[0]))
    filenames_list = list(map(lambda filename: "/".join([path, filename]), filenames_list))
    
    return filenames_list

def train_test_split(input_path, mask_path, test_portion):
    # Returns lists of train-test image-mask filenames
    input_fnames, mask_fnames = os.listdir(input_path), os.listdir(mask_path)

    num_images = len(input_fnames)
    num_test_images = int(num_images*test_portion)
    num_train_images = 1-num_test_images

    train_image_filenames = create_filenames(input_fnames[:num_train_images], input_path)
    train_mask_filenames = create_filenames(mask_fnames[:num_train_images], mask_path)

    test_image_filenames = create_filenames(input_fnames[num_train_images:], input_path)
    test_mask_filenames = create_filenames(mask_fnames[num_train_images:], mask_path)

    return train_image_filenames, test_image_filenames, train_mask_filenames, test_mask_filenames

def create_train_patches(
        samples_per_image,
        balance,
        mask_patch_size,
        high_resolution_patch_size,
        low_resolution_crop_size,
        low_resolution_patch_size,
        train_image_filenames,
        train_mask_filenames,
        mask_patch_path,
        high_resolution_patch_path,
        low_resolution_patch_path
    ):

    # Saves train patches
    print("Creating train patches...")

    # Create required directories
    for directory in [mask_patch_path, high_resolution_patch_path, low_resolution_patch_path]:
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            # First remove existed
            shutil.rmtree(directory)
            # Create new one
            os.makedirs(directory)
            os.makedirs(f"{directory}/train")
            os.makedirs(f"{directory}/test")

    # Calculate how many number of positive and negative patches to sample per image
    num_positive_samples = int(samples_per_image*balance)
    num_negative_samples = samples_per_image - num_positive_samples

    # Calculate minimum distance away from the boundary for a sampled patch to be valid
    min_boundary_width = max(
        mask_patch_size,
        high_resolution_patch_size,
        low_resolution_crop_size
    )

    # Iterative over image-mask pair
    patch_cnt = 0 # Patch count
    for (img_filename, mask_filename) in tqdm(zip(train_image_filenames, train_mask_filenames), total=len(train_image_filenames)):

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
                patches = create_train_patch(img, mask, coordinate, mask_patch_size, high_resolution_patch_size, low_resolution_crop_size, low_resolution_patch_size)

                # Save patches
                for patch, fpath in zip(patches, (mask_patch_path, high_resolution_patch_path, low_resolution_patch_path)):
                    with open(f"{fpath}/train/{patch_cnt}.npy", "wb") as file:
                        np.save(file, patch)
                
                patch_cnt+=1

    return None

def pad_image_multiple_of_output_shape(original_image, output_size=24):
    # Pad original image to make it multiple of output shape, default to 24
    x, y, z = original_image.shape
    
    xp = int(np.ceil(x/output_size)*output_size - x)
    yp = int(np.ceil(y/output_size)*output_size - y)
    zp = int(np.ceil(z/output_size)*output_size - z)
    
    return np.pad(original_image, ((0, xp), (0, yp), (0, zp)), 'constant')

def pad_image_for_patches(padded_image, crop_size, output_size=24):
    pad = (crop_size//2) - (output_size//2)
    return np.pad(padded_image, ((pad, pad), (pad, pad), (pad, pad)), "constant")

def extract_test_patches(image_name, padded_image, patch_size, fpath, output_size=24, is_resize=False, resize_size=None):
    patch_cnt = 0
    break_z = False
    break_x = False
    break_y = False
    for z_start_ind in range(0, padded_image.shape[2], 24):
        if not break_z:
            z_end_ind = z_start_ind+patch_size
            if z_end_ind==padded_image.shape[2]:
                break_z = True
        else:
            break
            
        for x_start_ind in range(0, padded_image.shape[0], 24):
            if not break_x:
                x_end_ind = x_start_ind+patch_size
                if x_end_ind==padded_image.shape[0]:
                    break_x = True
            else:
                break
                
            for y_start_ind in range(0, padded_image.shape[0], 24):
                if not break_y:
                    y_end_ind = y_start_ind+patch_size
                    if y_end_ind==padded_image.shape[0]:
                        break_y = True
                else:
                    break
                
                patch = padded_image[x_start_ind:x_end_ind,y_start_ind:y_end_ind,z_start_ind:z_end_ind]

                if is_resize:
                    patch = resize(patch, (resize_size, resize_size, resize_size))

                with open(f"{fpath}/test/{image_name}_{patch_cnt}.npy", "wb") as file:
                    np.save(file, patch)

                patch_cnt+=1

    return None

def create_test_patches(
        test_image_filenames,
        high_resolution_patch_size,
        low_resolution_crop_size,
        low_resolution_patch_size,
        high_resolution_patch_path,
        low_resolution_patch_path
    ):
    # Saves test patches
    print("Creating test patches...")
    for filename in tqdm(test_image_filenames, total=len(test_image_filenames)):
        test_image = np.load(filename)
        test_image = pad_image_multiple_of_output_shape(test_image)

        # Extract image name
        image_name = filename.split("/")[-1]

        # Pad the padded image for high resolution
        high_res_pad_img = pad_image_for_patches(test_image, high_resolution_patch_size)

        # Extract patches for high resolution patch
        extract_test_patches(image_name, high_res_pad_img, high_resolution_patch_size, high_resolution_patch_path, output_size=24)
        
        # Pad the padded image for low resolution (crop)
        low_res_pad_img = pad_image_for_patches(test_image, low_resolution_crop_size)

        # Extract patches for low resolution patch
        extract_test_patches(image_name, low_res_pad_img, low_resolution_crop_size, low_resolution_patch_path, output_size=24, is_resize=True, resize_size=low_resolution_patch_size)

    return None

def main():

    # Parameters
    input_path = config.INPUT_PATH
    mask_path = config.MASK_PATH
    mask_patch_size = config.MASK_PATCH_SIZE
    high_resolution_patch_size = config.HIGH_RESOLUTION_PATCH_SIZE
    low_resolution_crop_size = config.LOW_RESOLUTION_CROP_SIZE
    low_resolution_patch_size = config.LOW_RESOLUTION_PATCH_SIZE
    test_portion = config.TEST_PORTION
    samples_per_image = config.NUM_PATCHES
    balance = config.BALANCE
    mask_patch_path = config.MASK_PATCH_PATH
    high_resolution_patch_path = config.HIGH_RESOLUTION_PATCH_PATH
    low_resolution_patch_path = config.LOW_RESOLUTION_PATCH_PATH

    # Get filenames
    train_image_filenames, test_image_filenames, train_mask_filenames, test_mask_filenames = train_test_split(input_path, mask_path, test_portion)

    # Create and save train patches
    param_dict = {
        "samples_per_image" : samples_per_image,
        "balance" : balance,
        "samples_per_image" : samples_per_image,
        "mask_patch_size" : mask_patch_size,
        "high_resolution_patch_size" : high_resolution_patch_size,
        "low_resolution_crop_size" : low_resolution_crop_size,
        "low_resolution_patch_size" : low_resolution_patch_size,
        "train_image_filenames" : train_image_filenames,
        "train_mask_filenames" : train_mask_filenames,
        "mask_patch_path" : mask_patch_path,
        "high_resolution_patch_path" : high_resolution_patch_path,
        "low_resolution_patch_path" : low_resolution_patch_path
    }
    create_train_patches(**param_dict)

    # Create and save test patches
    param_dict = {
        "test_image_filenames" : test_image_filenames,
        "high_resolution_patch_size" : high_resolution_patch_size,
        "low_resolution_crop_size" : low_resolution_crop_size,
        "low_resolution_patch_size" : low_resolution_patch_size,
        "high_resolution_patch_path" : high_resolution_patch_path,
        "low_resolution_patch_path" : low_resolution_patch_path 
    }
    create_test_patches(**param_dict)

    return None

if __name__ == "__main__":
    main()