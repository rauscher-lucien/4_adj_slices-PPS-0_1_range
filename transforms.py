import numpy as np
import os
import sys
from skimage import filters, exposure
import matplotlib.pyplot as plt
import torch

class Normalize(object):
    """
    Normalize an image using mean and standard deviation.
    
    Args:
        mean (float or tuple): Mean for each channel.
        std (float or tuple): Standard deviation for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        """
        Args:
            data (tuple): Containing input and target images to be normalized.
        
        Returns:
            Tuple: Normalized input and target images.
        """
        input_img = data

        # Normalize input image
        input_normalized = (input_img - self.mean) / self.std

        return input_normalized


class RandomFlip(object):

    def __call__(self, data):

        input_img, target_img = data

        if np.random.rand() > 0.5:
            input_img = np.fliplr(input_img)
            target_img = np.fliplr(target_img)

        if np.random.rand() > 0.5:
            input_img = np.flipud(input_img)
            target_img = np.flipud(target_img)

        return input_img, target_img
    

class RandomHorizontalFlip:
    def __call__(self, data):
        """
        Apply random horizontal flipping to both the input stack of slices and the target slice.
        In 50% of the cases, only horizontal flipping is applied without vertical flipping.
        
        Args:
            data (tuple): A tuple containing the input stack and the target slice.
        
        Returns:
            Tuple: Horizontally flipped input stack and target slice, if applied.
        """
        input_stack = data

        # Apply horizontal flipping with a 50% chance
        if np.random.rand() > 0.5:
            # Flip along the width axis (axis 1), keeping the channel dimension (axis 2) intact
            input_stack = np.flip(input_stack, axis=1)

        return input_stack



class BatchRandomFlip(object):
    """
    Randomly flip patches in a batch of images horizontally and/or vertically.
    Works with batches of images in the format (batch_size, height, width, channels).
    """

    def __call__(self, batch_data):
        """
        Args:
            batch_data (tuple): A tuple containing two numpy arrays (input_patches, target_patches),
                                each with the shape (batch_size, height, width, channels).
        
        Returns:
            Tuple: Two numpy arrays with flipped images in the same shape as input.
        """
        input_patches, target_patches = batch_data
        batch_size = input_patches.shape[0]

        for i in range(batch_size):
            if np.random.rand() > 0.5:
                # Flip patches horizontally
                input_patches[i] = np.fliplr(input_patches[i])
                target_patches[i] = np.fliplr(target_patches[i])

            if np.random.rand() > 0.5:
                # Flip patches vertically
                input_patches[i] = np.flipud(input_patches[i])
                target_patches[i] = np.flipud(target_patches[i])

        return input_patches, target_patches



class RandomCrop:
    def __init__(self, output_size=(64, 64)):
        """
        RandomCrop constructor for cropping both the input stack of slices and the target slice.
        Args:
            output_size (tuple): The desired output size (height, width).
        """
        self.output_size = output_size

    def __call__(self, data):
        """
        Apply the cropping operation.
        Args:
            data (tuple): A tuple containing the input stack and the target slice.
        Returns:
            Tuple: Cropped input stack and target slice.
        """
        input_stack = data

        h, w, _ = input_stack.shape
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        input_cropped = input_stack[top:top+new_h, left:left+new_w, :]

        return input_cropped

    


class GenerateFixedPatches(object):
    """
    Generate 8 patches of size 64x64 from the input and target images.
    The images are first padded if they are smaller than 64x64.
    Patches are selected to cover the image as evenly as possible.
    """

    def __init__(self, patch_size=(80, 80), num_patches=8):
        self.patch_size = patch_size
        self.num_patches = num_patches

    def __call__(self, data):
        input_img, target_img = data
        h, w = input_img.shape[:2]
        new_h, new_w = self.patch_size

        # Ensure the image is at least 64x64 by padding if necessary
        pad_h = max(0, new_h - h)
        pad_w = max(0, new_w - w)
        if pad_h > 0 or pad_w > 0:
            input_img = np.pad(input_img, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)), mode='constant')
            target_img = np.pad(target_img, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)), mode='constant')
            h, w = input_img.shape[:2]

        patches_input = []
        patches_target = []

        # Calculate step sizes to try to distribute patches evenly across the image
        step_h = max(1, (h - new_h) // (self.num_patches // 2))
        step_w = max(1, (w - new_w) // (self.num_patches // 2))

        count = 0
        for i in range(0, h - new_h + 1, step_h):
            for j in range(0, w - new_w + 1, step_w):
                if count < self.num_patches:
                    patch_input = input_img[i:i+new_h, j:j+new_w, :]
                    patch_target = target_img[i:i+new_h, j:j+new_w, :]
                    patches_input.append(patch_input)
                    patches_target.append(patch_target)
                    count += 1
                else:
                    break
            if count >= self.num_patches:
                break

        # Convert lists to numpy arrays with an additional dimension for batch size
        patches_input = np.stack(patches_input, axis=0)
        patches_target = np.stack(patches_target, axis=0)

        return patches_input, patches_target


class GenerateRandomPatches(object):
    """
    Generate 8 patches of size 80x80 from the input and target images.
    The images are first padded if they are smaller than 80x80.
    Patches are selected at random positions within the image.
    """

    def __init__(self, patch_size=(128, 128), num_patches=8):
        self.patch_size = patch_size
        self.num_patches = num_patches

    def __call__(self, data):
        input_img, target_img = data
        h, w, _ = input_img.shape
        new_h, new_w = self.patch_size

        # Ensure the image is at least 80x80 by padding if necessary
        pad_h = max(0, new_h - h)
        pad_w = max(0, new_w - w)
        if pad_h > 0 or pad_w > 0:
            input_img = np.pad(input_img, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)), mode='constant', constant_values=0)
            target_img = np.pad(target_img, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)), mode='constant', constant_values=0)
            h, w = input_img.shape[:2]

        patches_input = []
        patches_target = []

        for _ in range(self.num_patches):
            # Randomly select the top-left corner of the patch
            i = np.random.randint(0, h - new_h + 1)
            j = np.random.randint(0, w - new_w + 1)

            patch_input = input_img[i:i+new_h, j:j+new_w, :]
            patch_target = target_img[i:i+new_h, j:j+new_w, :]
            patches_input.append(patch_input)
            patches_target.append(patch_target)

        # Convert lists to numpy arrays with an additional dimension for batch size
        patches_input = np.stack(patches_input, axis=0)
        patches_target = np.stack(patches_target, axis=0)

        return patches_input, patches_target




class GenerateRandomPatchesIntensity(object):
    """
    Generate patches of specified size from the input and target images.
    The images are first padded if they are smaller than the specified patch size.
    Patches are selected at random positions within the image, with a minimum average intensity threshold.
    If suitable patches are not found after a certain number of attempts, it defaults to selecting random patches.
    """

    def __init__(self, patch_size=(128, 128), num_patches=8, intensity_threshold=0.1, max_attempts=100):
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.intensity_threshold = intensity_threshold
        self.max_attempts = max_attempts

    def __call__(self, data):
        input_img, target_img = data
        h, w, _ = input_img.shape
        new_h, new_w = self.patch_size

        # Ensure the image is at least the size of patch_size by padding if necessary
        pad_h = max(0, new_h - h)
        pad_w = max(0, new_w - w)
        if pad_h > 0 or pad_w > 0:
            input_img = np.pad(input_img, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)), mode='constant', constant_values=0)
            target_img = np.pad(target_img, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)), mode='constant', constant_values=0)

        patches_input = []
        patches_target = []
        attempts = 0

        while len(patches_input) < self.num_patches and attempts < self.max_attempts:
            # Randomly select the top-left corner of the patch
            i = np.random.randint(0, h - new_h + 1)
            j = np.random.randint(0, w - new_w + 1)

            patch_input = input_img[i:i+new_h, j:j+new_w, :]
            patch_target = target_img[i:i+new_h, j:j+new_w, :]

            # Check if the patch meets the intensity threshold criteria
            if np.mean(patch_input) >= self.intensity_threshold or attempts >= self.max_attempts - self.num_patches:
                patches_input.append(patch_input)
                patches_target.append(patch_target)
            attempts += 1

        # If not enough patches meet the criteria, fill in with the last attempted patches to ensure num_patches are returned
        while len(patches_input) < self.num_patches:
            patches_input.append(patch_input)
            patches_target.append(patch_target)

        # Convert lists to numpy arrays with an additional dimension for batch size
        patches_input = np.stack(patches_input, axis=0)
        patches_target = np.stack(patches_target, axis=0)

        return patches_input, patches_target





class CropToMultipleOf32Inference:
    def __call__(self, img):
        # Assuming img is a 3D numpy array with shape (H, W, C) where C is the slice count
        h, w, _ = img.shape

        new_h = h - (h % 32)
        new_w = w - (w % 32)

        top = (h - new_h) // 2
        left = (w - new_w) // 2

        cropped_image = img[top:top + new_h, left:left + new_w]

        return cropped_image



class SquareCrop(object):
    def __call__(self, data):
        """
        Crop input and target images to form square images.

        Args:
        - data: a tuple containing input and target images as NumPy arrays.

        Returns:
        - A tuple containing cropped input and target images as NumPy arrays.
        """
        input_img, target_img = data

        # Get the minimum side length
        min_side_length = min(input_img.shape[0], input_img.shape[1])

        # Calculate the cropping boundaries
        start_row = (input_img.shape[0] - min_side_length) // 2
        start_col = (input_img.shape[1] - min_side_length) // 2

        # Crop input image
        cropped_input = input_img[start_row:start_row + min_side_length,
                                   start_col:start_col + min_side_length, :]

        # Crop target image
        cropped_target = target_img[start_row:start_row + min_side_length,
                                     start_col:start_col + min_side_length, :]

        return cropped_input, cropped_target




class ToTensor(object):
    """
    Convert images or batches of images to PyTorch tensors, handling both single images
    and tuples of images (input_img, target_img). The input is expected to be in the format
    (b, h, w, c) for batches or (h, w, c) for single images, and it converts them to
    PyTorch's (b, c, h, w) format or (c, h, w) for single images.
    """

    def __call__(self, data):
        """
        Convert input images or a tuple of images to PyTorch tensors, adjusting the channel position.

        Args:
            data (numpy.ndarray or tuple of numpy.ndarray): The input can be a single image (h, w, c),
            a batch of images (b, h, w, c), or a tuple of (input_img, target_img) in similar formats.

        Returns:
            torch.Tensor or tuple of torch.Tensor: The converted image(s) as PyTorch tensor(s) in the
            format (c, h, w) for single images or (b, c, h, w) for batches. If input is a tuple, returns
            a tuple of tensors.
        """
        def convert_image(img):
            # Convert a single image or a batch of images to a tensor, adjusting channel position
            if img.ndim == 4:  # Batch of images (b, h, w, c)
                return torch.from_numpy(img.transpose(0, 3, 1, 2).astype(np.float32))
            elif img.ndim == 3:  # Single image (h, w, c)
                return torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32))
            else:
                raise ValueError("Unsupported image format: must be (h, w, c) or (b, h, w, c).")

        # Check if the input is a tuple of images
        if isinstance(data, tuple):
            return tuple(convert_image(img) for img in data)
        else:
            return convert_image(data)



class ToNumpy(object):

    def __call__(self, data):

        return data.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    
    
class Denormalize(object):
    """
    Denormalize an image using mean and standard deviation.
    
    Args:
        mean (float or tuple): Mean for each channel used during normalization.
        std (float or tuple): Standard deviation for each channel used during normalization.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        """
        Args:
            img: A normalized image (as a numpy array) to be denormalized.
        
        Returns:
            The denormalized image as a numpy array.
        """
        # Denormalize the image
        denormalized_img = (img * self.std) + self.mean
        return denormalized_img


    

class MinMaxNormalize(object):
    """
    Normalize images to the 0-1 range using global minimum and maximum values provided at initialization.
    """

    def __init__(self, global_min, global_max):
        """
        Initializes the normalizer with global minimum and maximum values.

        Parameters:
        - global_min (float): The global minimum value used for normalization.
        - global_max (float): The global maximum value used for normalization.
        """
        self.global_min = global_min
        self.global_max = global_max

    def __call__(self, data):
        """
        Normalize input and target images to the 0-1 range using the global min and max.

        Args:
            data (tuple): Containing input and target images to be normalized.

        Returns:
            Tuple: Normalized input and target images.
        """
        input_img = data

        # Normalize input image
        input_normalized = (input_img - self.global_min) / (self.global_max - self.global_min)
        input_normalized = np.clip(input_normalized, 0, 1)  # Ensure within [0, 1] range

        return input_normalized.astype(np.float32)



class MinMaxNormalizeInference(object):
    """
    Normalize an image to the 0-1 range using global minimum and maximum values provided at initialization.
    This is adapted for inference where only one image is processed at a time.
    
    Args:
        global_min (float): The global minimum value used for normalization.
        global_max (float): The global maximum value used for normalization.
    """

    def __init__(self, global_min, global_max):
        self.global_min = global_min
        self.global_max = global_max

    def __call__(self, img):
        """
        Normalize a single image to the 0-1 range using the global min and max.

        Args:
            img (numpy.ndarray): Image to be normalized.

        Returns:
            numpy.ndarray: Normalized image.
        """
        # Normalize the image
        normalized_img = (img - self.global_min) / (self.global_max - self.global_min)
        normalized_img = np.clip(normalized_img, 0, 1)  # Ensure within [0, 1] range
        
        return normalized_img.astype(np.float32)
    
    
class NormalizeInference(object):
    """
    Normalize an image for inference using mean and standard deviation used during training.
    
    Args:
        mean (float or tuple): Mean for each channel used during normalization.
        std (float or tuple): Standard deviation for each channel used during normalization.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        """
        Normalize a single image using the mean and std.

        Args:
            img (numpy.ndarray): Image to be normalized. Shape should be (H, W, C) for a single image.

        Returns:
            numpy.ndarray: Normalized image.
        """
        # Assuming img is a single image of shape (H, W, C)
        # Convert to float for precise division
        img = img.astype(np.float32)
        
        # Normalize the image
        normalized_img = (img - self.mean) / self.std
        
        return normalized_img



class RandomSubsampleBatch(object):
    """
    Subsamples batches of images by applying a 2x2 sliding window across each image in the batch
    and randomly selecting one of the 4 pixels within each window. This process is applied
    independently to batches of input and target images, resulting in subsampled images
    with half the size of the originals.
    """

    def __call__(self, batch_data):
        """
        Subsample batches of input and target images.

        Args:
        - batch_data: a tuple containing two numpy arrays (input_patches, target_patches),
                      each with the shape (batch_size, height, width, channels).

        Returns:
        - A tuple containing subsampled batches of input and target images as NumPy arrays.
        """
        input_patches, target_patches = batch_data

        def subsample_batch(batch):
            subsampled_batch = []
            for img in batch:
                # Determine the size of the subsampled image
                new_height = img.shape[0] // 2
                new_width = img.shape[1] // 2

                # Initialize the subsampled image
                subsampled = np.zeros((new_height, new_width, img.shape[2]), dtype=img.dtype)

                # Apply 2x2 sliding window and randomly select one pixel from each window
                for i in range(new_height):
                    for j in range(new_width):
                        window = img[i*2:(i+1)*2, j*2:(j+1)*2, :]
                        random_row = np.random.randint(0, 2)
                        random_col = np.random.randint(0, 2)
                        subsampled[i, j, :] = window[random_row, random_col, :]
                
                subsampled_batch.append(subsampled)

            return np.array(subsampled_batch)

        # Subsample both input and target batches
        subsampled_input_patches = subsample_batch(input_patches)
        subsampled_target_patches = subsample_batch(target_patches)

        return subsampled_input_patches, subsampled_target_patches


class RandomSubsample(object):
    """
    Subsamples an image by applying a 2x2 sliding window across the image
    and randomly selecting one of the 4 pixels within each window. This process is applied
    independently to both an input and a target image, resulting in subsampled images
    with half the size of the originals.
    """

    def __call__(self, data):
        """
        Subsample an input and a target image.

        Args:
        - data: a tuple containing two numpy arrays (input_image, target_image),
                each with the shape (height, width, channels).

        Returns:
        - A tuple containing subsampled input and target images as NumPy arrays.
        """
        input_image, target_image = data

        def subsample_image(img):
            # Determine the size of the subsampled image
            new_height = img.shape[0] // 2
            new_width = img.shape[1] // 2

            # Initialize the subsampled image
            subsampled = np.zeros((new_height, new_width, img.shape[2]), dtype=img.dtype)

            # Apply 2x2 sliding window and randomly select one pixel from each window
            for i in range(new_height):
                for j in range(new_width):
                    window = img[i*2:(i+1)*2, j*2:(j+1)*2, :]
                    random_row = np.random.randint(0, 2)
                    random_col = np.random.randint(0, 2)
                    subsampled[i, j, :] = window[random_row, random_col, :]
            
            return subsampled

        # Subsample both the input and the target images
        subsampled_input = subsample_image(input_image)
        subsampled_target = subsample_image(target_image)

        return subsampled_input, subsampled_target
    
class BackTo01Range(object):
    """
    Normalize a tensor to the range [0, 1] based on its own min and max values.
    """

    def __call__(self, tensor):
        """
        Args:
            tensor: A tensor with any range of values.
        
        Returns:
            A tensor normalized to the range [0, 1].
        """
        min_val = tensor.min()
        max_val = tensor.max()
        
        # Avoid division by zero in case the tensor is constant
        if (max_val - min_val).item() > 0:
            # Normalize the tensor to [0, 1] based on its dynamic range
            normalized_tensor = (tensor - min_val) / (max_val - min_val)
        else:
            # If the tensor is constant, set it to a default value, e.g., 0, or handle as needed
            normalized_tensor = tensor.clone().fill_(0)  # Here, setting all values to 0

        return normalized_tensor
    


class PPS(object):
    """
    Custom subsampling for a stack of images. Applies a 2x2 window across each image
    in the stack and selects two adjacent pixels to form two new images, resulting in
    8 subsampled images from the original 4.
    """
    
    def __init__(self):
        # This can be expanded to include parameters for different subsampling strategies
        pass

    def select_pixels_2x2(self, window, pair_choice):
        """
        Selects two adjacent pixels based on the specified pair choice.

        Args:
        - window (numpy.ndarray): The 2x2 window of pixels.
        - pair_choice (str): One of 'top', 'bottom', 'left', 'right' indicating which pair of pixels to select.

        Returns:
        - (numpy.ndarray, numpy.ndarray): The selected input and target pixels.
        """
        if pair_choice == 'top':
            selected_input_pixel = window[0, np.random.choice([0, 1]), :]
            selected_target_pixel = window[0, 1 - np.random.choice([0, 1]), :]
        elif pair_choice == 'bottom':
            selected_input_pixel = window[1, np.random.choice([0, 1]), :]
            selected_target_pixel = window[1, 1 - np.random.choice([0, 1]), :]
        elif pair_choice == 'left':
            selected_input_pixel = window[np.random.choice([0, 1]), 0, :]
            selected_target_pixel = window[1 - np.random.choice([0, 1]), 0, :]
        else:  # 'right'
            selected_input_pixel = window[np.random.choice([0, 1]), 1, :]
            selected_target_pixel = window[1 - np.random.choice([0, 1]), 1, :]

        return selected_input_pixel, selected_target_pixel
    
    def select_pixels_4x4(self, window, pair_choice):
        """
        Selects two 2x2 blocks based on the specified pair choice for a 4x4 window.

        Args:
        - window (numpy.ndarray): The 4x4 window of pixels.
        - pair_choice (str): One of 'top', 'bottom', 'left', 'right' indicating which two 2x2 blocks to select.

        Returns:
        - (numpy.ndarray, numpy.ndarray): The selected input and target 2x2 blocks.
        """
        if pair_choice == 'top':
            block_1 = window[:2, :2]
            block_2 = window[:2, 2:]
        elif pair_choice == 'bottom':
            block_1 = window[2:, :2]
            block_2 = window[2:, 2:]
        elif pair_choice == 'left':
            block_1 = window[:2, 2:]
            block_2 = window[2:, 2:]
        elif pair_choice == 'right':
            block_1 = window[:2, :2]
            block_2 = window[2:, :2]
        else:
            raise ValueError("Invalid pair choice for 4x4 strategy")

        # Randomly assign the selected blocks to input and target
        if np.random.rand() > 0.5:
            return block_1, block_2
        else:
            return block_2, block_1

    def __call__(self, stack):
        assert stack.ndim == 3, "Expected 3D input (H, W, C)"
        h, w, c = stack.shape
        assert c == 4, "Expected 4 images in the stack"

        input_stack_2x2 = np.zeros((h // 2, w // 2, c), dtype=stack.dtype)
        target_stack_2x2 = np.zeros((h // 2, w // 2, c), dtype=stack.dtype)

        input_stack_4x4 = np.zeros((h // 2, w // 2, c), dtype=stack.dtype)
        target_stack_4x4 = np.zeros((h // 2, w // 2, c), dtype=stack.dtype)

        for i in range(0, h, 2):
            for j in range(0, w, 2):

                window_2x2 = stack[i:i+2, j:j+2, :]
                pair_choice_2x2 = np.random.choice(['top', 'bottom', 'left', 'right'])

                selected_input_pixel, selected_target_pixel = self.select_pixels_2x2(window_2x2, pair_choice_2x2)

                input_stack_2x2[i // 2, j // 2, :] = selected_input_pixel
                target_stack_2x2[i // 2, j // 2, :] = selected_target_pixel

        for i in range(0, h, 4):
            for j in range(0, w, 4):  # Process 4x4 windows
                window_4x4 = stack[i:i+4, j:j+4, :]
                pair_choice_4x4 = np.random.choice(['top', 'bottom', 'left', 'right'])

                selected_input_block, selected_target_block = self.select_pixels_4x4(window_4x4, pair_choice_4x4)

                input_stack_4x4[i//2 : (i//2)+2, j//2 : (j//2)+2, :] = selected_input_block
                target_stack_4x4[i//2 : (i//2)+2, j//2 : (j//2)+2, :] = selected_target_block

        return input_stack_2x2, target_stack_2x2, input_stack_4x4, target_stack_4x4


class ToTensorPPS(object):
    """
    Convert images or batches of images to PyTorch tensors, specifically designed to handle
    the output from the PPS class. It expects input in the format of a tuple containing four
    numpy arrays, each with dimensions (h, w, c), where 'c' refers to a stack of subsampled images.
    The conversion will adjust these to PyTorch's preferred format of (c, h, w) for each image in the stack.
    """

    def __call__(self, data_tuple):
        """
        Convert a tuple of numpy arrays to a tuple of PyTorch tensors, adjusting the channel position
        for each stack of images within the arrays.

        Args:
            data_tuple (tuple of numpy.ndarray): A tuple containing four numpy arrays, each with dimensions
            (h, w, c), representing the subsampled stacks of images.

        Returns:
            tuple of torch.Tensor: A tuple containing the converted stacks of images as PyTorch tensors,
            with each tensor in the format (c, h, w).
        """
        def convert_stack(stack):
            # Check if the stack is in the expected format (h, w, c) and convert to (c, h, w)
            if stack.ndim == 3:  # Stack of images (h, w, c)
                return torch.from_numpy(stack.transpose(2, 0, 1).astype(np.float32))
            else:
                raise ValueError("Unsupported stack format: must be (h, w, c).")

        return tuple(convert_stack(stack) for stack in data_tuple)



class PPSIndividualChannels(object):
    """
    Custom subsampling for a stack of images. Applies a 2x2 window across each image
    in the stack and selects two adjacent pixels to form two new images, resulting in
    8 subsampled images from the original 4.
    """
    
    def __init__(self):
        # This can be expanded to include parameters for different subsampling strategies
        pass

    def select_pixels_2x2(self, window, pair_choice, channel):
        """
        Selects two adjacent pixels based on the specified pair choice.

        Args:
        - window (numpy.ndarray): The 2x2 window of pixels.
        - pair_choice (str): One of 'top', 'bottom', 'left', 'right' indicating which pair of pixels to select.

        Returns:
        - (numpy.ndarray, numpy.ndarray): The selected input and target pixels.
        """
        if pair_choice == 'top':
            selected_input_pixel = window[0, np.random.choice([0, 1]), channel]
            selected_target_pixel = window[0, 1 - np.random.choice([0, 1]), channel]
        elif pair_choice == 'bottom':
            selected_input_pixel = window[1, np.random.choice([0, 1]), channel]
            selected_target_pixel = window[1, 1 - np.random.choice([0, 1]), channel]
        elif pair_choice == 'left':
            selected_input_pixel = window[np.random.choice([0, 1]), 0, channel]
            selected_target_pixel = window[1 - np.random.choice([0, 1]), 0, channel]
        else:  # 'right'
            selected_input_pixel = window[np.random.choice([0, 1]), 1, channel]
            selected_target_pixel = window[1 - np.random.choice([0, 1]), 1, channel]

        return selected_input_pixel, selected_target_pixel
    
    def select_pixels_4x4(self, window, pair_choice, channel):
        """
        Selects two 2x2 blocks based on the specified pair choice for a 4x4 window.

        Args:
        - window (numpy.ndarray): The 4x4 window of pixels.
        - pair_choice (str): One of 'top', 'bottom', 'left', 'right' indicating which two 2x2 blocks to select.

        Returns:
        - (numpy.ndarray, numpy.ndarray): The selected input and target 2x2 blocks.
        """
        if pair_choice == 'top':
            block_1 = window[:2, :2, channel]
            block_2 = window[:2, 2:, channel]
        elif pair_choice == 'bottom':
            block_1 = window[2:, :2, channel]
            block_2 = window[2:, 2:, channel]
        elif pair_choice == 'left':
            block_1 = window[:2, 2:, channel]
            block_2 = window[2:, 2:, channel]
        elif pair_choice == 'right':
            block_1 = window[:2, :2, channel]
            block_2 = window[2:, :2, channel]
        else:
            raise ValueError("Invalid pair choice for 4x4 strategy")

        # Randomly assign the selected blocks to input and target
        if np.random.rand() > 0.5:
            return block_1, block_2
        else:
            return block_2, block_1

    def __call__(self, stack):
        assert stack.ndim == 3, "Expected 3D input (H, W, C)"
        h, w, c = stack.shape
        assert c == 4, "Expected 4 images in the stack"

        input_stack_2x2 = np.zeros((h // 2, w // 2, c), dtype=stack.dtype)
        target_stack_2x2 = np.zeros((h // 2, w // 2, c), dtype=stack.dtype)

        input_stack_4x4 = np.zeros((h // 2, w // 2, c), dtype=stack.dtype)
        target_stack_4x4 = np.zeros((h // 2, w // 2, c), dtype=stack.dtype)

        for channel in range(0, c):
            for i in range(0, h, 2):
                for j in range(0, w, 2):

                    window_2x2 = stack[i:i+2, j:j+2, :]
                    pair_choice_2x2 = np.random.choice(['top', 'bottom', 'left', 'right'])

                    selected_input_pixel, selected_target_pixel = self.select_pixels_2x2(window_2x2, pair_choice_2x2, channel)

                    input_stack_2x2[i // 2, j // 2, channel] = selected_input_pixel
                    target_stack_2x2[i // 2, j // 2, channel] = selected_target_pixel

            for i in range(0, h, 4):
                for j in range(0, w, 4):  # Process 4x4 windows
                    window_4x4 = stack[i:i+4, j:j+4, :]
                    pair_choice_4x4 = np.random.choice(['top', 'bottom', 'left', 'right'])

                    selected_input_block, selected_target_block = self.select_pixels_4x4(window_4x4, pair_choice_4x4, channel)

                    input_stack_4x4[i//2 : (i//2)+2, j//2 : (j//2)+2, channel] = selected_input_block
                    target_stack_4x4[i//2 : (i//2)+2, j//2 : (j//2)+2, channel] = selected_target_block

        return input_stack_2x2, target_stack_2x2, input_stack_4x4, target_stack_4x4