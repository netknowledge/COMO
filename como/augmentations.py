"""
COMO Image Augmentations
========================

Albumentations-based image transforms for both training and inference.
Includes document-scanning artefacts, white-border cropping, and adaptive
resizing with keypoint tracking.

Public API:
  - ``ScanEdgeLines``     — simulate dark lines at scanned page edges
  - ``CropWhiteTrain``    — trim white borders, re-pad (training)
  - ``PadWhite``          — randomly pad one side with white space
  - ``PadToSquare``       — pad image to square, adjust keypoints
  - ``AdaptiveLongestMaxSize`` — resize longest edge adaptively
  - ``AdaptiveResize``    — resize with adaptive interpolation + keypoint track
  - ``CropWhiteInference`` — crop white borders (inference)
"""

import random

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np


# ======================== Scan Edge Lines Augmentation ========================

class ScanEdgeLines(A.ImageOnlyTransform):
    """Simulate dark lines / shadows at page edges from document scanning.

    When a book or patent page is scanned, the scanner bed often captures
    dark bands near the binding edge or page borders.  This augmentation
    draws 1-3 such lines on random edges of the image to make the model
    robust to this common artefact in real patent images.

    Each line is a semi-transparent dark band whose width, intensity, and
    edge are randomised independently.
    """

    def __init__(self, max_lines: int = 3, p: float = 0.3):
        super().__init__(p=p)
        self.max_lines = max_lines

    def apply(self, img, **params):
        h, w = img.shape[:2]
        n_lines = random.randint(1, self.max_lines)
        result = img.copy()

        edges = random.sample(['top', 'bottom', 'left', 'right'], k=min(n_lines, 4))

        for edge in edges:
            # Band width: 1-6 % of the relevant dimension
            if edge in ('top', 'bottom'):
                band_w = random.randint(max(1, int(h * 0.01)), max(2, int(h * 0.06)))
            else:
                band_w = random.randint(max(1, int(w * 0.01)), max(2, int(w * 0.06)))

            # Darkness: 0 (black) – 80 (dark grey)
            intensity = random.randint(0, 80)
            # Alpha blend factor for the band
            alpha = random.uniform(0.3, 1.0)

            # Build the band region slice
            if edge == 'top':
                roi = result[0:band_w, :]
            elif edge == 'bottom':
                roi = result[h - band_w:h, :]
            elif edge == 'left':
                roi = result[:, 0:band_w]
            else:  # right
                roi = result[:, w - band_w:w]

            # Blend: roi = roi * (1-alpha) + intensity * alpha
            band = np.full_like(roi, intensity, dtype=np.uint8)
            blended = cv2.addWeighted(roi, 1.0 - alpha, band, alpha, 0)
            # Apply a gradient fade so the edge isn't perfectly sharp
            if edge in ('top', 'bottom'):
                grad = np.linspace(1.0, 0.0, band_w).reshape(-1, 1)
                if edge == 'bottom':
                    grad = grad[::-1]
                if roi.ndim == 3:
                    grad = grad[:, :, np.newaxis]
            else:
                grad = np.linspace(1.0, 0.0, band_w).reshape(1, -1)
                if edge == 'right':
                    grad = grad[:, ::-1]
                if roi.ndim == 3:
                    grad = grad[:, :, np.newaxis]
            grad = grad.astype(np.float32)
            # Where grad~1 use blended (dark), where grad~0 keep original
            final = (blended.astype(np.float32) * grad
                     + roi.astype(np.float32) * (1.0 - grad))
            final = np.clip(final, 0, 255).astype(np.uint8)

            if edge == 'top':
                result[0:band_w, :] = final
            elif edge == 'bottom':
                result[h - band_w:h, :] = final
            elif edge == 'left':
                result[:, 0:band_w] = final
            else:
                result[:, w - band_w:w] = final

        return result

    def get_transform_init_args_names(self):
        return ('max_lines',)


# ======================== CropWhite for Training ========================

class CropWhiteTrain(A.DualTransform):
    """Crop white borders and re-pad with a small margin.
    
    Adapted from MolScribe's CropWhite for albumentations 2.0 API.
    Supports keypoint tracking (DualTransform).
    
    After geometric augmentations (e.g., rotation with fit_output=True),
    the image may have large white corners. This transform trims them so
    the molecule fills the canvas, then adds a small uniform padding.
    """
    
    def __init__(self, value=(255, 255, 255), pad=5, p=1.0):
        super().__init__(p=p)
        self.value = value
        self.pad = pad

    @property
    def targets_as_params(self):
        return ["image"]

    def get_params_dependent_on_data(self, params, data):
        img = data["image"]
        height, width = img.shape[:2]
        
        # Find non-white bounding box
        if img.ndim == 3:
            x = (img != np.array(self.value)).any(axis=2)
        else:
            x = (img != self.value[0])
        
        if not x.any():
            return {"crop_top": 0, "crop_bottom": 0, "crop_left": 0, "crop_right": 0}
        
        row_sum = x.any(axis=1)
        col_sum = x.any(axis=0)
        
        top = int(row_sum.argmax())
        bottom = height - int(row_sum[::-1].argmax())
        left = int(col_sum.argmax())
        right = width - int(col_sum[::-1].argmax())
        
        return {
            "crop_top": top,
            "crop_bottom": height - bottom,
            "crop_left": left,
            "crop_right": width - right,
        }

    def apply(self, img, crop_top=0, crop_bottom=0, crop_left=0, crop_right=0, **params):
        height, width = img.shape[:2]
        cropped = img[crop_top:height - max(crop_bottom, 0) or height,
                      crop_left:width - max(crop_right, 0) or width]
        # Re-pad with uniform small margin
        if self.pad > 0:
            if cropped.ndim == 3:
                fill_val = self.value
            else:
                fill_val = (self.value[0],)
            cropped = cv2.copyMakeBorder(
                cropped, self.pad, self.pad, self.pad, self.pad,
                cv2.BORDER_CONSTANT, value=fill_val
            )
        return cropped

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        crop_top=0, crop_bottom=0, crop_left=0, crop_right=0,
        **params,
    ) -> np.ndarray:
        result = keypoints.copy()
        result[:, 0] = result[:, 0] - crop_left + self.pad
        result[:, 1] = result[:, 1] - crop_top + self.pad
        return result

    def get_transform_init_args_names(self):
        return ('value', 'pad')


class PadWhite(A.DualTransform):
    """Randomly pad ONE edge with white space.

    Mimics the original MolScribe PadWhite: pick one of the four sides
    at random and pad it by up to ``pad_ratio`` of the image dimension.
    """

    def __init__(self, pad_ratio=0.4, p=0.2, value=(255, 255, 255)):
        super().__init__(p=p)
        self.pad_ratio = pad_ratio
        self.value = value

    @property
    def targets_as_params(self):
        return ["image"]

    def get_params_dependent_on_data(self, params, data):
        img = data["image"]
        height, width = img.shape[:2]
        pad_top = pad_bottom = pad_left = pad_right = 0
        side = random.randrange(4)
        if side == 0:
            pad_top = int(height * self.pad_ratio * random.random())
        elif side == 1:
            pad_bottom = int(height * self.pad_ratio * random.random())
        elif side == 2:
            pad_left = int(width * self.pad_ratio * random.random())
        elif side == 3:
            pad_right = int(width * self.pad_ratio * random.random())
        return {"pad_top": pad_top, "pad_bottom": pad_bottom,
                "pad_left": pad_left, "pad_right": pad_right}

    def apply(self, img, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        return cv2.copyMakeBorder(
            img, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=self.value)

    def apply_to_keypoints(
        self, keypoints: np.ndarray,
        pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params,
    ) -> np.ndarray:
        result = keypoints.copy()
        result[:, 0] = result[:, 0] + pad_left
        result[:, 1] = result[:, 1] + pad_top
        return result

    def get_transform_init_args_names(self):
        return ('value', 'pad_ratio')


# ======================== PadToSquare ========================

class PadToSquare:
    """
    Simple callable that pads an image to be square and adjusts keypoints.
    Not an Albumentations transform - used separately.
    """
    def __init__(self, fill=255):
        self.fill = fill

    def __call__(self, image, keypoints=None):
        h, w = image.shape[:2]
        target_dim = max(h, w)
        
        pad_h = target_dim - h
        pad_w = target_dim - w
        
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        # Handle fill value for color images - convert to tuple of floats
        if image.ndim == 3 and image.shape[2] == 3:
            fill_value = (float(self.fill), float(self.fill), float(self.fill))
        else:
            fill_value = (float(self.fill),)
        
        padded_img = cv2.copyMakeBorder(
            image, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=fill_value
        )
        
        # Adjust keypoints if provided
        if keypoints is not None:
            adjusted_kps = []
            for kp in keypoints:
                x, y = kp[0], kp[1]
                adjusted_kps.append([x + left, y + top])
            return padded_img, adjusted_kps
        
        return padded_img, None


# ======================== Inference-Specific Transforms ========================

class AdaptiveLongestMaxSize(A.ImageOnlyTransform):
    """LongestMaxSize with adaptive interpolation.
    
    Uses INTER_LINEAR (bilinear) when upscaling and INTER_AREA when downscaling,
    matching OpenCV best-practice for each direction.
    """
    
    def __init__(self, max_size, p=1.0):
        super().__init__(p=p)
        self.max_size = max_size

    def apply(self, img, **params):
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest == self.max_size:
            return img
        scale = self.max_size / longest
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))
        interpolation = cv2.INTER_AREA if longest > self.max_size else cv2.INTER_LINEAR
        return cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    def get_transform_init_args_names(self):
        return ('max_size',)


class AdaptiveResize(A.DualTransform):
    """Resize with adaptive interpolation.
    
    Uses INTER_LINEAR (bilinear) when upscaling and INTER_AREA when downscaling,
    matching OpenCV best-practice for each direction.
    Supports keypoint tracking (DualTransform).
    """

    def __init__(self, height, width, p=1.0):
        super().__init__(p=p)
        self.height = height
        self.width = width

    @property
    def targets_as_params(self):
        return ["image"]

    def get_params_dependent_on_data(self, params, data):
        img = data["image"]
        h, w = img.shape[:2]
        interpolation = cv2.INTER_AREA if (h > self.height or w > self.width) else cv2.INTER_LINEAR
        return {
            "scale_x": self.width / w,
            "scale_y": self.height / h,
            "interpolation": interpolation,
        }

    def apply(self, img, scale_x=1, scale_y=1, interpolation=cv2.INTER_LINEAR, **params):
        return cv2.resize(img, (self.width, self.height), interpolation=interpolation)

    def apply_to_keypoints(self, keypoints, scale_x=1, scale_y=1, **params):
        if keypoints.size == 0:
            return keypoints
        result = keypoints.copy()
        result[:, 0] *= scale_x
        result[:, 1] *= scale_y
        return result

    def get_transform_init_args_names(self):
        return ('height', 'width')


class CropWhiteInference(A.ImageOnlyTransform):
    """Crop white borders from images during inference.
    
    Finds the bounding box of non-white pixels and crops the image,
    keeping a small padding. This removes wasted white space so that
    the molecule content fills more of the final resized image.
    """
    
    def __init__(self, value=(255, 255, 255), pad=5, p=1.0):
        super().__init__(p=p)
        self.value = value
        self.pad = pad
    
    def apply(self, img, **params):
        height, width = img.shape[:2]
        # Find non-white pixels
        if img.ndim == 3:
            non_white = (img != self.value).any(axis=2)
        else:
            non_white = (img != 255)
        
        if not non_white.any():
            return img
        
        # Find bounding box of non-white region
        rows = non_white.any(axis=1)
        cols = non_white.any(axis=0)
        top = rows.argmax()
        bottom = height - rows[::-1].argmax()
        left = cols.argmax()
        right = width - cols[::-1].argmax()
        
        # Crop with padding
        top = max(0, top - self.pad)
        bottom = min(height, bottom + self.pad)
        left = max(0, left - self.pad)
        right = min(width, right + self.pad)
        
        return img[top:bottom, left:right]
    
    def get_transform_init_args_names(self):
        return ('value', 'pad')
