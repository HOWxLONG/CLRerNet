import cv2
import numpy as np
from mmcv.transforms.base import BaseTransform
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class ResizePad(BaseTransform):
    """Resize an image with unchanged aspect ratio, then pad to a fixed size.

    This keeps full-frame highway images uncropped and avoids warping non-16:9
    samples. Lane point coordinates are scaled and shifted with the image.
    """

    def __init__(self, width, height, pad_value=0):
        self.width = int(width)
        self.height = int(height)
        self.pad_value = pad_value

    def _transform_points(self, lanes, scale, pad_left, pad_top):
        transformed = []
        for lane in lanes:
            coords = []
            for i in range(len(lane) // 2):
                x = float(lane[2 * i]) * scale + pad_left
                y = float(lane[2 * i + 1]) * scale + pad_top
                coords.extend([x, y])
            transformed.append(coords)
        return transformed

    def _transform_mask(self, mask, new_w, new_h, pad_left, pad_top):
        resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        out = np.zeros((self.height, self.width), dtype=resized.dtype)
        out[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
        return out

    def transform(self, results):
        img = results['img']
        src_h, src_w = img.shape[:2]
        scale = min(self.width / src_w, self.height / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        pad_left = (self.width - new_w) // 2
        pad_top = (self.height - new_h) // 2
        pad_right = self.width - new_w - pad_left
        pad_bottom = self.height - new_h - pad_top

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=self.pad_value,
        )
        results['img'] = padded
        results['img_shape'] = padded.shape
        results['pad_shape'] = padded.shape
        results['resize_pad_scale'] = scale
        results['resize_pad_pad'] = (pad_left, pad_top, pad_right, pad_bottom)
        results['resize_pad_size'] = (new_w, new_h)

        if 'gt_points' in results:
            results['gt_points'] = self._transform_points(
                results['gt_points'], scale, pad_left, pad_top
            )
        if results.get('gt_masks') is not None:
            results['gt_masks'] = self._transform_mask(
                results['gt_masks'], new_w, new_h, pad_left, pad_top
            )
        return results
