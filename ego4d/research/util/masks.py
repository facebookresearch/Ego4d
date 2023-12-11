from dataclasses import dataclass

import numpy as np

from ego4d.research.util.lzstring import decompress_from_encoded_uri
from pycocotools import mask as mask_utils


@dataclass
class Mask:
    width: int
    height: int
    encoded_mask: str


def decode_mask(mask: dict) -> np.ndarray:
    w = mask["width"]
    h = mask["height"]
    encoded_mask = mask["encodedMask"]
    return decode_mask_obj(Mask(width=w, height=h, encoded_mask=encoded_mask))


def decode_mask_obj(mask: Mask) -> np.ndarray:
    decomp_string = decompress_from_encoded_uri(mask.encoded_mask)
    decomp_encoded = decomp_string.encode()  # pyre-ignore
    rle_obj = {
        "size": [mask.height, mask.width],
        "counts": decomp_encoded,
    }

    output = mask_utils.decode(rle_obj)
    return output


def blend_mask(
    input_img: np.ndarray, binary_mask: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    if input_img.ndim == 2:
        return input_img

    mask_image = np.zeros(input_img.shape, np.uint8)
    mask_image[:, :, 1] = 255
    mask_image = mask_image * np.repeat(binary_mask[:, :, np.newaxis], 3, axis=2)

    blend_image = input_img[:, :, :]
    pos_idx = binary_mask > 0
    for ind in range(input_img.ndim):
        ch_img1 = input_img[:, :, ind]
        ch_img2 = mask_image[:, :, ind]
        ch_img3 = blend_image[:, :, ind]
        ch_img3[pos_idx] = alpha * ch_img1[pos_idx] + (1 - alpha) * ch_img2[pos_idx]
        blend_image[:, :, ind] = ch_img3
    return blend_image
