import numpy as np
import cv2
import mediapipe as mp
import torch


     
class MaskCrop:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"mask": ("MASK",),
				"top": ("INT", {"default": 0, "min": 0}),
				"left": ("INT", {"default": 0, "min": 0}),
				"right": ("INT", {"default": 100, "min": 0}),
				"bottom": ("INT", {"default": 100, "min": 0})
			}
		}

	RETURN_TYPES = ("MASK",)
	RETURN_NAMES = ("cropped_mask",)
	FUNCTION = "crop_mask"
	CATEGORY = "AIB/Mask"

	def crop_mask(self, mask, top, left, right, bottom):
		cropped_mask = mask[0, top:bottom, left:right]
		cropped_mask = np.expand_dims(cropped_mask, axis=0)
		return (torch.from_numpy(cropped_mask),) 


# ComfyUI节点映射（必须）
NODE_CLASS_MAPPINGS = {
    "MaskCrop": MaskCrop
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskCrop": "Mask Crop"
}