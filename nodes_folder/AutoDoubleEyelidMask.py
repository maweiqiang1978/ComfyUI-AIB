import numpy as np
import cv2
import mediapipe as mp
import torch


def get_px(lm, idx, w, h):
    """提取关键点的像素坐标"""
    x = int(lm.landmark[idx].x * w)
    y = int(lm.landmark[idx].y * h)
    return (x, y)


def calculate_midpoint(point1, point2):
    """计算两个点的中点"""
    return ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)


def calculate_ratio_point(point1, point2, ratio):
    """按比例计算点的坐标"""
    x = int(point1[0] + ratio * (point2[0] - point1[0]))
    y = int(point1[1] + ratio * (point2[1] - point1[1]))
    return (x, y)


def gen_double_eyelid_mask(h, w, lm, is_right_eye=False):
    mask = np.zeros((h, w), dtype=np.float32)
    if is_right_eye:
        single_points = [257, 259, 260]
        midpoint_pairs = [(467, 342), (342, 265), (265, 446), (446, 359), (263, 467), (466, 467), (258, 385), (362, 464)]
        ratio_pairs = [
            (414, 413, 1 / 3),
            (286, 441, 1 / 3),
            (258, 442, 1 / 3),
            (388, 260, 1 / 3),
            (387, 259, 1 / 3),
            (386, 257, 1 / 3),
            (286, 384, 1 / 3),
            (414, 398, 1 / 3)
        ]
        connection_order = [
            (0, 'Ratio'),
            (1, 'Ratio'),
            (2, 'Ratio'),
            (0, 'Single'), (1, 'Single'), (2, 'Single'),
            (0, 'Midpoint'), (1, 'Midpoint'), (2, 'Midpoint'), (3, 'Midpoint'), (4, 'Midpoint'), (5, 'Midpoint'),
            (3, 'Ratio'), (4, 'Ratio'), (5, 'Ratio'),
            (6, 'Midpoint'),
            (6, 'Ratio'),
            (7, 'Ratio'),
            (7, 'Midpoint')
        ]
        points = []
        for index, point_type in connection_order:
            if point_type == 'Single':
                points.append(get_px(lm, single_points[index], w, h))
            elif point_type == 'Midpoint':
                point1, point2 = midpoint_pairs[index]
                midpoint = calculate_midpoint(get_px(lm, point1, w, h), get_px(lm, point2, w, h))
                points.append(midpoint)
            elif point_type == 'Ratio':
                point1, point2, ratio = ratio_pairs[index]
                ratio_point = calculate_ratio_point(get_px(lm, point1, w, h), get_px(lm, point2, w, h), ratio)
                points.append(ratio_point)
        cnt = np.array(points, dtype=np.int32)
    else:
        single_points = [56, 28, 27, 29, 30, 190]
        midpoint_pairs = [(247, 113), (35, 226), (226, 130), (33, 247), (246, 247)]
        ratio_pairs = [
            (161, 30, 1 / 3),
            (160, 29, 1 / 3),
            (159, 27, 1 / 3),
            (158, 28, 1 / 3),
            (157, 56, 1 / 3),
            (173, 190, 1 / 3),
            (133, 243, 1 / 3)
        ]
        connection_order = [
            (0, 'Single'), (1, 'Single'), (2, 'Single'), (3, 'Single'), (4, 'Single'),
            (0, 'Midpoint'), (1, 'Midpoint'), (2, 'Midpoint'), (3, 'Midpoint'), (4, 'Midpoint'),
            (0, 'Ratio'), (1, 'Ratio'), (2, 'Ratio'), (3, 'Ratio'), (4, 'Ratio'),
            (5, 'Ratio'), (6, 'Ratio'), (5, 'Single')
        ]
        points = []
        for index, point_type in connection_order:
            if point_type == 'Single':
                points.append(get_px(lm, single_points[index], w, h))
            elif point_type == 'Midpoint':
                point1, point2 = midpoint_pairs[index]
                midpoint = calculate_midpoint(get_px(lm, point1, w, h), get_px(lm, point2, w, h))
                points.append(midpoint)
            elif point_type == 'Ratio':
                point1, point2, ratio = ratio_pairs[index]
                ratio_point = calculate_ratio_point(get_px(lm, point1, w, h), get_px(lm, point2, w, h), ratio)
                points.append(ratio_point)
        cnt = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [cnt], 1.0)
    return mask


class AutoDoubleEyelidMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("left_eyelid_mask", "right_eyelid_mask")
    FUNCTION = "run"
    CATEGORY = "AIB/Mask"

    def run(self, image):
        img_np = image[0].cpu().numpy().copy()
        h, w = img_np.shape[:2]
        img_rgb = (img_np * 255).astype(np.uint8)
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
        ) as fm:
            res = fm.process(img_rgb)
            if not res.multi_face_landmarks:
                left_mask = np.zeros((h, w), dtype=np.float32)
                right_mask = np.zeros((h, w), dtype=np.float32)
            else:
                lm = res.multi_face_landmarks[0]
                left_mask = gen_double_eyelid_mask(h, w, lm)
                right_mask = gen_double_eyelid_mask(h, w, lm, is_right_eye=True)
        left_mask = np.expand_dims(left_mask, axis=0)
        right_mask = np.expand_dims(right_mask, axis=0)
        left_mask = np.clip(left_mask, 0.0, 1.0)
        right_mask = np.clip(right_mask, 0.0, 1.0)
        left_tensor = torch.from_numpy(left_mask)
        right_tensor = torch.from_numpy(right_mask)
        return (left_tensor, right_tensor)


# ComfyUI节点映射（必须）
NODE_CLASS_MAPPINGS = {
    "AutoDoubleEyelidMask": AutoDoubleEyelidMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoDoubleEyelidMask": "Auto Double Eyelid Mask"
}