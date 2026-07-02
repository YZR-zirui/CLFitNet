__author__ = 'YaoHan'
import numpy as np
import os
import cv2
from PIL import Image
import logging
import random
from typing import List, Dict

logger = logging.getLogger(__name__)

# 新增：掩码文件夹名称配置
MASK_FOLDER_NAME = "drop2_1"
# 仅保留后2个时间序列
KEEP_SEQ_NUM = 2

class InputHandle:
    def __init__(self, datas, masks, indices, group_mapping, input_param):
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.minibatch_size = input_param['minibatch_size']
        self.image_width = input_param['image_width']
        self.datas = datas          # 原始图像数据
        self.masks = masks          # 泪液层掩码数据
        self.indices = indices
        self.group_mapping = group_mapping          # 序列起始索引 -> group_id
        self.current_position = 0
        self.current_batch_indices = []
        self.current_input_length = input_param['seq_length']

    def total_batch(self):
        return int(len(self.indices) / self.minibatch_size)

    def total(self):
        return len(self.indices)

    def begin(self, do_shuffle=False, shuffle_within_group=True):
        logger.info("Initialization for read data ")
        if do_shuffle:
            random.shuffle(self.indices)
        elif shuffle_within_group and self.group_mapping:
            # 按组聚合索引
            group_dict: Dict[str, List[int]] = {}
            for idx in self.indices:
                group_id = self.group_mapping.get(idx)
                if group_id is None:
                    group_id = "default"
                group_dict.setdefault(group_id, []).append(idx)
            # 组内打乱
            for group_id in group_dict:
                random.shuffle(group_dict[group_id])
            # 按组 id 排序后展平
            new_indices = []
            for group_id in sorted(group_dict.keys()):
                new_indices.extend(group_dict[group_id])
            self.indices = new_indices
        elif shuffle_within_group:
            # 没有 group_mapping 时退化为全局打乱
            random.shuffle(self.indices)

        self.current_position = 0
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.minibatch_size]

    def next(self):
        self.current_position += self.minibatch_size
        if self.no_batch_left():
            return None
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.minibatch_size]

    def no_batch_left(self):
        return self.current_position + self.minibatch_size > self.total()

    def get_batch(self):
            if self.no_batch_left():
                logger.error(f"No batch left in {self.name}. Call begin() to restart.")
                return None

            # 图像保持完整的输入长度（5帧，用于3进2出）
            input_batch = np.zeros(
                (self.minibatch_size, self.current_input_length, self.image_width, self.image_width, 1),
                dtype=self.input_data_type)
            # 掩码只取最后两帧 (KEEP_SEQ_NUM = 2)
            mask_batch = np.zeros(
                (self.minibatch_size, KEEP_SEQ_NUM, self.image_width, self.image_width, 1),
                dtype=self.input_data_type)

            for i, batch_ind in enumerate(self.current_batch_indices):
                begin = batch_ind
                end = begin + self.current_input_length
                
                input_batch[i, :, :, :, :] = self.datas[begin:end, :, :, :]
                mask_batch[i, :, :, :, :] = self.masks[begin:end, :, :, :][-KEEP_SEQ_NUM:]

            # 返回：[batch, 7, H, W, 1] (前5个通道是完整图像序列，后2个通道是预测目标的掩码)
            return np.concatenate([input_batch, mask_batch], axis=1)
    
    def print_stat(self):
        logger.info(f"Iterator Name: {self.name}")
        logger.info(f"current_position: {self.current_position}")
        logger.info(f"Minibatch Size: {self.minibatch_size}")
        logger.info(f"total Size: {self.total()}")
        logger.info(f"current_input_length: {self.current_input_length}")
        logger.info(f"Input Data Type: {self.input_data_type}")


class DataProcess:
    def __init__(self, input_param):
        self.paths = os.path.join(input_param['paths'], input_param['mode'])
        self.image_width = input_param['image_width']
        self.input_param = input_param
        self.seq_len = input_param['seq_length']

    def load_mask(self, tp_path):
            """加载单个时间点的泪液层掩码"""
            # 【关键修复】：根据实际目录结构，添加 'scan 1' 层级
            mask_path = os.path.join(tp_path, 'scan 1', MASK_FOLDER_NAME)
            if not os.path.isdir(mask_path):
                return None
            
            masks = [f for f in os.listdir(mask_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            if not masks:
                return None
            masks.sort()
            
            mask_list = []
            for mask_name in masks:
                mask_img_path = os.path.join(mask_path, mask_name)
                mask_img = Image.open(mask_img_path).convert('L')
                mask_np = np.array(mask_img, dtype=np.float32)
                # 同样需要归一化到 0-1 之间，这对于后面的质心计算非常重要
                resized = cv2.resize(mask_np, (self.image_width, self.image_width)) / 255.0  
                mask_list.append(resized.reshape(self.image_width, self.image_width, 1))
            return mask_list
    
    
    def load_data(self, paths, mode='train'):
        print(f"Mode: {mode} || Loading images from {paths}")

        data_list = []          # 存储所有图像数据
        mask_list = []          # 存储所有掩码数据
        seq_group_mapping = {}  # 序列起始索引 -> group_id

        for case_name in sorted(os.listdir(paths)):
            case_path = os.path.join(paths, case_name)
            if not os.path.isdir(case_path):
                continue

            for side in ['OD', 'OS']:
                side_path = os.path.join(case_path, side)
                if not os.path.isdir(side_path):
                    continue

                # 获取时间点文件夹并按名称排序
                timepoints = sorted([d for d in os.listdir(side_path)
                                     if os.path.isdir(os.path.join(side_path, d))])
                if len(timepoints) < self.seq_len:
                    continue

                group_id = f"{case_name}_{side}"
                side_images = []
                side_masks = []

                for tp in timepoints:
                    tp_path = os.path.join(side_path, tp)
                    new_path = os.path.join(tp_path, 'scan 1', 'new')
                    if not os.path.isdir(new_path):
                        continue

                    # 加载图像
                    images = [f for f in os.listdir(new_path)
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                    if not images:
                        continue
                    images.sort()

                    tp_images = []
                    for img_name in images:
                        img_path = os.path.join(new_path, img_name)
                        img = Image.open(img_path).convert('L')
                        img_np = np.array(img, dtype=np.float32)
                        resized = cv2.resize(img_np, (self.image_width, self.image_width)) / 255.0
                        tp_images.append(resized.reshape(self.image_width, self.image_width, 1))
                    
                    # 加载对应掩码
                    tp_masks = self.load_mask(tp_path)
                    if tp_masks is None or len(tp_masks) != len(tp_images):
                        # 掩码数量不匹配则跳过该时间点
                        continue
                    
                    side_images.extend(tp_images)
                    side_masks.extend(tp_masks)

                if len(side_images) < self.seq_len:
                    continue

                group_start_idx = len(data_list)
                data_list.extend(side_images)
                mask_list.extend(side_masks)

                # 滑动窗口生成序列起始位置（步长 = seq_len，不重叠）
                step = self.seq_len
                for start in range(0, len(side_images) - self.seq_len + 1, step):
                    global_start = group_start_idx + start
                    seq_group_mapping[global_start] = group_id

        total_frames = len(data_list)
        if total_frames == 0:
            raise RuntimeError(f"No valid frames found in {paths}. Check directory structure.")

        data = np.array(data_list, dtype=np.float32)
        masks = np.array(mask_list, dtype=np.float32) if mask_list else np.zeros_like(data)
        indices = sorted(seq_group_mapping.keys())
        group_mapping = {idx: seq_group_mapping[idx] for idx in indices}

        print(f"Built {len(indices)} sequences (each length {self.seq_len}) from {total_frames} frames.")
        print(f"Loaded {len(mask_list)} mask frames.")
        return data, masks, indices, group_mapping

    def get_train_input_handle(self):
        train_data, train_masks, train_indices, train_group_map = self.load_data(self.paths, mode='train')
        return InputHandle(train_data, train_masks, train_indices, train_group_map, self.input_param)

    def get_test_input_handle(self):
        test_data, test_masks, test_indices, test_group_map = self.load_data(self.paths, mode='test')
        return InputHandle(test_data, test_masks, test_indices, test_group_map, self.input_param)

    def get_valid_input_handle(self):
        valid_data, valid_masks, valid_indices, valid_group_map = self.load_data(self.paths, mode='valid')
        return InputHandle(valid_data, valid_masks, valid_indices, valid_group_map, self.input_param)