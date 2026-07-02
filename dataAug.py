import random
import numpy as np


def rotate_and_flip(frames, angle_range=15):
    rotated_flipped_frames = []
    batch_size, seq_len, height, width, channels = frames.shape
    for i in range(batch_size):
        angle = random.uniform(-angle_range, angle_range)
        flip = random.choice([-1, 0, 1])  # -1:水平垂直翻转, 0:水平翻转, 1:垂直翻转
        for j in range(seq_len):
            frame = np.rot90(frames[i, j], k=int(angle))
            if flip != -1:
                frame = np.flip(frame, axis=flip)
            rotated_flipped_frames.append(frame.copy())
    return np.array(rotated_flipped_frames).reshape(frames.shape).copy()

def adjust_brightness_contrast(frames, brightness=0.2, contrast=0.2):
    adjusted_frames = []
    batch_size, seq_len, height, width, channels = frames.shape
    for i in range(batch_size):
        b_factor = 1 + random.uniform(-brightness, brightness)
        c_factor = 1 + random.uniform(-contrast, contrast)
        for j in range(seq_len):
            frame = frames[i, j] * b_factor
            frame = (frame - 127.5 * (1 - c_factor)) / c_factor + 127.5
            adjusted_frames.append(np.clip(frame, 0, 255))
    return np.array(adjusted_frames).reshape(frames.shape).copy()

def temporal_reverse(frames):
    return frames[:, ::-1, :, :, :].copy()

def add_noise(frames, noise_level=15):
    noise_frames = []
    batch_size, seq_len, height, width, channels = frames.shape
    for i in range(batch_size):
        for j in range(seq_len):
            noise = np.random.normal(0, noise_level, (height, width, channels))
            frame = frames[i, j] + noise
            noise_frames.append(np.clip(frame, 0, 255))
    return np.array(noise_frames).reshape(frames.shape).copy()

def synthetic_augmentation(frames, alpha=0.7):
    synthetic_frames = []
    batch_size, seq_len, height, width, channels = frames.shape
    synthetic_image = np.random.uniform(0, 256, (height, width, channels))
    for i in range(batch_size):
        for j in range(seq_len):
            frame = alpha * frames[i, j] + (1 - alpha) * synthetic_image
            synthetic_frames.append(np.clip(frame, 0, 255))
    return np.array(synthetic_frames).reshape(frames.shape).copy()

def apply_augmentations(frames):
    if random.random() > 0.8:
        frames = rotate_and_flip(frames)
    if random.random() > 0.6:
        frames = adjust_brightness_contrast(frames)
    if random.random() > 0.8:
        frames = temporal_reverse(frames)
    if random.random() > 0.7:
        frames = add_noise(frames)
    if random.random() > 0.7:
        frames = synthetic_augmentation(frames)
    return frames
