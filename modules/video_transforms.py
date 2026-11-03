"""
MMRT: Video Transforms for Inference
Video preprocessing utilities for sign language recognition.
"""
import cv2
import numpy as np
import torch
import PIL.Image


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label=None, file_info=None):
        for t in self.transforms:
            image = t(image)
        return image, label


class ToTensor:
    def __call__(self, video):
        if isinstance(video, list):
            video = np.array(video)
            video = torch.from_numpy(video.transpose((0, 3, 1, 2))).float()
        elif isinstance(video, np.ndarray):
            video = torch.from_numpy(video.transpose((0, 3, 1, 2)))
        return video


class CenterCrop:
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, clip):
        im_h, im_w, im_c = clip[0].shape
        new_h, new_w = self.size
        new_h = im_h if new_h >= im_h else new_h
        new_w = im_w if new_w >= im_w else new_w
        top = int(round((im_h - new_h) / 2.))
        left = int(round((im_w - new_w) / 2.))
        return [img[top:top + new_h, left:left + new_w] for img in clip]


class Resize:
    def __init__(self, rate=1.0, interp='bilinear'):
        self.rate = rate
        self.interpolation = interp

    def __call__(self, clip):
        if self.rate == 1.0:
            return clip
        scaling_factor = self.rate
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError(f'Expected numpy.ndarray or PIL.Image but got {type(clip[0])}')

        new_w = int(im_w * scaling_factor) if 0 < scaling_factor <= 1 else int(scaling_factor)
        new_h = int(im_h * scaling_factor) if 0 < scaling_factor <= 1 else int(scaling_factor)
        new_size = (new_w, new_h)

        if isinstance(clip[0], np.ndarray):
            return [np.array(PIL.Image.fromarray(img).resize(new_size)) for img in clip]
        return [img.resize(size=new_size) for img in clip]


def extract_video_frames(video_path, target_size=256, crop_size=224):
    """
    Extract frames from video for MMRT inference.

    Args:
        video_path: Path to video file
        target_size: Resize to this size (default 256)
        crop_size: Center crop to this size (default 224)

    Returns:
        Tensor of shape [T, C, H, W] normalized to [-1, 1]
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize to target_size
        frame = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
        frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError("No frames extracted from video")

    # Apply transforms
    transform = Compose([CenterCrop(crop_size), Resize(1.0), ToTensor()])
    video, _ = transform(frames, None)

    # Normalize to [-1, 1]
    video = video.float() / 127.5 - 1
    return video
