"""Decode uniformly sampled frames from a STAR video interval."""

from pathlib import Path


def decode_clip(path: Path, start: float, end: float, count: int, image_size: int):
    import cv2
    import numpy as np

    if not path.is_file():
        raise FileNotFoundError(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    try:
        duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / capture.get(cv2.CAP_PROP_FPS)
        if not (0 <= start < end) or duration <= 0:
            raise ValueError(
                f"Invalid clip or video duration: {path} [{start}, {end}], duration={duration}"
            )
        clip_end = min(end, duration)
        if start >= clip_end:
            raise ValueError(
                f"Clip starts beyond video end: {path} [{start}, {end}], duration={duration}"
            )
        timestamps = [
            start + (clip_end - start) * (i + 0.5) / count for i in range(count)
        ]
        frames = []
        for timestamp in timestamps:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot decode {path} at {timestamp:.3f}s")
            height, width = frame.shape[:2]
            scale = min(1.0, image_size / max(height, width))
            if scale < 1:
                frame = cv2.resize(
                    frame,
                    (round(width * scale), round(height * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return np.stack(frames), timestamps, clip_end - start
    finally:
        capture.release()
