"""Monocular depth estimation wrapper.

Wraps HuggingFace Depth-Anything-V2 (metric, outdoor) as our default. Any
transformers-compatible depth model will work — pass `model_id` to use a
newer checkpoint (e.g., a future DA3 release).

This module is only used at camera-calibration time, not at per-frame
inference. Runtime TIM uses the pre-computed homography in `schema.py`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

# Default model: DA-V2 Metric-Outdoor-Large. Metric depth in meters, trained on
# outdoor / driving data. Roughly matches our traffic-junction viewpoint.
DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"


class DepthEstimator:
    """Lazy-loads a HF depth model and runs it on BGR images.

    Example:
        est = DepthEstimator()                       # default DA-v2-metric-outdoor
        est = DepthEstimator("other/depth-model-id") # override
        depth = est.predict(bgr_frame)                # (H, W) float32 meters
    """

    def __init__(
        self,
        model_id: str = DEFAULT_DEPTH_MODEL,
        device: Optional[str | torch.device] = None,
    ) -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.model_id = model_id
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = (
            AutoModelForDepthEstimation.from_pretrained(model_id)
            .to(self.device)
            .eval()
        )

    @torch.no_grad()
    def predict(self, bgr: np.ndarray) -> np.ndarray:
        """Return depth map of shape (H, W) in meters (float32).

        Handles BGR → RGB conversion internally; accepts any OpenCV-style
        uint8 HxWx3 array.
        """
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 image, got shape {bgr.shape}")

        rgb = bgr[:, :, ::-1]  # BGR → RGB
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        # predicted_depth has shape (B, h, w) at model's internal resolution
        depth = outputs.predicted_depth  # (1, h, w) tensor
        # Resize back to input resolution
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=(bgr.shape[0], bgr.shape[1]),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        return depth.detach().cpu().numpy().astype(np.float32)

    def predict_batch(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Convenience wrapper: run `predict` on each frame."""
        return [self.predict(f) for f in frames]
