"""TIM inference package.

Contains the production inference wrapper that turns a video frame (or video)
into the per-pedestrian info consumed by TDM.
"""

from embed_traffic.inference.schema import (
    PedestrianInfo,
    TIMFrameOutput,
    serialize_frame_output,
    deserialize_frame_output,
)
from embed_traffic.inference.tim import TIM
from embed_traffic.inference.topdown import (
    render_topdown_frame,
    render_topdown_video,
)

__all__ = [
    "TIM",
    "PedestrianInfo",
    "TIMFrameOutput",
    "serialize_frame_output",
    "deserialize_frame_output",
    "render_topdown_frame",
    "render_topdown_video",
]
