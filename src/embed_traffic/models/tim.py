"""Legacy TIM module — kept only as a re-export shim for backwards compatibility.

The canonical TIM inference API now lives in `embed_traffic.inference`. Import
from there instead:

    from embed_traffic.inference import TIM, PedestrianInfo, TIMFrameOutput

This shim remains because older scripts/notebooks may `from embed_traffic.models.tim import TIM`.
It intentionally does NOT re-introduce the deprecated `traffic_light_state` field
or `view="traffic_light"` path — see plan.txt [VII].
"""

from __future__ import annotations

import warnings

from embed_traffic.inference.schema import PedestrianInfo, TIMFrameOutput as TIMOutput
from embed_traffic.inference.tim import TIM

warnings.warn(
    "embed_traffic.models.tim is deprecated; import from embed_traffic.inference instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["TIM", "PedestrianInfo", "TIMOutput"]
