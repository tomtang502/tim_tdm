"""Traffic Decision Model (TDM).

Consumes TIM's world-space pedestrian output together with an ego-vehicle
`CarState` and emits an alert: {NONE, CAUTION, SLOW_DOWN, BRAKE}.

Public entry points:

    from embed_traffic.tdm import TDM, CarState, AlertLevel, TDMOutput
    from embed_traffic.tdm.simulator import make_scenario

    tdm = TDM()                                    # default thresholds
    car = make_scenario("approaching")(0.0)        # synthetic car state at t=0
    out = tdm.decide(tim_frame_output, car)
    print(out.alert, out.reason)
"""

from embed_traffic.tdm.schema import (
    AlertLevel,
    CarState,
    CollisionPrediction,
    TDMOutput,
)
from embed_traffic.tdm.tdm import TDM

__all__ = [
    "TDM",
    "AlertLevel",
    "CarState",
    "CollisionPrediction",
    "TDMOutput",
]
