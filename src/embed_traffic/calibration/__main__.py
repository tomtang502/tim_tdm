"""CLI entry point: python -m embed_traffic.calibration"""

from embed_traffic.calibration.calibrate import main
import sys

if __name__ == "__main__":
    sys.exit(main())
