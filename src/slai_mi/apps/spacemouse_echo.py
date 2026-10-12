"""Print SpaceMouse input and worker diagnostics without opening robot hardware."""

from __future__ import annotations

import argparse
import json
import time

from slai_mi.devices.spacemouse.client import SpaceMouseProcess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--duration-s", type=float, default=0.0, help="0 means run until Ctrl-C")
    args = parser.parse_args()
    if args.rate_hz <= 0.0 or args.duration_s < 0.0:
        parser.error("--rate-hz must be positive and --duration-s must be non-negative")

    mouse = SpaceMouseProcess(
        backend="spnav",
        rate_hz=max(125.0, args.rate_hz * 4.0),
        stale_timeout=0.05,
    )
    started = time.monotonic()
    try:
        mouse.start()
        while args.duration_s == 0.0 or time.monotonic() - started < args.duration_s:
            motion, buttons = mouse.state()
            diagnostics = mouse.diagnostics()
            print(
                json.dumps(
                    {
                        "motion": [round(float(value), 5) for value in motion],
                        "buttons": {
                            str(key): value for key, value in sorted(buttons.items()) if value
                        },
                        **diagnostics,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            time.sleep(1.0 / args.rate_hz)
    except KeyboardInterrupt:
        pass
    finally:
        mouse.stop()


if __name__ == "__main__":
    main()
