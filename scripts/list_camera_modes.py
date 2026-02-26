#!/usr/bin/env python3
"""
list_camera_modes.py

Queries a V4L2 camera device and prints all available
pixel formats, resolutions, and frame-rate intervals.

Usage:
  python3 list_camera_modes.py [/dev/video0]
  ros2 run arducam_dual_camera list_camera_modes.py
"""

import sys
import subprocess
import re


def list_modes(device: str = "/dev/video0") -> None:
    print(f"\n{'='*60}")
    print(f"  Arducam camera modes — {device}")
    print(f"{'='*60}\n")

    # ── Raw v4l2-ctl output ────────────────────────────────────────────────
    result = subprocess.run(
        ["v4l2-ctl", "-d", device, "--list-formats-ext"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"ERROR: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)

    raw = result.stdout
    print(raw)

    # ── Parse and probe frame intervals per mode ───────────────────────────
    # Extract (format, width, height) pairs
    current_fmt = None
    sizes: list[tuple[str, int, int]] = []

    for line in raw.splitlines():
        fmt_match = re.search(r"'(\w+)'\s+\(", line)
        size_match = re.search(r"Size: Discrete (\d+)x(\d+)", line)
        if fmt_match:
            current_fmt = fmt_match.group(1)
        elif size_match and current_fmt:
            sizes.append((current_fmt, int(size_match.group(1)), int(size_match.group(2))))

    if not sizes:
        print("No resolutions found.")
        return

    print(f"\n{'─'*60}")
    print(f"  Frame-interval probe")
    print(f"{'─'*60}")

    seen: set[tuple[int, int]] = set()
    for fmt, w, h in sizes:
        if (w, h) in seen:
            continue
        seen.add((w, h))

        r2 = subprocess.run(
            ["v4l2-ctl", "-d", device,
             f"--list-frameintervals=width={w},height={h},pixelformat={fmt}"],
            capture_output=True, text=True
        )
        fps_info = r2.stdout.strip()
        if "Interval" in fps_info:
            intervals = re.findall(r"\((\S+ fps)\)", fps_info)
            fps_str = ", ".join(intervals) if intervals else fps_info
        else:
            fps_str = "intervals not reported by driver"

        half_w = w // 2
        print(f"  {fmt}  combined={w}×{h}  →  per-eye={half_w}×{h}  |  {fps_str}")

    print()


if __name__ == "__main__":
    dev = sys.argv[1] if len(sys.argv) > 1 else "/dev/video0"
    list_modes(dev)
