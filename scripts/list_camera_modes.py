#!/usr/bin/env python3


"""
  MIT License -------------------------------------------------------------------------------
  Copyright (c) 2026 W.M. Nipun Dhananjaya

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.

  --------------------------------------------------------------------------------------------

  * @file arducam_b0573_node.cpp
  * @author: WM Nipun Dhananjaya (nipun.dhananjaya@gmail.com)
  * @date: 27.02.2026
  * @brief Python script for listing all V4L2 pixel formats, resolutions, and frame-rate 
         intervals available on the Arducam device.   

  * Usage:
        python3 list_camera_modes.py [/dev/video0]
        ros2 run arducam_dual_camera list_camera_modes.py
----------------------------------------------------------------------------------------------
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
