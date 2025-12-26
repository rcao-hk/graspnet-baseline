#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pick crop ROI by clicking 4 points on an image (cv2).
- Left click: add point
- Right click: remove last point
- Press 'r': reset
- Press 's': save ROI to JSON
- Press 'q' or ESC: quit

Output:
- crop_x_left, crop_x_right, crop_y_top, crop_y_bottom (int, clamped)
- Also prints a suggested config snippet
"""

import os
import json
import argparse
import cv2
import numpy as np


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def draw_overlay(img, pts):
    vis = img.copy()
    h, w = vis.shape[:2]

    # draw points
    for i, (x, y) in enumerate(pts):
        cv2.circle(vis, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(vis, f"{i+1}", (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # draw bbox if possible
    if len(pts) >= 2:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0, x1 = clamp(min(xs), 0, w - 1), clamp(max(xs), 0, w - 1)
        y0, y1 = clamp(min(ys), 0, h - 1), clamp(max(ys), 0, h - 1)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 0, 0), 2)
        cv2.putText(vis, f"ROI: x[{x0},{x1}] y[{y0},{y1}]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # instructions
    lines = [
        "LClick: add point | RClick: undo | r: reset | s: save | q/ESC: quit",
        f"Points: {len(pts)}/4 (any 2+ points already define bbox; use 4 for convenience)"
    ]
    y = h - 40
    for line in lines:
        cv2.putText(vis, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y += 28

    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--out", default="crop_roi.json", help="Output JSON path")
    parser.add_argument("--window", default="pick_roi", help="OpenCV window name")
    args = parser.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {args.image}")

    h, w = img.shape[:2]
    pts = []

    def on_mouse(event, x, y, flags, param):
        nonlocal pts
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(pts) < 4:
                pts.append((int(x), int(y)))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(pts) > 0:
                pts.pop()

    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(args.window, on_mouse)

    saved = False
    while True:
        vis = draw_overlay(img, pts)
        cv2.imshow(args.window, vis)
        key = cv2.waitKey(20) & 0xFF

        if key in (27, ord('q')):  # ESC or q
            break
        if key == ord('r'):
            pts = []
            saved = False
        if key == ord('s'):
            if len(pts) < 2:
                print("[WARN] Need at least 2 points to define a crop bbox.")
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x0, x1 = clamp(min(xs), 0, w - 1), clamp(max(xs), 0, w - 1)
            y0, y1 = clamp(min(ys), 0, h - 1), clamp(max(ys), 0, h - 1)

            roi = {
                "camera_crop_x_left": int(x0),
                "camera_crop_x_right": int(x1),
                "camera_crop_y_top": int(y0),
                "camera_crop_y_bottom": int(y1),
                "image_path": os.path.abspath(args.image),
                "image_size_wh": [int(w), int(h)],
                "picked_points_xy": pts,
            }
            with open(args.out, "w") as f:
                json.dump(roi, f, indent=2)

            print("\n[SAVED]", os.path.abspath(args.out))
            print("crop params:")
            print(f"  camera_crop_x_left   = {x0}")
            print(f"  camera_crop_x_right  = {x1}")
            print(f"  camera_crop_y_top    = {y0}")
            print(f"  camera_crop_y_bottom = {y1}")
            print("\nconfig snippet:")
            print(json.dumps({
                "camera_crop_x_left": int(x0),
                "camera_crop_x_right": int(x1),
                "camera_crop_y_top": int(y0),
                "camera_crop_y_bottom": int(y1),
            }, indent=2))
            saved = True

    cv2.destroyAllWindows()
    if not saved and len(pts) >= 2:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0, x1 = clamp(min(xs), 0, w - 1), clamp(max(xs), 0, w - 1)
        y0, y1 = clamp(min(ys), 0, h - 1), clamp(max(ys), 0, h - 1)
        print("\n[EXIT] You picked ROI (not saved):")
        print(f"camera_crop_x_left={x0}, camera_crop_x_right={x1}, "
              f"camera_crop_y_top={y0}, camera_crop_y_bottom={y1}")


if __name__ == "__main__":
    main()
