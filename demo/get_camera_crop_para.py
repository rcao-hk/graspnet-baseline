#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pick crop ROI by clicking 4 points on a live RealSense color stream (cv2).
- Left click: add point (up to 4)
- Right click: remove last point
- Press 'r': reset points
- Press 's': save ROI to JSON (and print config snippet)
- Press 'q' or ESC: quit

Output JSON keys (same as your demo cfg):
  camera_crop_x_left, camera_crop_x_right, camera_crop_y_top, camera_crop_y_bottom
"""

import os
import json
import argparse
import cv2
import numpy as np
import pyrealsense2 as rs


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def draw_overlay(img, pts):
    vis = img.copy()
    h, w = vis.shape[:2]

    # points
    for i, (x, y) in enumerate(pts):
        cv2.circle(vis, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(vis, f"{i+1}", (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # bbox
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
        f"Points: {len(pts)}/4 (any 2+ points define bbox; 4 points is convenient)"
    ]
    y = h - 40
    for line in lines:
        cv2.putText(vis, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y += 28

    return vis


def rs_init(w, h, fps):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)

    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)

    # warmup
    for _ in range(10):
        pipe.wait_for_frames()

    return pipe, align


def rs_get_color(pipe, align):
    frames = pipe.wait_for_frames()
    aligned = align.process(frames)
    color_frame = aligned.get_color_frame()
    if not color_frame:
        return None
    color = np.asanyarray(color_frame.get_data())  # BGR uint8
    return color


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="crop_roi.json", help="Output JSON path")
    parser.add_argument("--window", default="pick_roi_realsense", help="OpenCV window name")
    parser.add_argument("--w", type=int, default=1280, help="RealSense stream width")
    parser.add_argument("--h", type=int, default=720, help="RealSense stream height")
    parser.add_argument("--fps", type=int, default=30, help="RealSense stream FPS")
    args = parser.parse_args()

    pipe, align = rs_init(args.w, args.h, args.fps)

    pts = []
    saved = False
    last_frame = None

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

    try:
        while True:
            frame = rs_get_color(pipe, align)
            if frame is None:
                continue
            last_frame = frame
            h, w = frame.shape[:2]

            vis = draw_overlay(frame, pts)
            cv2.imshow(args.window, vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):  # ESC or q
                break

            if key == ord("r"):
                pts = []
                saved = False

            if key == ord("s"):
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
                    "stream_size_wh": [int(w), int(h)],
                    "picked_points_xy": pts,
                }
                os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
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

                # optional: also dump a snapshot for reference
                snap_path = os.path.splitext(args.out)[0] + "_snapshot.png"
                cv2.imwrite(snap_path, frame)
                print("[SAVED] snapshot:", os.path.abspath(snap_path))

                saved = True

    finally:
        try:
            pipe.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()

    # if user quits without saving but has ROI, print it
    if (not saved) and (last_frame is not None) and (len(pts) >= 2):
        h, w = last_frame.shape[:2]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0, x1 = clamp(min(xs), 0, w - 1), clamp(max(xs), 0, w - 1)
        y0, y1 = clamp(min(ys), 0, h - 1), clamp(max(ys), 0, h - 1)
        print("\n[EXIT] You picked ROI (not saved):")
        print(f"camera_crop_x_left={x0}, camera_crop_x_right={x1}, "
              f"camera_crop_y_top={y0}, camera_crop_y_bottom={y1}")


if __name__ == "__main__":
    main()
