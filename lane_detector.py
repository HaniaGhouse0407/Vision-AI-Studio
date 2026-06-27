"""
Lane Detection Pipeline using OpenCV Canny + Hough Transform.
Implements full lane detection for dashcam / ADAS applications.
"""
from __future__ import annotations
import cv2, numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class LaneLines:
    left: Optional[np.ndarray]    # (x1, y1, x2, y2)
    right: Optional[np.ndarray]
    curvature: Optional[float] = None  # pixels
    offset: Optional[float] = None     # lane center offset in pixels


class LaneDetector:
    """
    Classical lane detection pipeline:
    1. Grayscale → Gaussian blur → Canny edge detection
    2. Region-of-interest (trapezoid) masking
    3. Probabilistic Hough transform
    4. RANSAC line fitting + averaging
    5. Extrapolate to full lane lines
    """

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        blur_kernel: int = 5,
        hough_rho: float = 2,
        hough_theta: float = np.pi / 180,
        hough_threshold: int = 20,
        hough_min_line_len: int = 40,
        hough_max_line_gap: int = 20,
        roi_bottom_pct: float = 0.95,
        roi_top_pct: float = 0.60,
    ):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.blur_kernel = blur_kernel
        self.hough_rho = hough_rho
        self.hough_theta = hough_theta
        self.hough_threshold = hough_threshold
        self.hough_min_line_len = hough_min_line_len
        self.hough_max_line_gap = hough_max_line_gap
        self.roi_bottom_pct = roi_bottom_pct
        self.roi_top_pct = roi_top_pct
        # Temporal smoothing buffer
        self._history: List[LaneLines] = []
        self._history_len = 7

    def detect(self, frame: np.ndarray) -> LaneLines:
        """Full detection pipeline on a single frame."""
        h, w = frame.shape[:2]

        # 1. Pre-process
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # 2. ROI mask — trapezoidal region
        mask = np.zeros_like(edges)
        roi_pts = np.array([[
            (int(w * 0.05), int(h * self.roi_bottom_pct)),
            (int(w * 0.45), int(h * self.roi_top_pct)),
            (int(w * 0.55), int(h * self.roi_top_pct)),
            (int(w * 0.95), int(h * self.roi_bottom_pct)),
        ]], dtype=np.int32)
        cv2.fillPoly(mask, roi_pts, 255)
        masked = cv2.bitwise_and(edges, mask)

        # 3. Hough lines
        raw_lines = cv2.HoughLinesP(
            masked, self.hough_rho, self.hough_theta, self.hough_threshold,
            minLineLength=self.hough_min_line_len, maxLineGap=self.hough_max_line_gap,
        )

        # 4. Separate & fit left/right
        left_pts, right_pts = [], []
        if raw_lines is not None:
            for line in raw_lines:
                x1, y1, x2, y2 = line[0]
                if x1 == x2:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.3:   # ignore near-horizontal
                    continue
                (left_pts if slope < 0 else right_pts).extend([(x1, y1), (x2, y2)])

        lane = LaneLines(
            left=self._fit_line(left_pts, h, self.roi_top_pct),
            right=self._fit_line(right_pts, h, self.roi_top_pct),
        )
        lane = self._smooth(lane)
        if lane.left is not None and lane.right is not None:
            lane.offset = self._calc_offset(lane, w)
        return lane

    def annotate(self, frame: np.ndarray, lane: LaneLines) -> np.ndarray:
        out = frame.copy()
        overlay = np.zeros_like(frame)
        h = frame.shape[0]

        if lane.left is not None:
            cv2.line(out, (lane.left[0], lane.left[1]), (lane.left[2], lane.left[3]), (0, 255, 0), 4)
        if lane.right is not None:
            cv2.line(out, (lane.right[0], lane.right[1]), (lane.right[2], lane.right[3]), (0, 255, 0), 4)

        # Fill lane polygon
        if lane.left is not None and lane.right is not None:
            pts = np.array([
                [lane.left[0], lane.left[1]],
                [lane.left[2], lane.left[3]],
                [lane.right[2], lane.right[3]],
                [lane.right[0], lane.right[1]],
            ], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 200, 0))
            out = cv2.addWeighted(out, 0.85, overlay, 0.15, 0)

        # Offset info
        if lane.offset is not None:
            side = "LEFT" if lane.offset < 0 else "RIGHT"
            text = f"Offset: {abs(lane.offset):.0f}px {side}"
            cv2.putText(out, text, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
        return out

    @staticmethod
    def _fit_line(pts: List, h: int, top_pct: float) -> Optional[np.ndarray]:
        if len(pts) < 2:
            return None
        xs, ys = zip(*pts)
        try:
            poly = np.polyfit(ys, xs, 1)
        except np.RankWarning:
            return None
        y_bottom = int(h * 0.95)
        y_top = int(h * top_pct)
        x_bottom = int(np.polyval(poly, y_bottom))
        x_top = int(np.polyval(poly, y_top))
        return np.array([x_bottom, y_bottom, x_top, y_top])

    def _smooth(self, lane: LaneLines) -> LaneLines:
        self._history.append(lane)
        if len(self._history) > self._history_len:
            self._history.pop(0)
        valid_left = [h.left for h in self._history if h.left is not None]
        valid_right = [h.right for h in self._history if h.right is not None]
        return LaneLines(
            left=np.mean(valid_left, axis=0).astype(int) if valid_left else None,
            right=np.mean(valid_right, axis=0).astype(int) if valid_right else None,
        )

    @staticmethod
    def _calc_offset(lane: LaneLines, w: int) -> float:
        lane_center = (lane.left[0] + lane.right[0]) / 2
        return lane_center - w / 2
