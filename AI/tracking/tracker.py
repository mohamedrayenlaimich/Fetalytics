from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


def compute_bbox_iou(box_a: Optional[Dict[str, int]], box_b: Optional[Dict[str, int]]) -> float:
    if not box_a or not box_b:
        return 0.0
    xa1, ya1 = box_a["x"], box_a["y"]
    xa2, ya2 = xa1 + box_a["w"], ya1 + box_a["h"]
    xb1, yb1 = box_b["x"], box_b["y"]
    xb2, yb2 = xb1 + box_b["w"], yb1 + box_b["h"]

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = float(inter_w * inter_h)
    if inter_area <= 0.0:
        return 0.0

    area_a = float(max(1, box_a["w"] * box_a["h"]))
    area_b = float(max(1, box_b["w"] * box_b["h"]))
    return inter_area / (area_a + area_b - inter_area + 1e-8)


@dataclass
class TrackState:
    track_id: int
    structure_type: str
    last_frame_index: int
    last_bbox: Dict[str, int]
    hits: int = 1
    misses: int = 0
    consecutive_ious: List[float] = field(default_factory=list)


class TemporalStructureTracker:
    """
    Lightweight temporal IoU tracker for anatomical structures (fetal_head, pubic_symphysis)
    across consecutive extracted ultrasound video frames.
    Tracks identity consistency and inter-frame spatial smoothness.
    """

    def __init__(self, iou_threshold: float = 0.25, max_missed_frames: int = 5) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self._next_track_id = 1
        self.active_tracks: Dict[str, TrackState] = {}
        self.identity_switches: int = 0
        self.all_ious: List[float] = []

    def update(
        self,
        frame_index: int,
        structure_type: str,
        bbox: Optional[Dict[str, int]],
    ) -> Optional[int]:
        """Update tracker for a structure in the current frame and return assigned track_id."""
        existing = self.active_tracks.get(structure_type)
        if bbox is None:
            if existing is not None:
                existing.misses += 1
                if existing.misses > self.max_missed_frames:
                    self.active_tracks.pop(structure_type, None)
            return None

        if existing is None:
            tid = self._next_track_id
            self._next_track_id += 1
            self.active_tracks[structure_type] = TrackState(
                track_id=tid,
                structure_type=structure_type,
                last_frame_index=frame_index,
                last_bbox=dict(bbox),
            )
            return tid

        iou = compute_bbox_iou(existing.last_bbox, bbox)
        if iou >= self.iou_threshold:
            existing.last_frame_index = frame_index
            existing.last_bbox = dict(bbox)
            existing.hits += 1
            existing.misses = 0
            existing.consecutive_ious.append(iou)
            self.all_ious.append(iou)
            return existing.track_id
        else:
            # Re-initialize track if object jumped drastically
            self.identity_switches += 1
            tid = self._next_track_id
            self._next_track_id += 1
            self.active_tracks[structure_type] = TrackState(
                track_id=tid,
                structure_type=structure_type,
                last_frame_index=frame_index,
                last_bbox=dict(bbox),
            )
            return tid

    def get_summary_metrics(self, total_evaluated_frames: int) -> Dict[str, float]:
        mean_iou = float(np.mean(self.all_ious)) if self.all_ious else 0.0
        total_hits = sum(t.hits for t in self.active_tracks.values())
        continuity = (
            min(1.0, float(total_hits) / float(max(1, total_evaluated_frames)))
            if total_evaluated_frames > 0
            else 0.0
        )
        return {
            "mean_temporal_iou": round(mean_iou, 4),
            "identity_switches": int(self.identity_switches),
            "track_continuity_ratio": round(continuity, 4),
        }
