import ast
import csv
import hashlib
import json
import logging
import mimetypes
import random
import re
import struct
import zlib
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from sqlalchemy.orm import Session

from Backend.app.core.config import BASE_DIR, settings
from Backend.app.models import (
    Annotation,
    Dataset,
    DatasetFile,
    Examination,
    ExtractedFrame,
    Patient,
    UltrasoundVideo,
)
from Backend.app.services.b2_storage import (
    B2StorageService,
    compute_file_sha256,
    safe_inspect_and_extract_zip,
)

logger = logging.getLogger("fetalytics.ingestion")

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SUPPORTED_VIDEO_EXTS = {".avi", ".mp4", ".mov"}
SUPPORTED_ANNOTATION_EXTS = {".csv", ".json"}


@dataclass
class DatasetInventoryReport:
    dataset_name: str
    dataset_version: str
    b2_bucket: str
    b2_source_key: str
    dry_run: bool
    timestamp: str
    total_files_discovered: int = 0
    videos_valid: int = 0
    images_or_masks_valid: int = 0
    annotations_parsed: int = 0
    extracted_frames_count: int = 0
    corrupt_files: List[str] = field(default_factory=list)
    duplicate_files: List[str] = field(default_factory=list)
    unsupported_files: List[str] = field(default_factory=list)
    unlabelled_files: List[str] = field(default_factory=list)
    supported_tasks: List[str] = field(default_factory=list)
    subject_counts_by_split: Dict[str, int] = field(default_factory=dict)
    file_counts_by_split: Dict[str, int] = field(default_factory=dict)
    subject_leakage_detected: bool = False
    processing_steps: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def extract_subject_id_from_filename(filename: str) -> str:
    """
    Derive a pseudonymous patient/subject identifier from an ultrasound video/image filename.
    Ensures all frames, X-suffixed variants, and multi-clip acquisitions from the same
    patient study session stay strictly in the same split.
    Examples:
      '202101141947512000310I1.avi' -> 'SUBJ-IUGC-20210114194751'
      '202101141947512000310I2.avi' -> 'SUBJ-IUGC-20210114194751'
      '240617091744.avi'            -> 'SUBJ-IUGC-2024061709'
      '240617091744X.avi'           -> 'SUBJ-IUGC-2024061709'
      'test_10_80.avi'              -> 'SUBJ-IUGC-test_10'
      '20190909T155747I1.avi'       -> 'SUBJ-IUGC-20190909T155747'
    """
    stem = Path(filename).stem
    # Strip frame suffix if formatted like 202012132137152159400I1_6
    stem_no_frame = re.sub(r"_\d+(?:_6)?$", "", stem)
    # Match 14-digit YYYYMMDDHHMMSS timestamp prefix shared by patient study clips
    m_ts = re.match(r"^(\d{14})\d*I\d+$", stem_no_frame)
    if m_ts:
        return f"SUBJ-IUGC-{m_ts.group(1)}"
    # Match 12-digit YYMMDDHHMMSS (2024 IUGC cohort) with optional trailing 'X' -> group by session hour 20YYMMDDHH
    m_yy = re.match(r"^(24\d{6})\d{4}X?$", stem_no_frame, re.IGNORECASE)
    if m_yy:
        return f"SUBJ-IUGC-20{m_yy.group(1)}"
    # Match YYYYMMDDTHHMMSS prefix
    m_iso = re.match(r"^(\d{8}T\d{6})(?:I\d+|_\d+)?$", stem_no_frame)
    if m_iso:
        return f"SUBJ-IUGC-{m_iso.group(1)}"
    m = re.match(r"^(test_\d+|val_\d+|train_\d+|patient_\d+|subj_\d+)_\d+$", stem, re.IGNORECASE)
    if m:
        return f"SUBJ-IUGC-{m.group(1)}"
    return f"SUBJ-IUGC-{stem_no_frame.rstrip('Xx')}"


def assign_subject_splits(
    subject_ids: List[str],
    seed: int = 42,
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> Dict[str, str]:
    """
    Deterministically partition unique subject IDs into train, val, and test sets
    so that zero subject leakage occurs across splits.
    """
    unique_subjects = sorted(set(subject_ids))
    rng = random.Random(seed)
    shuffled = list(unique_subjects)
    rng.shuffle(shuffled)

    n = len(shuffled)
    if n == 0:
        return {}
    if n == 1:
        return {shuffled[0]: "train"}
    if n == 2:
        return {shuffled[0]: "train", shuffled[1]: "test"}

    n_train = max(1, int(round(n * ratios[0])))
    n_val = max(1, int(round(n * ratios[1])))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1

    split_map: Dict[str, str] = {}
    for i, sid in enumerate(shuffled):
        if i < n_train:
            split_map[sid] = "train"
        elif i < n_train + n_val:
            split_map[sid] = "val"
        else:
            split_map[sid] = "test"
    return split_map


def verify_no_subject_leakage(subject_to_splits: Dict[str, Set[str]]) -> bool:
    """Return True if any subject appears in more than one split."""
    for _, splits in subject_to_splits.items():
        clean_splits = {s for s in splits if s in ("train", "val", "test")}
        if len(clean_splits) > 1:
            return True
    return False


def _parse_index_list(raw_val: Any) -> List[int]:
    if raw_val is None:
        return []
    s = str(raw_val).strip()
    if not s or s.upper() in ("NULL", "NONE", "NAN"):
        return []
    if s.upper() == "ALL":
        return [-1]  # Marker for all frames
    try:
        if s.startswith("["):
            parsed = ast.literal_eval(s)
            return [int(x) for x in parsed]
        return [int(x.strip()) for x in s.split(",") if x.strip().isdigit()]
    except Exception:
        return []


def compute_mask_gt_biometry(mask_img: np.ndarray) -> Dict[str, float]:
    """
    Compute ground-truth fetal head contour ellipse parameters (HC_px, BPD_px, OFD_px, area_px)
    from a segmentation mask where fetal head pixels equal 255 (or > 200).
    """
    if mask_img is None:
        return {}
    if mask_img.ndim == 3:
        mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask_img

    fh_binary = (mask_gray >= 200).astype(np.uint8) * 255
    ps_binary = ((mask_gray >= 100) & (mask_gray < 200)).astype(np.uint8) * 255

    metrics: Dict[str, float] = {
        "fh_area_px": float(np.count_nonzero(fh_binary)),
        "ps_area_px": float(np.count_nonzero(ps_binary)),
    }

    contours, _ = cv2.findContours(fh_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if len(largest) >= 5:
            (cx, cy), (axis1, axis2), angle = cv2.fitEllipse(largest)
            major_axis = float(max(axis1, axis2))
            minor_axis = float(min(axis1, axis2))
            a = major_axis / 2.0
            b = minor_axis / 2.0
            # Ramanujan's second approximation for ellipse circumference
            h = ((a - b) ** 2) / ((a + b) ** 2 + 1e-8)
            hc_px = float(np.pi * (a + b) * (1.0 + (3.0 * h) / (10.0 + np.sqrt(4.0 - 3.0 * h))))
            metrics.update(
                {
                    "bpd_px": round(minor_axis, 3),
                    "ofd_px": round(major_axis, 3),
                    "hc_px": round(hc_px, 3),
                    "ellipse_cx": round(float(cx), 2),
                    "ellipse_cy": round(float(cy), 2),
                    "ellipse_angle": round(float(angle), 2),
                }
            )
    return metrics


class DatasetIngestionPipeline:
    """
    End-to-end dataset ingestion pipeline connecting Backblaze B2, controlled local extraction,
    and Neon PostgreSQL metadata registration.
    """

    def __init__(
        self,
        db: Optional[Session] = None,
        storage: Optional[B2StorageService] = None,
        temp_dir: Optional[Path] = None,
    ):
        self.db = db
        self.storage = storage or B2StorageService()
        self.temp_dir = Path(temp_dir or settings.TEMP_STORAGE_DIR)
        self.frames_output_dir = BASE_DIR / "AI" / "datasets" / "extracted" / "frames"

    def ensure_sample_archive_from_b2(
        self,
        bucket: str,
        outer_key: str = "datasets/DatasetV3.zip",
        inner_subset: str = "val",
    ) -> Path:
        """
        Ensure the authorized sub-archive (e.g. val.zip inside datasets/DatasetV3.zip)
        is downloaded from B2 to controlled temporary storage.
        """
        target_dir = self.temp_dir / "DatasetV3"
        target_dir.mkdir(parents=True, exist_ok=True)
        local_zip = target_dir / f"{inner_subset}.zip"

        if local_zip.exists() and local_zip.stat().st_size > 1_000_000:
            return local_zip

        # Locate inner zip entry in outer B2 archive central directory via S3 Range request
        s3 = self.storage.client
        head = s3.head_object(Bucket=bucket, Key=outer_key)
        total_size = int(head["ContentLength"])
        tail_size = min(65536, total_size)
        start_byte = total_size - tail_size
        resp = s3.get_object(Bucket=bucket, Key=outer_key, Range=f"bytes={start_byte}-{total_size - 1}")
        tail_bytes = resp["Body"].read()

        eocd_pos = tail_bytes.rfind(b"PK\x05\x06")
        if eocd_pos == -1:
            raise RuntimeError(f"Could not locate ZIP central directory in {outer_key}")

        _, _, _, _, _, cd_size, cd_offset, _ = struct.unpack("<4sHHHHIIH", tail_bytes[eocd_pos : eocd_pos + 22])
        cd_bytes = tail_bytes[cd_offset - start_byte : cd_offset - start_byte + cd_size]

        pos = 0
        target_entry = None
        while pos < len(cd_bytes) and cd_bytes[pos : pos + 4] == b"PK\x01\x02":
            fields = struct.unpack("<4sHHHHHHIIIHHHHHII", cd_bytes[pos : pos + 46])
            comp_size, uncomp_size, fname_len, extra_len, comment_len = (
                fields[8],
                fields[9],
                fields[10],
                fields[11],
                fields[12],
            )
            local_header_offset = fields[16]
            fname = cd_bytes[pos + 46 : pos + 46 + fname_len].decode("utf-8", errors="replace")
            if fname.endswith(".zip") and f"/{inner_subset}-" in fname:
                target_entry = (fname, comp_size, uncomp_size, local_header_offset)
                break
            pos += 46 + fname_len + extra_len + comment_len

        if not target_entry:
            raise RuntimeError(f"Subset '{inner_subset}' not found inside {outer_key}")

        _, comp_size, _, offset = target_entry
        end_byte = offset + 30 + 256 + comp_size
        stream_resp = s3.get_object(Bucket=bucket, Key=outer_key, Range=f"bytes={offset}-{end_byte}")
        body = stream_resp["Body"]
        first_chunk = body.read(65536)
        header = struct.unpack("<4sHHHHHIIIHH", first_chunk[:30])
        fn_l, ex_l = header[9], header[10]
        data_start = 30 + fn_l + ex_l

        decomp = zlib.decompressobj(-zlib.MAX_WBITS)
        remaining = comp_size - (len(first_chunk) - data_start)
        with open(local_zip, "wb") as f_out:
            f_out.write(decomp.decompress(first_chunk[data_start:]))
            while remaining > 0:
                chunk = body.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                f_out.write(decomp.decompress(chunk))
            f_out.write(decomp.flush())

        return local_zip

    def ingest_extracted_directory(
        self,
        extracted_root: Path,
        dataset_name: str = "IUGC2024-Fetal-Ultrasound-V3",
        dataset_version: str = "3.0.0",
        b2_bucket: Optional[str] = None,
        b2_source_key: str = "datasets/DatasetV3.zip",
        dry_run: bool = False,
        max_videos: Optional[int] = None,
        seed: int = 42,
    ) -> DatasetInventoryReport:
        """
        Inspect, validate, parse annotations, extract labeled video frames, assign subject-level
        train/val/test splits without leakage, and register records in PostgreSQL.
        """
        bucket = b2_bucket or settings.B2_BUCKET_NAME or "fetalytics-videos"
        report = DatasetInventoryReport(
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            b2_bucket=bucket,
            b2_source_key=b2_source_key,
            dry_run=dry_run,
            timestamp=datetime.utcnow().isoformat(),
            supported_tasks=[
                "fetal_head_segmentation",
                "standard_plane_classification",
                "fetal_biometry_and_landmarks",
            ],
        )
        report.processing_steps.append(f"Initialized ingestion for {dataset_name} (dry_run={dry_run}).")

        extracted_root = Path(extracted_root).resolve()
        if not extracted_root.exists():
            raise FileNotFoundError(f"Extracted dataset directory does not exist: {extracted_root}")

        # 1. Parse CSV and JSON annotation files across extracted_root
        seg_info_map: Dict[str, Dict[str, Any]] = {}
        plane_info_map: Dict[str, Dict[str, Any]] = {}
        landmarks_map: Dict[str, Dict[str, Any]] = {}

        for meta_file in sorted(extracted_root.rglob("*")):
            if not meta_file.is_file():
                continue
            name_lower = meta_file.name.lower()
            if name_lower == "seg_info.csv":
                with open(meta_file, "r", encoding="utf-8", errors="replace") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        fn = (row.get("filename") or "").strip()
                        if fn:
                            idx_raw = row.get("labeled_frame_index") or row.get("labeled_index")
                            seg_info_map[fn] = {
                                "frame_count": int(row.get("frame_count") or 0),
                                "labeled_indices": _parse_index_list(idx_raw),
                            }
                report.processing_steps.append(f"Parsed segmentation index CSV: {meta_file.name} ({len(seg_info_map)} entries).")

            elif name_lower in ("val_info.csv", "train_info.csv", "test_info.csv", "cls_label.csv", "class_label.csv"):
                with open(meta_file, "r", encoding="utf-8", errors="replace") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        fn = (row.get("filename") or "").strip()
                        if not fn:
                            continue
                        sp_idx = _parse_index_list(row.get("SP_index") or row.get("pos_index"))
                        nsp_idx = _parse_index_list(row.get("NSP_index") or row.get("neg_index"))
                        existing = plane_info_map.get(fn, {})
                        plane_info_map[fn] = {
                            "frame_count": int(row.get("frame_count") or existing.get("frame_count") or 0),
                            "sp_indices": sp_idx or existing.get("sp_indices", []),
                            "nsp_indices": nsp_idx or existing.get("nsp_indices", []),
                        }
                report.processing_steps.append(f"Parsed plane classification CSV: {meta_file.name}.")

            elif name_lower in ("landmarks.json", "landmark.json"):
                data = json.loads(meta_file.read_text(encoding="utf-8", errors="replace"))
                if isinstance(data, dict):
                    landmarks_map.update(data)
                report.processing_steps.append(f"Parsed landmark JSON: {meta_file.name} ({len(landmarks_map)} entries).")

        # 2. Index mask PNGs by stem and filename
        mask_paths_by_name: Dict[str, Path] = {}
        for p in sorted(extracted_root.rglob("*.png")):
            if p.is_file():
                mask_paths_by_name[p.name] = p

        # 3. Discover all files, check duplicates, corruption, unsupported extensions
        seen_hashes: Dict[str, str] = {}
        all_files = [p for p in sorted(extracted_root.rglob("*")) if p.is_file()]
        report.total_files_discovered = len(all_files)

        video_files: List[Path] = []
        for fpath in all_files:
            rel_str = str(fpath.relative_to(extracted_root)).replace("\\", "/")
            ext = fpath.suffix.lower()
            fsize = fpath.stat().st_size

            if fsize == 0:
                report.corrupt_files.append(rel_str)
                continue

            sha = compute_file_sha256(fpath)
            if sha in seen_hashes:
                report.duplicate_files.append(rel_str)
                continue
            seen_hashes[sha] = rel_str

            if ext in SUPPORTED_VIDEO_EXTS:
                video_files.append(fpath)
            elif ext in SUPPORTED_IMAGE_EXTS:
                img = cv2.imread(str(fpath), cv2.IMREAD_UNCHANGED)
                if img is None or img.size == 0:
                    report.corrupt_files.append(rel_str)
                else:
                    report.images_or_masks_valid += 1
            elif ext in SUPPORTED_ANNOTATION_EXTS:
                pass
            else:
                report.unsupported_files.append(rel_str)

        if max_videos is not None:
            video_files = video_files[:max_videos]

        # 4. Subject-level split assignment (70% train / 15% val / 15% test across unique subjects)
        subject_ids = [extract_subject_id_from_filename(vf.name) for vf in video_files]
        subject_split_map = assign_subject_splits(subject_ids, seed=seed)

        subject_to_splits_check: Dict[str, Set[str]] = {}
        for sid, sp in subject_split_map.items():
            subject_to_splits_check.setdefault(sid, set()).add(sp)
            report.subject_counts_by_split[sp] = report.subject_counts_by_split.get(sp, 0) + 1

        report.subject_leakage_detected = verify_no_subject_leakage(subject_to_splits_check)

        # Prepare DB objects if not dry_run
        dataset_obj: Optional[Dataset] = None
        if not dry_run and self.db is not None:
            dataset_obj = self.db.query(Dataset).filter(Dataset.name == dataset_name).first()
            if not dataset_obj:
                dataset_obj = Dataset(
                    name=dataset_name,
                    version=dataset_version,
                    description="IUGC2024 Intrapartum Ultrasound Fetal Head & Pubic Symphysis Dataset",
                    license_info="IUGC2024 Grand Challenge Research License",
                    b2_bucket=bucket,
                    b2_prefix=b2_source_key,
                    status="ingesting",
                    supported_tasks=report.supported_tasks,
                )
                self.db.add(dataset_obj)
                self.db.flush()
            self.frames_output_dir.mkdir(parents=True, exist_ok=True)

        # 5. Process each video, validate OpenCV stream, link annotations, and extract labeled keyframe
        for vf in video_files:
            rel_str = str(vf.relative_to(extracted_root)).replace("\\", "/")
            vname = vf.name
            stem = vf.stem
            subj_id_str = extract_subject_id_from_filename(vname)
            split_name = subject_split_map.get(subj_id_str, "train")

            cap = cv2.VideoCapture(str(vf))
            if not cap.isOpened():
                report.corrupt_files.append(rel_str)
                cap.release()
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_s = (total_frames / fps) if fps > 0 else 0.0

            if total_frames <= 0 or width <= 0 or height <= 0:
                report.corrupt_files.append(rel_str)
                cap.release()
                continue

            report.videos_valid += 1
            report.file_counts_by_split[split_name] = report.file_counts_by_split.get(split_name, 0) + 1

            # Determine labeled segmentation frame index and mask file
            seg_entry = seg_info_map.get(vname, {})
            labeled_indices = seg_entry.get("labeled_indices", [])
            mask_name_candidates = [f"{stem}.png"]
            for l_idx in labeled_indices:
                mask_name_candidates.append(f"{stem}_{l_idx}.png")
                mask_name_candidates.append(f"{stem}_{l_idx}_6.png")

            matched_mask_path: Optional[Path] = None
            for cand in mask_name_candidates:
                if cand in mask_paths_by_name:
                    matched_mask_path = mask_paths_by_name[cand]
                    break

            plane_entry = plane_info_map.get(vname, {})
            has_any_label = bool(matched_mask_path or plane_entry)
            if not has_any_label:
                report.unlabelled_files.append(rel_str)

            target_frame_idx = labeled_indices[0] if labeled_indices else 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame_bgr = cap.read()
                target_frame_idx = 0
            cap.release()

            if not ok or frame_bgr is None:
                report.corrupt_files.append(rel_str)
                continue

            report.extracted_frames_count += 1
            if matched_mask_path:
                report.annotations_parsed += 1

            if not dry_run and self.db is not None and dataset_obj is not None:
                # Upsert Subject / Patient
                subj = self.db.query(Patient).filter(Patient.anonymized_id == subj_id_str).first()
                if not subj:
                    subj = Patient(
                        anonymized_id=subj_id_str,
                        dataset_source=dataset_name,
                        clinical_notes=f"Ingested subject from {dataset_name} ({split_name} split)",
                    )
                    self.db.add(subj)
                    self.db.flush()

                exam_code = f"EXAM-{dataset_name}-{stem}"
                exam = self.db.query(Examination).filter(Examination.examination_code == exam_code).first()
                if not exam:
                    exam = Examination(
                        examination_code=exam_code,
                        patient_id=subj.id,
                        dataset_id=dataset_obj.id,
                        standard_plane="transperineal_sagittal",
                        status="ingested",
                    )
                    self.db.add(exam)
                    self.db.flush()

                v_sha = compute_file_sha256(vf)
                b2_obj_key = f"{b2_source_key}#{rel_str}"
                dfile = (
                    self.db.query(DatasetFile)
                    .filter(DatasetFile.dataset_id == dataset_obj.id, DatasetFile.original_filename == vname)
                    .first()
                )
                if not dfile:
                    dfile = DatasetFile(
                        dataset_id=dataset_obj.id,
                        subject_id=subj.id,
                        examination_id=exam.id,
                        b2_bucket=bucket,
                        b2_object_key=b2_obj_key,
                        relative_path=rel_str,
                        original_filename=vname,
                        media_type="video",
                        mime_type=mimetypes.guess_type(vname)[0] or "video/x-msvideo",
                        file_size_bytes=vf.stat().st_size,
                        sha256_checksum=v_sha,
                        width=width,
                        height=height,
                        split_subset=split_name,
                        is_corrupt=False,
                        is_duplicate=False,
                        has_annotation=has_any_label,
                        status="ingested",
                    )
                    self.db.add(dfile)
                    self.db.flush()

                uvideo = (
                    self.db.query(UltrasoundVideo)
                    .filter(UltrasoundVideo.examination_id == exam.id, UltrasoundVideo.original_filename == vname)
                    .first()
                )
                if not uvideo:
                    uvideo = UltrasoundVideo(
                        examination_id=exam.id,
                        dataset_file_id=dfile.id,
                        b2_bucket=bucket,
                        b2_object_key=b2_obj_key,
                        local_file_path=str(vf),
                        original_filename=vname,
                        mime_type="video/x-msvideo",
                        file_size_bytes=vf.stat().st_size,
                        sha256_checksum=v_sha,
                        fps=fps,
                        duration_seconds=duration_s,
                        total_frames=total_frames,
                        width=width,
                        height=height,
                        upload_status="verified",
                        processing_status="ingested",
                    )
                    self.db.add(uvideo)
                    self.db.flush()

                # Save extracted labeled frame to disk
                frame_filename = f"{stem}_frame_{target_frame_idx}.png"
                frame_disk_path = self.frames_output_dir / frame_filename
                cv2.imwrite(str(frame_disk_path), frame_bgr)
                frame_sha = compute_file_sha256(frame_disk_path)
                ts_ms = (float(target_frame_idx) / fps) * 1000.0 if fps > 0 else 0.0

                sp_list = plane_entry.get("sp_indices", [])
                is_sp = (-1 in sp_list) or (target_frame_idx in sp_list) or bool(matched_mask_path)

                eframe = (
                    self.db.query(ExtractedFrame)
                    .filter(
                        ExtractedFrame.video_id == uvideo.id,
                        ExtractedFrame.frame_index == target_frame_idx,
                    )
                    .first()
                )
                if not eframe:
                    eframe = ExtractedFrame(
                        video_id=uvideo.id,
                        examination_id=exam.id,
                        dataset_file_id=dfile.id,
                        frame_index=target_frame_idx,
                        timestamp_ms=round(ts_ms, 2),
                        width=width,
                        height=height,
                        local_cache_path=str(frame_disk_path),
                        sha256_checksum=frame_sha,
                        is_standard_plane=is_sp,
                        processing_status="extracted",
                    )
                    self.db.add(eframe)
                    self.db.flush()

                # Register Ground-Truth Annotation if mask & landmarks exist
                if matched_mask_path:
                    mask_arr = cv2.imread(str(matched_mask_path), cv2.IMREAD_UNCHANGED)
                    biometry_gt = compute_mask_gt_biometry(mask_arr)
                    lm_info = landmarks_map.get(matched_mask_path.name, {})
                    if lm_info:
                        biometry_gt["hsd_px"] = float(lm_info.get("hsd", 0.0))
                        biometry_gt["aop_deg"] = float(lm_info.get("aop", 0.0))

                    existing_ann = (
                        self.db.query(Annotation)
                        .filter(Annotation.frame_id == eframe.id, Annotation.dataset_id == dataset_obj.id)
                        .first()
                    )
                    if not existing_ann:
                        ann = Annotation(
                            dataset_id=dataset_obj.id,
                            dataset_file_id=dfile.id,
                            frame_id=eframe.id,
                            annotation_type="segmentation_and_landmarks",
                            task_name="fetal_head_segmentation",
                            provenance="IUGC2024_expert_ground_truth",
                            mask_local_path=str(matched_mask_path),
                            label_value="standard_plane" if is_sp else "non_standard_plane",
                            geometry_json={
                                "ps_points_yx": lm_info.get("ps_points"),
                                "hsd_point_yx": lm_info.get("hsd_point"),
                                "aop_tangency_yx": lm_info.get("aop_tangency"),
                                "sp_frame_indices": sp_list[:50],
                            },
                            measurements_json=biometry_gt,
                            is_verified=True,
                        )
                        self.db.add(ann)

        if not dry_run and self.db is not None and dataset_obj is not None:
            dataset_obj.status = "ready"
            dataset_obj.total_files = report.total_files_discovered
            dataset_obj.split_summary = report.subject_counts_by_split
            dataset_obj.inventory_report = asdict(report)
            self.db.commit()
            report.processing_steps.append("Committed dataset, subjects, examinations, frames, and annotations to PostgreSQL.")

        return report
