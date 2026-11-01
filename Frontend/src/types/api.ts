export interface MeasurementRecord {
  id: number;
  uuid: string;
  measurement_type: string;
  value: number;
  unit: string;
  value_mm: number | null;
  value_px: number;
  pixel_spacing_mm: number | null;
  is_calibrated: boolean;
  confidence_score: number;
  uncertainty_px: number;
  uncertainty_mm: number | null;
  estimated_ga_weeks?: number | null;
  method: string;
  model_version_tag: string;
  quality_warnings?: string[];
}

export interface AnalysisJobRecord {
  id: number;
  uuid: string;
  scan_id: number | null;
  examination_id: number | null;
  job_type: "image_analysis" | "video_analysis";
  status: "queued" | "processing" | "completed" | "failed";
  progress_percent: number;
  source_filename: string;
  source_sha256: string;
  pixel_spacing_mm: number | null;
  is_calibrated: boolean;
  processing_config?: Record<string, unknown>;
  quality_warnings: string[];
  result_summary?: {
    head_detected?: boolean;
    ps_detected?: boolean;
    standard_plane_probability?: number;
    is_standard_plane?: boolean;
    frames_evaluated?: number;
    standard_plane_frames_count?: number;
    keyframe_index?: number;
    keyframe_timestamp_ms?: number;
    overall_confidence?: number;
    quality_score?: number;
  };
  error_message?: string | null;
  processing_time_ms?: number | null;
  created_at: string;
  completed_at?: string | null;
  disclaimer: string;
  measurements?: MeasurementRecord[];
}

export interface ModelVersionRecord {
  id: number;
  uuid: string;
  model_name: string;
  version_tag: string;
  task_name: string;
  architecture: string;
  dataset_version: string;
  b2_bucket: string | null;
  b2_artifact_key: string | null;
  artifact_sha256: string | null;
  evaluation_summary?: {
    segmentation?: {
      fetal_head_dice_mean: number;
      fetal_head_dice_ci95: [number, number];
      fetal_head_iou_mean: number;
      pubic_symphysis_dice_mean: number;
      baseline_otsu_fetal_head_dice_mean: number;
    };
    standard_plane_classification?: {
      sensitivity: number;
      specificity: number;
      f1_score: number;
      brier_calibration_score: number;
    };
  };
  limitations_notes?: string | null;
  is_active: boolean;
  created_at: string;
}

export interface DatasetRecord {
  id: number;
  uuid: string;
  name: string;
  version: string;
  status: string;
  b2_bucket: string;
  b2_prefix: string;
  total_files: number;
  supported_tasks: string[];
  split_summary: Record<string, number>;
  created_at: string;
}
