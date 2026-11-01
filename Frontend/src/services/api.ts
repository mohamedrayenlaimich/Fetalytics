import {
  AnalysisJobRecord,
  DatasetRecord,
  MeasurementRecord,
  ModelVersionRecord,
} from "../types/api";

const API_BASE =
  (import.meta as unknown as { env?: { VITE_API_URL?: string } }).env?.VITE_API_URL ||
  "http://127.0.0.1:8000/api/v1";

const DEFAULT_HEADERS: Record<string, string> = {
  "X-API-Key": "fetalytics-dev-clinical-key-2026",
};

export async function fetchDatasets(): Promise<DatasetRecord[]> {
  const res = await fetch(`${API_BASE}/datasets/`, { headers: DEFAULT_HEADERS });
  if (!res.ok) throw new Error(`Failed to load datasets (${res.status})`);
  return res.json();
}

export async function fetchModelVersions(): Promise<ModelVersionRecord[]> {
  const res = await fetch(`${API_BASE}/models/versions`, { headers: DEFAULT_HEADERS });
  if (!res.ok) throw new Error(`Failed to load model versions (${res.status})`);
  return res.json();
}

export async function fetchJobs(): Promise<AnalysisJobRecord[]> {
  const res = await fetch(`${API_BASE}/jobs/`, { headers: DEFAULT_HEADERS });
  if (!res.ok) throw new Error(`Failed to load analysis jobs (${res.status})`);
  return res.json();
}

export async function fetchJobDetail(jobId: number): Promise<AnalysisJobRecord> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}`, { headers: DEFAULT_HEADERS });
  if (!res.ok) throw new Error(`Failed to fetch job #${jobId} (${res.status})`);
  return res.json();
}

export async function fetchJobMeasurements(jobId: number): Promise<MeasurementRecord[]> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}/measurements`, {
    headers: DEFAULT_HEADERS,
  });
  if (!res.ok) throw new Error(`Failed to fetch measurements for job #${jobId}`);
  return res.json();
}

export async function fetchJobReport(jobId: number): Promise<Record<string, unknown>> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}/report`, { headers: DEFAULT_HEADERS });
  if (!res.ok) throw new Error(`Failed to fetch report for job #${jobId}`);
  return res.json();
}

export function getJobMediaUrl(
  jobId: number,
  kind: "original" | "annotated" | "keyframe"
): string {
  return `${API_BASE}/jobs/${jobId}/media/${kind}`;
}

export function uploadAndAnalyzeWithProgress(
  file: File,
  patientAnonymizedId: string,
  pixelSpacingMm: number | null,
  frameStep: number,
  onUploadProgress: (percent: number) => void
): Promise<AnalysisJobRecord> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/jobs/analyze`);
    xhr.setRequestHeader("X-API-Key", DEFAULT_HEADERS["X-API-Key"]);

    xhr.upload.onprogress = (ev) => {
      if (ev.lengthComputable) {
        onUploadProgress(Math.round((ev.loaded / ev.total) * 100));
      }
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch (err) {
          reject(err);
        }
      } else {
        reject(new Error(xhr.responseText || `Upload failed with status ${xhr.status}`));
      }
    };

    xhr.onerror = () => reject(new Error("Network error during upload."));

    const formData = new FormData();
    formData.append("file", file);
    formData.append("patient_anonymized_id", patientAnonymizedId);
    formData.append("frame_step", String(frameStep));
    if (pixelSpacingMm !== null && !Number.isNaN(pixelSpacingMm) && pixelSpacingMm > 0) {
      formData.append("pixel_spacing_mm", String(pixelSpacingMm));
    }

    xhr.send(formData);
  });
}
