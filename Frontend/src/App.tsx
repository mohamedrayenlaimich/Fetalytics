import React, { useEffect, useState } from "react";
import {
  AnalysisJobRecord,
  DatasetRecord,
  ModelVersionRecord,
} from "./types/api";
import {
  fetchDatasets,
  fetchJobDetail,
  fetchJobReport,
  fetchJobs,
  fetchModelVersions,
  getJobMediaUrl,
  uploadAndAnalyzeWithProgress,
} from "./services/api";
import "./App.css";

export function App() {
  const [datasets, setDatasets] = useState<DatasetRecord[]>([]);
  const [models, setModels] = useState<ModelVersionRecord[]>([]);
  const [jobs, setJobs] = useState<AnalysisJobRecord[]>([]);
  const [selectedJob, setSelectedJob] = useState<AnalysisJobRecord | null>(null);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [subjectId, setSubjectId] = useState<string>("SUBJ-CLINICAL-001");
  const [pixelSpacingInput, setPixelSpacingInput] = useState<string>("");
  const [frameStep, setFrameStep] = useState<number>(4);

  const [uploadPercent, setUploadPercent] = useState<number>(0);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [errorBanner, setErrorBanner] = useState<string | null>(null);

  const loadOverview = async () => {
    try {
      const [dsList, mvList, jobList] = await Promise.all([
        fetchDatasets(),
        fetchModelVersions(),
        fetchJobs(),
      ]);
      setDatasets(dsList);
      setModels(mvList);
      setJobs(jobList);
      if (jobList.length > 0 && !selectedJob) {
        const detail = await fetchJobDetail(jobList[0].id);
        setSelectedJob(detail);
      }
    } catch (err) {
      setErrorBanner(err instanceof Error ? err.message : "Unable to connect to Fetalytics API.");
    }
  };

  useEffect(() => {
    loadOverview();
  }, []);

  // Poll active job progress when queued or processing
  useEffect(() => {
    if (!selectedJob || !["queued", "processing"].includes(selectedJob.status)) {
      return;
    }
    const timer = window.setInterval(async () => {
      try {
        const updated = await fetchJobDetail(selectedJob.id);
        setSelectedJob(updated);
        if (["completed", "failed"].includes(updated.status)) {
          const jobList = await fetchJobs();
          setJobs(jobList);
        }
      } catch {
        // Ignore transient poll errors
      }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [selectedJob]);

  const handleAnalyzeSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) {
      setErrorBanner("Please select an ultrasound image (.png, .jpg) or video (.avi, .mp4) file.");
      return;
    }
    setErrorBanner(null);
    setIsSubmitting(true);
    setUploadPercent(0);

    const parsedSpacing =
      pixelSpacingInput.trim() !== "" ? parseFloat(pixelSpacingInput) : null;

    try {
      const createdJob = await uploadAndAnalyzeWithProgress(
        selectedFile,
        subjectId.trim() || "SUBJ-CLINICAL-001",
        parsedSpacing,
        frameStep,
        (pct) => setUploadPercent(pct)
      );
      const detail = await fetchJobDetail(createdJob.id);
      setSelectedJob(detail);
      const jobList = await fetchJobs();
      setJobs(jobList);
    } catch (err) {
      setErrorBanner(err instanceof Error ? err.message : "Upload or analysis initiation failed.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleDownloadReport = async (jobId: number) => {
    try {
      const rep = await fetchJobReport(jobId);
      const blob = new Blob([JSON.stringify(rep, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `fetalytics_report_job_${jobId}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setErrorBanner(err instanceof Error ? err.message : "Report download failed.");
    }
  };

  const activeModel = models[0] || null;

  return (
    <div className="fetalytics-app">
      <header className="top-header">
        <div>
          <h1>Fetalytics — Clinical Decision-Support & MLOps Platform</h1>
          <p className="subtitle">
            Traceable Fetal Ultrasound Image & Video Biometry · Backblaze B2 · Neon PostgreSQL
          </p>
        </div>
        <div className="disclaimer-pill">
          ⚠️ RESEARCH & CLINICAL DECISION-SUPPORT ONLY — NOT AN AUTONOMOUS DIAGNOSTIC SYSTEM
        </div>
      </header>

      {errorBanner && (
        <div className="alert-banner error">
          <span>{errorBanner}</span>
          <button onClick={() => setErrorBanner(null)}>Dismiss</button>
        </div>
      )}

      <main className="main-grid">
        {/* Left column: Upload & Job Queue + Model & Dataset Registry */}
        <section className="panel">
          <h2>1. Upload & Start Ultrasound Analysis</h2>
          <form onSubmit={handleAnalyzeSubmit} className="upload-form">
            <label>
              Pseudonymous Subject ID
              <input
                type="text"
                value={subjectId}
                onChange={(e) => setSubjectId(e.target.value)}
                required
              />
            </label>

            <label>
              Ultrasound Image (.png, .jpg) or Video (.avi, .mp4)
              <input
                type="file"
                accept=".png,.jpg,.jpeg,.bmp,.tif,.tiff,.webp,.mp4,.avi,.mov"
                onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
              />
            </label>

            <div className="row-inputs">
              <label>
                Calibration (mm / pixel) <small>(optional)</small>
                <input
                  type="number"
                  step="0.001"
                  min="0.01"
                  placeholder="Leave blank if uncalibrated"
                  value={pixelSpacingInput}
                  onChange={(e) => setPixelSpacingInput(e.target.value)}
                />
              </label>

              <label>
                Video Frame Step
                <input
                  type="number"
                  min="1"
                  max="30"
                  value={frameStep}
                  onChange={(e) => setFrameStep(Number(e.target.value))}
                />
              </label>
            </div>

            {isSubmitting && (
              <div className="progress-box">
                <div className="progress-label">Upload Progress: {uploadPercent}%</div>
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: `${uploadPercent}%` }} />
                </div>
              </div>
            )}

            <button type="submit" className="primary-btn" disabled={isSubmitting}>
              {isSubmitting ? "Uploading & Queuing..." : "Upload & Run AI Analysis"}
            </button>
          </form>

          <h3>Recent Analysis Jobs</h3>
          <div className="job-list">
            {jobs.map((j) => (
              <div
                key={j.id}
                className={`job-item ${selectedJob?.id === j.id ? "active" : ""}`}
                onClick={async () => {
                  const detail = await fetchJobDetail(j.id);
                  setSelectedJob(detail);
                }}
              >
                <div>
                  <strong>Job #{j.id}</strong> — {j.source_filename}
                  <div className="job-meta">
                    {j.job_type} · {j.is_calibrated ? `${j.pixel_spacing_mm} mm/px` : "Uncalibrated (px)"}
                  </div>
                </div>
                <span className={`status-badge ${j.status}`}>
                  {j.status} ({j.progress_percent}%)
                </span>
              </div>
            ))}
          </div>

          <h3>Registered Model Version & Validation Summary</h3>
          {activeModel ? (
            <div className="meta-card">
              <div>
                <strong>Version Tag:</strong> <code>{activeModel.version_tag}</code>
              </div>
              <div>
                <strong>Architecture:</strong> {activeModel.architecture} ({activeModel.task_name})
              </div>
              <div>
                <strong>B2 Artifact Key:</strong>{" "}
                <code>{activeModel.b2_artifact_key || "Local checkpoint verified"}</code>
              </div>
              {activeModel.evaluation_summary?.segmentation && (
                <div className="metrics-mini">
                  <span>
                    FH Dice: {activeModel.evaluation_summary.segmentation.fetal_head_dice_mean} (95% CI:{" "}
                    {activeModel.evaluation_summary.segmentation.fetal_head_dice_ci95.join(" – ")})
                  </span>
                  <span>
                    Baseline Otsu Dice:{" "}
                    {activeModel.evaluation_summary.segmentation.baseline_otsu_fetal_head_dice_mean}
                  </span>
                </div>
              )}
              {activeModel.limitations_notes && (
                <p className="limitations-text">{activeModel.limitations_notes}</p>
              )}
            </div>
          ) : (
            <p className="muted">No trained model version registered yet.</p>
          )}

          <h3>Registered Cloud Datasets (Backblaze B2 → Neon DB)</h3>
          {datasets.map((d) => (
            <div key={d.id} className="meta-card">
              <strong>
                {d.name} (v{d.version})
              </strong>{" "}
              — <span className="status-badge completed">{d.status}</span>
              <div className="job-meta">
                B2: <code>{d.b2_bucket}/{d.b2_prefix}</code> · Total files: {d.total_files}
              </div>
              {d.split_summary && (
                <div className="job-meta">
                  Subject Splits: Train={d.split_summary.train || 0}, Val={d.split_summary.val || 0}, Test=
                  {d.split_summary.test || 0} (0 leakage)
                </div>
              )}
            </div>
          ))}
        </section>

        {/* Right column: Selected Job Results, Visualizations, Measurements & Warnings */}
        <section className="panel">
          <h2>2. Analysis Visualization, Measurements & Traceability Report</h2>
          {!selectedJob ? (
            <p className="muted">Select or submit an ultrasound scan to view analysis results.</p>
          ) : (
            <div className="job-detail">
              <div className="detail-header">
                <div>
                  <h3>
                    Job #{selectedJob.id}: {selectedJob.source_filename}
                  </h3>
                  <div className="job-meta">
                    UUID: <code>{selectedJob.uuid}</code> · SHA-256:{" "}
                    <code>{selectedJob.source_sha256?.slice(0, 16)}…</code>
                  </div>
                </div>
                <button
                  className="secondary-btn"
                  onClick={() => handleDownloadReport(selectedJob.id)}
                  disabled={selectedJob.status !== "completed"}
                >
                  Download JSON Report
                </button>
              </div>

              {/* Analysis Progress Bar */}
              <div className="progress-box">
                <div className="progress-label">
                  Status: <strong>{selectedJob.status.toUpperCase()}</strong> ({selectedJob.progress_percent}%)
                  {selectedJob.processing_time_ms
                    ? ` · Processed in ${selectedJob.processing_time_ms} ms`
                    : ""}
                </div>
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{ width: `${selectedJob.progress_percent}%` }}
                  />
                </div>
              </div>

              {selectedJob.status === "failed" && (
                <div className="alert-banner error">
                  Analysis Failed: {selectedJob.error_message || "Unknown processing error."}
                </div>
              )}

              {selectedJob.result_summary?.overall_confidence !== undefined &&
                selectedJob.result_summary.overall_confidence < 0.55 && (
                  <div className="alert-banner warning">
                    ⚠️ Insufficient AI Confidence (
                    {(selectedJob.result_summary.overall_confidence * 100).toFixed(1)}%): Manual
                    sonographer verification is strictly required before clinical interpretation.
                  </div>
                )}

              {selectedJob.quality_warnings && selectedJob.quality_warnings.length > 0 && (
                <div className="warnings-box">
                  <strong>Quality Control & Calibration Warnings:</strong>
                  <ul>
                    {selectedJob.quality_warnings.map((w, idx) => (
                      <li key={idx}>{w}</li>
                    ))}
                  </ul>
                </div>
              )}

              {selectedJob.status === "completed" && (
                <>
                  <div className="media-comparison">
                    <div className="media-card">
                      <h4>Original Input ({selectedJob.job_type === "video_analysis" ? "Video" : "Image"})</h4>
                      {selectedJob.job_type === "image_analysis" ? (
                        <img
                          src={getJobMediaUrl(selectedJob.id, "original")}
                          alt="Original Ultrasound"
                        />
                      ) : (
                        <video
                          src={getJobMediaUrl(selectedJob.id, "original")}
                          controls
                          muted
                        />
                      )}
                    </div>

                    <div className="media-card">
                      <h4>AI Annotated Output (Contour, Ellipse & Landmarks)</h4>
                      <img
                        src={getJobMediaUrl(selectedJob.id, "keyframe")}
                        alt="AI Annotated Keyframe"
                      />
                    </div>
                  </div>

                  <h3>Reproducible Biometric & Intrapartum Measurements</h3>
                  {selectedJob.measurements && selectedJob.measurements.length > 0 ? (
                    <table className="measurements-table">
                      <thead>
                        <tr>
                          <th>Metric</th>
                          <th>Value</th>
                          <th>Unit</th>
                          <th>Calibrated</th>
                          <th>Uncertainty</th>
                          <th>AI Confidence</th>
                          <th>Method</th>
                        </tr>
                      </thead>
                      <tbody>
                        {selectedJob.measurements.map((m) => (
                          <tr key={m.id}>
                            <td>
                              <strong>{m.measurement_type}</strong>
                            </td>
                            <td>{m.value.toFixed(2)}</td>
                            <td>
                              <code>{m.unit}</code>
                            </td>
                            <td>{m.is_calibrated ? "Yes" : "No (Pixels only)"}</td>
                            <td>
                              ±{m.uncertainty_mm !== null ? `${m.uncertainty_mm} mm` : `${m.uncertainty_px} px`}
                            </td>
                            <td>{(m.confidence_score * 100).toFixed(1)}%</td>
                            <td>
                              <small>{m.method}</small>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : (
                    <p className="muted">
                      No anatomical measurements extracted (structure not detected or insufficient plane quality).
                    </p>
                  )}
                </>
              )}

              <div className="clinical-disclaimer-footer">{selectedJob.disclaimer}</div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
export default App;
