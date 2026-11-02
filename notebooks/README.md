# ☁️ Fetalytics Free Cloud GPU Training Guide

This guide explains how to train the **Fetalytics FetalHeadUNet-v2.0** model completely in the cloud for **free** on high-performance GPUs, without straining your laptop CPU or battery.

---

## 🚀 Option 1: Google Colab (Recommended — 100% Free)

Google Colab provides a free **Nvidia T4 GPU (16 GB VRAM)** that trains the 340 ultrasound cases in **~2 to 3 minutes** (compared to 45–60+ minutes on a laptop CPU).

### Steps:
1. Open [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and select [`Fetalytics_Cloud_Training_Colab.ipynb`](file:///c:/Users/MSI/OneDrive/Documents/Fetalytics/notebooks/Fetalytics_Cloud_Training_Colab.ipynb) (or open directly from your GitHub repository).
3. In Colab, enable the GPU:
   - Go to **Runtime** > **Change runtime type**.
   - Under *Hardware accelerator*, select **T4 GPU** (Free tier).
   - Click **Save**.
4. Check your Backblaze B2 & Neon database credentials in **Step 3** of the notebook.
5. Click **Runtime** > **Run all** (or `Ctrl + F9`).

### What Happens Automatically:
- Fast download of `val.zip` and `test.zip` directly from your Backblaze B2 bucket (`fetalytics-videos`) into Colab's fast NVMe scratch disk.
- Extracts and verifies 340 ultrasound video and mask pairs.
- Partitions into 100 training, 21 validation, and 22 held-out test subjects (zero patient data leakage).
- Trains `FetalHeadUNet-v2.0` on the Nvidia T4 GPU in ~2 minutes with Cosine Annealing and balanced multi-class Dice loss.
- Evaluates strictly on the held-out test split (computes Dice, IoU, 95% Bootstrap CI, Sensitivity, Specificity).
- Uploads the trained `.pt` checkpoint to Backblaze B2 (`fetalytics-videos/models/...`).
- Registers the `TrainingRun` and `ModelVersion` in your Neon PostgreSQL database.
- Renders the diagnostic overlay comparison sheets directly in your browser.

---

## ⚡ Option 2: Kaggle Notebooks (30 Free GPU Hours / Week)

If you prefer Kaggle:
1. Go to [kaggle.com/code](https://www.kaggle.com/code) and click **New Notebook**.
2. In the right sidebar under **Settings**:
   - Turn **Internet** ON.
   - Set **Accelerator** to **GPU T4 x2** or **GPU P100**.
3. Upload `notebooks/Fetalytics_Cloud_Training_Colab.ipynb` (File > Import Notebook).
4. Run all cells.

---

## 💻 Standalone Cloud Script: `AI/training/train_cloud.py`

If you are using an SSH terminal or custom cloud instance (e.g., Paperspace, Lightning AI, AWS/GCP instance):

```bash
python AI/training/train_cloud.py --epochs 12 --batch-size 16 --lr 0.003 --seed 42 --patience 4
```

