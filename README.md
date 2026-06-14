# Video Super-Resolution

Project for the **Artificial Intelligence II: Deep Learning Methods** course.  
Topic: **#23 Video Super-Resolution**

## 1. Project Description

The goal of this project is to study and compare deep learning methods for **Video Super-Resolution (VSR)**.

Video Super-Resolution aims to reconstruct high-resolution video frames from low-resolution input frames. Unlike single-image super-resolution, VSR can exploit temporal information from neighboring frames in order to recover more details and improve reconstruction quality.

In this project, we focus on **x4 super-resolution**, where the input video frames are downsampled by a factor of 4 and the models must reconstruct the corresponding high-resolution frames.

## 2. Project Requirements

According to the project requirements for topic #23, the project must include:

- Experiments on at least **2 datasets**
- Comparison of at least **6 methods**
- At least **3 methods trained by us**, either from scratch or using transfer learning
- No direct usage of reported results from papers; all comparisons must be based on our own experiments
- A study of the effect of **transfer learning from one dataset to another**
- At least **3 ablation studies**
- Implementation in Python, with **PyTorch mandatory**
- A final report written in LaTeX, maximum 6 pages
- A short final presentation based on the report

## 3. Selected Datasets

We selected the following datasets:

### 3.1 Vimeo-90K Septuplet

Vimeo-90K is a widely used dataset for video restoration tasks, including video super-resolution.  
We use the **septuplet** version, where each sample contains 7 consecutive video frames.

Expected usage:

- High-resolution frames are used as ground truth
- Low-resolution frames are generated through x4 downsampling
- Used mainly for training and validation

### 3.2 REDS

REDS is a video restoration dataset used for tasks such as video super-resolution and deblurring.

Expected usage:

- `train_sharp` / `val_sharp` are used as high-resolution ground truth frames
- `train_sharp_bicubic/X4` / `val_sharp_bicubic/X4` are used as low-resolution inputs
- Used for training, validation, and transfer learning experiments

## 4. Task Definition

Given a sequence of low-resolution frames:

```text
LR[t-k], ..., LR[t-1], LR[t], LR[t+1], ..., LR[t+k]
```

the model must reconstruct the high-resolution center frame:

```text
HR[t]
```

The main setting is:

```text
Scale factor: x4
Input: multiple low-resolution neighboring frames
Output: one reconstructed high-resolution frame
```

For frame-by-frame baselines, the model receives only one low-resolution frame and reconstructs the corresponding high-resolution frame.

## 5. Methods to Compare

We plan to compare at least 6 methods:

| Method | Type | Trained by us |
|---|---|---|
| Bicubic interpolation | Classical baseline | No |
| SRCNN frame-by-frame | CNN image super-resolution baseline | Yes |
| EDSR-lite frame-by-frame | Residual CNN image super-resolution | Yes |
| Simple VSR-CNN | Multi-frame video super-resolution model | Yes |
| EDVR | Advanced video restoration / VSR model | Fine-tune or inference |
| BasicVSR / BasicVSR++ | Advanced recurrent video super-resolution model | Fine-tune or inference |

The first baseline, bicubic interpolation, provides a simple reference point.  
SRCNN and EDSR-lite process each frame independently, while the VSR-CNN, EDVR, and BasicVSR-based approaches use temporal information from multiple frames.

## 6. Evaluation Metrics

The following metrics will be used:

- **PSNR** — Peak Signal-to-Noise Ratio
- **SSIM** — Structural Similarity Index
- Optional: **LPIPS** — perceptual similarity metric

The main quantitative comparison will be based on PSNR and SSIM.  
Visual examples will also be included in the report to compare qualitative reconstruction results.

## 7. Planned Ablation Studies

We plan to include at least 3 ablation studies:

### 7.1 Number of Input Frames

We compare models using different temporal windows:

- 1 frame
- 3 frames
- 5 frames
- 7 frames, if computationally feasible

This shows how much temporal information helps video super-resolution.

### 7.2 Loss Function

We compare different loss functions:

- L1 loss
- MSE loss
- L1 loss + perceptual loss

This helps analyze the influence of the training objective on reconstruction quality.

### 7.3 Transfer Learning Between Datasets

We study transfer learning in both directions:

- Train on Vimeo-90K, evaluate or fine-tune on REDS
- Train on REDS, evaluate or fine-tune on Vimeo-90K

This is required by the project statement and helps evaluate how well the models generalize across datasets.

## 8. Repository Structure

Proposed project structure:

```text
video-super-resolution/
├── data/
│   ├── vimeo90k/
│   └── REDS/
├── src/
│   ├── datasets.py
│   ├── metrics.py
│   ├── train_srcnn.py
│   ├── train_edsr_lite.py
│   ├── train_simple_vsr.py
│   ├── evaluate.py
│   ├── bicubic_baseline.py
│   ├── compare_results.py
│   └── models/
│       ├── srcnn.py
│       ├── edsr_lite.py
│       └── simple_vsr.py
├── configs/
│   ├── srcnn.yaml
│   ├── edsr_lite.yaml
│   └── simple_vsr.yaml
├── results/
│   ├── images/
│   ├── tables/
│   └── checkpoints/
├── report/
├── requirements.txt
└── README.md
```

## 9. Dataset Preparation

### 9.1 Vimeo-90K

Expected structure:

```text
data/
└── vimeo90k/
    ├── GT/
    ├── sep_trainlist.txt
    └── sep_testlist.txt
```

The `GT` folder should contain the original high-resolution frame sequences.  
Low-resolution frames can be generated through x4 downsampling.

### 9.2 REDS

Expected structure:

```text
data/
└── REDS/
    ├── train_sharp/
    ├── train_sharp_bicubic/
    │   └── X4/
    ├── val_sharp/
    └── val_sharp_bicubic/
        └── X4/
```

## 10. First Implementation Milestone

The first milestone is to implement the full evaluation pipeline using bicubic interpolation:

1. Load HR video frames
2. Generate or load LR x4 frames
3. Upscale LR frames using bicubic interpolation
4. Compare the output with HR ground truth
5. Compute PSNR and SSIM
6. Save visual examples

This baseline will be used as the reference for all deep learning methods.

## 11. Expected Results

We expect deep learning methods to outperform bicubic interpolation in terms of PSNR and SSIM.

Frame-by-frame methods such as SRCNN and EDSR-lite should improve spatial detail reconstruction, while video-based methods such as Simple VSR-CNN, EDVR, and BasicVSR should further improve results by using temporal information from neighboring frames.

## 12. Technologies

Main technologies:

- Python
- PyTorch
- NumPy
- OpenCV
- Pillow
- scikit-image
- Matplotlib
- tqdm

## 13. Authors

Project developed for the Artificial Intelligence II course, Mihai Modi and Predescu Radu

National University of Science and Technology POLITEHNICA Bucharest.