# Video Super-Resolution

This repository contains the implementation and experimental evaluation for a Video Super-Resolution project. The objective is to reconstruct high-resolution video frames from low-resolution inputs using classical interpolation, single-frame deep learning methods, and a multi-frame video super-resolution model.

The project evaluates six super-resolution methods on two datasets: **Vimeo-90K** and **REDS**. It also includes transfer learning experiments and ablation studies to analyze the influence of temporal context, learning configuration, and pretrained initialization.

---

## Project Goal

Video Super-Resolution aims to recover high-resolution frames from low-resolution video sequences. Unlike single-image super-resolution, video super-resolution can exploit temporal information from neighboring frames.

In this project, we compare both frame-by-frame image super-resolution methods and multi-frame video super-resolution methods.

The target task is **×4 super-resolution**.

---

## Datasets

### Vimeo-90K

Vimeo-90K is used as one of the main training and evaluation datasets. The project uses video septuplets, where each sample contains seven consecutive frames.

For Vimeo-90K, the low-resolution frames are generated dynamically by bicubic downsampling from the high-resolution frames. The target frame is the center frame, `im4.png`.

Current subset used in experiments:

| Split | Number of samples |
|---|---:|
| Train | 6206 |
| Test | 814 |

### REDS

REDS is used as the second dataset for training, evaluation, and transfer learning experiments.

Unlike Vimeo-90K, REDS already provides bicubic low-resolution frames for ×4 super-resolution. The experiments use the high-resolution frames from `train_sharp` / `val_sharp` and the corresponding low-resolution frames from `train_sharp_bicubic/X4` / `val_sharp_bicubic/X4`.

---

## Methods

The project compares six methods:

| Method | Type | Description |
|---|---|---|
| Bicubic interpolation | Classical baseline | Non-learning interpolation baseline |
| SRCNN | Frame-by-frame CNN | Early convolutional neural network for image super-resolution |
| EDSR-lite | Frame-by-frame residual CNN | Lightweight residual super-resolution model |
| VSR-CNN | Multi-frame VSR model | Uses multiple neighboring LR frames to reconstruct the HR center frame |
| FSRCNN | Frame-by-frame CNN | Fast super-resolution CNN operating directly on LR input |
| ESPCN | Frame-by-frame CNN | Efficient sub-pixel convolutional network using PixelShuffle upsampling |

EDVR, BasicVSR, and BasicVSR++ are discussed in the report as related state-of-the-art video super-resolution methods, but the implemented experimental comparison focuses on the six methods above.

---

## Method Details

### Bicubic Interpolation

Bicubic interpolation is used as the classical non-learning baseline. It requires no training and serves as the reference point for all neural models.

### SRCNN

SRCNN is a convolutional neural network applied frame-by-frame. The model receives a bicubic-upscaled low-resolution frame and predicts the high-resolution output.

In this project, SRCNN is used for:

- Training on Vimeo-90K.
- Training directly on REDS.
- Transfer learning from Vimeo-90K to REDS.
- Comparing fine-tuned and non-fine-tuned models.

### EDSR-lite

EDSR-lite is a lightweight residual CNN inspired by EDSR. It removes batch normalization and uses residual blocks for improved reconstruction quality.

Unlike SRCNN, EDSR-lite receives the low-resolution image directly and performs learned upsampling internally.

### VSR-CNN

VSR-CNN is the main multi-frame video super-resolution model implemented in this project. It uses a sequence of neighboring low-resolution frames and reconstructs the high-resolution center frame.

The model supports different temporal input sizes:

| Variant | Input |
|---|---|
| VSR-CNN 1F | 1 frame |
| VSR-CNN 3F | 3 frames |
| VSR-CNN 5F | 5 frames |
| VSR-CNN 7F | 7 frames |

This allows the project to study whether temporal information improves reconstruction quality.

### FSRCNN

FSRCNN is a fast CNN-based super-resolution method. It operates directly on the low-resolution input and performs upsampling at the end of the network.

It is included as an efficient frame-by-frame neural baseline.

### ESPCN

ESPCN is an efficient sub-pixel convolutional network. It processes features in the low-resolution space and uses PixelShuffle for final upsampling.

It is included as a lightweight frame-by-frame method suitable for image and video super-resolution.

---

## Experimental Setup

All neural methods are implemented in PyTorch.

The project evaluates models using:

| Metric | Meaning |
|---|---|
| PSNR | Peak Signal-to-Noise Ratio |
| SSIM | Structural Similarity Index |

Higher values are better for both metrics.

Evaluation is performed at scale factor ×4. For metric computation, border cropping is applied according to the scale factor.

---

## Main Comparison

The main comparison evaluates all six methods on Vimeo-90K and REDS.

| Method | Vimeo PSNR | Vimeo SSIM | REDS PSNR | REDS SSIM |
|---|---:|---:|---:|---:|
| Bicubic | TBD | TBD | 26.5785 | 0.7626 |
| SRCNN | 31.4058 | 0.8783 | TBD | TBD |
| EDSR-lite | TBD | TBD | TBD | TBD |
| VSR-CNN | TBD | TBD | TBD | TBD |
| FSRCNN | TBD | TBD | TBD | TBD |
| ESPCN | TBD | TBD | TBD | TBD |

Current confirmed results:

| Experiment | PSNR | SSIM |
|---|---:|---:|
| Bicubic on REDS | 26.5785 | 0.7626 |
| SRCNN on Vimeo-90K | 31.4058 | 0.8783 |

The remaining values are filled after the corresponding model evaluations are completed.

---

## Transfer Learning

The project includes transfer learning from **Vimeo-90K to REDS**.

The main transfer learning experiment is performed with SRCNN:

| Experiment | Description |
|---|---|
| SRCNN Vimeo → REDS without fine-tuning | Model trained on Vimeo-90K and evaluated directly on REDS |
| SRCNN REDS from scratch | Model trained directly on REDS |
| SRCNN Vimeo → REDS fine-tuned | Model pretrained on Vimeo-90K and then fine-tuned on REDS |

This experiment analyzes whether pretraining on Vimeo-90K improves performance on REDS.

### SRCNN Transfer Learning Results

| Experiment | REDS PSNR | REDS SSIM |
|---|---:|---:|
| SRCNN Vimeo → REDS without fine-tuning | TBD | TBD |
| SRCNN REDS from scratch | TBD | TBD |
| SRCNN Vimeo → REDS fine-tuned | TBD | TBD |

The fine-tuned SRCNN checkpoint was trained using a Vimeo-90K pretrained model and continued training on REDS.

Final fine-tuning losses:

| Model | Final train loss | Final validation loss |
|---|---:|---:|
| SRCNN Vimeo → REDS fine-tuned | 0.002837 | 0.001700 |

The final reported comparison is based on PSNR and SSIM, not training loss.

---

## Ablation Studies

The project includes three ablation studies.

### 1. Temporal Context Ablation

This ablation studies the effect of the number of input frames for VSR-CNN.

| Model | Number of input frames | Vimeo PSNR | Vimeo SSIM |
|---|---:|---:|---:|
| VSR-CNN | 1 | TBD | TBD |
| VSR-CNN | 3 | TBD | TBD |
| VSR-CNN | 5 | TBD | TBD |
| VSR-CNN | 7 | TBD | TBD |

This experiment shows whether using neighboring video frames improves super-resolution performance compared to using only the center frame.

### 2. Learning Rate Ablation

This ablation studies the effect of different learning rates on VSR-CNN training.

| Learning rate | Vimeo PSNR | Vimeo SSIM |
|---:|---:|---:|
| 1e-2 | TBD | TBD |
| 1e-3 | TBD | TBD |
| 1e-4 | TBD | TBD |
| 1e-5 | TBD | TBD |

The goal is to analyze training stability and reconstruction quality under different optimization settings.

### 3. Transfer Learning Ablation

This ablation studies the effect of pretrained initialization when transferring from Vimeo-90K to REDS.

| Setup | Description | REDS PSNR | REDS SSIM |
|---|---|---:|---:|
| From scratch | Model trained directly on REDS | TBD | TBD |
| No fine-tuning | Model trained on Vimeo-90K and evaluated on REDS | TBD | TBD |
| Fine-tuning | Model trained on Vimeo-90K and then fine-tuned on REDS | TBD | TBD |

This experiment measures whether pretraining on one dataset helps adaptation to another dataset.

---

## Expected Analysis

The expected behavior is:

- Bicubic interpolation should provide the lowest neural-free baseline.
- SRCNN should improve over bicubic in most settings.
- EDSR-lite should improve over SRCNN due to residual learning and learned upsampling.
- VSR-CNN should benefit from temporal information when multiple frames are used.
- FSRCNN and ESPCN should provide efficient frame-by-frame alternatives.
- Fine-tuning from Vimeo-90K to REDS should improve REDS performance compared to direct cross-dataset evaluation without fine-tuning.

---

## Results Files

Evaluation outputs are saved as summary files and per-frame metric CSV files.

The main result artifacts are:

| Output type | Description |
|---|---|
| Summary files | Mean PSNR and SSIM for each evaluated model |
| CSV files | Per-frame PSNR and SSIM |
| Example images | Visual comparisons between bicubic, model output, and HR target |
| Checkpoints | Trained model weights |
| TensorBoard logs | Training and validation loss curves |

---

## Project Status

Current implemented methods:

| Method | Implemented | Trained | Evaluated |
|---|---:|---:|---:|
| Bicubic | Yes | Not required | Partially |
| SRCNN | Yes | Yes | Partially |
| EDSR-lite | Yes | In progress | In progress |
| VSR-CNN | Yes | In progress | In progress |
| FSRCNN | Yes | In progress | In progress |
| ESPCN | Yes | In progress | In progress |

Current completed highlights:

- REDS bicubic baseline evaluated.
- SRCNN trained on Vimeo-90K.
- SRCNN trained directly on REDS.
- SRCNN fine-tuned from Vimeo-90K to REDS.
- VSR-CNN, FSRCNN, and ESPCN added as additional comparison methods.

---

## Conclusion

This project builds a complete experimental pipeline for video super-resolution using two datasets and six methods. It combines classical interpolation, frame-by-frame CNN models, and a custom multi-frame VSR-CNN model.

The experiments are designed to evaluate:

- Performance differences between classical and neural methods.
- The benefit of residual and sub-pixel upsampling architectures.
- The importance of temporal context in video super-resolution.
- The effect of transfer learning from Vimeo-90K to REDS.
- The sensitivity of model performance to training choices such as learning rate and number of input frames.

Final conclusions will be drawn after all PSNR and SSIM results are collected.