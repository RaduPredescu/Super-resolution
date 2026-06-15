# Video Super-Resolution

This repository contains the implementation and experimental evaluation for a **Video Super-Resolution** project. The goal is to reconstruct high-resolution video frames from low-resolution inputs using classical interpolation, frame-by-frame neural super-resolution models, and a custom multi-frame video super-resolution model.

The project evaluates six methods on two datasets: **Vimeo-90K** and **REDS**. It also includes transfer learning experiments and ablation studies.

---

## Project Goal

Video Super-Resolution aims to recover high-resolution video frames from low-resolution video sequences. Compared to single-image super-resolution, video super-resolution can also use temporal information from neighboring frames.

The target task in this project is **x4 super-resolution**.

The project studies:

* classical interpolation;
* frame-by-frame convolutional super-resolution models;
* residual and sub-pixel upsampling architectures;
* multi-frame video super-resolution;
* transfer learning from Vimeo-90K to REDS;
* ablation studies on temporal context and learning rate.

---

## Datasets

### Vimeo-90K

Vimeo-90K is used as the main video super-resolution dataset. Each sample contains seven consecutive frames.

For Vimeo-90K, the low-resolution inputs are generated dynamically by bicubic downsampling from the high-resolution frames. The target frame is the center frame, `im4.png`.

| Split | Number of samples |
| ----- | ----------------: |
| Train |              6206 |
| Test  |               814 |

### REDS

REDS is used as the second dataset for training, evaluation, and transfer learning.

Unlike Vimeo-90K, REDS already provides bicubic low-resolution frames for x4 super-resolution. The experiments use the high-resolution frames from `train_sharp` / `val_sharp` and the low-resolution frames from `train_sharp_bicubic/X4` / `val_sharp_bicubic/X4`.

---

## Methods

The project compares six super-resolution methods.

| Method    | Type                            | Description                                                          |
| --------- | ------------------------------- | -------------------------------------------------------------------- |
| Bicubic   | Interpolation baseline          | Classical non-learning baseline                                      |
| SRCNN     | Frame-by-frame CNN              | CNN image super-resolution model applied independently on each frame |
| EDSR-lite | Frame-by-frame residual CNN     | Lightweight residual CNN with learned upsampling                     |
| VSR-CNN   | Multi-frame CNN                 | Custom video super-resolution CNN using multiple neighboring frames  |
| FSRCNN    | Frame-by-frame CNN              | Fast CNN super-resolution model operating directly on LR input       |
| ESPCN     | Frame-by-frame PixelShuffle CNN | Efficient sub-pixel convolutional model using PixelShuffle           |

EDVR, BasicVSR, and BasicVSR++ are discussed in the report as related state-of-the-art video super-resolution methods, but the implemented comparison in this repository focuses on the six methods above.

---

## Method Details

### Bicubic

Bicubic interpolation is used as the classical baseline. It requires no training and provides a reference point for the neural models.

### SRCNN

SRCNN is applied frame-by-frame. The low-resolution input is first upscaled with bicubic interpolation, and the CNN predicts the final high-resolution output.

SRCNN is used for the main transfer learning experiment:

* trained on Vimeo-90K;
* evaluated directly on REDS;
* trained from scratch on REDS;
* fine-tuned from Vimeo-90K to REDS.

### EDSR-lite

EDSR-lite is a lightweight residual CNN inspired by EDSR. It removes batch normalization and uses residual blocks to improve reconstruction quality.

Unlike SRCNN, EDSR-lite receives the low-resolution image directly and performs learned upsampling internally.

### VSR-CNN

VSR-CNN is the custom multi-frame video super-resolution model implemented in this project. It uses multiple neighboring low-resolution frames and reconstructs the high-resolution center frame.

The model supports different numbers of input frames:

| Variant    | Input    |
| ---------- | -------- |
| VSR-CNN 1F | 1 frame  |
| VSR-CNN 3F | 3 frames |
| VSR-CNN 5F | 5 frames |
| VSR-CNN 7F | 7 frames |

This allows us to study the effect of temporal context.

### FSRCNN

FSRCNN is a fast CNN-based super-resolution model. It operates directly on the low-resolution input and performs upsampling at the end of the network.

It is included as an efficient frame-by-frame baseline.

### ESPCN

ESPCN is an efficient sub-pixel convolutional network. It processes features in low-resolution space and performs final upsampling using PixelShuffle.

It is included as a lightweight frame-by-frame model suitable for image and video super-resolution.

---

## Evaluation Metrics

The models are evaluated using:

| Metric | Meaning                     |
| ------ | --------------------------- |
| PSNR   | Peak Signal-to-Noise Ratio  |
| SSIM   | Structural Similarity Index |

Higher values are better for both metrics.

All evaluations are performed for **x4 super-resolution**. Border cropping is applied during metric computation according to the scale factor.

---

## Main Results

| Method    | Type                            | Trained by us | Epochs | Learning rate | Batch | Max train samples | Vimeo PSNR ↑ | Vimeo SSIM ↑ | REDS PSNR ↑ | REDS SSIM ↑ |
| --------- | ------------------------------- | ------------- | -----: | ------------: | ----: | ----------------: | -----------: | -----------: | ----------: | ----------: |
| Bicubic   | Interpolation baseline          | No            |      - |             - |     - |                 - |      29.9324 |       0.8544 |     26.5785 |      0.7626 |
| SRCNN     | Frame-by-frame CNN              | Yes - Vimeo   |     20 |          1e-4 |     4 |              6206 |      31.4058 |       0.8783 |         TBD |         TBD |
| EDSR-lite | Frame-by-frame residual CNN     | Yes - Vimeo   |     20 |          1e-4 |     4 |              6206 |      32.3560 |       0.9003 |         TBD |         TBD |
| VSR-CNN   | Multi-frame CNN - 7 frames      | Yes - Vimeo   |     20 |          1e-4 |     4 |              3000 |      30.5926 |       0.8810 |         TBD |         TBD |
| FSRCNN    | Frame-by-frame CNN              | Yes - Vimeo   |     20 |          1e-4 |     2 |              6206 |      30.6435 |       0.8617 |         TBD |         TBD |
| ESPCN     | Frame-by-frame PixelShuffle CNN | Yes - Vimeo   |     20 |          1e-4 |     4 |              6206 |      30.6031 |       0.8663 |         TBD |         TBD |

---

## Confirmed Results

| Experiment                   |  PSNR ↑ | SSIM ↑ |
| ---------------------------- | ------: | -----: |
| Bicubic on REDS              | 26.5785 | 0.7626 |
| SRCNN on Vimeo-90K           | 31.4058 | 0.8783 |
| EDSR-lite on Vimeo-90K       | 32.3560 | 0.9003 |
| VSR-CNN 7-frame on Vimeo-90K | 30.5926 | 0.8810 |
| FSRCNN on Vimeo-90K          | 30.6435 | 0.8617 |

---

## Transfer Learning Results

The project studies transfer learning from **Vimeo-90K to REDS**.

| Model   | Training Setup               | REDS PSNR ↑ | REDS SSIM ↑ |
| ------- | ---------------------------- | ----------: | ----------: |
| SRCNN   | Vimeo → REDS, no fine-tuning |         TBD |         TBD |
| SRCNN   | REDS from scratch            |         TBD |         TBD |
| SRCNN   | Vimeo → REDS fine-tuned      |         TBD |         TBD |
| VSR-CNN | Vimeo → REDS, no fine-tuning |         TBD |         TBD |
| VSR-CNN | REDS from scratch            |         TBD |         TBD |
| VSR-CNN | Vimeo → REDS fine-tuned      |         TBD |         TBD |

The SRCNN fine-tuned model was initialized from a Vimeo-90K checkpoint and then trained further on REDS.

| Model                         | Final train loss | Final validation loss |
| ----------------------------- | ---------------: | --------------------: |
| SRCNN Vimeo → REDS fine-tuned |         0.002837 |              0.001700 |

The final comparison is based on PSNR and SSIM, not on training loss.

---

## Ablation Study 1: Number of Input Frames

This ablation studies how the number of input frames affects VSR-CNN performance.

| Model   | Dataset | Input frames |  PSNR ↑ | SSIM ↑ |
| ------- | ------- | -----------: | ------: | -----: |
| VSR-CNN | Vimeo   |            1 |     TBD |    TBD |
| VSR-CNN | Vimeo   |            3 |     TBD |    TBD |
| VSR-CNN | Vimeo   |            5 |     TBD |    TBD |
| VSR-CNN | Vimeo   |            7 | 30.5926 | 0.8810 |

This experiment measures whether using neighboring video frames improves reconstruction quality compared to using only the center frame.

---

## Ablation Study 2: Learning Rate

This ablation studies the effect of different learning rates on VSR-CNN training.

| Model   | Dataset | Learning rate |  PSNR ↑ | SSIM ↑ |
| ------- | ------- | ------------: | ------: | -----: |
| VSR-CNN | Vimeo   |          1e-3 |     TBD |    TBD |
| VSR-CNN | Vimeo   |          1e-4 | 30.5926 | 0.8810 |
| VSR-CNN | Vimeo   |          1e-5 |     TBD |    TBD |
| VSR-CNN | Vimeo   |          1e-2 |     TBD |    TBD |

The goal is to analyze training stability and reconstruction quality under different optimization settings.

---

## Ablation Study 3: Transfer Learning

This ablation studies the effect of pretrained initialization when transferring from Vimeo-90K to REDS.

| Setup          | Description                                            | REDS PSNR ↑ | REDS SSIM ↑ |
| -------------- | ------------------------------------------------------ | ----------: | ----------: |
| From scratch   | Model trained directly on REDS                         |         TBD |         TBD |
| No fine-tuning | Model trained on Vimeo-90K and evaluated on REDS       |         TBD |         TBD |
| Fine-tuning    | Model trained on Vimeo-90K and then fine-tuned on REDS |         TBD |         TBD |

This experiment measures whether pretraining on one dataset helps the model adapt to another dataset.

---

## Expected Observations

The expected behavior is:

* Bicubic interpolation provides the classical non-learning baseline.
* Neural models should generally improve over bicubic interpolation.
* EDSR-lite is expected to outperform SRCNN due to residual learning and learned upsampling.
* VSR-CNN is expected to benefit from neighboring frames when temporal information is useful.
* FSRCNN and ESPCN provide efficient frame-by-frame alternatives.
* Fine-tuning from Vimeo-90K to REDS is expected to improve REDS performance compared to direct cross-dataset evaluation without fine-tuning.

---

## Result Artifacts

The experimental pipeline produces:

| Artifact         | Description                                                    |
| ---------------- | -------------------------------------------------------------- |
| Summary files    | Mean PSNR and SSIM for each evaluated model                    |
| CSV files        | Per-frame PSNR and SSIM                                        |
| Example images   | Visual comparison between bicubic, model output, and HR target |
| Checkpoints      | Trained model weights                                          |
| TensorBoard logs | Training and validation loss curves                            |

---

## Project Status

| Method    | Implemented |      Trained |   Evaluated |
| --------- | ----------: | -----------: | ----------: |
| Bicubic   |         Yes | Not required |   Yes       |
| SRCNN     |         Yes |          Yes |   Yes       |
| EDSR-lite |         Yes |          Yes |   Yes       |
| VSR-CNN   |         Yes |          Yes |   Yes       |
| FSRCNN    |         Yes |          Yes |   Yes       |
| ESPCN     |         Yes |          Yes |   Yes       |

Current completed highlights:

* REDS bicubic baseline evaluated.
* SRCNN trained on Vimeo-90K.
* SRCNN trained directly on REDS.
* SRCNN fine-tuned from Vimeo-90K to REDS.
* EDSR-lite trained and evaluated on Vimeo-90K.
* VSR-CNN trained and evaluated on Vimeo-90K with 7 input frames.
* FSRCNN trained and evaluated on Vimeo-90K.
* ESPCN added as the sixth method.

---

## Conclusion

This project builds a complete experimental pipeline for video super-resolution using two datasets and six methods. It combines classical interpolation, frame-by-frame CNN models, residual CNN models, sub-pixel convolutional models, and a custom multi-frame VSR-CNN model.

The experiments are designed to evaluate:

* the difference between classical and neural super-resolution;
* the effect of residual learning and learned upsampling;
* the importance of temporal context in video super-resolution;
* the impact of transfer learning from Vimeo-90K to REDS;
* the sensitivity of VSR-CNN to learning rate and number of input frames.

Final conclusions will be drawn after all PSNR and SSIM results are collected.
