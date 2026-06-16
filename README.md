# Video Super-Resolution

This repository contains the implementation and experimental evaluation for a **Video Super-Resolution** project. The goal is to reconstruct high-resolution video frames from low-resolution inputs using classical interpolation, frame-by-frame deep learning models, and a custom multi-frame video super-resolution model.

The project studies **x4 video super-resolution** on two datasets: **Vimeo-90K** and **REDS**. The evaluation compares six methods: **Bicubic**, **SRCNN**, **EDSR-lite**, **VSR-CNN**, **FSRCNN**, and **ESPCN**.

---

## Project Overview

Video Super-Resolution aims to recover high-resolution frames from low-resolution video sequences. Compared to single-image super-resolution, video super-resolution can also exploit temporal information from neighboring frames.

This project focuses on:

* classical interpolation;
* frame-by-frame CNN-based super-resolution;
* residual CNN-based super-resolution;
* sub-pixel upsampling with PixelShuffle;
* multi-frame video super-resolution;
* transfer learning from Vimeo-90K to REDS;
* ablation studies on input frames, learning rate, and loss function.

All neural models are implemented in **PyTorch**.

---

## Datasets

### Vimeo-90K

Vimeo-90K is used as the main video super-resolution dataset. Each sample contains seven consecutive high-resolution frames.

For this project, low-resolution frames are generated dynamically using bicubic downsampling with scale factor **x4**. The target frame is the center frame, `im4.png`.

| Split | Number of samples |
| ----- | ----------------: |
| Train |              6206 |
| Test  |               814 |

### REDS

REDS is used as the second dataset for evaluation and transfer learning.

Unlike Vimeo-90K, REDS provides both high-resolution frames and bicubic low-resolution frames for **x4 super-resolution**.

The project uses:

| Split      | HR folder     | LR folder                |
| ---------- | ------------- | ------------------------ |
| Train      | `train_sharp` | `train_sharp_bicubic/X4` |
| Validation | `val_sharp`   | `val_sharp_bicubic/X4`   |

---

## Compared Methods

| Method    | Type             | Description                                                           |
| --------- | ---------------- | --------------------------------------------------------------------- |
| Bicubic   | Interpolation    | Classical non-learning baseline                                       |
| SRCNN     | Frame CNN        | CNN applied frame-by-frame after bicubic upsampling                   |
| EDSR-lite | Residual CNN     | Lightweight residual CNN with learned upsampling                      |
| VSR-CNN   | Multi-frame CNN  | Custom video super-resolution model using multiple neighboring frames |
| FSRCNN    | Frame CNN        | Fast CNN super-resolution model operating directly on LR input        |
| ESPCN     | PixelShuffle CNN | Efficient sub-pixel convolutional network using PixelShuffle          |

Advanced video super-resolution methods such as EDVR, BasicVSR, BasicVSR++, and VRT are discussed as related work, but the experimental comparison focuses on lightweight methods that can be trained and evaluated with the available computational resources.

---

## Evaluation Metrics

The models are evaluated using **PSNR** and **SSIM**.

| Metric | Meaning                     |
| ------ | --------------------------- |
| PSNR   | Peak Signal-to-Noise Ratio  |
| SSIM   | Structural Similarity Index |

Higher values indicate better reconstruction quality.

All experiments are performed for **x4 super-resolution**. During metric computation, border cropping equal to the scale factor is applied.

---

## Main Results

The following table compares all evaluated methods on Vimeo-90K and REDS.

| Model     | Type             | Trained by us | Epochs | Learning rate | Vimeo PSNR ↑ | Vimeo SSIM ↑ | REDS PSNR ↑ | REDS SSIM ↑ |
| --------- | ---------------- | ------------- | -----: | ------------: | -----------: | -----------: | ----------: | ----------: |
| Bicubic   | Interpolation    | No            |    N/A |           N/A |      29.9324 |       0.8544 |     26.5785 |      0.7626 |
| SRCNN     | Frame CNN        | Yes           |     20 |          1e-4 |      31.4058 |       0.8783 |     27.6228 |      0.7972 |
| EDSR-lite | Residual CNN     | Yes           |     20 |          1e-4 |      32.3560 |       0.9003 |     28.3731 |      0.8242 |
| VSR-CNN   | Multi-frame CNN  | Yes           |     20 |          1e-4 |      30.5926 |       0.8810 |     28.1483 |      0.8149 |
| FSRCNN    | Frame CNN        | Yes           |     20 |          1e-4 |      30.6435 |       0.8617 |     27.1607 |      0.7802 |
| ESPCN     | PixelShuffle CNN | Yes           |     20 |          1e-4 |      30.6031 |       0.8663 |     27.1563 |      0.7839 |

---

## Main Result Analysis

On **Vimeo-90K**, the best result is obtained by **EDSR-lite**, with:

| Dataset   | Best model |  PSNR ↑ | SSIM ↑ |
| --------- | ---------- | ------: | -----: |
| Vimeo-90K | EDSR-lite  | 32.3560 | 0.9003 |

On **REDS**, the best result in the main comparison is also obtained by **EDSR-lite**, with:

| Dataset | Best model |  PSNR ↑ | SSIM ↑ |
| ------- | ---------- | ------: | -----: |
| REDS    | EDSR-lite  | 28.3731 | 0.8242 |

The results show that all trained neural models improve over the bicubic baseline on REDS. EDSR-lite performs best overall because residual learning and learned upsampling are effective for super-resolution.

---

## Transfer Learning Results

Transfer learning is studied from **Vimeo-90K to REDS**. The goal is to analyze whether a model trained on Vimeo-90K generalizes to REDS and whether fine-tuning on REDS improves the final performance.

| Model   | Training setup                            | REDS PSNR ↑ | REDS SSIM ↑ |
| ------- | ----------------------------------------- | ----------: | ----------: |
| SRCNN   | Train Vimeo, eval REDS                    |     27.6693 |      0.7991 |
| SRCNN   | Train REDS, eval REDS                     |     27.6228 |      0.7972 |
| SRCNN   | Train Vimeo, fine-tune on REDS, eval REDS |     27.8155 |      0.8032 |
| VSR-CNN | Train Vimeo, eval REDS                    |     28.2629 |      0.8223 |
| VSR-CNN | Train REDS, eval REDS                     |     28.1483 |      0.8149 |
| VSR-CNN | Train Vimeo, fine-tune on REDS, eval REDS |     28.5620 |      0.8299 |

---

## Transfer Learning Analysis

For **SRCNN**, fine-tuning improves performance from:

| Setup                        | REDS PSNR ↑ | REDS SSIM ↑ |
| ---------------------------- | ----------: | ----------: |
| Vimeo → REDS, no fine-tuning |     27.6693 |      0.7991 |
| Vimeo → REDS, fine-tuned     |     27.8155 |      0.8032 |

For **VSR-CNN**, fine-tuning gives the best transfer learning result:

| Setup                        | REDS PSNR ↑ | REDS SSIM ↑ |
| ---------------------------- | ----------: | ----------: |
| Vimeo → REDS, no fine-tuning |     28.2629 |      0.8223 |
| Vimeo → REDS, fine-tuned     |     28.5620 |      0.8299 |

The best transfer learning result is obtained by **VSR-CNN fine-tuned on REDS**, reaching **28.5620 PSNR** and **0.8299 SSIM**.

This shows that pretraining on Vimeo-90K provides useful initialization, while fine-tuning on REDS helps the model adapt to the target dataset.

---

## Ablation Study 1: Number of Input Frames

This ablation studies how the number of input frames affects VSR-CNN performance on Vimeo-90K.

| Model   | Input frames |  PSNR ↑ | SSIM ↑ |
| ------- | -----------: | ------: | -----: |
| VSR-CNN |            1 | 30.7825 | 0.8878 |
| VSR-CNN |            3 | 31.4055 | 0.8846 |
| VSR-CNN |            5 | 31.2995 | 0.8826 |
| VSR-CNN |            7 | 30.5926 | 0.8810 |

### Analysis

The best PSNR is obtained with **3 input frames**, reaching **31.4055 PSNR**.

The best SSIM is obtained with **1 input frame**, reaching **0.8878 SSIM**.

The 7-frame model performs worse than the 3-frame and 5-frame variants. This suggests that simply adding more temporal information does not automatically improve performance. Without explicit motion alignment, additional neighboring frames can make training harder and may introduce temporal inconsistencies.

---

## Ablation Study 2: Learning Rate

This ablation studies the effect of the learning rate on VSR-CNN training on Vimeo-90K.

| Model   | Learning rate |  PSNR ↑ | SSIM ↑ |
| ------- | ------------: | ------: | -----: |
| VSR-CNN |          1e-3 | 31.4212 | 0.8891 |
| VSR-CNN |          1e-4 | 30.5926 | 0.8810 |
| VSR-CNN |          1e-5 | 28.5765 | 0.8209 |
| VSR-CNN |          1e-2 |  3.4688 | 0.3003 |

### Analysis

The best learning rate is **1e-3**, which obtains:

| Learning rate |  PSNR ↑ | SSIM ↑ |
| ------------: | ------: | -----: |
|          1e-3 | 31.4212 | 0.8891 |

A very small learning rate, **1e-5**, leads to weaker convergence. A very large learning rate, **1e-2**, causes unstable training and produces very poor reconstruction quality.

---

## Ablation Study 3: Loss Function

This ablation studies the effect of different loss functions on VSR-CNN performance on Vimeo-90K.

| Model   | Loss function    |  PSNR ↑ | SSIM ↑ |
| ------- | ---------------- | ------: | -----: |
| VSR-CNN | L1 loss          | 30.5926 | 0.8810 |
| VSR-CNN | MSE loss         | 31.5703 | 0.8822 |
| VSR-CNN | Smooth L1 loss   | 31.2018 | 0.8802 |
| VSR-CNN | Charbonnier loss | 30.3973 | 0.8822 |

### Analysis

The best PSNR is obtained with **MSE loss**, reaching **31.5703 PSNR**.

The best SSIM is shared by **MSE loss** and **Charbonnier loss**, both reaching **0.8822 SSIM**.

Although L1 loss is commonly used in image restoration, in this experiment MSE loss produced the strongest PSNR result for VSR-CNN. Smooth L1 also improves over L1 in PSNR, while Charbonnier obtains a similar SSIM but lower PSNR.

---

## Best Results Summary

| Experiment                   | Best model / setting       |  PSNR ↑ | SSIM ↑ |
| ---------------------------- | -------------------------- | ------: | -----: |
| Main comparison on Vimeo-90K | EDSR-lite                  | 32.3560 | 0.9003 |
| Main comparison on REDS      | EDSR-lite                  | 28.3731 | 0.8242 |
| Transfer learning on REDS    | VSR-CNN fine-tuned on REDS | 28.5620 | 0.8299 |
| Input frames ablation        | VSR-CNN, 3 frames          | 31.4055 | 0.8846 |
| Learning rate ablation       | VSR-CNN, 1e-3              | 31.4212 | 0.8891 |
| Loss function ablation       | VSR-CNN, MSE loss          | 31.5703 | 0.8822 |

---

## Result Artifacts

The experimental pipeline produces the following outputs:

| Artifact         | Description                                                     |
| ---------------- | --------------------------------------------------------------- |
| Checkpoints      | Trained model weights                                           |
| Summary files    | Mean PSNR and SSIM for each evaluated model                     |
| CSV files        | Per-frame PSNR and SSIM values                                  |
| Example images   | Visual comparisons between LR/Bicubic, SR output, and HR target |
| TensorBoard logs | Training and validation loss curves                             |

---

## Project Status

| Method    | Implemented | Trained | Evaluated on Vimeo | Evaluated on REDS |
| --------- | ----------: | ------: | -----------------: | ----------------: |
| Bicubic   |         Yes |     N/A |                Yes |               Yes |
| SRCNN     |         Yes |     Yes |                Yes |               Yes |
| EDSR-lite |         Yes |     Yes |                Yes |               Yes |
| VSR-CNN   |         Yes |     Yes |                Yes |               Yes |
| FSRCNN    |         Yes |     Yes |                Yes |               Yes |
| ESPCN     |         Yes |     Yes |                Yes |               Yes |

Additional completed experiments:

* transfer learning from Vimeo-90K to REDS;
* VSR-CNN ablation on number of input frames;
* VSR-CNN ablation on learning rate;
* VSR-CNN ablation on loss function.

---

## Conclusion

This project implemented and evaluated six methods for **x4 video super-resolution** on Vimeo-90K and REDS.

The compared methods include a classical bicubic baseline, frame-by-frame CNN models, residual super-resolution models, sub-pixel convolutional models, and a custom multi-frame VSR-CNN model.

The results show that learned methods improve over bicubic interpolation on both datasets. **EDSR-lite** achieves the best result in the main comparison, reaching **32.3560 PSNR / 0.9003 SSIM** on Vimeo-90K and **28.3731 PSNR / 0.8242 SSIM** on REDS.

The transfer learning experiments show that fine-tuning improves performance on REDS. The best transfer result is achieved by **VSR-CNN fine-tuned on REDS**, with **28.5620 PSNR / 0.8299 SSIM**.

The ablation studies show that more input frames do not always improve performance. The best VSR-CNN PSNR for the input-frame ablation is obtained with **3 frames**, while the learning-rate ablation shows that **1e-3** gives the best optimization behavior. The loss-function ablation shows that **MSE loss** gives the best PSNR among the tested losses.

Overall, the project demonstrates the importance of learned upsampling, residual learning, transfer learning, and careful training configuration for video super-resolution.
