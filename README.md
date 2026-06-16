# Video Super-Resolution

This repository contains the implementation and experimental evaluation for a **Video Super-Resolution** project. The goal is to reconstruct high-resolution video frames from low-resolution inputs using classical interpolation, frame-by-frame neural super-resolution models, and a custom multi-frame video super-resolution model.

The project evaluates six methods on two datasets: **Vimeo-90K** and **REDS**. It also includes transfer learning experiments and ablation studies on temporal context, learning rate, and cross-dataset adaptation.

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

Vimeo-90K is used as one of the main datasets. Each sample contains seven consecutive frames.

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

SRCNN is used for:

* training on Vimeo-90K;
* training directly on REDS;
* evaluating a Vimeo-trained model directly on REDS;
* fine-tuning from Vimeo-90K to REDS.

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

The models are evaluated using **PSNR** and **SSIM**.

| Metric | Meaning                     |
| ------ | --------------------------- |
| PSNR   | Peak Signal-to-Noise Ratio  |
| SSIM   | Structural Similarity Index |

Higher values are better for both metrics.

All evaluations are performed for **x4 super-resolution**. Border cropping is applied during metric computation according to the scale factor.

The Mean Squared Error between the super-resolved image and the high-resolution target is computed as:

$$
\mathrm{MSE} = \frac{1}{HWC}
\sum_{i=1}^{H}
\sum_{j=1}^{W}
\sum_{c=1}^{C}
\left( I^{HR}*{i,j,c} - I^{SR}*{i,j,c} \right)^2
$$

The PSNR metric is computed as:

$$
\mathrm{PSNR} = 10 \log_{10}
\left(
\frac{MAX_I^2}{\mathrm{MSE}}
\right)
$$

Since the images are normalized to the range $[0,1]$, $MAX_I = 1$, therefore:

$$
\mathrm{PSNR} = -10 \log_{10}(\mathrm{MSE})
$$

The SSIM metric is computed as:

$$
\mathrm{SSIM}(x,y) =
\frac{
(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)
}{
(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)
}
$$

where $I^{HR}$ is the ground-truth high-resolution image, $I^{SR}$ is the super-resolved image, $H$, $W$, and $C$ are the image height, width, and number of channels.

---

## Main Results

The following table shows the main comparison between all six methods on Vimeo-90K and REDS.

| Model     | Type                            | Trained by us | Epochs | Learning rate | Vimeo PSNR ↑ | Vimeo SSIM ↑ | REDS PSNR ↑ | REDS SSIM ↑ |
| --------- | ------------------------------- | ------------- | -----: | ------------: | -----------: | -----------: | ----------: | ----------: |
| Bicubic   | Interpolation baseline          | No            |    N/A |           N/A |      29.9324 |       0.8544 |     26.5785 |      0.7626 |
| SRCNN     | Frame-by-frame CNN              | Yes           |     20 |          1e-4 |      31.4058 |       0.8783 |     27.6228 |      0.7972 |
| EDSR-lite | Frame-by-frame residual CNN     | Yes           |     20 |          1e-4 |      32.3560 |       0.9003 |     28.3731 |      0.8242 |
| VSR-CNN   | Multi-frame CNN                 | Yes           |     20 |          1e-4 |      30.5926 |       0.8810 |     28.1483 |      0.8149 |
| FSRCNN    | Frame-by-frame CNN              | Yes           |     20 |          1e-4 |      30.6435 |       0.8617 |     27.1607 |      0.7802 |
| ESPCN     | Frame-by-frame PixelShuffle CNN | Yes           |     20 |          1e-4 |      30.6031 |       0.8663 |     27.1563 |      0.7839 |

---

## Confirmed Results

| Experiment                   |  PSNR ↑ | SSIM ↑ |
| ---------------------------- | ------: | -----: |
| Bicubic on Vimeo-90K         | 29.9324 | 0.8544 |
| Bicubic on REDS              | 26.5785 | 0.7626 |
| SRCNN on Vimeo-90K           | 31.4058 | 0.8783 |
| SRCNN on REDS                | 27.6228 | 0.7972 |
| EDSR-lite on Vimeo-90K       | 32.3560 | 0.9003 |
| EDSR-lite on REDS            | 28.3731 | 0.8242 |
| VSR-CNN 7-frame on Vimeo-90K | 30.5926 | 0.8810 |
| VSR-CNN 7-frame on REDS      | 28.1483 | 0.8149 |
| FSRCNN on Vimeo-90K          | 30.6435 | 0.8617 |
| FSRCNN on REDS               | 27.1607 | 0.7802 |
| ESPCN on Vimeo-90K           | 30.6031 | 0.8663 |
| ESPCN on REDS                | 27.1563 | 0.7839 |

---

## Transfer Learning Results

The project studies transfer learning from **Vimeo-90K to REDS**.

| Model   | Training setup                 | REDS PSNR ↑ | REDS SSIM ↑ |
| ------- | ------------------------------ | ----------: | ----------: |
| SRCNN   | Train Vimeo, evaluate REDS     |     27.6693 |      0.7991 |
| SRCNN   | REDS from scratch              |     27.6228 |      0.7972 |
| SRCNN   | Vimeo → REDS transfer learning |     27.8155 |      0.8032 |
| VSR-CNN | Train Vimeo, evaluate REDS     |     28.2629 |      0.8223 |
| VSR-CNN | REDS from scratch              |     28.1483 |      0.8149 |
| VSR-CNN | Vimeo → REDS transfer learning |     28.5620 |      0.8299 |

The transfer learning results show that fine-tuning a Vimeo-trained model on REDS improves performance compared to both direct cross-dataset evaluation and REDS training from scratch.

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
| VSR-CNN | Vimeo   |          1e-2 |     TBD |    TBD |
| VSR-CNN | Vimeo   |          1e-3 |     TBD |    TBD |
| VSR-CNN | Vimeo   |          1e-4 | 30.5926 | 0.8810 |
| VSR-CNN | Vimeo   |          1e-5 |     TBD |    TBD |

The goal is to analyze training stability and reconstruction quality under different optimization settings.

---

## Ablation Study 3: Transfer Learning

This ablation studies the effect of pretrained initialization when transferring from Vimeo-90K to REDS.

| Model   | Setup                      | REDS PSNR ↑ | REDS SSIM ↑ |
| ------- | -------------------------- | ----------: | ----------: |
| SRCNN   | REDS from scratch          |     27.6228 |      0.7972 |
| SRCNN   | Train Vimeo, evaluate REDS |     27.6693 |      0.7991 |
| SRCNN   | Vimeo → REDS fine-tuned    |     27.8155 |      0.8032 |
| VSR-CNN | REDS from scratch          |     28.1483 |      0.8149 |
| VSR-CNN | Train Vimeo, evaluate REDS |     28.2629 |      0.8223 |
| VSR-CNN | Vimeo → REDS fine-tuned    |     28.5620 |      0.8299 |

This experiment measures whether pretraining on one dataset helps the model adapt to another dataset.

---

## Expected Observations

The obtained results show the following trends:

* Bicubic interpolation provides the classical non-learning baseline.
* Neural methods improve over bicubic interpolation on REDS.
* EDSR-lite obtains the best REDS result among the frame-by-frame methods.
* VSR-CNN obtains the best REDS result after transfer learning.
* FSRCNN and ESPCN provide efficient frame-by-frame alternatives, but their REDS results are lower than SRCNN and EDSR-lite in this setup.
* Transfer learning from Vimeo-90K to REDS improves both SRCNN and VSR-CNN performance.

---

## Result Artifacts

The experimental pipeline produces:

| Artifact         | Description                                          |
| ---------------- | ---------------------------------------------------- |
| Summary files    | Mean PSNR and SSIM for each evaluated model          |
| CSV files        | Per-frame PSNR and SSIM                              |
| Example images   | Visual comparison between model output and HR target |
| Checkpoints      | Trained model weights                                |
| TensorBoard logs | Training and validation loss curves                  |

Example images are generated during evaluation and saved in the corresponding `results/.../examples/` folder. These images are used for qualitative comparison between the reconstructed super-resolved output and the high-resolution ground truth.

---

## Project Status

| Method    | Implemented |      Trained | Evaluated |
| --------- | ----------: | -----------: | --------: |
| Bicubic   |         Yes | Not required |       Yes |
| SRCNN     |         Yes |          Yes |       Yes |
| EDSR-lite |         Yes |          Yes |       Yes |
| VSR-CNN   |         Yes |          Yes |       Yes |
| FSRCNN    |         Yes |          Yes |       Yes |
| ESPCN     |         Yes |          Yes |       Yes |

Current completed highlights:

* REDS bicubic baseline evaluated.
* SRCNN trained and evaluated on Vimeo-90K and REDS.
* SRCNN transfer learning from Vimeo-90K to REDS completed.
* EDSR-lite trained and evaluated on Vimeo-90K and REDS.
* VSR-CNN trained and evaluated on Vimeo-90K and REDS.
* VSR-CNN transfer learning from Vimeo-90K to REDS completed.
* FSRCNN trained and evaluated on Vimeo-90K and REDS.
* ESPCN trained and evaluated on Vimeo-90K and REDS.

---

## Conclusion

This project builds a complete experimental pipeline for video super-resolution using two datasets and six methods. It combines classical interpolation, frame-by-frame CNN models, residual CNN models, sub-pixel convolutional models, and a custom multi-frame VSR-CNN model.

The experiments evaluate:

* the difference between classical and neural super-resolution;
* the effect of residual learning and learned upsampling;
* the importance of temporal context in video super-resolution;
* the impact of transfer learning from Vimeo-90K to REDS;
* the sensitivity of VSR-CNN to learning rate and number of input frames.

The best REDS result obtained in the current experiments is achieved by **VSR-CNN with Vimeo → REDS transfer learning**, reaching **28.5620 PSNR** and **0.8299 SSIM**.
