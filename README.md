# CLFitNet

CLFitNet is a SwinLSTM-based OCT sequence prediction project. The main task is to use the first 3 frames of an OCT sequence to predict the next 2 frames. The project contains both a standard SwinLSTM training pipeline and a full GAN-based pipeline with PatchGAN discriminators, perceptual loss, SSIM loss, and differentiable dynamic thickness loss.

## Entry Points

| File | Description |
| --- | --- |
| `train_GAN.py` | Training entry point for the full model. It uses a SwinLSTM generator, two PatchDiscriminator networks, GAN adversarial loss, VGG perceptual loss, SSIM loss, and dynamic thickness loss. |
| `test_GAN.py` | Inference and evaluation entry point for the full model. It loads trained GAN weights, reports MSE, thickness MAE, RMSE, and PSNR, and saves prediction, ground-truth, and mask images. |
| `train.py` | Training entry point without the GAN module. It trains the SwinLSTM sequence prediction model mainly with MSE loss. |
| `functions_GAN.py` | Training, validation, testing, and multi-step prediction functions used by `train_GAN.py` and `test_GAN.py`. |
| `functions.py` | Non-GAN training, validation, and multi-step prediction functions used by `train.py`. |

## Project Structure

```text
CLFitNet/
+-- train_GAN.py              # Full GAN training entry point
+-- test_GAN.py               # Full GAN inference and visualization entry point
+-- train.py                  # Non-GAN training entry point
+-- functions_GAN.py          # GAN training/testing functions
+-- functions.py              # Non-GAN training/testing functions
+-- SwinLSTM_MU_new.py        # SwinLSTM-M model
+-- SwinLSTM_D.py             # SwinLSTM-D model
+-- Discriminator.py          # PatchGAN discriminator
+-- Loss.py                   # Perceptual, SSIM, Sobel, and dynamic thickness losses
+-- dataAug.py                # Data augmentation for non-GAN training
+-- utils.py                  # Seeds, logging, metrics, visualization, and directories
+-- DataSet/
|   +-- Data_Center.py        # Dataset registry and data_provider
|   +-- data_SCL.py           # SCL dataset loading logic
+-- Gate_Unit/                # Spatiotemporal LSTM cell
+-- models/                   # PredRNN-related model code
+-- core/                     # Legacy/helper training utilities
```

## Dependencies

The code depends on PyTorch and common image-processing libraries. Python 3.8+ is recommended.

```bash
pip install torch torchvision numpy opencv-python pillow matplotlib scikit-image tqdm
```

If CUDA is used, install the `torch` and `torchvision` versions that match your local CUDA environment.

## Dataset Format

The current code uses the `SCL` dataset by default. The loading logic is implemented in `DataSet/data_SCL.py`. The dataset root directory should contain `train`, `valid`, and `test` subdirectories:

```text
SCL/
+-- train/
|   +-- case_xxx/
|       +-- OD/
|       |   +-- timepoint_xxx/
|       |       +-- scan 1/
|       |           +-- new/       # OCT images
|       |           +-- drop2_1/   # tear-film/thickness masks
|       +-- OS/
|           +-- ...
+-- valid/
|   +-- ...
+-- test/
    +-- ...
```

Notes:

- Images and masks can be `.png`, `.jpg`, `.jpeg`, or `.bmp`.
- Images are converted to grayscale and resized to `--img_width`.
- Pixel values are normalized to `[0, 1]`.
- Sequence length is controlled by `--total_length`, with a default value of `5`.
- The default task uses the first 3 frames as input and predicts the last 2 frames.
- The GAN pipeline also reads the masks of the last 2 frames for dynamic thickness loss and thickness-error evaluation.

## Train the Full GAN Model

Example:

```bash
python train_GAN.py \
  --train_data_path /path/to/SCL \
  --res_dir ./results \
  --model SwinLSTM-M \
  --img_width 512 \
  --batch_size 1 \
  --total_length 5 \
  --epochs 400
```

Default behavior of `train_GAN.py`:

- Uses `SwinLSTM-M` as the generator.
- Uses two `PatchDiscriminator` networks, one for each predicted frame.
- The generator loss combines adversarial loss, VGG perceptual loss, SSIM loss, and dynamic thickness loss.
- The dynamic thickness loss weight is controlled by `--lambda_thick`, with a default value of `0.001`.
- Models, logs, and intermediate visualizations are saved under `./results/model`, `./results/log`, and `./results/cache`.

Every 5 epochs, the best checkpoint within the current validation interval is saved with names similar to:

```text
results/model/Swin_GAN_Gnew_<epoch>_best
results/model/Swin_GAN_D1_<epoch>_best
results/model/Swin_GAN_D2_<epoch>_best
```

## Run Inference with the Full GAN Model

`test_GAN.py` currently loads checkpoints from the following hard-coded paths:

```text
results/ssim_thick_model/Swin_GAN_Gnew_202_best
results/ssim_thick_model/Swin_GAN_D1_202_best
results/ssim_thick_model/Swin_GAN_D2_202_best
```

If your checkpoints are stored elsewhere, update the corresponding `torch.load(...)` paths in `test_GAN.py`.

Example:

```bash
python test_GAN.py \
  --train_data_path /path/to/SCL \
  --res_dir ./results \
  --save_dir ./results/saved_images \
  --model SwinLSTM-M \
  --img_width 512 \
  --test_batch_size 1 \
  --total_length 5
```

The inference script reports:

- `MSE`
- `Thick_MAE(um)`
- `RMSE`
- `PSNR`

It also saves prediction, ground-truth, and mask images under `--save_dir`, for example:

```text
seq0000_frame4_pred.png
seq0000_frame4_gt.png
seq0000_frame4_mask.png
seq0000_frame5_pred.png
seq0000_frame5_gt.png
seq0000_frame5_mask.png
```

## Train the Non-GAN Model

Example:

```bash
python train.py \
  --train_data_path /path/to/SCL \
  --res_dir ./results \
  --model SwinLSTM-D \
  --img_width 256 \
  --batch_size 1 \
  --total_length 5 \
  --epochs 5
```

The non-GAN pipeline mainly uses the training logic in `functions.py`:

- The first 3 frames are used as input, and later frames are predicted.
- `dataAug.py` applies random rotation/flipping, brightness/contrast changes, temporal reversal, noise, and synthetic augmentation.
- The main training loss is MSE.
- Checkpoints are saved under `./results/model` by default.

## Common Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--device` | `cuda` | Runtime device. Use `cpu` if no GPU is available, although training will be much slower. |
| `--dataset_name` | `SCL` | Dataset name. Currently, only `SCL` is registered in `Data_Center.py`. |
| `--train_data_path` | Script-specific default | Dataset root directory. It should contain `train/valid/test`. |
| `--batch_size` | `1` | Training batch size. |
| `--test_batch_size` | `1` | Testing batch size. |
| `--img_width` | GAN: `512`, non-GAN: `256` | Image width and height after resizing. |
| `--total_length` | `5` | Total number of frames per sequence. |
| `--model` | GAN: `SwinLSTM-M`, non-GAN: `SwinLSTM-D` | Model type. |
| `--patch_size` | GAN: `4`, non-GAN: `2` | Patch size for patch embedding. |
| `--embed_dim` | GAN: `160`, non-GAN: `128` | Patch embedding dimension. |
| `--lr` | `0.0001` | Learning rate. |
| `--epochs` | GAN: `400`, non-GAN: `5` | Number of training epochs. |
| `--lambda_thick` | `0.001` | Weight of the dynamic thickness loss in GAN training. |
| `--scale_um` | About `10.04` | Physical scale conversion from pixels to micrometers. |

## Output Directories

The scripts call `utils.make_dir(args)` to create:

```text
results/
+-- cache/         # Cached validation/testing visualizations
+-- model/         # Model checkpoints
+-- log/           # Log files
```

`test_GAN.py` also uses:

```text
results/saved_images/   # Saved pred/gt/mask inference images
```

## Notes

1. Some scripts contain hard-coded GPU settings such as `CUDA_VISIBLE_DEVICES`. Adjust them if your GPU IDs are different.
2. The checkpoint paths in `test_GAN.py` are hard-coded and should be updated when testing different experiments.
3. The VGG perceptual loss in `Loss.py` uses ImageNet-pretrained `torchvision.models.vgg16` weights. If the weights are not cached locally, the first run may require network access.
4. Dataset loading is sensitive to directory names, especially `scan 1/new` and `scan 1/drop2_1`.
5. If the number of images and masks does not match for a time point, that time point is skipped.

## Recommended Workflow

1. Prepare `SCL/train`, `SCL/valid`, and `SCL/test` according to the dataset format above.
2. Run `train.py` or `train_GAN.py` with a small `--epochs` value first to verify data loading, GPU memory usage, and output directories.
3. Train the full model with `train_GAN.py`.
4. Update the checkpoint paths in `test_GAN.py`.
5. Run `test_GAN.py` to compute metrics and save visualization results.
