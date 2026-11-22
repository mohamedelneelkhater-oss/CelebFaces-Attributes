# Kaggle guide: training the GAN models

This guide matches the notebook `notebooks/gan_face_generation.ipynb` and the data you used on Kaggle.

## 1. Your setup

| Item | Value |
|---|---|
| Notebook | `notebookfc6363d81f` (Kaggle user `mohamedelneel`) |
| Dataset | **CelebFaces Attributes (CelebA) Dataset** by Jessica Li (over 200k celebrity images with 40 binary attribute annotations, about 202,599 images) |
| Accelerator | GPU T4 x1 (one GPU is enough) |
| Internet | Off for training; On only for the public Gradio link (and `pip install gradio` if it is missing) |

> Add the dataset from **Add Input -> Datasets**. The list with a `<>` icon is **Notebooks** (other people's code), not datasets.
> Do not add notebooks as input: the loader searches `/kaggle/input` for the first folder with many images and could pick a wrong one.

## 2. Import and run the notebook

1. Kaggle -> **Create -> New Notebook -> File -> Import Notebook**, and upload `notebooks/gan_face_generation.ipynb`.
2. Settings -> Accelerator: **GPU T4 x1**.
3. Add Input -> Datasets -> search `CelebA` -> add **CelebFaces Attributes (CelebA) Dataset** (Jessica Li).
4. Run the cells in order.

Expected check after the loading cell: the image folder is printed and about **202,599** images are found.
Only a random subset of 20,000 images is used (see the configuration below).

## 3. What each section produces

| Section | Output |
|---|---|
| 4. Load and preprocess | `DATA` tensor of 20,000 face crops, 64x64, uint8 (about 245 MB in RAM) |
| 6. Real faces | A grid of preprocessed real faces (sanity check) |
| 9-11. Training | Per epoch: losses, sample grid in `samples/<model>/epoch_XXX.png`; every 2 epochs: quality metric and best checkpoint |
| 12. Curves | `training_curves.png` |
| 13. Progress | `progress_comparison.png` |
| 14. Best model | `models/DCGAN_best.pth`, `models/LSGAN_best.pth`, `models/best_overall.pth`, `models/registry.json` |
| 15. Final faces | `final_faces.png` |
| 16. Gradio | Local URL `http://0.0.0.0:7860` and a public `gradio.live` link |

## 4. Configuration used (cell 2 of the notebook)

| Parameter | Value | Why |
|---|---|---|
| `img_size` | 64 | Small images keep training fast and light |
| `latent_dim` | 100 | Standard DCGAN noise size |
| `gen_features` / `disc_features` | 64 / 64 | About 3.6M Generator parameters |
| `num_images` | 20000 | Subset of CelebA, enough for recognizable faces |
| `batch_size` | 128 | Fits easily on a T4 |
| `epochs` | 20 | Few minutes per model |
| `lr` / `beta1` | 2e-4 / 0.5 | Standard DCGAN optimizer settings |
| `eval_every` / `eval_samples` | 2 / 3000 | Quality metric every 2 epochs on 3000 generated faces |

For better faces, raise `epochs` (for example 40-50) and/or `num_images`. Time grows roughly in proportion.

## 5. Download the results for GitHub

From the **Output** panel (right side) download:

- `/kaggle/working/models/DCGAN_best.pth` and `LSGAN_best.pth` -> put in `models/`
- `/kaggle/working/final_faces.png`, `progress_comparison.png`, `training_curves.png` -> put in `assets/`

If the Output panel does not show them, click **Save Version -> Save & Run All**, then open the version's **Output** tab.

## 6. Common Kaggle problems

| Problem | Cause and fix |
|---|---|
| `OSError: Cannot find empty port in range: 7860-7860` | An older app still holds the port. The notebook now calls `gr.close_all()` before launching. If it persists, use **Run -> Restart session** or set `PORT = 7861` |
| Public link shows `504 Gateway Time-out` | The link belongs to an older run, or the cell is no longer running. Copy the **newest** link from the output and keep the cell in Running state |
| Link keeps loading in the browser | Try the link from a phone on mobile data. If it works there, your network or ISP is blocking `gradio.live`; deploy to Hugging Face Spaces instead (see `docs/DEPLOYMENT.md`) |
| `No face images found` | The dataset was not added, or a Notebook was added instead of the Dataset |
| Out of memory | Lower `num_images` or `batch_size` |
