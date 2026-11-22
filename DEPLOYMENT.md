# Deployment guide

The app is `app.py` (Gradio). It reads the checkpoints in `models/`, shows a **Model** dropdown
(`Best overall (...)`, `DCGAN`, `LSGAN`), and generates faces on the CPU. No GPU is needed.

## Before any deployment

1. Train the models on Kaggle (see `KAGGLE_GUIDE.md`).
2. Download `DCGAN_best.pth` and `LSGAN_best.pth` from `/kaggle/working/models/` and put them in `models/`.
3. (Optional) Download `final_faces.png`, `progress_comparison.png`, `training_curves.png` into `assets/`.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `PORT` or `GRADIO_SERVER_PORT` | `7860` | Port the app listens on |
| `GRADIO_SERVER_NAME` | `0.0.0.0` | Host to bind (`0.0.0.0` = reachable from outside the machine/container) |
| `GRADIO_SHARE` | `0` | Set to `1` to also create a temporary public `gradio.live` link |
| `MODELS_DIR` | `./models` | Folder containing the `*.pth` checkpoints |

## Option A: run locally

```bash
git clone https://github.com/<your-username>/gan-face-generation.git
cd gan-face-generation
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://localhost:7860. To use another port: `PORT=8080 python app.py` (Windows PowerShell: `$env:PORT=8080; python app.py`).

## Option B: Hugging Face Spaces (recommended: permanent link, free CPU)

This avoids the temporary `gradio.live` tunnel (which can show `504 Gateway Time-out` when the Kaggle cell is not running).

1. Create an account at https://huggingface.co and click **New -> Space**.
2. Name it (for example `gan-face-generator`), choose **SDK: Gradio**, hardware **CPU basic (free)**, and create it.
3. Upload these files to the Space (**Files -> Add file -> Upload files**, or `git push`):
   - `app.py`, `requirements.txt`
   - the folder `gan_faces/`
   - the folder `models/` with `DCGAN_best.pth` and `LSGAN_best.pth`
   - `deployment/huggingface_space_README.md`, uploaded as **`README.md`** (it contains the required Space configuration block)
4. The Space builds automatically. When the status is **Running**, your app is live at
   `https://huggingface.co/spaces/<your-username>/gan-face-generator`.

Notes:
- Files above 10 MB need Git LFS on Hugging Face. The web upload handles this for you. If you use `git push`, run `git lfs install` and `git lfs track "*.pth"` first.
- Hugging Face reads Python packages from `requirements.txt`. If the build reports a Gradio version conflict, remove the `gradio` line from `requirements.txt` for the Space and let the Space's SDK provide it.

## Option C: Docker

```bash
docker build -t gan-faces .
docker run --rm -p 7860:7860 gan-faces
```

Open http://localhost:7860. To use another host port: `docker run --rm -p 8080:7860 gan-faces` then open http://localhost:8080.
The image contains the `models/` folder, so the checkpoints must exist before you build.

## Option D: Render, Railway or similar

These platforms give your app a `PORT` environment variable, which `app.py` already reads.

- Build command: `pip install -r requirements.txt`
- Start command: `python app.py`
- Make sure the `models/` folder (about 28 MB for both checkpoints) is part of the repository.

## Option E: directly from the Kaggle notebook (temporary)

Section 16 of the notebook launches the app on `PORT = 7860` with `SHARE = True`. Kaggle's `localhost` is not reachable from your browser,
so the public `gradio.live` link is the only way in. It works only while the cell is **Running**; stop the cell and the link dies.
Internet must be On in the notebook settings.

## Troubleshooting

| Problem | Fix |
|---|---|
| `OSError: Cannot find empty port in range: 7860-7860` | Another process uses the port. Stop it, or choose another port with `PORT=7861` (Kaggle: call `gr.close_all()` or restart the session) |
| `No checkpoints found in 'models'` | Put `DCGAN_best.pth` and `LSGAN_best.pth` into `models/` |
| `504 Gateway Time-out` on a `gradio.live` link | The link is from an old run or the cell stopped. Use the newest link, or deploy with Option B |
| Install is very slow or huge | Use the CPU PyTorch index already set in `requirements.txt` |
| Port 7860 not reachable from another computer | Keep `GRADIO_SERVER_NAME=0.0.0.0` and open the port in your firewall or cloud security group |
| Slow first generation | The first click loads the model from disk; later clicks use the in-memory cache |

## Publishing the repository to GitHub

```bash
cd gan-face-generation
git init
git add .
git commit -m "Initial commit: GAN face generation with Gradio deployment"
git branch -M main
git remote add origin https://github.com/<your-username>/gan-face-generation.git
git push -u origin main
```

Create the empty repository first on https://github.com/new (do not add a README there, you already have one).
Each checkpoint is about 14 MB, so normal Git is fine and Git LFS is not needed for GitHub.
