# models/

Put the trained Generator checkpoints from Kaggle in this folder:

| File | What it is | Needed? |
|---|---|---|
| `DCGAN_best.pth` | Best DCGAN (BCE loss) Generator checkpoint | Yes |
| `LSGAN_best.pth` | Best LSGAN (least-squares loss) Generator checkpoint | Yes |
| `best_overall.pth` | Copy of whichever of the two scored best | No (git-ignored) |
| `registry.json` | Summary written by the notebook | No |

## Where to get them

In your Kaggle notebook, after running the training cells, the files are saved in `/kaggle/working/models/`.
Open the **Output** panel on the right, expand `/kaggle/working/models`, and download `DCGAN_best.pth` and `LSGAN_best.pth`
(or click **Save Version**, open the version's **Output** tab, and download them from there).

## Size and Git

Each checkpoint is about 14 MB (a 3.6M-parameter Generator in float32), which is below GitHub's 100 MB limit,
so a normal `git add models/*.pth` works and Git LFS is not required.

## What is inside a checkpoint

`name`, `loss_type`, `state_dict`, `latent_dim`, `gen_features`, `epoch`, `metric`.
The app reads `name`, `epoch` and `metric` directly from the files, so it knows which model is the best overall
without needing `registry.json`.
