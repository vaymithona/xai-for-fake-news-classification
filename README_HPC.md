# Running the Fake News XAI Notebook on an HPC Server (Ubuntu + A40 GPU)

Step-by-step guide to run `Fake_News_Classification_Final copy.ipynb` on an HPC
cluster after cloning the repository from GitHub. Target environment: **Ubuntu**,
**NVIDIA A40 (48 GB)**, typically with a **SLURM** scheduler and either the
`module` system or `conda`.

The notebook trains classical models (LogReg, Random Forest, XGBoost, LightGBM),
an LSTM, and a frozen **DistilBERT** encoder + MLP head, then produces the XAI
section (SHAP, LIME, attention, Integrated Gradients). It auto-detects the GPU and
falls back to CPU if none is found.

---

## 0. Prerequisites / things to know first

- **Login node vs. compute node.** You log in to a *login node* that usually
  **has internet**. The *compute nodes* (where the GPU lives) often **do not**.
  This matters: the notebook downloads the DistilBERT weights from Hugging Face
  and the NLTK stopwords list on first run. We therefore **pre-download them on the
  login node** (Step 5) so the GPU job can run offline.
- **The dataset is not in Git.** `WELFake_Dataset.csv` (~63k+ articles, > 100 MB)
  exceeds GitHub's file limit and is normally excluded from the repo. You must place
  it in the project root yourself (Step 3).
- **Never run heavy compute on the login node.** Always request a GPU via SLURM
  (`srun`/`sbatch`) — Steps 6a / 6b.

---

## 1. Connect to the HPC

```bash
ssh <username>@<hpc-login-hostname>
```

Move to your work/scratch area (use scratch for large data, not your home quota):

```bash
cd /scratch/$USER        # or wherever your cluster wants project files
```

---

## 2. Clone (or pull) the code from GitHub

First time:

```bash
git clone https://github.com/<your-account>/<your-repo>.git fake-news-xai
cd fake-news-xai
```

Already cloned — just update:

```bash
cd fake-news-xai
git pull
```

---

## 3. Get the dataset

Place `WELFake_Dataset.csv` in the **project root** (same folder as the notebook).
The notebook reads it from the current directory: `DATASET_PATH = 'WELFake_Dataset.csv'`.

Options:

```bash
# A) Copy from your laptop (run this FROM your laptop, not the HPC):
scp "WELFake_Dataset.csv" <username>@<hpc-login-hostname>:/scratch/$USER/fake-news-xai/

# B) Download with the Kaggle CLI on the login node (needs ~/.kaggle/kaggle.json):
pip install --user kaggle
kaggle datasets download -d saurabhshahane/fake-news-classification
unzip fake-news-classification.zip      # yields WELFake_Dataset.csv
```

Verify it is there:

```bash
ls -lh WELFake_Dataset.csv
```

---

## 4. Create the Python environment

Use the cluster's Conda/Anaconda module if available (recommended), otherwise a venv.

### Option A — Conda (recommended)

```bash
module load anaconda3        # or: module load miniconda; check `module avail`
conda create -y -n fakenews python=3.11
conda activate fakenews
```

### Option B — venv + a CUDA module

```bash
module load python/3.11 cuda/12.1     # names vary per cluster; check `module avail cuda`
python -m venv ~/envs/fakenews
source ~/envs/fakenews/bin/activate
pip install --upgrade pip
```

### Install the packages

Install a **CUDA-enabled PyTorch** that matches the cluster's driver. The A40 is an
Ampere GPU (compute capability 8.6) and works with the CUDA 12.1 or 11.8 wheels:

```bash
# CUDA 12.1 build (most modern clusters). Use cu118 if your driver is older.
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Everything else (these are CPU/GPU-agnostic):
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm nltk \
            transformers shap lime captum \
            jupyterlab ipykernel nbconvert
```

> The notebook's own `!pip install` cell installs the base libraries too, but doing it
> here gives you the correct **GPU** PyTorch build and the XAI extras (`shap`, `lime`,
> `captum`) up front. You can safely skip the in-notebook install cell.

Register the env as a Jupyter kernel (only needed for the interactive route, Step 6a):

```bash
python -m ipykernel install --user --name fakenews --display-name "fakenews"
```

---

## 5. Pre-download the model + NLTK data (on the LOGIN node, has internet)

This caches everything so the GPU compute node can run offline.

```bash
# Cache the DistilBERT weights and tokenizer
python -c "from transformers import AutoTokenizer, AutoModel; \
m='distilbert-base-uncased'; AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m); \
print('DistilBERT cached')"

# Cache the NLTK stopwords list
python -c "import nltk; nltk.download('stopwords'); print('stopwords cached')"
```

By default these go to `~/.cache/huggingface` and `~/nltk_data`. If your home directory
is small, point the caches at scratch **before** the commands above and reuse the same
values in your job script (Step 6):

```bash
export HF_HOME=/scratch/$USER/hf_cache
export NLTK_DATA=/scratch/$USER/nltk_data
```

---

## 6. Run the notebook on the GPU

Pick **one** of the two routes.

### Route 6a — Interactive (Jupyter Lab via SSH tunnel)

Good for exploring plots and the XAI figures live.

1. Grab an A40 interactively (flags vary by cluster — ask your admin for the partition
   and `gres` string):

   ```bash
   srun --partition=gpu --gres=gpu:a40:1 --cpus-per-task=8 --mem=32G \
        --time=02:00:00 --pty bash
   ```

2. On the allocated compute node, activate the env and start Jupyter (no browser):

   ```bash
   conda activate fakenews          # or: source ~/envs/fakenews/bin/activate
   export HF_HOME=/scratch/$USER/hf_cache NLTK_DATA=/scratch/$USER/nltk_data   # if set in Step 5
   hostname                          # note the compute node name, e.g. gpu-node07
   jupyter lab --no-browser --ip=0.0.0.0 --port=8888
   ```

3. From your **laptop**, open an SSH tunnel through the login node to that compute node:

   ```bash
   ssh -N -L 8888:<gpu-node-name>:8888 <username>@<hpc-login-hostname>
   ```

4. Open the printed `http://127.0.0.1:8888/lab?token=...` URL in your browser, open the
   notebook, select the **fakenews** kernel, and **Run All**.

### Route 6b — Batch job (headless, recommended for the full run)

Runs the whole notebook end-to-end and saves an executed copy with all outputs.
Create `run_notebook.slurm` in the project root:

```bash
#!/bin/bash
#SBATCH --job-name=fakenews-xai
#SBATCH --partition=gpu                 # <-- your GPU partition
#SBATCH --gres=gpu:a40:1                # <-- request one A40
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=fakenews_%j.log

set -e
module load anaconda3                   # or your cuda/python modules
source activate fakenews                # or: source ~/envs/fakenews/bin/activate

# Reuse the caches populated on the login node (offline-safe)
export HF_HOME=/scratch/$USER/hf_cache
export NLTK_DATA=/scratch/$USER/nltk_data
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

nvidia-smi                              # confirm the A40 is visible

jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  "Fake_News_Classification_Final copy.ipynb"

# Also export a static HTML report of the run (figures included):
jupyter nbconvert --to html "Fake_News_Classification_Final copy.ipynb"
```

Submit and monitor:

```bash
sbatch run_notebook.slurm
squeue -u $USER                 # watch the queue
tail -f fakenews_<jobid>.log    # follow progress
```

When it finishes, the notebook is updated in place with all outputs, and
`Fake_News_Classification_Final copy.html` holds a shareable report.

---

## 7. Confirm the GPU was actually used

In the executed notebook, the environment cell prints:

```
Using GPU: NVIDIA A40  (47.5 GiB)
```

and the DistilBERT config cell prints `FP16: True`, `BERT batch size: 256`. If you
instead see `No CUDA GPU detected; using CPU.`, the job did not get a GPU — recheck the
`--gres` flag (Step 6) and that you installed the **CUDA** PyTorch wheel (Step 4).

---

## 8. Notes & troubleshooting

- **A40 memory.** 48 GB is ample for this workload. The transformer step defaults to
  `BATCH_SIZE = 256` + FP16 on GPU. If you ever hit out-of-memory, lower `BATCH_SIZE`
  or set `SAMPLE_SIZE` (in the §8 config cell) to e.g. `10000` for a quick run.
- **Compute node has no internet.** Symptoms: the run hangs or errors trying to reach
  `huggingface.co`. Fix: do Step 5 on the login node and keep `TRANSFORMERS_OFFLINE=1`
  / `HF_HUB_OFFLINE=1` in the job (as in 6b).
- **`CUDA error: no kernel image is available`** usually means a PyTorch/CUDA mismatch.
  Reinstall PyTorch with the wheel matching the cluster driver (`cu121` or `cu118`),
  and load the matching `cuda` module if you use a venv.
- **LightGBM / XGBoost errors about libgomp / OpenMP**: `conda install -c conda-forge
  lightgbm xgboost` instead of pip if your cluster lacks the system OpenMP runtime.
- **Slow / CPU-only fallback** in Route 6b: confirm with `nvidia-smi` inside the job
  (it is in the script) — an empty GPU list means the partition/gres request was wrong.
```
