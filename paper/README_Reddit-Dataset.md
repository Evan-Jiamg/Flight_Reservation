# Reddit stance corpus

Stage 1 of the H-COG study. Reddit r/politics comments from April 2019 are
clustered by topic, a RoBERTa regression head is trained to place each comment
on a stance scale, and the scored comments become the empirical priors for the
agents in `../Hybrid-Network/`.

Two topics are carried forward: **gun control** (n = 3,581) and **abortion**
(n = 1,618), drawn from clusters identified in Stage 1.

`Workflow.md` documents each stage in detail; this file covers the layout, what
is and is not in the repository, and how to reproduce the corpus.

---

## Layout

```
Reddit-Dataset/
├── pipeline/                     the stages that produce data
│   ├── fetch_dataset.py            0. download the Pushshift dump
│   ├── stage1_cluster.py           1. SBERT embedding + HDBSCAN clustering
│   ├── stage2_train.py             2. train the stance regressor (local)
│   ├── stage2_train_colab.ipynb    2. the same, on Colab GPU
│   ├── stage2_infer.py             3. score the corpus with a checkpoint
│   └── run_pipeline.sh             all of the above, end to end
│
├── analysis/                     exploration and inspection
│   ├── eda.py                      streaming EDA over the full dump
│   ├── visualize.py                sampled EDA charts
│   ├── visualize_stance.py         stance distribution charts
│   ├── review_comments.py          read or delete individual comments
│   ├── figures/                    generated charts
│   └── reports/                    generated text reports
│
├── data/                         cluster assignments and metadata
├── model/                        RoBERTa checkpoints (not tracked)
└── stance_scores/                the scored corpus, one Parquet per cluster
```

Scripts in `pipeline/` and `analysis/` resolve the dataset root from their own
location, so they run from anywhere.

---

## What is in the repository

Tracked: all code, the cluster assignments in `data/`, the six EDA charts, the
EDA report, and the scored corpus in `stance_scores/`.

Not tracked, and why:

| | Size | Reason |
|---|---|---|
| `model/*.pt` | 2.8 GB | six 476 MB checkpoints; each exceeds GitHub's 100 MB file limit |
| the Pushshift dump | 15 GB | `RC_2019-04.zst`, redistributed by Pushshift, not by us |
| `data/embeddings_politics.npy` | large | recomputed by `stage1_cluster.py` |

`analysis/figures/*.png` **is** tracked even though it is generated: `eda.py`
reads the 15 GB dump, which no clone has, so those six figures cannot be
rebuilt here. `analysis/figures/stance/` is not tracked, because
`visualize_stance.py` rebuilds it from `stance_scores/`.

---

## Requirements

```bash
pip install -r requirements.txt
```

Versions are pinned to the environment that produced the corpus. Only Stage 4
needs a GPU; inspecting the scored corpus needs nothing beyond pandas, pyarrow
and matplotlib.

Stage 3 is much faster with RAPIDS (`cuml`), but both its UMAP and HDBSCAN
steps fall back to CPU when it is absent. Set `CUML_PATH` if RAPIDS lives
outside the default site-packages.

---

## Reproducing the corpus

```bash
export REDDIT_ZST=/path/to/RC_2019-04.zst
bash pipeline/run_pipeline.sh
```

Or one stage at a time:

```bash
python3 pipeline/fetch_dataset.py                     # 0. download
python3 pipeline/stage1_cluster.py --output-dir data  # 1. cluster
python3 pipeline/stage2_train.py                      # 2. train
python3 pipeline/stage2_infer.py \
        --clusters data/clusters_gun_abortion.json \
        --checkpoint model/final_model.pt             # 3. score
```

Environment variables the pipeline honours:

| | |
|---|---|
| `REDDIT_ZST` | the Pushshift dump |
| `PYTHON` | interpreter for the stages |
| `STANCE_SCORES_DIR` | where scores are written and read |
| `STANCE_CKPT` | checkpoint used for scoring |
| `CUML_PATH` | RAPIDS site-packages, if available |

## Inspection

```bash
python3 analysis/visualize_stance.py                  # stance charts
python3 analysis/review_comments.py --topic gun --order desc --limit 20
```

`analysis/eda.py` and `analysis/visualize.py` both need the raw dump.

---

## Downstream

`../Hybrid-Network/data/init_agents.py` reads
`stance_scores/cluster_<id>.parquet`, samples 50 agents per topic stratified by
stance, and writes their intrinsic opinions, stubbornness coefficients and
generated backgrounds into `../Hybrid-Network/data/agents/`.
