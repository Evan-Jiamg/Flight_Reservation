# Pipeline

How the stance corpus is built, stage by stage. `README.md` covers the layout
and what is tracked; this file covers what each stage does and why.

The source is the Pushshift Reddit comment dump for **April 2019**,
`RC_2019-04.zst`: 15.5 GB compressed, **138,473,643 comments**, one JSON object
per line. Comments are clustered by topic, then each comment in the clusters of
interest is placed on a continuous stance scale from −1 (oppose) to +1
(support).

---

## Stage 0 — Acquire the dump

**Script** `pipeline/fetch_dataset.py`
**Input** Kaggle dataset `i221113hadiyatanveer/the-pushshift-reddit-dataset-submissions`
**Output** the archive on disk, and a printed summary of its structure

Set `REDDIT_ZST` to the resulting `RC_2019-04.zst` for every later stage.

---

## Stage 1 — Streaming EDA over the full dump

**Script** `analysis/eda.py`
**Input** the complete archive
**Output** `analysis/reports/eda_report.txt`, `analysis/figures/` (charts A–F)
**Cost** about 66 minutes at ~35k comments/second

The dump does not fit in memory, so nothing is materialised as a DataFrame. It
is read in 8 MB chunks and parsed line by line; means and variances use
Welford's online algorithm, which is numerically stable in a single pass;
per-subreddit deletion rates and mean scores accumulate in counters. Distinct
authors are capped at two million to bound memory.

What the pass established:

- 138,473,643 comments, zero parse failures
- 43 fields per record; the ones that matter downstream are `body`, `score`,
  `subreddit`, `author`, `created_utc`
- score is heavily right-skewed: mean 7.62, sd 107.47, range −9,090 to 66,762
- overall deletion rate 11.56%
- activity peaks at 17:00–19:00 UTC, North American afternoon
- `removal_reason` and `archived` are empty in essentially every record and are
  dropped downstream
- a few livestream subreddits are deleted at ~100% and are excluded

---

## Stage 2 — Sampled visualisation

**Script** `analysis/visualize.py`
**Input** the first 200,000 comments
**Output** seven charts in `analysis/figures/`

A fast look while the full pass runs: score distribution and CDF, the 25
busiest subreddits, comment length, per-field missingness, hourly volume,
score against length, and deletion rates overall and per subreddit.

---

## Stage 3 — Semantic clustering

**Script** `pipeline/stage1_cluster.py`
**Goal** group comments within one subreddit so that a cluster is a single
concrete topic rather than a broad theme.

Comments are streamed from the archive and filtered to a subreddit, excluding
deleted comments and bodies shorter than 20 characters. Each body is embedded
into 384 dimensions with the `all-MiniLM-L6-v2` bi-encoder, optionally reduced
with UMAP, then clustered with HDBSCAN. TF-IDF over each cluster supplies a
`topic_label` and `topic_description`.

HDBSCAN is used rather than k-means because the number of topics is not known
in advance and most comments belong to no coherent topic at all; HDBSCAN
assigns those to noise instead of forcing them into a cluster.

```bash
python3 pipeline/stage1_cluster.py --subreddit politics --min-cluster-size 50
```

Output, one record per cluster in `data/clusters_<subreddit>.json`:

```json
{
  "cluster_id": 42,
  "topic_label": "trump immigration wall funding",
  "topic_description": "Discussion about: trump wall, immigration policy, border funding",
  "comment_ids": ["ejualnb", "ejualnd", "..."],
  "size": 312
}
```

Nine clusters are carried forward, six on gun control and two on abortion, in
`data/clusters_gun_abortion.json`. One further cluster is scored but excluded
from both topics.

RAPIDS `cuml` accelerates both UMAP and HDBSCAN when present; both steps fall
back to CPU otherwise. `CUML_PATH` points at a RAPIDS install outside the
default site-packages.

---

## Stage 4 — Train the stance regressor

**Script** `pipeline/stage2_train.py`, or `stage2_train_colab.ipynb` on a GPU
**Output** `model/final_model.pt`, `model/cv_results.json`

A cross-encoder over RoBERTa-base:

```
[CLS] <topic_description> [SEP] <comment_body> [SEP]
                  |
        RoBERTa-base (12 layers, 768 dim)
                  |
             Linear(768 -> 1)
                  |
                 tanh
                  |
          stance in [-1, +1]
```

Topic and comment attend to each other across every layer, so the score is a
stance *toward that topic* rather than a sentiment reading of the comment
alone. The `tanh` bounds the output to the interval the downstream simulation
expects.

**Training data.** IBM Argument Quality Ranking 30K
(`ibm-research/argument_quality_ranking_30k`): 30,497 arguments across 71
debate topics, each with a `stance_WA` label in {−1, +1} and a confidence
`stance_WA_conf`. Arguments below 0.8 confidence are dropped. `stance_WA` is
used directly as the regression target, so no pseudo-labelling is involved.

**Validation.** Four-fold `GroupKFold` grouped by topic, so every fold is
evaluated on debate topics it never saw. Grouping by topic rather than at
random is the point: a random split leaks topic-specific vocabulary between
folds and overstates generalisation to the Reddit topics, which are unseen by
construction.

```bash
python3 pipeline/stage2_train.py --epochs 3 --batch-size 16
```

---

## Stage 5 — Score the clusters

**Script** `pipeline/stage2_infer.py`
**Input** a cluster JSON from Stage 3 and a checkpoint from Stage 4
**Output** one Parquet file per cluster under `stance_scores/<variant>/`

The archive is scanned once to recover the bodies of every comment named in the
clusters. Each is paired with its cluster's `topic_description`, passed through
the model, and written out as
`(comment_id, stance, reddit_score, subreddit, created_utc, cluster_id)`.

```bash
python3 pipeline/stage2_infer.py \
        --clusters data/clusters_gun_abortion.json \
        --checkpoint model/final_model_bws.pt \
        --output-dir stance_scores/bws
```

---

## Stage 6 — Recalibration

The regressor trained in Stage 4 is accurate in ordering but badly calibrated
in magnitude: 43.7% of gun-control comments and 55.4% of abortion comments came
back with |stance| > 0.9, pinned against the ends of the scale. A stance
variable that is nearly binary cannot support the continuous opinion dynamics
downstream.

Recalibrating the head against Best-Worst Scaling judgements produces
`model/final_model_bws.pt`. Rescoring with it drops saturation to 0.8% and 1.8%
and roughly halves the standard deviation.

Both scored sets are kept, for reasons set out in `stance_scores/README.md`:
the recalibrated one is what the paper reports, and the uncalibrated one is
what the simulation's agent priors were actually sampled from.

---

## Stage 7 — Inspection

```bash
python3 analysis/visualize_stance.py                  # charts from bws/
STANCE_VARIANT=uncalibrated python3 analysis/visualize_stance.py
python3 analysis/review_comments.py --topic gun --order desc --limit 20
```

`review_comments.py` lists the highest- or lowest-scoring comments for a topic
and can delete individual records, for pruning misclassified comments by hand.

---

## End to end

```bash
export REDDIT_ZST=/path/to/RC_2019-04.zst
bash pipeline/run_pipeline.sh
```

The script runs Stages 3 to 5, derives the gun-control and abortion cluster ids
from the Stage 3 labels by keyword, and finishes with the stance charts. Stages
0 to 2 are run once by hand.

---

## Downstream

`../Hybrid-Network/data/init_agents.py` reads
`stance_scores/uncalibrated/cluster_<id>.parquet`, samples 50 agents per topic
stratified across the stance range, and writes their intrinsic opinions,
stubbornness coefficients and generated backgrounds into
`../Hybrid-Network/data/agents/`. From there the simulation takes over; see
`../Hybrid-Network/README.md`.
