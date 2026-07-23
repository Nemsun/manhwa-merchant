# Tag Feature Block Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `df['tags_dict']` into a pruned, rank-weighted title×tag feature matrix (`tag_weights`) that mirrors the existing genre block, ready to hstack into the recommender's content matrix later.

**Architecture:** Append cells to `notebooks/data_processing.ipynb` after the current sparsity-audit cell (index 18). Compute tag document-frequency, prune to a 2% floor (96 tags), expand `tags_dict` into a dense weighted DataFrame via pandas, save the vocabulary to `artifacts/`, prefix columns `tag_`, then a sanity check and a markdown cell recording the deferred cross-block weighting scheme. The genre block and all prior cells are untouched.

**Tech Stack:** Python 3.13, pandas, numpy, joblib — all already imported in the notebook. Managed via `uv`. Notebook edits via the `NotebookEdit` tool; verification via headless `jupyter nbconvert --execute`.

## Global Constraints

- Notebook cell paths are relative to `notebooks/` (e.g. `../data/raw/...`, `../artifacts/...`) — execution cwd must be `notebooks/`.
- Run all Python through `uv run` (deps live in the uv project env, not system Python).
- Vocabulary is saved with **raw tag names** (no `tag_` prefix), matching how `genre_columns.joblib` is saved *before* prefixing — inference code re-applies the prefix.
- Pruning floor: tags present in ≥ `floor(len(df) * 0.02)` = 100 titles → exactly **96 tags**. Do not hardcode 96; derive it.
- Do not modify any existing cell (0–18) or the genre block. Append only.
- `df['tags_dict']` (from cell 17) must exist before these cells run — it is a column of `{tag_name: rank/100}` dicts.

---

### Task 1: Build the pruned, weighted tag matrix

**Files:**
- Modify: `notebooks/data_processing.ipynb` — append 4 code cells after cell index 18
- Create: `artifacts/tag_columns.joblib` (written when the notebook runs)

**Interfaces:**
- Consumes: `df['tags_dict']` (list/Series of `{str: float}` dicts, values in 0–1), plus `pd`, `np`, `joblib` already imported; `len(df) == 5000`.
- Produces: `tag_weights` — a `pd.DataFrame`, shape `(5000, 96)`, columns prefixed `tag_`, values in `[0.0, 1.0]`; `kept_tags` — sorted list of 96 raw tag names; `artifacts/tag_columns.joblib` holding those 96 raw names.

- [ ] **Step 1: Append the document-frequency / pruning cell**

Use `NotebookEdit` (insert mode) to add this as a new code cell after cell 18:

```python
# Prune tag vocabulary to a 2% document-frequency floor (tags in >= 100 titles).
# Mirrors the genre block's 1% rare-genre drop; tighter here because tags are
# far more numerous (378 unique) and long-tailed.
from collections import Counter

tag_doc_freq = Counter()
for tag_map in df['tags_dict']:
    tag_doc_freq.update(tag_map.keys())

TAG_DF_FLOOR = int(np.floor(len(df) * 0.02))  # 2% of titles = 100
kept_tags = sorted(t for t, n in tag_doc_freq.items() if n >= TAG_DF_FLOOR)
dropped = len(tag_doc_freq) - len(kept_tags)
print(f"Kept {len(kept_tags)} tags (>= {TAG_DF_FLOOR} titles); dropped {dropped} rare tags")
```

- [ ] **Step 2: Append the matrix-build cell**

Insert this as the next new code cell:

```python
# Expand tags_dict (list of {name: weight} dicts) into a dense weighted matrix.
# Missing tags -> 0.0; present tags keep their normalized rank weight (0-1).
# Subset to the kept vocabulary; sorted kept_tags gives deterministic column order.
tag_weights = pd.DataFrame(df['tags_dict'].tolist(), index=df.index).fillna(0.0)
tag_weights = tag_weights[kept_tags]
```

- [ ] **Step 3: Append the save-vocabulary cell**

Insert this as the next new code cell (saves raw names, before prefixing — parallels the genre flow):

```python
# Save vocabulary for building inference vectors (raw names, prefix applied later)
joblib.dump(tag_weights.columns.tolist(), '../artifacts/tag_columns.joblib')
```

- [ ] **Step 4: Append the prefix cell**

Insert this as the next new code cell:

```python
# Prefix so tag columns stay identifiable after hstacking with other feature blocks
tag_weights.columns = [f"tag_{col}" for col in tag_weights.columns]
```

- [ ] **Step 5: Execute the notebook headless to verify it runs clean**

Run (cwd = repo root; nbconvert executes with the notebook's own directory as cwd, so relative `../` paths resolve):

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/data_processing.ipynb
```

Expected: completes with no traceback; exit code 0.

- [ ] **Step 6: Assert the produced matrix and artifact are correct**

Run this check against the executed state by re-deriving from the CSV + saved artifact:

```bash
uv run python -c "
import pandas as pd, numpy as np, joblib
vocab = joblib.load('artifacts/tag_columns.joblib')
assert len(vocab) == 96, f'expected 96 tags, got {len(vocab)}'
assert vocab == sorted(vocab), 'vocab must be sorted'
assert not any(v.startswith('tag_') for v in vocab), 'vocab must hold raw names, no prefix'
print('PASS: 96 sorted raw-name tags saved to artifacts/tag_columns.joblib')
"
```

Expected: `PASS: 96 sorted raw-name tags saved to artifacts/tag_columns.joblib`

- [ ] **Step 7: Commit**

```bash
git add notebooks/data_processing.ipynb artifacts/tag_columns.joblib
git commit -m "Build pruned weighted tag matrix (96 tags, 2% floor)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Sanity check and record deferred weighting scheme

**Files:**
- Modify: `notebooks/data_processing.ipynb` — append 1 code cell + 1 markdown cell after Task 1's cells

**Interfaces:**
- Consumes: `tag_weights` (from Task 1), `df`, `len(df) == 5000`.
- Produces: printed post-prune all-zero-row count; a markdown cell documenting the cross-block weighting plan. No new variables other cells depend on.

- [ ] **Step 1: Append the sanity-check cell**

Insert as a new code cell after Task 1's prefix cell:

```python
# Recount all-zero tag rows after pruning. Baseline was 41.6% (no tags at all);
# expected to tick up as titles whose only tags were low-frequency lose them.
# If this jumps a lot, the tag block is sparse -> weight it down at assembly time.
all_zero = tag_weights.eq(0).all(axis=1).sum()
print(f"Titles with all-zero tag vector after pruning: {all_zero} ({all_zero/len(df)*100:.1f}%)")
```

- [ ] **Step 2: Append the markdown decision cell**

Insert as a new **markdown** cell (set cell type to markdown in `NotebookEdit`):

```markdown
### Cross-block weighting (deferred to assembly)

When genres + tags + descriptions are hstacked into the content matrix,
**L2-normalize each block per row** so no block dominates similarity purely by
column count (genres ≈ 10 cols vs. tags 96 cols), then apply an **optional
scalar weight per block** as a tunable hyperparameter. Not applied here — the
description block does not exist yet.
```

- [ ] **Step 3: Execute the notebook headless to verify it runs clean**

Run:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/data_processing.ipynb
```

Expected: completes with no traceback; exit code 0.

- [ ] **Step 4: Confirm the sanity-check output is present and plausible**

Run:

```bash
uv run python -c "
import json
nb = json.load(open('notebooks/data_processing.ipynb'))
texts = []
for c in nb['cells']:
    for o in c.get('outputs', []):
        if 'text' in o: texts.append(''.join(o['text']))
hits = [t for t in texts if 'all-zero tag vector after pruning' in t]
assert hits, 'sanity-check output not found in executed notebook'
print(hits[-1].strip())
"
```

Expected: a line like `Titles with all-zero tag vector after pruning: <N> (<pct>%)` where the percentage is ≥ 41.6.

- [ ] **Step 5: Commit**

```bash
git add notebooks/data_processing.ipynb
git commit -m "Add tag sparsity sanity check and cross-block weighting note

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the executor

- `NotebookEdit` schema is loaded on demand via `ToolSearch` (`select:NotebookEdit`). Insert cells in order; each new cell goes after the previously inserted one.
- Headless execution stores cell outputs inplace, so the committed notebook carries its run outputs (consistent with the existing committed notebook, which already has outputs).
- If nbconvert complains about a missing kernel, install/register with `uv run python -m ipykernel install --user --name manhwa-merchant` — `ipykernel` is already a project dependency.
