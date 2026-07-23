# Tag Feature Block — Design

**Date:** 2026-07-23
**Component:** `notebooks/data_processing.ipynb` — tag feature engineering
**Status:** Approved

## Goal

Turn the parsed, rank-weighted tag data (`df['tags_dict']`) into a pruned,
weighted title×tag feature matrix that mirrors the existing genre one-hot
block, so it can later be hstacked with the genre and description blocks into
the content-based recommender's feature matrix.

## Context

- Prior work (committed): `parse_tag_weights()` builds `df['tags_dict']`, a
  column of `{tag_name: rank/100}` dicts (rank normalized to 0–1). Tags are
  stored **unfiltered** in the CSV so the pruning cutoff stays tunable in the
  notebook without re-hitting the AniList API.
- Data shape: 5000 titles, **378 unique tags**, mean **8.9 tags/title**.
  **41.6%** of titles have no tag data at all (all-zero tag vector).
- The genre block (`genre_dummies`) established the pattern this block follows:
  one-hot/weighted DataFrame → drop rare columns → `joblib.dump` the vocabulary
  to `artifacts/` → prefix columns → hstack later.

## Decisions

1. **Vocabulary pruning: 2% document-frequency floor.** Keep tags appearing in
   ≥ 100 titles (2% of 5000) → **96 tags**. Denser and lower-noise than the
   genre-matching 1% floor (which would keep 154).
2. **No blocklist.** Keep all 96 tags, including format/demographic tags
   (`Full Color`, `Long Strip`, `Male Protagonist`, etc.). Rank weighting plus
   the recommender are relied on to discount low-signal tags. Revisit only if
   recommendations turn out format-driven.
3. **Weighted, not binary.** Matrix cells hold the normalized rank weight
   (0–1), preserving the signal that `parse_tag_weights()` deliberately kept.
   This is the key difference from the binary genre block.
4. **Build the block now; defer cross-block weighting.** Build the weighted tag
   matrix in this work. The cross-block normalization scheme is *recorded* but
   *not applied* here, because the description block does not exist yet.

## Cross-block weighting scheme (recorded, applied later)

At final content-matrix assembly (separate future work), when genres + tags +
descriptions are hstacked:

- **L2-normalize each block per row** so no block dominates similarity purely
  by column count (genres ≈ 10 cols vs. tags 96 cols).
- Then apply an **optional scalar weight per block** as a tunable
  hyperparameter.

This is explicitly out of scope for the tag-block work.

## Approach (chosen: pandas)

`pd.DataFrame(df['tags_dict'].tolist(), index=df.index).fillna(0.0)` expands the
list of dicts into a dense title×tag frame; missing tags become 0.0, present
tags keep their rank weight. This mirrors the `genre_dummies` DataFrame idiom
exactly (prune columns, `tag_`-prefix, dump `.columns`, hstack). At 5000×96
dense (~4 MB) memory is a non-issue.

Rejected: `DictVectorizer` (emits sparse, foreign to the dense-DataFrame
pipeline, no frequency pruning) and `MultiLabelBinarizer` (binary-only, discards
the weights).

## Implementation — cells appended after the sparsity-audit cell (18)

1. **Compute tag document-frequency** across `df['tags_dict']`; build the kept
   set (tags in ≥ 100 titles). Print kept/dropped counts, paralleling the genre
   "Dropped N rare genres" cell.
2. **Build weighted matrix** —
   `tag_weights = pd.DataFrame(df['tags_dict'].tolist(), index=df.index).fillna(0.0)`,
   then subset to the kept columns.
3. **Save vocabulary** —
   `joblib.dump(tag_weights.columns.tolist(), '../artifacts/tag_columns.joblib')`,
   parallel to `genre_columns.joblib`, for building inference vectors.
4. **Prefix columns** — rename to `tag_{name}` so blocks stay identifiable
   after hstack.
5. **Sanity check** — recount titles with an all-zero tag row after pruning
   (baseline 41.6%; expected to tick up as titles whose only tags were
   low-frequency lose them). Confirms nothing silently broke.
6. **Markdown cell** recording the cross-block weighting scheme above, so the
   decision is captured at the point it will be applied.

## Out of scope

- The hstack / cross-block L2 normalization and scalar weighting (assembly-time
  work).
- The description feature block.
- Any recommender / similarity computation.

## Success criteria

- `tag_weights` is a 5000-row DataFrame with 96 `tag_`-prefixed columns holding
  0–1 weights.
- `artifacts/tag_columns.joblib` exists with the 96-tag vocabulary.
- Sanity-check cell prints the post-prune all-zero-row count without error.
- The genre block and prior cells are untouched.
