# DS605 - Lab 5: ML with Scikit-learn and From Scratch

Dataset: [UCI Garment Employee Productivity dataset](https://archive.ics.uci.edu/dataset/597/productivity+prediction+of+garment+employees)

Predicting `actual_productivity` (regression) and `MeetsTarget` (classification, a
column I made myself: 1 if `actual_productivity >= targeted_productivity` else 0).

Three versions, all in `lab5.ipynb`:
- Part A - scikit-learn
- Part B - from scratch, numpy/pandas only, first attempt
- Part C - from scratch again, cleaned up

## Running it

```
pip install -r requirements.txt
jupyter notebook lab5.ipynb
```

Run top to bottom (or Run All) - Part C's comparison table needs variables from A and B.

## Cleaning notes

- `wip` missing in ~42% of rows -> filled with median.
- `department` had `finishing` and `finishing ` (trailing space) as separate values by
  mistake, just stripped it.
- Dropped `date`, kept `quarter`/`day` for the time info.
- `team` (1-12) left as a number instead of one-hot, it's more of an ID than a category.

## Split

Manual - shuffle indexes with seed 42, first 20% is test. Same split function called
in every part so A/B/C all train and test on the same 958/239 rows. No sklearn split
function used, even in Part A.

## Results

| Metric | A (sklearn) | B (scratch) | C (scratch, optimized) |
|---|---|---|---|
| MAE | 0.1090 | 0.1090 | 0.1083 |
| RMSE | 0.1484 | 0.1484 | 0.1483 |
| R2 | 0.1735 | 0.1735 | 0.1748 |
| Accuracy | 0.7490 | 0.7490 | 0.7490 |
| Precision | 0.7685 | 0.7636 | 0.7710 |
| Recall | 0.9432 | 0.9545 | 0.9375 |
| F1 | 0.8469 | 0.8485 | 0.8462 |

Timing isn't in the table since it jumps around a bit each run, but it's printed in the
notebook - regression is basically instant everywhere, classification training goes from
~0.005s (sklearn) to ~0.08s (Part C), since Part C runs way more gradient descent steps.

## Part C changes

- Fixed a leak from Part B - median/mean/std were computed on the whole dataset before
  splitting, now computed from train only.
- Dropped `idle_time` / `idle_men` - both 0 in ~98.5% of rows, barely correlated with
  the target.
- Added ridge (L2) to linear regression, small R2 bump (0.1735 -> 0.1748).
- Logistic regression: more epochs (1000->5000) and higher lr (0.1->0.5) so it actually
  converges instead of stopping early - gets closer to sklearn's numbers.

## Notes

Regression matches sklearn almost exactly, same normal equation either way. R2 stays low
(~0.17) - not a bug, this dataset just isn't very linear. Logistic regression is where
scratch and sklearn actually diverge, since gradient descent doesn't land on the exact
same weights as sklearn's solver unless run long enough - accuracy stays at 0.7490 in
all 3 but precision/recall shift a bit. Scratch logistic is also a lot slower (plain
loop vs sklearn's solver) - that's the speed-for-accuracy tradeoff the lab's about.

`MeetsTarget` is ~73/27, not balanced, hence tracking precision/recall/F1 too.
