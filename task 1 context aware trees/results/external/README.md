# External development comparisons

All methods are evaluated on identical common genomic sites within each panel. `metrics_deployed_fit.csv` describes the released seed-4262 fit. `metrics_by_seed.csv` contains three independent training seeds; `metrics_mean.csv` reports their arithmetic mean. Run `python evaluate_external_panels.py` from the Task 1 folder to regenerate these tables from the included per-site scores.

The panels comprise mouse ELIGOS MeRIP-seq, miCLIP and miCLIP2 labels; mouse Zhong GLORI labels; and two human HeLa comparisons of supported native positives against IVT controls at the same loci. The HeLa comparisons test native/control separation and are not general human native-cell classification tests. The three ELIGOS assays share one input run, and both HeLa comparisons share one IVT control.

The final three-seed mean exceeds released m6Anet in ROC AUC and trapezoidal PR AUC on all six panels. This is a numerical result on these development panels, not a claim of statistical significance or universal superiority. Compared with the original 42-feature model, PR AUC improves on all six panels; MeRIP-seq ROC AUC decreases. Individual-seed results and the deployed fit remain visible separately.

External labels were consulted during feature/recipe/blend selection. Model parameters were fitted only using Datasets 0–2. A later untouched evaluation is needed for an unbiased estimate of performance after this selection. SingleMod scores are included where available, with training-overlap limitations; they are not treated as independent evidence of human transfer.

PR AUC uses trapezoidal integration; average precision is reported separately as `ap`. m6Anet ranking uses negative log survival to avoid numerical saturation. `protocol.json` records the source accessions, sample sizes, alignment rules and interpretation of each panel. The compressed tables retain genomic coordinates, labels, gene groups and every compared score.
