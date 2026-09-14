# Model card — DeepSeaNet

## Overview

| | |
|---|---|
| **Task** | Object detection, six marine classes |
| **Domain** | Underwater video, brackish coastal water, fixed camera |
| **Input** | RGB frames, 960×540 native; 320²–800² at train time depending on run |
| **Output** | Bounding boxes with class and confidence |
| **Paper** | ICAPAI 2024, IEEE. [doi:10.1109/ICAPAI61893.2024.10541265](https://doi.org/10.1109/ICAPAI61893.2024.10541265) |
| **Licence** | MIT (code). Dataset licensed separately by Aalborg University. |

## Intended use

Research on underwater detection, adversarial robustness and explainability.
Suitable as a baseline or starting point for marine monitoring work.

**Not suitable, without revalidation on your own data, for:**

- Navigation, collision avoidance or any safety-critical decision.
- Population counts or ecological claims. Detection counts are not abundance
  estimates; they carry all the biases below.
- Deployment at a different site, depth, season or camera. Every frame in
  training came from one fixed rig in Limfjorden.

## Training data

Brackish dataset (Pedersen et al., CVPR Workshops 2019). ~10,000 annotated
frames after preprocessing, six classes, one recording site. See
[`DATASET.md`](DATASET.md).

## Metrics

See [`RESULTS.md`](RESULTS.md) for the paper's tables and the committed runs'
logs, which differ; [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) explains why.

## Limitations and biases

**Split leakage inflates all published numbers.** Frames from continuous video
are near-duplicates. The published 70:20:10 split is random, so near-identical
frames appear in both training and test. Use `--strategy grouped` for honest
estimates and expect lower scores.

**Single site and camera.** No evidence of transfer to other water bodies,
depths, lighting conditions or hardware.

**Class imbalance.** Shrimp and jellyfish are 3–4% of validation boxes. A high
mean mAP can coexist with poor performance on exactly the rare species that
monitoring programmes care most about.

**Small objects are hardest.** `small_fish` is consistently the weakest class
across every detector tested. Objects a few dozen pixels wide, moving between
frames, in low contrast.

**Visibility varies.** Performance degrades with turbidity. The models were not
evaluated stratified by visibility, so the reported means average over easy and
hard conditions without distinguishing them.

**Adversarial robustness is bounded.** Adversarial training against UAPs at a
chosen `ε` does not confer general robustness. A stronger or differently
constructed attack may still succeed.

## Ethical considerations

Marine monitoring models can inform fishing quotas, conservation decisions and
protected-area enforcement. Overstated accuracy has real downstream cost.
Report per-class numbers, state the split strategy, and validate on the water
you intend to monitor before any of it informs a decision.

## Maintenance

Maintained as a research artefact accompanying a published paper. Issues and
pull requests welcome; see [`../CONTRIBUTING.md`](../CONTRIBUTING.md).
