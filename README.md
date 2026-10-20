# STOOD-X

Reference implementation of **STOOD-X**, a two-step protocol that decides whether an input is out of distribution and, at the same time, explains which features drove that decision.

> Sevillano-García, I., Luengo, J., Herrera, F. (2026).
> **STOOD-X: Explainable out-of-distribution detection via nonparametric statistical testing on large-scale datasets.**
> *Pattern Recognition*, 177. [doi:10.1016/j.patcog.2026.113254](https://doi.org/10.1016/j.patcog.2026.113254)

The test does not train a detector and does not assume a distribution for the scores. It compares the distance from the input to a reference set of in-distribution samples against the distances among the reference samples themselves, using a Bayesian comparison of two samples with a region of practical equivalence derived from the observed differences.

## How it works

1. **Representation.** A classifier is wrapped by `FeatureExtractor`, which records the activations of one internal layer and normalises them by their maximum absolute value. `feature_activations()` additionally multiplies them by their relevance, so the representation reflects what the layer contributes to the prediction.
2. **Comparison.** `OODTest` stores the representation of a validation set. For a new input it keeps the `k_neighbors` closest reference samples, and for each of them their own `k_neighbors`; then it runs a paired Bayesian test between the distances from the input and the distances among reference samples. The output is, per neighbour, the posterior probability that the input is farther than, equivalent to, or closer than the reference set's own members.

Feature selection is built in: with `n_features`, only the features with the largest cumulative presence in the reference subset are kept, and the rest are zeroed in both sides of the comparison.

## Install

```bash
git clone https://github.com/isega24/STOOD-X.git
cd STOOD-X
pip install -e .
```

`torch`, `torchvision` and `zennit-crp` are heavy. For a CPU-only install, install torch first from the CPU index and then the package.

## Quickstart

```python
import torch, torchvision
from stoodx import FeatureExtractor, OODTest

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.resnet18(weights="DEFAULT")
extractor = FeatureExtractor(model=model, device=device, feature_name="avgpool")

test = OODTest(extractor, k_neighbors=10)

# 1) build the reference set from in-distribution samples
for batch, _ in in_distribution_loader:
    test.add_features(batch.to(device))

# 2) score a single sample (no batch dimension)
x = next(iter(test_loader))[0][0].to(device)
scores = test.test(x, n_features=256, intraclass=True)
print(scores)
```

Attribution needs gradients, so do **not** wrap the calls above in `torch.no_grad()`.

### Output

One row per neighbour, three columns:

| column | meaning |
|---|---|
| `d(x,VAL)>d(x_i,VAL)` | posterior probability that the input is **farther** from the reference set than its members are from each other — evidence for out-of-distribution |
| `d(x,VAL)~d(x_i,VAL)` | posterior probability that the two distances are **equivalent** within the region of practical equivalence |
| `d(x,VAL)<d(x_i,VAL)` | posterior probability that the input is **closer** than the reference members |

Aggregate the rows into a single score before comparing against a baseline; the
raw table is per neighbour on purpose, because the disagreement across neighbours
is itself informative.

### Constructor arguments

`FeatureExtractor(model, device, feature_name, composite=None)`

- `feature_name` — the layer to record. `"avgpool"` for the torchvision ResNet and EfficientNet families, `"encoder"` for a Vision Transformer (the class token is kept).
- `composite` — zennit composite for the attribution. `None` uses the zennit-crp default, the LRP configuration used in the paper.

`OODTest(model, distance=..., k_neighbors=10)`

- `distance` — callable over two feature vectors. Default is the Euclidean norm over the feature dimension.
- `k_neighbors` — neighbours per comparison.

`test(x, n_features=None, intraclass=True)`

- `x` — one sample, without batch dimension.
- `n_features` — features kept by presence; `None` keeps all.
- `intraclass` — restrict the reference subset to the predicted class.

## Scope and status

This is **research code**, extracted from the experimental repository used for the paper. It ships the method, not the experiment: datasets, trained checkpoints, the dataset grid and the sweep scripts are not included. The API is enough to reproduce the protocol and to rerun it on your own data.

Deviations from the original experimental code, all of them fixes needed to make the entry points run:

- the package is `stoodx` (was `XAIOODTest`), `FeatureExtractor` (was `FeatureEstractor`) and the public methods are snake_case;
- the reference-subset filter compared integer labels against the prediction's logits inside an `or`, which raised on any tensor with more than one element; it now compares against the predicted class index, and raises a clear error when that class is absent from the reference set;
- `test()` zeroed the stored reference features in place, so a second call compared against corrupted features; it now works on a copy;
- the default distance used `dim=1`, which raises for the one-dimensional feature vectors this code produces; it is now the norm over the last dimension;
- `attribute()` passed `record_layer` as a string, which `zennit-crp` 0.6 rejects, and returned the layer's raw shape `(B, C, H, W)` although its own docstring promises `(B, F)`; it now passes a list and flattens, which is the contract `OODTest` relies on;
- **the comparison loop was rewritten.** The original built a neighbour-distance matrix, sorted it per column, sliced the result to `k-1` rows and then indexed it `k` times, so it raised `IndexError` on every input and never executed. The current version follows the protocol the original docstring describes: take the `k` nearest reference samples to the input, then each of those neighbours' own `k` nearest samples, and run the Bayesian comparison per neighbour. If you are reproducing the paper's numbers, review this function against the experiment code before trusting them;
- empty modules and tests that only asserted `True` were removed, and the licence declared in the packaging metadata now matches the `LICENSE` file.

Known limitation: the KNN search is a full pairwise loop, so the reference set should stay in the thousands, not the millions. The paper's large-scale results come from the experiment pipeline, not from this loop.

## Citation

```bibtex
@article{sevillano2026stoodx,
  title   = {STOOD-X: Explainable out-of-distribution detection via nonparametric
             statistical testing on large-scale datasets},
  author  = {Sevillano-Garc{\'i}a, Iv{\'a}n and Luengo, Juli{\'a}n and Herrera, Francisco},
  journal = {Pattern Recognition},
  volume  = {177},
  year    = {2026},
  doi     = {10.1016/j.patcog.2026.113254}
}
```

`CITATION.cff` is included for GitHub's "Cite this repository".

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

The `OODTest` tests run on a stub extractor and need no weights. The `FeatureExtractor` test downloads torchvision weights and is skipped when that is not possible.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).

The Bayesian comparison is delegated to [baycomp](https://github.com/janezd/baycomp) (Benavoli et al., 2017); the relevance propagation to [zennit](https://github.com/chr5tphr/zennit) and [zennit-crp](https://github.com/rachtibat/zennit-crp).
