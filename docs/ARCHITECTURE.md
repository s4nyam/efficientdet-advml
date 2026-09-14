# Architecture

A one-stage detector is three parts: a **backbone** that sees, a **neck** that
mixes scales, a **head** that decides. DeepSeaNet keeps EfficientDet's recipe
and changes the neck.

```
frame 960x540
      |
      v
  [ backbone ]   EfficientNet-style MBConv blocks, Swish
      |          emits P3 P4 P5 P6 P7  (strides 8 16 32 64 128)
      v
  [   neck   ]   BiFPN  ->  BiSkFPN     <- the proposed change
      |          bidirectional fusion + deconv branch + skip branch
      v
  [   head   ]   class subnet  : which species
                 box subnet    : where, how big
      |
      v
  decode + NMS -> detections
```

---

## Backbone — `models/backbone.py`

`MBConv` (mobile inverted bottleneck) is the repeating unit: a 1×1 expansion by
factor `T`, a depthwise `k×k` convolution, squeeze-and-excitation, a 1×1
projection back down, and a residual shortcut when the shapes allow it. The
depthwise convolution is where the parameter saving comes from — it filters
each channel independently instead of mixing all of them.

`BackboneConfig.from_phi(phi)` applies EfficientNet's compound scaling rule:
width `1.2^phi`, depth `1.1^phi`.

**This is a reference backbone, not a re-implementation of EfficientNet-B0.**
It exists so the neck and head can be exercised and tested end to end. For real
training, pass a pretrained backbone:

```python
from deepseanet.models import DeepSeaNet, DeepSeaNetConfig

model = DeepSeaNet(DeepSeaNetConfig(num_classes=6), backbone=my_pretrained_backbone)
```

Any module works as long as it returns a list of feature maps (finest first)
and exposes an `out_channels` attribute.

### Swish

`Swish(x) = x · σ(βx)`, with `β` optionally learnable. Against ReLU it is
smooth and keeps a non-zero gradient for small negative inputs, so weak noisy
activations are **damped rather than clipped to zero**. On murky frames a large
share of activations sit near zero, which is exactly where the two differ.

`swish_derivative` gives the analytic form `σ(βx) + βx·σ(βx)(1−σ(βx))`; a test
checks it against autograd.

---

## Neck — `models/biskfpn.py`

### BiFPN, the baseline

A top-down pass followed by a bottom-up pass. Every input edge carries a
learned scalar weight, normalised by fast fusion:

```
O = Σᵢ  ReLU(wᵢ) / (ε + Σⱼ ReLU(wⱼ))  ·  Iᵢ
```

The ReLU keeps weights non-negative so the normalisation is stable — that is
what makes it "fast" fusion rather than a softmax.

### BiSkFPN, the proposal

Each output node is built from **three** sources instead of one:

```
BiSkFPN(P)ᵢ = concat( Pᵢ ,  deconv(Pᵢ₊₁) ,  skip_{i→i+1}(Pᵢ₋₁) )
```

| Branch | What it is | Why |
|---|---|---|
| `Pᵢ` | the level itself, after BiFPN fusion | the baseline signal |
| `deconv(Pᵢ₊₁)` | the coarser level, up-sampled by a **transposed convolution** | the filter is learned, so some detail lost in down-sampling is recovered — unlike a parameter-free resize |
| `skip(Pᵢ₋₁)` | the finer level, strided across, **bypassing the deconv branch** | low-level cues (edges, small blobs, the few dozen pixels of a shrimp) reach the head without surviving every fusion stage |

The concatenation is projected back to `channels` with a separable 1×1
convolution. Boundary levels have no `Pᵢ₊₁` or no `Pᵢ₋₁`; those branches are
dropped and the fuse convolution is sized accordingly.

The robustness argument: when the input is perturbed, the fine-detail path does
not have to survive repeated re-mixing to influence the prediction.

**Cost.** At `phi=0` the neck grows from 187 K to 1.26 M parameters — about
1.07 M extra, 12% of the whole model. Run the ablation before assuming it pays
for itself on your data:

```bash
deepseanet model --neck bifpn   --image-size 512
deepseanet model --neck biskfpn --image-size 512
```

### Skip connections against residual connections

A residual connection **adds** the input to the output; a skip connection
**concatenates** them. Addition forces the two signals to share a channel
budget and lets them cancel; concatenation keeps both intact and lets the next
convolution decide how to weigh them. That flexibility costs parameters, which
is the trade being made here.

---

## Head — `models/head.py`

Two subnets, applied to every pyramid level with **shared convolution weights
but per-level batch-norm statistics**. Sharing weights is what makes the head
cheap (39 K parameters); keeping batch-norm separate stops a level with an
unusual activation scale from dragging the others.

The classification subnet's final bias is initialised to
`−log((1−π)/π)` with `π = 0.01`, so training starts with a low objectness
prior. Without it, the tens of thousands of background anchors swamp the first
few hundred steps.

`AnchorGenerator` produces anchors in `(x1, y1, x2, y2)` pixels: 3 aspect
ratios × 3 scales = 9 per location, base size `4 × stride`.

---

## Loss — `losses.py`

```
L = L_cls + α · L_box + β · L_reg
```

| Term | Form | Notes |
|---|---|---|
| `L_cls` | sigmoid focal loss | `α=0.25`, `γ=2`. Reduces to weighted BCE at `γ=0`. |
| `L_box` | smooth L1 on encoded offsets | computed over positive anchors only |
| `L_reg` | mean squared L2 norm of weights | skips biases and norm parameters |

Both `L_cls` and `L_box` are normalised by the number of positive anchors, so
the loss scale does not depend on how crowded a frame is. The committed
EfficientDet run logged `α = 50`, `β = 1`.

---

## Attacks — `attacks/uap.py`

A universal adversarial perturbation is one image-agnostic pattern added to
every frame.

- `random_uap` — the closed form printed in the paper,
  `U = ξ · sign(Σ rᵢ · δᵢ)`: a fixed random sign pattern at amplitude `ξ`.
  Cheap and model-independent, fine as a noise augmentation.
- `optimize_uap` — the actual attack. Gradient ascent on real frames, projected
  back onto the `ℓ∞` or `ℓ2` ball after each step. Use this when **measuring**
  robustness; a random pattern is a weak adversary and will flatter the model.
- `curriculum_epsilon` — keeps the first 20% of training clean, then ramps `ε`
  to its maximum. Training at full strength from step zero tends to stall,
  because the model never gets a clean signal to learn the task from.

---

## Explainability — `explain/gradcampp.py`

`GradCAMPlusPlus` implements

```
Lᶜ(x, y) = ReLU( Σₖ αₖᶜ · Aₖ(x, y) )
```

with `αₖᶜ` from the standard closed form in gradient powers. For detectors the
output is a tuple, so pass a `score_fn` to reduce it to one scalar per image:

```python
with GradCAMPlusPlus(model, model.neck.layers[-1]) as cam:
    heat = cam(images, score_fn=lambda out: out[0][..., class_id].amax(dim=1))
```

`EigenCAM` is also provided, because that is what the original notebooks
actually ran — it projects activations onto their first principal component,
needs no gradients and no class score, and therefore shows *where the layer is
active* rather than *what drove a prediction*. Having both makes the comparison
explicit instead of accidental.
