# EdgeAttNet: a purpose-built barb-aware architecture, reimplemented cleanly

## What and why

Reimplemented from "EdgeAttNet: Towards Barb-Aware Filament Segmentation" (arXiv:2509.02964), which reports
improved barb recognition over plain U-Net baselines, evaluated directly on MAGFiLO. This is architecturally
unrelated to V1-V5 (Mask R-CNN, instance segmentation) -- it's the "regardless of price, what's our best option"
answer: more data (not pursued here) plus a model actually designed for thin, branching structures, rather than
a generic detector adapted to them.

**Reimplemented from the paper's architecture description, NOT from the authors' released weights**
(github.com/dasjar/EdgeAttNet). Their reported split (1,295 train / 45 val / 99 test of 1,439 filtered MAGFiLO
images) almost certainly overlaps heavily with our competition's held-out test set -- we already know 77% of our
test images are in the public MAGFiLO release. Using their weights would risk the same contamination we've
deliberately avoided throughout this project. We train fresh, on only our own permitted 601/106 split.

## Architecture

4-stage U-Net (channels 64/128/256/512, each stage its own 2x2 maxpool -> 16x reduction at the bottleneck). A
tiny edge branch (3x3 conv + sigmoid on the raw image) is resized to the bottleneck and added into the
self-attention Query and Key (not Value): `Q = K = bottleneck_features + edge_prior`, `V = bottleneck_features`,
4 heads x 128 dim. Loss: BCE + Dice. ~11.1M parameters.

**This is semantic segmentation, not instance segmentation** -- one binary "is this pixel part of any filament"
mask per image, no notion of separate filaments. `edgeattnet_predict.py` converts this into scored instances via
connected-component labelling (each component = one predicted filament, 8-connectivity) with each instance's
score set to the mean pixel probability inside it -- a design decision the paper doesn't make for us, since
semantic segmentation has no instance-confidence concept. One real consequence: two real, separate filaments
that touch in the binary mask get merged into one prediction here, a different failure mode from Mask R-CNN's
tendency to fragment one filament into several -- not obviously better or worse without testing.

## Two real bugs caught while reimplementing (both fixed and verified)

1. Initially built with only 3 pooling stages before the bottleneck (8x reduction) instead of 4 (16x, as the
   paper states) -- caused a 64GB attention-matrix allocation to crash immediately.
2. After adding the 4th stage, forgot to actually pool its output before attention -- same reduction bug, one
   level later. Fixed by explicitly pooling `e4` before it reaches the attention module.

## Resolution: the paper doesn't say what it trained at, and it likely matters

Full self-attention over the bottleneck is quadratic in token count. At our native 2048x2048 resolution, that's
4.3GB for the attention weights alone (128x128 = 16,384 tokens), before backprop storage or the rest of the
network -- confirmed real by an instant OOM in local testing (this Mac's MPS backend has ~30GB total). The paper
never states its training resolution and almost certainly used something much smaller. We're testing at native
resolution first on the rented H100 (80GB) rather than presuming we need to downsize and lose the fine barb
detail that's the whole point of this architecture -- `train.resize` in the config is a tested fallback if the
GPU probe shows otherwise.

## Status

Every component built and independently tested for correctness (model shapes/gradients, dataset construction,
connected-components conversion, full training loop, PQ-evaluation glue code) -- all locally verified except
real-resolution GPU feasibility, which needs actual hardware to answer honestly rather than guess at.

```bash
python scripts/train_edgeattnet.py --config EdgeAttNet/configs/edgeattnet.yaml
```
