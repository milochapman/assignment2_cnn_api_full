# Assignment 2 — CNN Theory (with Derivations)

**Output size formula (per spatial dimension):**
\[
\text{out} = \left\lfloor \frac{H + 2P - K}{S} \right\rfloor + 1
\]

**Q1**: Input 32×32×3; conv with 8 filters, 5×5 kernel, stride 1, **no padding**. Output size?  
`out = ⌊(32 − 5)/1⌋ + 1 = 27 + 1 = 28` → **28×28×8**.

**Q2**: If padding is **same** (stride 1), how does output size change?  
Spatial size is preserved → **32×32×8**.

**Q3**: Apply a 3×3 filter with stride 2 and **no padding** to a 64×64 input. Output spatial size?  
`out = ⌊(64 − 3)/2⌋ + 1 = ⌊61/2⌋ + 1 = 30 + 1 = 31` → **31×31**.

**Q4**: Max-pooling of size 2×2 with stride 2 on a 16×16 feature map. Output size?  
**8×8**.

**Q5**: Two successive conv layers on a 128×128 image, each 3×3, stride 1, **same** padding. Output shape?  
Spatial size unchanged each time → **128×128** (channel count depends on number of filters).

**Q6**: What happens if `model.train()` is removed before the training loop?  
Training-time behaviors are disabled for certain layers: **Dropout** won’t drop units, **BatchNorm** won’t update batch statistics (if left in eval semantics), leading to mismatch between training and inference behavior and degraded/unstable training.