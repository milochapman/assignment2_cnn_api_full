# Assignment 2 — Part 3: CNN Theory (with Derivations)

This section shows the formula I used, the step-by-step substitutions, and the final answers.

---

## General Output-Size Formula (per spatial dimension)

For a 2D convolution/pooling with input height/width \(H\), kernel \(K\), stride \(S\), and padding \(P\):
\[
H_{\text{out}}=\Big\lfloor \frac{H + 2P - K}{S} \Big\rfloor + 1,
\quad
W_{\text{out}} \text{ similarly.}
\]

- **Channels:** After a *convolution*, the output channel count equals the **number of filters**.
- **Pooling:** Pooling **does not change** the number of channels.

> Note: The floor \(\lfloor\cdot\rfloor\) matters when \((H + 2P - K)\) is not divisible by \(S\).  
> For “**same**” padding with \(S=1\), the spatial size is preserved.

---

## Q1
**Given:** Input \(32 \times 32 \times 3\); conv with **8** filters, \(K=5\), \(S=1\), \(P=0\).

**Derivation:**
\[
H_{\text{out}}=W_{\text{out}}
= \Big\lfloor \frac{32 + 0 - 5}{1} \Big\rfloor + 1
= \lfloor 27 \rfloor + 1
= 28.
\]

**Answer:** \(\boxed{28 \times 28 \times 8}\).  
(*8 comes from the number of filters.*)

---

## Q2
**Change:** padding \(\to\) “same”, still \(S=1\).

**Reasoning:** With “same” and stride 1, padding is chosen to keep spatial size.

**Answer:** \(\boxed{32 \times 32 \times 8}\).

---

## Q3
**Given:** Input \(64 \times 64\); conv \(K=3\), \(S=2\), \(P=0\). (Spatial size only.)

**Derivation:**
\[
H_{\text{out}}=W_{\text{out}}
= \Big\lfloor \frac{64 - 3}{2} \Big\rfloor + 1
= \lfloor 30.5 \rfloor + 1
= 31.
\]

**Answer:** \(\boxed{31 \times 31}\).

---

## Q4
**Given:** MaxPool \(2 \times 2\), \(S=2\) on a \(16 \times 16\) feature map.

**Derivation:**
\[
H_{\text{out}}=W_{\text{out}}
= \Big\lfloor \frac{16 - 2}{2} \Big\rfloor + 1
= \lfloor 7 \rfloor + 1
= 8.
\]

**Answer:** \(\boxed{8 \times 8}\).  
(*Pooling does not change channels.*)

---

## Q5
**Given:** Two successive conv layers on a \(128 \times 128\) input; each with \(K=3\), \(S=1\), padding = “same”.

**Reasoning:** Each “same” conv with stride 1 preserves spatial size; two in a row still preserve it.

**Answer:** \(\boxed{128 \times 128}\).  
(*Channel count depends on the number of filters in those layers; not specified here.*)

---

## Q6
**Question:** What happens if we remove `model.train()` before the training loop?

**Role of `model.train()`:** Switches the model to **training mode**:
- **Dropout**: Enabled in train mode (randomly zeroes activations), disabled in eval mode.
- **BatchNorm**: Uses **mini-batch** statistics and **updates** running stats in train mode; in eval mode it uses frozen running means/vars and *does not* update them.

**Consequence of removing it:**
- If the model was previously put in `eval()` (e.g., after validation), it may **stay in eval mode**: Dropout won’t drop, BatchNorm won’t update running stats → mismatch between train/inference behaviors, degraded/unstable training.
- Even if `eval()` wasn’t called, being explicit with `model.train()` is good hygiene to avoid state leakage.

> Clarification: `model.train()` does **not** directly control gradient computation (that’s governed by `requires_grad` and contexts like `torch.no_grad()`). It toggles module behaviors (Dropout/BatchNorm).

---
