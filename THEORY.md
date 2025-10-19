# Assignment 2 — Part 3: CNN Theory (Derivations)

This section shows the formula used, the step-by-step substitutions, and the final answers.

---

## General Output-Size Formula (per spatial dimension)

For a 2D convolution/pooling with input size H (or W), kernel K, stride S, padding P:

    out = floor((H + 2P - K) / S) + 1

Notes:
- After a **convolution**, the output **channels** = number of **filters**.
- **Pooling** does not change the number of channels.

---

## Q1
**Given:** input 32×32×3; conv with 8 filters, K=5, S=1, P=0  
**Compute (per spatial dim):**

    out = floor((32 + 0 - 5) / 1) + 1
        = floor(27) + 1
        = 28

**Answer:** 28×28×8

---

## Q2
**Change:** padding → "same", still S=1  
**Reasoning:** "same" with stride 1 preserves spatial size.  
**Answer:** 32×32×8

---

## Q3
**Given:** input 64×64; conv K=3, S=2, P=0 (spatial only)  
**Compute:**

    out = floor((64 - 3) / 2) + 1
        = floor(61 / 2) + 1
        = floor(30.5) + 1
        = 31

**Answer:** 31×31

---

## Q4
**Given:** max-pool K=2, S=2 on 16×16  
**Compute:**

    out = floor((16 - 2) / 2) + 1
        = floor(7) + 1
        = 8

**Answer:** 8×8  (channels unchanged)

---

## Q5
**Given:** two successive convs on 128×128; each K=3, S=1, padding="same"  
**Reasoning:** each layer preserves spatial size with S=1 and "same" padding.  
**Answer:** 128×128  (channel count depends on #filters, not specified)

---

## Q6
**Question:** What if `model.train()` is removed before the training loop?

- `model.train()` switches modules to **training mode**:
  - **Dropout**: enabled in train mode; disabled in eval mode.
  - **BatchNorm**: uses mini-batch stats and **updates** running mean/var in train; eval uses **frozen** running stats and does not update.
- If you omit it (especially after having called `eval()` during validation), the model can remain in **eval mode**:
  - Dropout won’t drop; BatchNorm won’t update → mismatch between training and inference behavior → degraded/unstable training.
- Note: `model.train()` does **not** control gradient computation (that’s via `requires_grad` / `torch.no_grad()`); it toggles module behavior.

---
