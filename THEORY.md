# Assignment 2 — CNN Theory (with Derivations & Rationale)

> This section shows the *process* I used to solve each item: the general formula, the substitution, sanity checks, and common pitfalls. Answers are highlighted at the end of each item.

---

## Quick Checklist

| Item | What I show | Done |
|---|---|---|
| Formula | Output-size formula with floor and parameters
| Channels | Note that conv output channels = #filters; pooling keeps channels
| Derivations | Step-by-step substitutions for Q1–Q5
| Sanity checks | “Does it make sense?” mini-checks per item
| Training mode | Clear explanation of `model.train()` effects

---

## General output-size formula (per spatial dimension)

For a 2D conv/pool with input height/width \(H\), kernel \(K\), stride \(S\), and padding \(P\):
\[
H_{\text{out}}
= \Big\lfloor \frac{H + 2P - K}{S} \Big\rfloor + 1,
\qquad
W_{\text{out}} \text{ similarly.}
\]

- **Channels:** After a **convolution**, the output channel count equals the **number of filters**.  
- **Pooling:** Pooling **does not change** channel count.

> **Edge cases:** The floor \(\lfloor\cdot\rfloor\) matters when \((H + 2P - K)\) is not divisible by \(S\).  
> For “**same**” padding with \(S=1\), spatial size is preserved. For “same” with \(S>1\), the preserved-size intuition no longer holds exactly; always use the formula.

---

## Q1
**Given**: Input \(32\times 32\times 3\); conv with **8** filters, \(K=5\), \(S=1\), \(P=0\).

**Derivation**
\[
H_{\text{out}}=W_{\text{out}}
= \Big\lfloor \frac{32 + 0 - 5}{1} \Big\rfloor + 1
= \lfloor 27 \rfloor + 1
= 28.
\]

**Answer**: \(\boxed{28 \times 28 \times 8}\).  
*(8 comes from the number of filters.)*

**Sanity check**: No padding shrinks each spatial dim by \(K-1=4\) with \(S=1\): \(32\to28\).

---

## Q2
**Change**: padding \(\to\) “same”, still \(S=1\).

**Reasoning**: With “same” and \(S=1\), padding is chosen to keep spatial size.

**Answer**: \(\boxed{32 \times 32 \times 8}\).

**Sanity check**: Channels unchanged from Q1 (still 8), size preserved (same + \(S=1\)).

---

## Q3
**Given**: Input \(64\times64\), conv \(K=3\), \(S=2\), \(P=0\). (Spatial size only.)

**Derivation**
\[
H_{\text{out}}=W_{\text{out}}
= \Big\lfloor \frac{64 - 3}{2} \Big\rfloor + 1
= \lfloor 30.5 \rfloor + 1
= 31.
\]

**Answer**: \(\boxed{31 \times 31}\).

**Note**: The floor is critical because stride 2 skips positions; odd remainders truncate.

---

## Q4
**Given**: MaxPool \(2\times2\), \(S=2\) on \(16\times16\).

**Derivation**
\[
H_{\text{out}}=W_{\text{out}}
= \Big\lfloor \frac{16 - 2}{2} \Big\rfloor + 1
= \lfloor 7 \rfloor + 1
= 8.
\]

**Answer**: \(\boxed{8 \times 8}\) (channels unchanged).

**Mnemonic**: \(2\times2\) pool with stride 2 halves the spatial size when the input is even.

---

## Q5
**Given**: Two successive convs on \(128\times128\), each \(K=3\), \(S=1\), **same** padding.

**Reasoning**: Each “same” conv with \(S=1\) preserves the spatial size; two in a row still preserve it.

**Answer**: \(\boxed{128 \times 128}\) (channel count depends on the number of filters in those layers).

**Sanity check**: If either layer had no padding, size would shrink by 2 across the two layers: \(128\to126\to124\).

---

## Q6
**Question**: What if we remove `model.train()` before the training loop?

**Role of `model.train()`**: Switches the module to **training mode**:
- **Dropout**: Enabled in train mode (randomly zeroes activations); disabled in eval mode.
- **BatchNorm**: Uses **mini-batch** statistics and **updates** running stats in train mode; in eval it uses frozen running means/vars and does **not** update them.

**Consequence of removing it**:
- If the model was previously put in `eval()` (e.g., after validation), the network may **stay in eval mode**.  
- Then **Dropout won’t drop**, **BatchNorm won’t update** running stats, producing a mismatch between “training” and “inference” behaviors → degraded or unstable training, sometimes failure to converge.
- Even if you *didn’t* call `eval()`, being explicit with `model.train()` is good hygiene to avoid state leaks.

> **Clarification**: `model.train()` does **not** directly control gradient computation; that’s governed by `requires_grad` and context managers like `torch.no_grad()`. Its job is to toggle modules (Dropout/BatchNorm) into their training behavior.

---

## Common pitfalls & quick checks
- Forgetting that output **channels** equal the number of **filters**.
- Assuming “same” keeps size for \(S>1\) — not necessarily; always fall back to the formula.
- Ignoring the **floor** in the formula; stride often introduces truncation.
- Mixing up H/W with C: the formula applies per spatial dimension; channel logic is separate.
- For pooling, kernel/stride act like conv but **don’t change channels**.
