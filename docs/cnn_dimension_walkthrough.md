# 1D CNN Dimension Walkthrough: Input → Block 4

## Starting point: input **(B, 12, 101)**

The general formula for Conv1d output length is:

$$L_{out} = \left\lfloor \frac{L_{in} + 2 \cdot padding - kernel\_size}{stride} \right\rfloor + 1$$

For MaxPool1d(kernel\_size=2, stride=2):

$$L_{out} = \left\lfloor \frac{L_{in} - 2}{2} \right\rfloor + 1$$

---

## Block 1 — Conv1d(12→32, k=9, s=2, p=4) + MaxPool1d(2)

**After Conv:**

$$L = \left\lfloor \frac{101 + 2 \times 4 - 9}{2} \right\rfloor + 1 = \left\lfloor \frac{100}{2} \right\rfloor + 1 = 51$$

Shape: **(B, 32, 51)**

**After MaxPool1d(2):**

$$L = \left\lfloor \frac{51 - 2}{2} \right\rfloor + 1 = \left\lfloor \frac{49}{2} \right\rfloor + 1 = 25$$

Shape: **(B, 32, 25)**

---

## Block 2 — Conv1d(32→64, k=7, s=2, p=3) + MaxPool1d(2)

**After Conv:**

$$L = \left\lfloor \frac{25 + 2 \times 3 - 7}{2} \right\rfloor + 1 = \left\lfloor \frac{24}{2} \right\rfloor + 1 = 13$$

Shape: **(B, 64, 13)**

**After MaxPool1d(2):**

$$L = \left\lfloor \frac{13 - 2}{2} \right\rfloor + 1 = \left\lfloor \frac{11}{2} \right\rfloor + 1 = 6$$

Shape: **(B, 64, 6)**

---

## Block 3 — Conv1d(64→128, k=5, s=2, p=2) + MaxPool1d(2)

**After Conv:**

$$L = \left\lfloor \frac{6 + 2 \times 2 - 5}{2} \right\rfloor + 1 = \left\lfloor \frac{5}{2} \right\rfloor + 1 = 3$$

Shape: **(B, 128, 3)**

**After MaxPool1d(2):**

$$L = \left\lfloor \frac{3 - 2}{2} \right\rfloor + 1 = \left\lfloor \frac{1}{2} \right\rfloor + 1 = 1$$

Shape: **(B, 128, 1)**

---

## Block 4 — Conv1d(128→256, k=3, s=1, p=1)  *(no MaxPool)*

**After Conv:**

$$L = \left\lfloor \frac{1 + 2 \times 1 - 3}{1} \right\rfloor + 1 = \left\lfloor \frac{0}{1} \right\rfloor + 1 = 1$$

Shape: **(B, 256, 1)**

---

## Summary

| Stage | Operation | Output shape |
|-------|-----------|-------------|
| Input | — | (B, 12, 101) |
| Block 1 Conv | k=9, s=2, p=4 | (B, 32, 51) |
| Block 1 Pool | MaxPool(2) | (B, 32, **25**) |
| Block 2 Conv | k=7, s=2, p=3 | (B, 64, 13) |
| Block 2 Pool | MaxPool(2) | (B, 64, **6**) |
| Block 3 Conv | k=5, s=2, p=2 | (B, 128, 3) |
| Block 3 Pool | MaxPool(2) | (B, 128, **1**) |
| Block 4 Conv | k=3, s=1, p=1 | (B, 256, **1**) |
| AdaptiveAvgPool | pool to 1 | (B, 256, 1) |
| Flatten | — | **(B, 256)** |

Each block performs **two rounds of downsampling** — Conv (stride=2) then MaxPool (stride=2) — giving an effective 4x stride per block.
Block 4 uses stride=1 so the length stays at 1 once Block 3 has already reduced it there.
