# 02 — Introduction to ONNX

This chapter establishes the conceptual foundation for the rest of this tutorial. You will learn what ONNX is, why it exists, how the surrounding ecosystem is organized, and how to prepare a working Python environment for hands-on exercises in later chapters.

---

## Learning Order

| # | Section | Focus | Time |
|---|---------|-------|:----:|
| 1 | [What is ONNX](01_What_is_ONNX/) | Definition, history, interoperability story, features, comparison | ~35 min |
| 2 | [Why ONNX Matters](02_Why_ONNX_Matters/) | N×M fragmentation, N+M hub pattern, benefits, trade-offs | ~35 min |
| 3 | [ONNX Ecosystem Overview](03_ONNX_Ecosystem_Overview/) | Model Zoo, ONNX Runtime, converters, Netron, hardware partners | ~35 min |
| 4 | [Installation and Setup](04_Installation_and_Setup/) | Environments, packages, converters, verification, troubleshooting | ~35 min |

Each section contains:
- **`{Topic}_Deep_Dive.ipynb`** — Theory and internals (~20 min)
- **`{Topic}_Apply.ipynb`** — Hands-on exercises (~15 min)
- **`README.md`** — Enhanced reference with interview questions

---

## Learning Objectives

By the end of this chapter, you should be able to:

1. **Explain** ONNX in plain language: full name, purpose, and the interoperability gap it addresses
2. **Describe** the historical context (Facebook and Microsoft, 2017) and evolution into an open standard
3. **Reason** about the N×M fragmentation problem and how ONNX shifts it to N+M
4. **Navigate** the ONNX ecosystem: models, runtimes, converters, visualization
5. **Prepare** a reproducible Python environment with `onnx` and `onnxruntime`
6. **Validate** your setup by building a tiny ONNX model and running inference

---

## Prerequisites

### Knowledge
- Python: comfortable running scripts and using pip
- Machine learning basics: training vs inference distinction

### Software
- Python 3.8+ (3.10+ recommended)
- Internet access for package installation

### How to Use This Chapter
Read sections **01**–**03** sequentially; they build a shared vocabulary. Complete section **04** before advancing to later chapters.

---

## Files and Assets

| Path | Description |
|------|-------------|
| `04_Installation_and_Setup/verify_installation.py` | Minimal end-to-end verification script |
| `diagrams/onnx_overview.md` | Mermaid diagram: frameworks → ONNX → runtimes |

---

*Next: [01 — What is ONNX →](01_What_is_ONNX/)*

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
