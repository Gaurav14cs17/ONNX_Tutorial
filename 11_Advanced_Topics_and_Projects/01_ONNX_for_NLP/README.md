# ONNX for NLP: Transformers, Tokenizers, and Deployment-Grade Inference

> **Interview Relevance:** HIGH — Transformer ONNX export is one of the most asked topics in ML engineering interviews; expect questions on dynamic axes, tokenizer parity, and INT8 quantization for NLP.

---

## Visual Overview
*Diagrams will be added as the tutorial evolves.*

```
 +------------------+     +---------------------------+
 | Raw text         | --> | Tokenizer (pre-ONNX)      |
 | "Graphs are ..." |     | -> input_ids [B,T] int64  |
 +------------------+     | -> attention_mask [B,T]   |
                          +--------------+------------+
                                         |
                                         v
                          +------------------------------+
                          | ONNX model (transformer)      |
                          | multi-input / multi-output    |
                          +--------------+---------------+
                                         |
                   +---------------------+---------------------+
                   v                                           v
           +---------------+                        +---------------+
           | Task head     |                        | Embeddings    |
           | logits / NER  |                        | for retrieval |
           +---------------+                        +---------------+
```

## Why (Motivation)

NLP models in production are rarely "just matmuls" — they are **tokenization + tensor bundles + post-processing** with subtle training/serving skew traps. ONNX provides a portable graph for the neural portion, but you still own tokenizer parity, padding policy, and dynamic sequence behavior. Mastering NLP ONNX export is critical for deploying transformers at scale.

## When (Use Cases)

- **Text classification APIs** serving BERT/DistilBERT at high throughput
- **Named entity recognition** pipelines with sequence labeling outputs
- **Sentence embeddings** for search and retrieval systems
- **Question answering** systems with dynamic input lengths
- **Cross-framework deployment** (PyTorch model → ONNX → any runtime)

## How (Mechanism)

Export a transformer with `torch.onnx.export` (or Hugging Face Optimum), specifying dynamic axes for batch and sequence dimensions. The ONNX graph accepts `input_ids` and `attention_mask` as int64 tensors. At serving time, the tokenizer runs outside ONNX, producing tensors that must exactly match the training tokenizer's behavior. ORT then executes the graph with graph optimizations and optional INT8 quantization.

---

## Prerequisites
- [10_Deployment](../../10_Deployment/) — Deployment fundamentals
- [06_Exporting_Models_to_ONNX](../../06_Exporting_Models_to_ONNX/) — Export mechanics

## What You Will Learn

| Concept | Why It Matters | Interview Frequency |
|---------|---------------|:-------------------:|
| BERT/GPT-2 ONNX export patterns | Foundation for all transformer deployment | ⭐⭐⭐ |
| Dynamic axes for variable sequences | Production NLP requires variable-length input | ⭐⭐⭐ |
| Tokenizer handling and parity | #1 source of NLP serving bugs | ⭐⭐⭐ |
| Hugging Face Optimum integration | Industry-standard export tooling | ⭐⭐ |
| INT8 dynamic quantization for NLP | CPU throughput optimization | ⭐⭐⭐ |
| Numerical parity testing | Validating export correctness | ⭐⭐ |

## Key Interview Questions Answered Here

1. **What tensors does a BERT ONNX export need?** → `input_ids` [B,T] int64, `attention_mask` [B,T] int64, and optionally `token_type_ids` [B,T] int64. Outputs vary by task head.
2. **Where does tokenization happen relative to ONNX?** → Outside the graph. The tokenizer runs in Python/C++ before feeding tensors to ORT. This boundary is the #1 source of skew.
3. **How do you handle variable sequence lengths?** → Use `dynamic_axes` in `torch.onnx.export` to keep batch and sequence dimensions symbolic. Enforce max_length at the API boundary.
4. **How do you validate an NLP ONNX export?** → Run identical tokenized input through both PyTorch and ORT, then compare outputs with `np.allclose` using appropriate `atol`/`rtol`.

## Notebooks

| Notebook | Focus | Time |
|----------|-------|:----:|
| [ONNX_for_NLP_Deep_Dive.ipynb](01_ONNX_for_NLP_Deep_Dive.ipynb) | Theory & internals | ~20 min |
| [ONNX_for_NLP_Apply.ipynb](02_ONNX_for_NLP_Apply.ipynb) | Hands-on practice | ~15 min |

## Common Mistakes & Confusions

- **`max_length` mismatch between tokenizer and training** — Causes random accuracy collapses at serving time.
- **Wrong `padding_side` for decoder models** — GPT-2/causal LMs need left-padding; BERT needs right-padding.
- **Forgetting int64 dtype** — ONNX transformer graphs expect int64 inputs; int32 causes silent errors.
- **Loading the model per request** — Session creation is expensive; load once at startup.
- **`fast` vs `slow` tokenizer edge cases** — Rare tokenization mismatches can cause subtle accuracy drift.

---

## Next Steps
→ [02_ONNX_for_Computer_Vision](../02_ONNX_for_Computer_Vision/)

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
