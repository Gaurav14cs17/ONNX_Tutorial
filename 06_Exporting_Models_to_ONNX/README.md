# 06 — Exporting Models to ONNX

This chapter is the practical bridge between **training-time frameworks** and **deployment-ready ONNX graphs**. You will learn how to export trained models from common libraries, validate exports, and ensure numerical parity.

---

## Learning Order

| # | Topic | Focus | Time |
|:-:|-------|-------|:----:|
| 1 | [01_PyTorch_to_ONNX](./01_PyTorch_to_ONNX/) | torch.onnx.export, tracing vs scripting, dynamic axes | ~35 min |
| 2 | [02_TensorFlow_Keras_to_ONNX](./02_TensorFlow_Keras_to_ONNX/) | tf2onnx, SavedModel, layout handling | ~35 min |
| 3 | [03_Scikit_Learn_to_ONNX](./03_Scikit_Learn_to_ONNX/) | skl2onnx, pipelines, preprocessing | ~35 min |
| 4 | [04_Other_Frameworks](./04_Other_Frameworks/) | XGBoost, LightGBM, PaddlePaddle, comparison | ~35 min |

---

## Learning Objectives

By the end of this chapter, you should be able to:

1. **Export** PyTorch models with dynamic axes and proper eval mode
2. **Convert** TensorFlow/Keras artifacts via tf2onnx
3. **Convert** sklearn pipelines with embedded preprocessing
4. **Survey** additional converter tools and select the right one
5. **Validate** every export with checker + numerical parity testing

---

## Prerequisites

- Completion of chapters on ONNX IR, operators, and OpSets
- Familiarity with at least one of: PyTorch, TensorFlow, or sklearn

## Quality Checklist (Apply to Every Export)

- [ ] Model passes `onnx.checker.check_model`
- [ ] Opset matches target runtime capabilities
- [ ] I/O names, dtypes, and rank match serving expectations
- [ ] Numerical parity verified on realistic inputs
- [ ] Dynamic axes work for variable batch/sequence sizes

## Files and Assets

| Path | Description |
|------|-------------|
| [01_PyTorch_to_ONNX/pytorch_export_example.py](./01_PyTorch_to_ONNX/pytorch_export_example.py) | CNN export + ORT parity |
| [02_TensorFlow_Keras_to_ONNX/tf_export_example.py](./02_TensorFlow_Keras_to_ONNX/tf_export_example.py) | Keras → tf2onnx → ORT |
| [03_Scikit_Learn_to_ONNX/sklearn_export_example.py](./03_Scikit_Learn_to_ONNX/sklearn_export_example.py) | skl2onnx pipeline + parity |
| [diagrams/export_pipeline.md](./diagrams/export_pipeline.md) | Framework → ONNX paths diagram |

---

*Next chapter: ONNX Runtime and Inference*

---
**Author:** [Gaurav14cs17](https://github.com/Gaurav14cs17)
