# Export pipeline (all paths → ONNX)

This diagram summarizes major **framework-to-ONNX** export routes taught in Chapter 5. Arrow labels name the **typical converter or API**. Exact package names and supported opsets evolve; always check the upstream project for your version.

```mermaid
flowchart TB
  subgraph Training["Training frameworks"]
    PT["PyTorch nn.Module"]
    TF["TensorFlow / Keras"]
    SK["scikit-learn"]
    Trees["XGBoost / LightGBM"]
    PD["PaddlePaddle"]
    MX["Apache MXNet"]
    ML["MATLAB Deep Learning Toolbox"]
  end

  subgraph Intermediates["Intermediate representations"]
    TS["TorchScript graph\n(trace / script)"]
    SM["TensorFlow SavedModel\n(GraphDef / FuncDef)"]
    SKIR["sklearn parse_* / topology\n+ operator library"]
    GBM["Tree ensemble\nJSON / native booster"]
    PDProg["Paddle Program / IR"]
    MXSym["MXNet Symbol / Gluon"]
    MLNET["MATLAB network object\n→ ONNX exporter"]
  end

  ONNX["ONNX Model (.onnx)\nIR + weights + metadata"]

  PT -->|"torch.onnx.export"| TS
  TS --> ONNX

  TF -->|"tf2onnx.convert / Python API"| SM
  SM --> ONNX

  SK -->|"skl2onnx.convert_sklearn"| SKIR
  SKIR --> ONNX

  Trees -->|"Hummingbird-ML\nor ONNXMLTools"| GBM
  GBM --> ONNX

  PD -->|"paddle2onnx"| PDProg
  PDProg --> ONNX

  MX -->|"mxnet.contrib.onnx\n/ export APIs"| MXSym
  MXSym --> ONNX

  ML -->|"exportONNXNetwork"| MLNET
  MLNET --> ONNX

  subgraph Validation["Post-export validation"]
    CHK["onnx.checker.check_model"]
    ORT["onnxruntime.InferenceSession"]
  end

  ONNX --> CHK
  ONNX --> ORT
```

## How to read this diagram

- **PyTorch** usually goes through a **TorchScript** representation (trace or script) produced internally during export; you normally call one API: `torch.onnx.export`.
- **TensorFlow** conversion typically starts from a **SavedModel** (or a frozen graph in older workflows); **`tf2onnx`** is the mainstream open-source path today.
- **scikit-learn** conversion is **declarative**: parsers walk estimator types and emit ONNX nodes from **`skl2onnx`**’s operator mapping tables.
- **Tree ensembles** can be converted via **ONNXMLTools** (native tree → ONNX operators) or **Hummingbird** (compile to tensor pipelines → torch → ONNX in some workflows).
- **All paths** should converge on the same **deployment gate**: structural checking plus **runtime parity** tests.

---

*Return to [Chapter 5 README](../README.md)*
