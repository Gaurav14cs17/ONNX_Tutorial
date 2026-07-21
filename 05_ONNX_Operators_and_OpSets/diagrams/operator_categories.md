# ONNX Operator Categories — Mermaid Map

This diagram orients readers to the **practical categories** used throughout Chapter 4. It is **not** the internal C++ taxonomy of ONNX; it emphasizes **relationships** among operator families.

```mermaid
flowchart TB
    subgraph Root["ONNX Standard Operators (conceptual view)"]
        T[Tensor layout & movement]
        E[Elementwise & math]
        L[Linear algebra]
        R[Reductions & statistics]
        NN[Neural network primitives]
        CF[Control flow & graphs]
        S[Sequence / optional / advanced types]
        IO[IO & encoding helpers]
    end

    subgraph T_ops["Tensor layout & movement"]
        T1[Reshape / Flatten / Squeeze / Unsqueeze]
        T2[Transpose / DepthToSpace / SpaceToDepth]
        T3[Pad / Tile / Slice / Split / Concat]
        T4[Gather / GatherElements / Scatter*]
    end

    subgraph E_ops["Elementwise & math"]
        E1[Add / Sub / Mul / Div / Pow]
        E2[Sqrt / Exp / Log / Abs / Sign]
        E3[Clip / Min / Max / Where]
    end

    subgraph L_ops["Linear algebra"]
        L1[MatMul / Gemm / Einsum]
    end

    subgraph R_ops["Reductions"]
        R1[ReduceSum / ReduceMean / ReduceMax]
        R2[ArgMax / ArgMin / TopK]
    end

    subgraph NN_ops["NN primitives"]
        N1[Conv / ConvTranspose / Pool*]
        N2[Relu / Sigmoid / Tanh / Softmax / Gelu]
        N3[BatchNormalization / LayerNormalization / InstanceNormalization]
        N4[Dropout / ...]
        N5[LSTM / GRU / RNN]
    end

    subgraph CF_ops["Control flow"]
        C1[If / Loop / Scan]
    end

    T --> T_ops
    E --> E_ops
    L --> L_ops
    R --> R_ops
    NN --> NN_ops
    CF --> CF_ops

    %% Cross-links (relationships)
    E_ops -. "broadcasting rules feed NN activations" .-> NN_ops
    T_ops -. "layout changes precede Conv/Gemm in many exports" .-> NN_ops
    L_ops -. "often fused with E_ops biases" .-> E_ops
    R_ops -. "produce masks consumed by Where / Gather" .-> T_ops
    R_ops -. "normalize activations" .-> NN_ops

    S[Sequence / Optional ops] --- Root
    IO[Encoder / decoder & compressed formats\n(opset dependent)] --- Root
```

## How to read the diagram

- **Tensor movement** ops rarely change *values* (except `Pad`/`Slice`-like edge cases); they reindex memory views.
- **Elementwise** ops consume broadcasting; they glue **math** onto **tensor** layouts.
- **Linear algebra** ops contract indices; they dominate **parameter-heavy** compute density.
- **NN primitives** are frequently **fused** at runtime (`Conv+Relu`, `Gemm+Bias+Activation`).
- **Control flow** ops intersect with deployment support—always verify your EP.

If you need a tabular index instead, see [`../01_Standard_Operators/README.md`](../01_Standard_Operators/README.md).
