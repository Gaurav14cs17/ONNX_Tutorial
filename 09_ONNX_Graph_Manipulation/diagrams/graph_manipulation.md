# Graph manipulation map (Mermaid)

This diagram summarizes the **graph manipulation operations** covered in Chapter 8 and how they connect to validation and deployment.

```mermaid
flowchart TB
  subgraph inspect["Inspect"]
    I1[Load ModelProto]
    I2[Summarize IO / nodes / initializers]
    I3[Optional: subgraph extraction]
  end

  subgraph edit["Modify"]
    E1[Add / remove / replace nodes]
    E2[Rewire value name references]
    E3[Change graph inputs/outputs]
    E4[merge_models via onnx.compose]
  end

  subgraph infer["Shape inference"]
    S1[infer_shapes]
    S2[Inspect value_info / partial shapes]
  end

  subgraph validate["Validate"]
    V1[onnx.checker.check_model]
    V2[Policy tests: forbidden ops / budgets]
  end

  I1 --> I2 --> I3
  I3 --> E1
  E1 --> E2 --> E3 --> E4
  E4 --> S1 --> S2 --> V1 --> V2
```

---

## Notes

- **Merge** is powerful but is the most common place for **name collisions**—use **`prefix`** utilities when needed.
- **Shape inference** is best viewed as **informational** for fully dynamic graphs—still invaluable for catching mistakes.

---

## See also

- [../README.md](../README.md)
