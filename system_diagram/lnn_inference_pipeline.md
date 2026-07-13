# LNN Compressor Anomaly Detection — Inference Pipeline Architecture

```mermaid
flowchart TD
    A[Historian / DCS tags<br/>Multi-rate sensor streams] --> B[Feature engineering<br/>Resampling + schema validation]
    B --> C[LNN autoencoder<br/>Reconstruction error]
    REG[(Model registry<br/>MLflow)] -.->|supplies model| C
    C --> D{Anomaly threshold<br/>Score vs. calibrated limit}
    D -->|Normal| E[Logged, no alert]
    D -->|Anomalous| F[Secondary classifier<br/>Multi-label sigmoid head]
    F --> G[Diagnosis output<br/>Failure mode/s + confidence]
    G --> H[Inference API<br/>FastAPI serving layer]
    H --> I[Alert / work order<br/>Pushed to CMMS]
    H --> J[Dashboard<br/>Trend & review UI]
    C -.->|reconstruction error stream| MON[[Monitoring / drift detection]]
    MON -.->|triggers retrain| REG

    classDef dataStyle fill:#E6F1FB,stroke:#185FA5,color:#0C447C;
    classDef modelStyle fill:#EEEDFE,stroke:#534AB7,color:#3C3489;
    classDef neutralStyle fill:#F1EFE8,stroke:#5F5E5A,color:#2C2C2A;
    classDef anomalyStyle fill:#FAECE7,stroke:#993C1D,color:#712B13;
    classDef servingStyle fill:#E1F5EE,stroke:#0F6E56,color:#085041;

    class A,B dataStyle
    class C,REG modelStyle
    class D,E neutralStyle
    class F,G anomalyStyle
    class H,I,J servingStyle
    class MON neutralStyle
```

## Legend

- **Blue** — data ingestion / serving layer
- **Purple** — model artifacts (LNN autoencoder, registry)
- **Gray** — neutral / decision / monitoring
- **Coral** — anomaly diagnosis path
- **Teal** — feature engineering

Solid arrows = request-time data path. Dotted arrows = supporting infrastructure
(model registry supplying the deployed model; monitoring loop watching the
reconstruction-error stream to trigger retraining).