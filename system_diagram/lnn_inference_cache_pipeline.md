# LNN Compressor Anomaly Detection — Inference Pipeline Architecture

```mermaid
flowchart TD
    A[Historian / DCS tags<br/>Multi-rate sensor streams] --> B[Feature engineering<br/>Resampling + schema validation]
    B --> CK{Similarity cache lookup<br/>vector KNN, recent window only}
    CK -.->|KNN search| CACHE[(Redis Enterprise<br/>RediSearch vector index)]
    CK -->|hit: close + recent| H[Inference API response<br/>FastAPI serving layer]
    CK -->|miss: no close recent match| C[LNN autoencoder<br/>Reconstruction error]
    REG[(Model registry<br/>MLflow)] -.->|supplies model| C
    C --> D{Anomaly threshold<br/>Score vs. calibrated limit}
    D -->|Normal| E[Logged, no alert]
    D -->|Anomalous| F[Secondary classifier<br/>Multi-label sigmoid head]
    F --> G[Diagnosis output<br/>Failure mode/s + confidence]
    E --> WR[Write vector + result to cache<br/>expires after recency window]
    G --> WR
    WR -.->|index| CACHE
    WR --> H
    H --> I[Alert / work order<br/>Pushed to CMMS]
    H --> J[Dashboard<br/>Trend & review UI]
    C -.->|reconstruction error stream| MON[[Monitoring / drift detection]]
    MON -.->|triggers retrain| REG

    classDef dataStyle fill:#E6F1FB,stroke:#185FA5,color:#0C447C;
    classDef modelStyle fill:#EEEDFE,stroke:#534AB7,color:#3C3489;
    classDef neutralStyle fill:#F1EFE8,stroke:#5F5E5A,color:#2C2C2A;
    classDef anomalyStyle fill:#FAECE7,stroke:#993C1D,color:#712B13;
    classDef servingStyle fill:#E1F5EE,stroke:#0F6E56,color:#085041;
    classDef cacheStyle fill:#FBEAF0,stroke:#993556,color:#72243E;

    class A,B dataStyle
    class C,REG modelStyle
    class D,E neutralStyle
    class F,G anomalyStyle
    class H,I,J servingStyle
    class MON neutralStyle
    class CK,CACHE,WR cacheStyle
```

## Legend

- **Blue** — data ingestion / serving layer
- **Purple** — model artifacts (LNN autoencoder, registry)
- **Gray** — neutral / decision / monitoring
- **Coral** — anomaly diagnosis path
- **Teal** — feature engineering
- **Pink** — Redis similarity caching layer (vector KNN, cache-aside pattern)

Solid arrows = request-time data path. Dotted arrows = supporting infrastructure
(model registry supplying the deployed model; similarity cache lookup/write;
monitoring loop watching the reconstruction-error stream to trigger retraining).

## Caching notes

- **Cache type**: similarity-based, not exact-key. A new feature window is
  compared against recent cached windows using vector KNN (cosine or L2
  distance) — if a close-enough match exists, reuse its prediction instead of
  running the model.
- **Why recency matters as much as distance**: candidates must be both close
  *and* recent (e.g. last few hours). Old "normal" snapshots are never
  eligible matches no matter how close, so a genuinely drifting process
  eventually stops matching anything cached and forces fresh inference —
  this is what stops the cache from masking a slow-developing fault like
  DGS degradation.
- **Distance threshold**: calibrate empirically — measure how reconstruction
  error changes as a function of feature-space distance on known-normal data,
  and set the threshold well below where that relationship starts moving.
- **Infra requirement**: needs Redis with the RediSearch module for vector
  KNN/HNSW indexing — plain OSS Redis doesn't support this. On Azure, that
  means the Enterprise tier of Azure Cache for Redis (or Azure Managed
  Redis) specifically; Basic/Standard/Premium tiers don't support it.
- **What's still recomputed on every request**: the feature engineering step
  itself — only the model forward pass is skippable on a cache hit.