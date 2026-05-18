# PERFORMANCE_REPORT.md

Audit date: 2026-05-18

## Baseline Observations
- Startup path performs ingestion + embedding and index build synchronously.
- Request path is synchronous and lock-heavy.
- No caching, batching, streaming, or queueing.
- No profiling/telemetry to identify p95/p99 bottlenecks.

## Bottlenecks and Optimizations

| Area | Current Behavior | Bottleneck | Optimization | Expected Impact |
|---|---|---|---|---|
| API startup | Builds ingestion pipeline on import (`main.py:42-93`) | Long cold start and memory spikes | Precompute embeddings offline; load persisted index artifact | 5-20x faster startup |
| Request execution | Sync path with locks across components | Throughput collapse under concurrency | Async endpoint + worker pool + remove broad locks | 2-6x throughput |
| Retrieval | Intent not propagated, fallback broad matches | Low precision causes longer generation cycles | Intent/topic-aware retrieval and reranking | Better latency + quality |
| Validation | Similarity hardcoded | Invalid confidence, retry churn | Use real scores and calibrated threshold | Fewer low-value responses |
| Inference | Unoptimized model serving | High token latency | Quantization + vLLM/ONNX Runtime/TensorRT (GPU) | 1.5-8x inference speed |
| Caching | No cache layer | Repeated identical compute | Redis for embedding and response cache | Lower p50 latency |
| Background jobs | None | Heavy tasks compete with API path | Celery/RQ with RabbitMQ or Kafka | Stable API latency |
| Observability | No metrics/tracing | Unknown hotspots | Prometheus + OpenTelemetry + Grafana | Faster optimization cycles |

## Priority Performance Roadmap

### P0
1. Move ingestion/index build out of API startup.
2. Fix retrieval control flow and score propagation.
3. Add basic metrics: request latency, token generation time, retrieval duration.

### P1
1. Add Redis cache for hot queries and embeddings.
2. Convert `/query` to async and isolate model inference workers.
3. Add response streaming for long answers.

### P2
1. Add queue-backed workloads with Celery + RabbitMQ (or Kafka for high-volume streaming events).
2. Add reranker and token-budget manager.
3. Evaluate vector backend for scale: `pgvector` (relational-first), `Pinecone`/`Weaviate` (managed vector-first).

## Scaling Recommendations by Stage

### Stage 1 (single node)
- Keep FAISS local, persist index artifact to disk.
- Use Redis cache + gunicorn/uvicorn workers.

### Stage 2 (multi-tenant)
- Externalize vector store (`pgvector`, Pinecone, or Weaviate).
- Add tenant-aware filtering and request quotas.

### Stage 3 (high throughput)
- Inference server with vLLM (batching, paged attention).
- ONNX Runtime/TensorRT where hardware supports it.
- Queue decoupling for ingestion/re-index jobs (Kafka or RabbitMQ).

## Profiling Plan
1. Add timing spans for preprocess, intent detect, retrieval, generation, validation.
2. Track p50/p95/p99 latency, token/sec, error rate, timeout rate.
3. Track CPU, RAM, GPU utilization per request class.
4. Introduce load tests at 10/50/100 concurrent users and compare deltas after each optimization phase.
