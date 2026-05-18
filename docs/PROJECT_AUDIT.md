# PROJECT_AUDIT.md

Audit date: 2026-05-18
Project: dynamic-ai-customer-support
Audited scope: all git-tracked files (`git ls-files`), including backend code, data artifacts, docs, config, and lockfiles.

## 1. Executive Summary
Project health is **high risk for production**.

Key blockers:
1. Runtime bootstrap fails in current environment due missing runtime dependencies.
2. API has no authentication, no rate limiting, and leaks internal errors.
3. AI pipeline has intent-taxonomy mismatches that break routing and strategy logic.
4. Validation confidence is artificially inflated (`similarity=1.0` hardcoded).
5. No automated tests; no CI/CD, Docker, monitoring, or deployment scaffolding.

Readiness score (estimated): **34/100**.

## 2. Architecture Review
Current architecture is a single-process monolith with in-memory state:
- Offline ingestion is executed at API startup in `backend/app/main.py`.
- Inference/retrieval/feature extraction run synchronously in request path.
- Session memory is process-local and shared incorrectly across users.
- No database, queue, cache, or durable vector index lifecycle.

Documented architecture vs implementation gaps:
- Docs describe production-style modular system (`README.md`, `ARCHITECTURE_NOTES.md`), but implementation lacks deployment primitives and safety controls.
- `response_strategy` expects intents/features not emitted by `intent_classifier`/`query_preprocess`.
- `api/routes.py` is disconnected placeholder API (fake in-memory users).

## 3. Critical Bugs

| ID | Severity | Finding | Evidence | Impact | Fix |
|---|---|---|---|---|---|
| BUG-001 | Critical | Runtime import failure from missing libs | `ModuleNotFoundError: langchain_text_splitters` while importing `backend/app/main.py`; dependencies incomplete in `pyproject.toml:7-11` | Service cannot start in clean env | Add full runtime deps and lock; validate via startup smoke test in CI |
| BUG-002 | Critical | Hardcoded absolute paths and malformed path (`backend /data`) | `backend/app/data_ingestion/run_preprocessing.py:9`, `backend/app/new.py:22`, `backend/app/new.py:268`, `backend/app/main.py:32` | Non-portable; file-not-found failures | Use env-based paths + `pathlib`, validate on startup |
| BUG-003 | High | Shared session state across all users | Global `SESSION_ID` in `backend/app/main.py:35`; used in `main.py:110-113` | Cross-user context leakage, wrong follow-up behavior | Generate per-request/session IDs from client token/cookie/header |
| BUG-004 | High | Retrieval routing ignores detected intent | `retriever.retrieve(... top_k=5)` in `backend/app/main.py:137-140` without `intent` | Low retrieval precision; wrong top-k logic | Pass intent and topic features into retrieval router |
| BUG-005 | High | Empty retrieval check is logically incorrect | `if not retrieval:` at `backend/app/main.py:142` while retrieval is dict | Empty result treated as non-empty; downstream hallucination pressure | Check `if retrieval.get("count", 0) == 0` |
| BUG-006 | High | Validator signal is invalid (hardcoded similarity) | `similarity: 1.0` at `backend/app/main.py:167` | Confidence always inflated; fallback logic unreliable | Use actual top retrieval similarity score |
| BUG-007 | High | Response strategy never matches intended branches | `is_greeting` required in `response_strategy.py:30`, but not produced; `complexity == "complex"` in `response_strategy.py:82`, classifier emits `small|medium|big` at `intent_classifier.py:55` | Wrong prompts, degraded answer quality | Unify intent/feature schema across modules |
| BUG-008 | Medium | Typo in fallback text | `"Could you please clarify()"` at `response_generator.py:63` | User-facing quality regression | Fix string and add unit test |
| BUG-009 | Medium | Broad exception swallowing hides root causes | Bare `except:` and generic `except Exception` across pipeline, e.g., `llm_reasoner.py:172,213`, `intent_features.py:68` | Debugging impossible; silent degraded behavior | Replace with typed exceptions + structured logs |
| BUG-010 | Medium | Transaction handler modules are empty | `backend/app/transaction_handlers/*.py` | Dead architecture paths | Implement or remove placeholders |

## 4. Security Audit

### Findings
1. **No authN/authZ on core endpoint** (`/query` in `backend/app/main.py:101-181`).
2. **Internal exception details exposed to clients** (`HTTPException(... detail=str(e))` at `main.py:181`).
3. **PII committed in repository data and logs** (`backend/app/data/training_data.txt`, `new_training_data.txt`, `backend/output/output_view_*.txt`).
4. **No rate limiting / abuse protection** (no middleware/policy).
5. **Prompt-injection defense missing** (context injected directly in `context_assembler.py:52`).
6. **No secure config/secret lifecycle** (`.env` empty; no config validation layer).
7. **No security observability** (no audit logs, SIEM hooks, alerting, or anomaly detection).

### Recommended security controls
- Add JWT/OAuth2 middleware + RBAC per endpoint.
- Replace raw error detail with sanitized error codes.
- Remove/redact PII datasets from repo; use DLP scan in CI.
- Add request size limits + rate limiting (e.g., `slowapi`, API gateway throttling).
- Add prompt-injection guardrails: context delimiters, output policy checks, refusal templates.
- Introduce secret management (`Vault`/cloud secret manager), typed settings model.
- Add security tests: auth bypass, prompt injection, rate-limit enforcement.

## 5. Performance Audit

### Bottlenecks
- Startup performs full ingestion + embedding load (`main.py:42-64`, `69-93`).
- Request path is synchronous and lock-heavy (`threading.Lock` around key stages).
- Repeated language detection and feature extraction across modules.
- No cache for query embeddings or frequent retrieval results.
- No streaming token responses; full answer latency only.
- FAISS index lifecycle lacks persistent incremental update workflow.

### Optimization plan
- Move ingestion to offline batch job; load persisted index at startup.
- Use async endpoint and worker pool for model inference.
- Add Redis cache for frequent queries and embedding vectors.
- Add response streaming with timeout/circuit-breaker.
- Profile tokenization/model time; enable quantized inference.
- Scale inference with vLLM/TensorRT/ONNX Runtime where GPU present.

## 6. AI/LLM Audit

### Major LLM/RAG issues
1. Intent taxonomies diverge across classifier/router/validator/vector store.
2. Retrieval thresholds and fallback allow low-similarity documents.
3. Validator is heuristic-only and receives fake similarity value.
4. No reranker, no citation grounding, no source confidence propagation.
5. Prompt strategy inconsistency with generated features (e.g., greeting).
6. Logs show unstable outputs and NaN/inference errors (`backend/output/output_view_2.txt`, `output_view_3.txt`).

### Recommended AI architecture upgrades
- Standardize schema contract (`intent`, `topic`, `urgency`, `complexity`) as shared pydantic model.
- Add reranker (cross-encoder) after vector retrieval.
- Propagate retrieval scores into validation and response confidence.
- Add answer citation spans and confidence explanation.
- Add prompt-injection and hallucination checks pre/post generation.
- Consider pgvector/Pinecone/Weaviate for multi-tenant + metadata filtering at scale.

## 7. Backend Audit
- API layer is minimal and not production hardened.
- `api/routes.py` contains unauthenticated fake user CRUD (`backend/app/api/routes.py:5-21`) and is not integrated into app router.
- No persistence layer (no SQL/NoSQL), no migration framework, no repository/service boundary.
- No queue/async jobs for heavy operations.
- Error handling and logging are inconsistent.
- Root `main.py` is placeholder and unrelated to backend runtime.

## 8. Frontend Audit
No frontend code is present in this repository.

Risk:
- Full-stack delivery claims cannot be validated.
- No UI performance/accessibility/telemetry testing possible.

## 9. DevOps Audit
Missing production prerequisites:
- No Dockerfile / docker-compose
- No CI/CD workflows
- No environment matrix (dev/stage/prod)
- No metrics/logging/alerting stack
- No backup/rollback runbooks
- No infra-as-code/Kubernetes manifests

Recommended stack:
- Build: multi-stage Docker + slim runtime image
- CI: GitHub Actions (lint, tests, security scan, build)
- Deploy: Kubernetes + Helm (or ECS)
- Monitoring: Prometheus + Grafana
- Error tracking: Sentry
- Logs: ELK/OpenSearch

## 10. Testing Audit
Current state:
- `pytest` run result: **no tests discovered**.
- No unit/integration/load/security suites.

Required coverage additions:
1. Unit: preprocessors, intent classifier, retrieval filters, validator scoring.
2. Integration: end-to-end `/query` with mocked LLM/vector index.
3. Contract: response schema guarantees and error envelopes.
4. Load: concurrent query latency p50/p95/p99 + memory growth.
5. Security: auth bypass, rate-limit abuse, prompt injection scenarios.
6. AI eval: hallucination rate, groundedness, citation correctness.

## 11. Refactoring Recommendations
- Introduce `Settings` class (pydantic settings) for all paths/models/thresholds.
- Split runtime app bootstrap into:
  - `startup.py` (resource initialization)
  - `services/` (retrieval, reasoning, validation)
  - `api/routers/` (versioned endpoints)
- Remove or quarantine experimental `backend/app/new.py` into `experiments/`.
- Replace global mutable memory with bounded session store.
- Replace broad exceptions with typed error taxonomy.
- Add structured logging and correlation IDs.

## 12. Priority Roadmap

| Priority | Task | Impact | Difficulty |
|---|---|---|---|
| P0 | Fix dependency manifest and reproducible environment lock | Startup unblocked | Medium |
| P0 | Remove hardcoded paths; env-driven config | Portability + deployability | Low |
| P0 | Add auth + sanitized error handling + rate limits | Security baseline | Medium |
| P0 | Unify intent schema across all modules | Major AI quality gain | Medium |
| P1 | Pass real retrieval similarity to validator | Hallucination reduction | Low |
| P1 | Correct retrieval empty checks and pass intent into router | Retrieval precision | Low |
| P1 | Introduce tests + CI pipeline | Release confidence | Medium |
| P1 | Move ingestion to offline job + persistent index artifacts | Startup speed + scalability | Medium |
| P2 | Add Redis cache + async workers (Celery/RQ) | Throughput and latency | Medium |
| P2 | Add reranker + citation output | Answer trustworthiness | Medium |
| P2 | Add observability (Prometheus/Grafana/Sentry) | Ops stability | Medium |
| P3 | Evaluate vector backend (pgvector/Pinecone/Weaviate) for multi-tenant scale | Long-term scalability | Medium |
| P3 | Evaluate optimized inference (vLLM/ONNX Runtime/TensorRT/quantization) | Cost/performance | High |

---

## Improvement Matrix (Problem -> Root Cause -> Fix)

| Problem | Root Cause | Severity | Recommended Fix | Optimized Implementation | Best Practices | Estimated Impact | Priority |
|---|---|---|---|---|---|---|---|
| Runtime dependency failures | Missing libs in `pyproject.toml` | Critical | Add complete dependencies and lock | `uv sync --frozen` in CI + import smoke test | Reproducible builds | Service starts reliably | P0 |
| Non-portable file paths | Absolute and malformed paths | Critical | Use env vars and relative resolution | `Settings(DATA_PATH=...)` + startup validation | 12-factor config | Cross-env deploy works | P0 |
| Cross-user context leakage | Global `SESSION_ID` and in-memory shared state | High | Session-scoped ID + external store | Redis keyed by user/session with TTL | Privacy by design | Eliminates contamination | P0 |
| Weak retrieval routing | Intent not passed to retriever | High | Pass normalized intent/topic | typed RetrievalRequest model | Contract-driven design | Better relevance | P1 |
| False validator confidence | Similarity hardcoded to `1.0` | High | Use top similarity from retriever | response payload carries score/citations | Groundedness checks | Lower hallucination rate | P1 |
| Strategy mismatch | Divergent feature names and values | High | Standardize taxonomy | shared enums/constants module | Single source of truth | Stable prompt behavior | P0 |
| Security exposure | No auth/rate limit/error sanitization | Critical | Add auth middleware, throttling, safe errors | API gateway + middleware + audit logs | Zero-trust defaults | Major risk reduction | P0 |
| Prompt-injection vulnerability | Raw context and no policy filter | High | Add pre/post guardrails | allowlisted instruction scaffold + output validator | LLM safety layering | Fewer unsafe outputs | P1 |
| No automated quality gate | Zero tests/CI | High | Add unit/integration/security tests | GitHub Actions pipeline | Shift-left QA | Regression prevention | P1 |
| No ops visibility | Missing metrics/logging/alerts | Medium | Add monitoring stack | Prometheus + Grafana + Sentry + ELK | SLO-driven ops | Faster incident response | P2 |

