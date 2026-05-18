# TESTING_REPORT.md

Audit date: 2026-05-18

## Current Testing State
- `pytest -q` result: **no tests ran**.
- No automated QA gates for functionality, performance, or security.

## Recommended Test Architecture

Suggested structure:
- `tests/unit/`
- `tests/integration/`
- `tests/security/`
- `tests/load/`
- `tests/fixtures/`

Tooling:
- `pytest`, `pytest-asyncio`, `httpx`, `pytest-cov`
- `locust` or `k6` for load
- `schemathesis` for API contract/fuzz

## Unit Tests

### Services and Utilities
1. `QueryPreprocessor.invoke`
- small-talk detection
- punctuation normalization
- empty input behavior
- language detection fallback

2. `IntentClassifier.classify`
- deterministic keyword routes
- low-confidence fallback behavior
- invalid LLM output resilience

3. `FAISSIndex.retrieve`
- dimension mismatch rejection
- threshold filtering
- identity query filtering
- fallback selection behavior

4. `AnswerValidator.invoke`
- empty answer/context fallbacks
- length and guessing penalties
- confidence threshold behavior

5. `ResponseStrategyRouter.select`
- strategy precedence
- intent/emotion/urgency mappings

## Integration Tests
1. API integration (`/query`)
- valid query request/response shape
- empty query returns 400
- internal failure returns sanitized error

2. Pipeline integration
- ingest sample docs -> embed -> index -> retrieve -> generate -> validate
- verify real similarity score propagation

3. LLM integration (mocked and real optional profile)
- mocked deterministic tests in CI
- optional nightly integration with live model runtime

4. RAG integration
- retrieval relevance thresholds
- citation/context alignment checks

## Load Testing
1. Concurrent user profiles
- 10, 50, 100 concurrent clients
- mixed query types: greeting, identity, transactional, long-form

2. Metrics to capture
- p50/p95/p99 latency
- throughput (RPS)
- timeout/error rates
- CPU/RAM/GPU utilization

3. Stress scenarios
- long context and token-heavy prompts
- repeated identical queries (cache effectiveness)

## Security Testing
1. Auth bypass tests
- no token, malformed token, wrong role

2. Rate-limit tests
- burst traffic and sustained abuse

3. Prompt injection tests
- instruction override attempts
- data exfiltration prompts
- malformed context delimiters

4. Error handling tests
- ensure no stack traces/internal details in response bodies

## Edge Case Tests
- Empty input and whitespace-only input
- Very large payloads and long user prompts
- Invalid JSON body
- Timeout and downstream model failure simulation
- Unicode and multilingual inputs
- Hallucination challenge set (out-of-knowledge questions)

## Coverage Targets
- Unit coverage: >= 80%
- Critical path integration coverage: 100% (query endpoint + retrieval + validation)
- Security test pass rate: 100% required for release

## CI Quality Gates
1. Lint + type checks
2. Unit/integration tests
3. Coverage threshold enforcement
4. Security scans (secrets + dependency audit)
5. Build and startup smoke test

## First Sprint Test Backlog
1. Add unit tests for preprocessor, intent classifier, validator.
2. Add `/query` integration tests with mocked reasoner.
3. Add regression tests for bugs identified in `BUG_REPORT.md`.
4. Add CI workflow to run full test suite on pull requests.
