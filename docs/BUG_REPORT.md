# BUG_REPORT.md

Audit date: 2026-05-18

## Confirmed Runtime/Logic Bugs

### BUG-001: Runtime import crash in clean environment
- Severity: Critical
- Evidence:
  - Import command from `backend`: `../.venv/bin/python -c "import app.main"`
  - Failure: `ModuleNotFoundError: No module named 'langchain_text_splitters'`
  - Dependency list is incomplete in `pyproject.toml:7-11`
- Root cause:
  - Runtime modules are imported across app, but lock/manifest does not include required packages.
- Fix:
  - Declare all runtime dependencies and lock deterministically.
- Validation:
  - Add CI smoke test: `python -c "import app.main"`.

### BUG-002: Hardcoded path + malformed path with embedded space
- Severity: Critical
- Evidence:
  - `backend/app/data_ingestion/run_preprocessing.py:9`
  - `backend/app/new.py:22`
  - `backend/app/new.py:268`
- Root cause:
  - Machine-local absolute paths committed to source.
- Fix:
  - Replace with config-driven paths and `Path.resolve()` checks.

### BUG-003: Shared global session ID causes context leakage
- Severity: High
- Evidence:
  - Global `SESSION_ID` in `backend/app/main.py:35`
  - Used in request path at `backend/app/main.py:110-113`
- Root cause:
  - Session context not scoped per user/client request.
- Fix:
  - Generate session key from request/user identity, persist with TTL.

### BUG-004: Retrieval runs without detected intent
- Severity: High
- Evidence:
  - `backend/app/main.py:137-140` does not pass intent to `retriever.retrieve()`.
- Root cause:
  - API handler ignores intent data when calling retriever.
- Fix:
  - Pass normalized `intent` + `intent_topic` to retrieval layer.

### BUG-005: Empty retrieval check always passes for dict payload
- Severity: High
- Evidence:
  - `if not retrieval:` in `backend/app/main.py:142`
  - Retrieval returns dict shape like `{"docs": [], "count": 0, "status": "empty"}`.
- Root cause:
  - Truthiness check is against container existence, not count.
- Fix:
  - `if retrieval.get("count", 0) == 0:`.

### BUG-006: Validator confidence is invalid due hardcoded similarity
- Severity: High
- Evidence:
  - `similarity: 1.0` in `backend/app/main.py:167`
- Root cause:
  - Similarity score not propagated from vector retrieval.
- Fix:
  - Use top result score and fallback when score unavailable.

### BUG-007: Strategy routing mismatch prevents intended behavior
- Severity: High
- Evidence:
  - Greeting strategy checks `is_greeting` at `response_strategy.py:30`, feature not emitted.
  - BigIssue strategy checks `complexity == "complex"` at `response_strategy.py:82`, classifier emits `small|medium|big` (`intent_classifier.py:55`).
- Root cause:
  - Inconsistent taxonomy and feature contracts across modules.
- Fix:
  - Define shared enums/constants and migrate all components.

### BUG-008: User-visible fallback typo
- Severity: Medium
- Evidence:
  - `backend/app/reasoning/response_generator.py:63` returns `clarify()`.
- Root cause:
  - Literal string typo.
- Fix:
  - Correct copy and add assertion test.

### BUG-009: Broad exception swallowing obscures failures
- Severity: Medium
- Evidence:
  - `backend/app/reasoning/llm_reasoner.py:172,213`
  - `backend/app/intent_detection/intent_features.py:68`
  - `backend/app/main.py:180-181`
- Root cause:
  - Unscoped exception handling + no structured log context.
- Fix:
  - Use typed exceptions and structured error mapping.

### BUG-010: Placeholder modules left empty
- Severity: Medium
- Evidence:
  - `backend/app/transaction_handlers/account_support.py` (empty)
  - `backend/app/transaction_handlers/order_lookup.py` (empty)
  - `backend/app/transaction_handlers/refund_handler.py` (empty)
- Root cause:
  - Incomplete implementation.
- Fix:
  - Implement handlers or remove architecture references.

## Functional Regression Signals from Runtime Logs
- `backend/output/output_view_2.txt` and `backend/output/output_view_3.txt` show unstable outputs and inference errors:
  - `probability tensor contains either inf, nan or element < 0`
  - repeated low-confidence fallback despite long generated answers
- These logs indicate control-flow/validation inconsistency and model stability issues.

## Quick Fix Patch Order
1. Dependency and configuration cleanup.
2. Session scoping + retrieval intent propagation.
3. Retrieval empty-check + validator score propagation.
4. Taxonomy normalization across classifier/router/validator/vector store.
5. Exception handling and sanitized API errors.
6. Add tests for each bug above before further refactors.
