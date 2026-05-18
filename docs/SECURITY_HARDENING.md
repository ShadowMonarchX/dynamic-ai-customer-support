# SECURITY_HARDENING.md

Date: 2026-05-18
Project: `dynamic-ai-customer-support`

## Security Controls Implemented

## 1. Authentication and Authorization
- Implemented OAuth2 password grant endpoint: `/api/v1/auth/token`.
- Implemented refresh endpoint: `/api/v1/auth/refresh`.
- Implemented JWT issue/verify with `iat`, `exp`, `sub`, `role`, `type`.
- Enforced access-token-only authorization in request guard (`type == access`).
- Implemented RBAC guard (`require_roles`) across protected query endpoints.
- Added secure password hashing and verification using Passlib `pbkdf2_sha256`.

## 2. Secret and config hygiene
- Centralized config via `pydantic-settings` in `backend/app/core/settings.py`.
- Enforced strong key policy: `SECRET_KEY >= 32` characters.
- Removed hardcoded runtime paths and relied on env-driven settings + validation.
- Updated container/deployment defaults to compliant key placeholders.

## 3. API abuse protection
- Enabled rate limiting middleware using `slowapi`.
- Added request body-size enforcement middleware (`REQUEST_TOO_LARGE`/413).
- Added baseline route protection requiring valid bearer tokens.

## 4. Secure error handling and privacy
- Added structured app-level exception model (`AppError`).
- Replaced internal leak patterns with sanitized error envelopes:
  - `code`
  - `message`
  - `trace_id`
- Added correlation ID middleware for incident triage without exposing internals.

## 5. Prompt injection and LLM safety
- Added input policy checks in `SafetyService`:
  - jailbreak phrase detection
  - unsafe instruction patterns
  - empty/oversized input rejection
- Added retrieval context isolation wrapper (`<retrieved_context>...</retrieved_context>`).
- Added output moderation filters for secret-like patterns (`password`, `api_key`, `token`, etc.).

## 6. Session isolation
- Removed process-global session behavior from serving path.
- Added per-session context storage in cache/repository layer.
- Added TTL-based session expiration policy.

## 7. Security testing coverage
- Auth success/failure tests.
- Auth-required tests on query endpoints.
- Prompt-injection blocking tests.
- Rate-limit enforcement tests.
- Unsafe output filtering tests.

## Files of Interest
- `backend/app/core/security.py`
- `backend/app/core/settings.py`
- `backend/app/core/exceptions.py`
- `backend/app/api/v1/auth.py`
- `backend/app/api/v1/query.py`
- `backend/app/services/safety_service.py`
- `backend/app/middleware/body_limit.py`
- `backend/app/middleware/correlation.py`

## Follow-up Hardening (Recommended)
- Add refresh-token rotation + revocation list.
- Add account lockout/backoff on repeated failed auth.
- Move default admin credentials to one-time bootstrap flow.
- Enable secret scanning in CI and pre-commit.
- Add WAF/rate-limit policies at ingress level.
