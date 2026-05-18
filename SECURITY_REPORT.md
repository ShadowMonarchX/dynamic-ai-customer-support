# SECURITY_REPORT.md

Audit date: 2026-05-18

## Security Posture Summary
Current posture: **High risk** for internet-exposed deployment.

Primary risks:
- No authentication/authorization on primary query endpoint.
- Error detail leakage to clients.
- PII is committed in repository training data and output logs.
- No rate limiting, abuse prevention, or security monitoring.
- Prompt-injection and LLM output safety controls are absent.

## Attack Surface
- REST API (`backend/app/main.py`, `/query` endpoint)
- Ingestion and vector retrieval pipeline
- LLM prompt assembly and output generation
- Repository data artifacts (`backend/app/data/*`, `backend/output/*`)

## Findings

| ID | Severity | Finding | Evidence | Risk | Remediation |
|---|---|---|---|---|---|
| SEC-001 | Critical | Missing authN/authZ | `backend/app/main.py:101-181` | Unauthorized use, data exfiltration, abuse | Add OAuth2/JWT + RBAC + API gateway auth |
| SEC-002 | High | Internal errors leaked to user | `HTTPException(status_code=500, detail=str(e))` at `main.py:181` | Information disclosure | Return opaque error codes; log full stack internally |
| SEC-003 | High | PII committed in repo and logs | `backend/app/data/training_data.txt`, `new_training_data.txt`, `backend/output/output_view_*.txt` | Compliance/privacy risk | Remove/redact PII, rotate exposure, add DLP pre-commit/CI scans |
| SEC-004 | High | No rate limiting or request size guardrails | no middleware/policy in API | DoS cost amplification | Add per-IP/per-token throttles + body size limits |
| SEC-005 | High | Prompt injection controls missing | `context_assembler.py:50-52` direct context concatenation | Model instruction override/exfiltration | Add trusted system templates + retrieval sanitization + output policy checker |
| SEC-006 | Medium | Global shared session state | `SESSION_ID` global in `main.py:35` | Cross-user context bleed | Session isolation via user/session token and TTL store |
| SEC-007 | Medium | Broad exception swallowing | e.g. `llm_reasoner.py:172,213` | Security-relevant failures hidden | Typed exceptions + security event logging |
| SEC-008 | Medium | Placeholder unauthenticated user CRUD | `backend/app/api/routes.py:5-21` | Misuse if mounted | Remove placeholder routes or secure behind auth |

## Recommended Security Architecture
1. API protection
- JWT/OAuth2 auth middleware
- RBAC policy for all endpoints
- Rate limiting and WAF/API gateway rules

2. Data protection
- Remove personal identity corpus from VCS and logs
- Add data classification tags (public/internal/confidential)
- Encrypt at rest for vector index and any persisted data

3. LLM safety
- Prompt hardening with immutable instruction frame
- Input risk classifier (prompt injection, jailbreak, sensitive queries)
- Output safety validator (PII leakage, policy checks)

4. Secure ops
- Secrets in Vault/Secrets Manager
- Centralized audit logging
- Security alerting (Sentry + SIEM)

## Security Testing Plan
- Auth bypass tests (`/query` without token, invalid token, privilege escalation)
- Rate limit tests (burst and sustained)
- Prompt injection tests (instruction override, data exfiltration attempts)
- Error response tests (verify no stack traces/internal class names)
- Data governance tests (PII scanners on repo and runtime outputs)

## Immediate 7-Day Remediation
1. Add auth + rate limiting + safe error envelopes.
2. Remove or redact PII artifacts from repository.
3. Add prompt safety checks and output policy validation.
4. Add CI security scans (`bandit`, dependency audit, secret scan).
