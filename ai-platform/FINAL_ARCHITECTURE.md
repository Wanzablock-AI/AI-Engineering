# AI Engineering Platform — Final Architecture

## Status

**Architecture status:** Frozen after P11 Full E2E validation  
**Branch:** `main`  
**Latest platform commit:** `85d4403 Add Docker Compose platform stack`

The platform is a Dockerized AI orchestration layer integrating security,
model routing, evaluation/observability, production RAG, and multi-agent
research.

## Services

| Service | Container | Internal Port | Host Port |
|---|---|---:|---:|
| Gateway | `ai-platform-gateway` | 8000 | 8005 |
| Security | `ai-platform-security` | 8000 | — |
| Model Router | `ai-platform-model-router` | 8000 | — |
| Evaluation | `ai-platform-eval` | 8000 | — |
| Production RAG | `ai-platform-rag` | 8000 | — |
| Research | `ai-platform-research` | 8000 | — |

The Gateway is the public entry point.

## API

- `GET /health`
- `POST /v1/generate`
- `POST /v1/rag/query`
- `POST /v1/research`

## Generation Pipeline

Gateway → Security Middleware → Model Router → T440p Ollama → Evaluation → Gateway

## RAG Pipeline

Gateway → Production RAG → citations → Gateway

## Research Pipeline

Gateway → Multi-Agent Research → Ollama generation → synthesis → Gateway

## Request IDs

The platform uses `X-Request-ID`.

Client-supplied request IDs are preserved through the Gateway and returned
in response headers and response bodies.

If no request ID is supplied, the Gateway generates one.

## Security

The Security Middleware runs before model generation.

Security-blocked requests do not proceed to the Model Router.

Verified during P11:

`SECURITY_BLOCKED / prompt_injection`

## Ollama

Ollama runs on the Lenovo T440p server and is reached over the NetBird network.

Configured endpoint:

`http://100.118.104.245:11434`

Live Docker-to-Ollama connectivity was verified during P10/P11.

## Evaluation / Observability

Evaluation data is persisted in:

`eval-observability/data/eval-observability.db`

The `events` table records:

- request
- model
- provider
- latency
- success

The live P11 generation request was successfully persisted.

## RAG

Production RAG persists SQLite data under:

`production-rag/data/`

The Dockerized RAG service successfully completed a live query and returned
document/page citations.

## Research

The research service performs synchronous sequential LLM operations.

For multiple questions, the workflow performs research generation for each
question followed by final synthesis.

Long-running research requests can therefore take substantially longer than
simple generation requests.

A one-question live Gateway research request completed successfully during
P11 validation.

## Docker

The complete platform is defined by:

`compose.yaml`

Build:

`docker compose build`

Start:

`docker compose up -d`

Status:

`docker compose ps`

Logs:

`docker compose logs -f`

Stop:

`docker compose down`

Gateway:

`http://127.0.0.1:8005`

## Validation

P11 Full E2E validation completed successfully.

- Gateway health — PASS
- Live generation — PASS
- Security blocking — PASS
- RAG E2E — PASS
- Research E2E — PASS
- Request-ID propagation — PASS
- Evaluation persistence — PASS

Regression suite:

`43 passed in 0.61s`

## Known Operational Notes

### Model selection

The current Model Router returned `llama3.2:3b` even when the Gateway request
specified `granite4.1:3b`.

This is a known Model Router behavior and is outside the P11 freeze.

### Research latency

Multi-question research is synchronous and sequential. Long-running Ollama
generation can exceed short client timeouts.

This is a future optimization opportunity, not a P11 blocker.

## Freeze Boundary

Validated platform commits:

- `85d4403` Add Docker Compose platform stack
- `15311f4` Add platform integration test suite
- `4e75aea` Standardize errors and request IDs
- `4428f45` Integrate multi-agent research into AI platform
- `b9bac1d` Integrate RAG into AI platform
- `c6103fb` Add multi-agent research service
- `7b89ab8` Add evaluation and observability service
- `947c1f7` Dockerize security middleware

Independent `production-rag` and `model-router` working-tree changes are
intentionally excluded from this freeze.

## Operational Principle

Future changes should follow:

inspect → change one component → test → validate E2E → commit → freeze

The current architecture is the stable baseline.
