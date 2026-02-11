# Web API Question Templates

Use this template when the user requests building or modifying a web API.

## Quick Checklist

- [ ] Protocol: REST / GraphQL / gRPC / WebSocket
- [ ] Framework: Express / FastAPI / Django / Flask / Next.js
- [ ] Auth: None / JWT / OAuth2 / API Key / Session
- [ ] Data format: JSON / XML / MessagePack
- [ ] Documentation: OpenAPI / None
- [ ] Rate limiting: None / Basic / Advanced
- [ ] Validation: Pydantic / Zod / Joi / Manual

## Question Template

### Critical (P0)

- Q1 (P0) [Category: Integration]
  Question: Does this API need to integrate with external services?
  Options: Yes (specify below) / No
  Default if unanswered: No

- Q2 (P0) [Category: Security]
  Question: What authentication mechanism is required?
  Options: None / JWT / OAuth2 / API Key / Session-based
  Default if unanswered: JWT with env var `JWT_SECRET`

### Major Design (P1)

- Q3 (P1) [Category: UX/API Design]
  Question: REST or GraphQL?
  Options: REST / GraphQL / Both
  Default if unanswered: REST with OpenAPI docs

- Q4 (P1) [Category: Performance]
  Question: Expected request volume?
  Options: Low (<1k/day) / Medium (<100k/day) / High (>100k/day)
  Default if unanswered: Medium — include basic rate limiting

- Q5 (P1) [Category: Testing]
  Question: What test coverage is needed?
  Options: Unit only / Unit + Integration / Full (unit + int + e2e)
  Default if unanswered: Unit + Integration with pytest

### Polish (P2)

- Q6 (P2) [Category: UX/API Design]
  Question: Response envelope format?
  Options: Raw data / {data, meta} / {success, data, error}
  Default if unanswered: Raw data with appropriate HTTP status codes

- Q7 (P2) [Category: Deployment]
  Question: Containerization needed?
  Options: Yes (Dockerfile) / Yes (Docker Compose) / No
  Default if unanswered: Yes — Dockerfile with multi-stage build

- Q8 (P2) [Category: Correctness]
  Question: Input validation strictness?
  Options: Lenient / Strict (reject unknown fields) / Extra strict (custom rules)
  Default if unanswered: Strict with Pydantic v2
