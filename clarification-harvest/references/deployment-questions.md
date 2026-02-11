# Deployment Question Templates

Use this template when the user requests deployment, CI/CD, or infrastructure setup.

## Quick Checklist

- [ ] Platform: AWS / GCP / Azure / VPS / Self-hosted / Vercel / Fly.io
- [ ] Method: Docker / Native / Serverless / Kubernetes
- [ ] CI/CD: GitHub Actions / GitLab CI / CircleCI / None
- [ ] Secrets: ENV files / Vault / KMS / Parameter Store
- [ ] DB: Managed (RDS/Cloud SQL) / Self-hosted / Containerized
- [ ] Domain: Existing / New / Not needed

## Question Template

### Critical (P0)

- Q1 (P0) [Category: Integration]
  Question: Target deployment platform?
  Options: AWS / GCP / Azure / VPS (DigitalOcean/Linode/etc) / Vercel / Fly.io / Self-hosted / Not sure
  Default if unanswered: Docker setup (portable across platforms)

- Q2 (P0) [Category: Security]
  Question: Secrets management approach?
  Options: Environment variables / Secret manager (AWS SM/GCP SM) / Vault / Docker secrets
  Default if unanswered: Environment variables with `.env.example` template

### Major Design (P1)

- Q3 (P1) [Category: Deployment]
  Question: Container orchestration?
  Options: None (single container) / Docker Compose / Kubernetes / ECS / Cloud Run
  Default if unanswered: Docker Compose for multi-service, single container otherwise

- Q4 (P1) [Category: Deployment]
  Question: CI/CD pipeline needed?
  Options: GitHub Actions / GitLab CI / None (manual deploy)
  Default if unanswered: GitHub Actions with test + build on PR, deploy on main

- Q5 (P1) [Category: Performance]
  Question: Scaling requirements?
  Options: Single instance / Horizontal scaling (auto) / Vertical scaling
  Default if unanswered: Single instance with health checks

### Polish (P2)

- Q6 (P2) [Category: Deployment]
  Question: Database deployment included?
  Options: Yes (managed) / Yes (containerized) / No (use existing)
  Default if unanswered: No — assume existing or document connection setup

- Q7 (P2) [Category: Testing]
  Question: Health check endpoint required?
  Options: Simple (200 OK) / Detailed (DB, external deps) / None
  Default if unanswered: Detailed with dependency checks

- Q8 (P2) [Category: Deployment]
  Question: Monitoring/observability?
  Options: None / Basic logging / Structured logs / APM (Datadog/New Relic)
  Default if unanswered: Structured JSON logging
