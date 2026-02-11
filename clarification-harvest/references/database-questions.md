# Database Question Templates

Use this template when the user requests database schema design, migrations, or queries.

## Quick Checklist

- [ ] Type: PostgreSQL / MySQL / SQLite / MongoDB / DynamoDB
- [ ] ORM: SQLAlchemy / Prisma / TypeORM / Mongoose / Raw SQL
- [ ] Migrations: Alembic / Prisma Migrate / Flyway / Manual
- [ ] Relations: 1:1 / 1:N / N:M patterns needed
- [ ] Soft delete: Yes / No
- [ ] Audit fields: created_at, updated_at, created_by, updated_by

## Question Template

### Critical (P0)

- Q1 (P0) [Category: Integration]
  Question: Database connection string available?
  Options: Yes (stored in env) / No / SQLite (file-based)
  Default if unanswered: SQLite for local dev, connection string via `DATABASE_URL`

- Q2 (P0) [Category: Correctness]
  Question: Transaction boundaries — per-request or per-operation?
  Options: Per-request / Per-operation / Manual control
  Default if unanswered: Per-operation with explicit transactions where needed

### Major Design (P1)

- Q3 (P1) [Category: Scope]
  Question: Is this a new schema or modifying existing?
  Options: New schema / Add to existing / Migration only
  Default if unanswered: New schema with migration setup

- Q4 (P1) [Category: Performance]
  Question: Expected data volume?
  Options: Small (<10k rows) / Medium (<1M rows) / Large (>1M rows)
  Default if unanswered: Medium — include indexes on foreign keys and search fields

- Q5 (P1) [Category: Testing]
  Question: Test database strategy?
  Options: In-memory SQLite / Testcontainers / Shared test DB / Mocking
  Default if unanswered: In-memory SQLite with rollback after each test

### Polish (P2)

- Q6 (P2) [Category: Correctness]
  Question: Soft delete or hard delete?
  Options: Soft delete (deleted_at) / Hard delete / Both (configurable)
  Default if unanswered: Soft delete with `deleted_at` timestamp

- Q7 (P2) [Category: Correctness]
  Question: Audit fields needed?
  Options: None / Timestamps only (created_at, updated_at) / Full audit (+ created_by, updated_by)
  Default if unanswered: Timestamps only

- Q8 (P2) [Category: Deployment]
  Question: Migration execution strategy?
  Options: On startup / Manual CLI / CI/CD pipeline
  Default if unanswered: Manual CLI command (safer for production)
