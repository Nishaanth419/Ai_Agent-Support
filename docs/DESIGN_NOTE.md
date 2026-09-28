# Schema Design Note

## Timestamp convention

All timestamps are `TEXT` in ISO-8601 UTC (`YYYY-MM-DDTHH:MM:SSZ`).
SQLite has no native date type; TEXT lets standard string ordering work for
range queries, and a single documented format avoids mixed-zone bugs.

## Status enumerations

`CHECK` constraints enforce allowed values inline rather than through separate
lookup tables, keeping the schema simple.  If the set of values must grow
frequently, migrate to a lookup table (as done for `ticket_categories`).

| Table       | Column   | Allowed values                                        |
|-------------|----------|-------------------------------------------------------|
| customers   | plan     | free, pro, enterprise                                 |
| tickets     | status   | open, pending, resolved, closed, escalated            |
| replies     | status   | proposed, approved, rejected                          |

## Proposed vs approved replies

`replies.status = 'proposed'` marks an AI-drafted reply awaiting review.
`replies.status = 'approved'` means a human approved it and it was sent;
`approved_at` is set at that moment.  A rejected draft keeps its row for
audit purposes; the agent creates a new `proposed` row when asked to retry.

## Ticket categories

`ticket_categories` is a separate table (not a `CHECK` list) so new
categories can be added at runtime without a DDL migration.  The FK on
`tickets.category_id` is nullable—tickets may be uncategorised initially and
categorised later by the agent.

## Escalations

One open escalation per ticket is enforced by a partial unique index on
`(ticket_id) WHERE resolved_at IS NULL`.  When the human queue resolves the
issue they set `resolved_at`; a new escalation can then be opened if needed.
`assigned_to` is a free-text field to avoid coupling to a specific agent
directory; the application layer resolves it.

## Foreign-key enforcement

SQLite disables FK checks by default.  `configure_connection()` (called by
both `connect()` and `init_db()`) issues `PRAGMA foreign_keys = ON` so
constraints are checked for every connection opened through this module.
Application code **must** use `db.connect()` or `db.init_db()` rather than
calling `sqlite3.connect()` directly.

## Extensions assumed safe to leave for later

- Full-text search on `tickets.body` (FTS5 virtual table).
- Soft-delete flag on customers / tickets.
- Multi-agent audit log (who approved which reply).
