// Vercel serverless function: barrier feedback for the public (static) site.
//   GET  /api/feedback -> { barrier_id: { confirm, reject, status } }
//   POST /api/feedback { barrier_id, verdict: "confirm"|"reject", comment? } -> 201
// Storage: Neon Postgres via DATABASE_URL (set by the Vercel Neon integration). Without it
// the site stays read-only: GET returns {}, POST returns 503.
// Mirrors accessmap.web.app (FastAPI) so the frontend works the same locally and online.
import { createHash } from "node:crypto";
import { neon } from "@neondatabase/serverless";
import { BARRIER_IDS } from "./_ids.js";

const IDS = new Set(BARRIER_IDS);
const MAX_PER_HOUR = 30;   // per client, a simple guard against spam
let tableReady = null;

function db() {
  const url = process.env.DATABASE_URL || process.env.POSTGRES_URL;
  return url ? neon(url) : null;
}

async function ensureTable(sql) {
  tableReady ??= sql`CREATE TABLE IF NOT EXISTS feedback (
      id BIGSERIAL PRIMARY KEY,
      barrier_id TEXT NOT NULL,
      verdict TEXT NOT NULL CHECK (verdict IN ('confirm', 'reject')),
      comment TEXT NOT NULL DEFAULT '',
      ip_hash TEXT NOT NULL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now())`;
  try {
    await tableReady;
  } catch (e) {
    tableReady = null;
    throw e;
  }
}

// Same rule as accessmap.web.app.feedback_status.
const status = (c, r) => (r > c ? "rejected" : c > r ? "confirmed" : "disputed");

async function summary(sql, barrierId = null) {
  const rows = barrierId
    ? await sql`SELECT barrier_id, verdict, COUNT(*)::int AS n FROM feedback
                WHERE barrier_id = ${barrierId} GROUP BY barrier_id, verdict`
    : await sql`SELECT barrier_id, verdict, COUNT(*)::int AS n FROM feedback
                GROUP BY barrier_id, verdict`;
  const out = {};
  for (const { barrier_id, verdict, n } of rows) {
    out[barrier_id] ??= { confirm: 0, reject: 0 };
    out[barrier_id][verdict] = n;
  }
  for (const s of Object.values(out)) s.status = status(s.confirm, s.reject);
  return out;
}

function clientHash(req) {
  const ip = String(req.headers["x-forwarded-for"] || "").split(",")[0].trim() || "unknown";
  const salt = process.env.FEEDBACK_SALT || "access-map";
  return createHash("sha256").update(`${salt}:${ip}`).digest("hex");  // never store raw IPs
}

export default async function handler(req, res) {
  res.setHeader("Cache-Control", "no-store");
  const sql = db();
  try {
    if (req.method === "GET") {
      if (!sql) return res.status(200).json({});
      await ensureTable(sql);
      return res.status(200).json(await summary(sql));
    }
    if (req.method !== "POST") {
      res.setHeader("Allow", "GET, POST");
      return res.status(405).json({ detail: "method not allowed" });
    }
    if (!sql) return res.status(503).json({ detail: "feedback storage is not configured" });

    let body = req.body;
    if (typeof body === "string") {
      try { body = JSON.parse(body || "{}"); } catch { body = null; }
    }
    const { barrier_id: barrierId, verdict, comment = "" } = body || {};
    if (typeof barrierId !== "string" || !IDS.has(barrierId)) {
      return res.status(404).json({ detail: "unknown barrier" });
    }
    if (verdict !== "confirm" && verdict !== "reject") {
      return res.status(422).json({ detail: "verdict must be confirm or reject" });
    }
    if (typeof comment !== "string" || comment.length > 500) {
      return res.status(422).json({ detail: "comment must be a string of at most 500 characters" });
    }

    await ensureTable(sql);
    const ipHash = clientHash(req);
    const [{ n }] = await sql`SELECT COUNT(*)::int AS n FROM feedback
                              WHERE ip_hash = ${ipHash} AND created_at > now() - interval '1 hour'`;
    if (n >= MAX_PER_HOUR) return res.status(429).json({ detail: "too much feedback, try later" });

    const [{ id }] = await sql`INSERT INTO feedback (barrier_id, verdict, comment, ip_hash)
                               VALUES (${barrierId}, ${verdict}, ${comment.trim()}, ${ipHash})
                               RETURNING id`;
    const s = await summary(sql, barrierId);
    return res.status(201).json({ id: Number(id), barrier: s[barrierId] });
  } catch (e) {
    console.error("feedback error", e);
    return res.status(500).json({ detail: "feedback storage error" });
  }
}
