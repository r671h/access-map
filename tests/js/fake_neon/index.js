// In-memory fake of neon(): understands only the queries feedback.js sends.
export const rows = [];
export function neon() {
  return async (strings, ...v) => {
    const q = strings.join("?");
    if (q.includes("CREATE TABLE")) return [];
    if (q.includes("INSERT")) { rows.push({ id: rows.length + 1, barrier_id: v[0], verdict: v[1], comment: v[2], ip_hash: v[3] }); return [{ id: String(rows.length) }]; }
    if (q.includes("ip_hash =")) return [{ n: rows.filter(r => r.ip_hash === v[0]).length }];
    if (q.includes("GROUP BY")) {
      const src = q.includes("WHERE barrier_id") ? rows.filter(r => r.barrier_id === v[0]) : rows;
      const m = {}; for (const r of src) { const k = r.barrier_id + "|" + r.verdict; m[k] = (m[k] || 0) + 1; }
      return Object.entries(m).map(([k, n]) => { const [barrier_id, verdict] = k.split("|"); return { barrier_id, verdict, n }; });
    }
    throw new Error("unexpected query " + q);
  };
}
