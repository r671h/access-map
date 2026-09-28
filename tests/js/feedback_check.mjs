import assert from "node:assert/strict";
const { default: handler } = await import("./api/feedback.js");
const { rows } = await import("@neondatabase/serverless");
function call(method, body, ip = "1.2.3.4") {
  return new Promise(resolve => {
    const res = { headers: {}, code: 0, setHeader(k, v) { this.headers[k] = v; },
      status(c) { this.code = c; return this; }, json(o) { resolve({ code: this.code, body: o }); } };
    handler({ method, body, headers: { "x-forwarded-for": ip } }, res);
  });
}
delete process.env.DATABASE_URL;
assert.deepEqual(await call("GET"), { code: 200, body: {} });
assert.equal((await call("POST", { barrier_id: "stairs-1", verdict: "reject" })).code, 503);
process.env.DATABASE_URL = "postgres://fake";
assert.equal((await call("POST", { barrier_id: "nope", verdict: "reject" })).code, 404);
assert.equal((await call("POST", { barrier_id: "stairs-1", verdict: "maybe" })).code, 422);
assert.equal((await call("POST", { barrier_id: "stairs-1", verdict: "reject", comment: "x".repeat(501) })).code, 422);
assert.equal((await call("POST", "not json")).code, 404);
let r = await call("POST", JSON.stringify({ barrier_id: "stairs-1", verdict: "reject", comment: " kerb " }));
assert.equal(r.code, 201); assert.deepEqual(r.body.barrier, { confirm: 0, reject: 1, status: "rejected" });
assert.equal(rows[0].comment, "kerb"); assert.ok(!rows[0].ip_hash.includes("1.2.3.4") && rows[0].ip_hash.length === 64);
r = await call("POST", { barrier_id: "stairs-1", verdict: "confirm" }, "5.6.7.8");
assert.deepEqual(r.body.barrier, { confirm: 1, reject: 1, status: "disputed" });
assert.deepEqual((await call("GET")).body, { "stairs-1": { confirm: 1, reject: 1, status: "disputed" } });
assert.equal((await call("DELETE")).code, 405);
for (let i = 0; i < 29; i++) await call("POST", { barrier_id: "curb_ramp-1", verdict: "confirm" }, "9.9.9.9");
assert.equal((await call("POST", { barrier_id: "curb_ramp-1", verdict: "confirm" }, "9.9.9.9")).code, 201);
assert.equal((await call("POST", { barrier_id: "curb_ramp-1", verdict: "confirm" }, "9.9.9.9")).code, 429);
console.log("feedback.js: all checks passed");
