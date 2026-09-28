// Browser port of accessmap/routing/router.py (same algorithm, same answers; checked by
// tests/test_routing.py::test_js_router_matches_python). Keep the two in step.
// Exposes globalThis.AccessRouter; works as a <script> and under Node (require/import).
(function () {
  "use strict";
  const MAX_SNAP_M = 250.0;
  const SNAP_PREFER_USABLE_M = 30.0;
  const MIN_COST_FACTOR = 0.1;
  const VIOLATION_M = 100000.0;
  const M_PER_DEG_LAT = 110574.0;
  const M_PER_DEG_LON_EQ = 111320.0;

  const round = (x, n) => Number(x.toFixed(n));

  class RoutingError extends Error {}

  // Binary min-heap of [cost, node], ordered like Python tuples.
  class Heap {
    constructor() { this.a = []; }
    get size() { return this.a.length; }
    static less(x, y) { return x[0] < y[0] || (x[0] === y[0] && x[1] < y[1]); }
    push(item) {
      const a = this.a; a.push(item);
      let i = a.length - 1;
      while (i > 0) {
        const p = (i - 1) >> 1;
        if (!Heap.less(a[i], a[p])) break;
        [a[i], a[p]] = [a[p], a[i]]; i = p;
      }
    }
    pop() {
      const a = this.a, top = a[0], last = a.pop();
      if (a.length) {
        a[0] = last;
        let i = 0;
        for (;;) {
          const l = 2 * i + 1, r = l + 1;
          let m = i;
          if (l < a.length && Heap.less(a[l], a[m])) m = l;
          if (r < a.length && Heap.less(a[r], a[m])) m = r;
          if (m === i) break;
          [a[i], a[m]] = [a[m], a[i]]; i = m;
        }
      }
      return top;
    }
  }

  class AccessRouter {
    constructor(graph) {
      this.g = graph;
      this.nodes = graph.nodes;
      this.edges = graph.edges;
      this.barriers = graph.barriers;
      this.profiles = graph.profiles;
      this.defaults = graph.defaults;
      this.aliases = graph.surface_aliases;
      this.maxDetour = graph.max_detour ?? null;
      const lat0 = this.nodes.reduce((s, n) => s + n[1], 0) / this.nodes.length;
      this.kx = M_PER_DEG_LON_EQ * Math.cos(lat0 * Math.PI / 180);
      this.adj = this.nodes.map(() => []);
      this.edges.forEach((e, i) => { this.adj[e.u].push([i, e.v]); this.adj[e.v].push([i, e.u]); });
    }

    // -- geometry ---------------------------------------------------------------------
    xy(p) { return [p[0] * this.kx, p[1] * M_PER_DEG_LAT]; }

    project(edge, lonlat) {
      const [px, py] = this.xy(lonlat);
      const pts = edge.coords.map(c => this.xy(c));
      const segLen = [];
      for (let i = 0; i + 1 < pts.length; i++) segLen.push(Math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]));
      const total = segLen.reduce((s, x) => s + x, 0) || 1.0;
      let best = [Infinity, 0.0, edge.coords[0]];
      let run = 0.0;
      for (let i = 0; i + 1 < pts.length; i++) {
        const a = pts[i], b = pts[i + 1];
        const dx = b[0] - a[0], dy = b[1] - a[1];
        const L2 = dx * dx + dy * dy;
        const f = L2 === 0 ? 0.0 : Math.max(0.0, Math.min(1.0, ((px - a[0]) * dx + (py - a[1]) * dy) / L2));
        const qx = a[0] + f * dx, qy = a[1] + f * dy;
        const d = Math.hypot(px - qx, py - qy);
        if (d < best[0]) best = [d, (run + f * segLen[i]) / total, [qx / this.kx, qy / M_PER_DEG_LAT]];
        run += segLen[i];
      }
      return best;
    }

    snap(lonlat, canLeave) {
      const cands = this.edges.map((e, i) => {
        const [d, t, q] = this.project(e, lonlat);
        return { edge: i, t, lonlat: q, snap_m: d };
      });
      cands.sort((a, b) => a.snap_m - b.snap_m || a.edge - b.edge);
      if (!cands.length || cands[0].snap_m > MAX_SNAP_M) {
        throw new RoutingError(`point is more than ${MAX_SNAP_M} m from the walking network`);
      }
      const limit = Math.min(cands[0].snap_m + SNAP_PREFER_USABLE_M, MAX_SNAP_M);
      for (const c of cands) {
        if (c.snap_m > limit) break;
        if (canLeave(c.edge, c.t)) return c;
      }
      return cands[0];
    }

    // -- costs ------------------------------------------------------------------------
    surfaceFactor(edge, profile) {
      const table = profile.surface_multiplier;
      const unknown = table.unknown ?? 1.0;
      const vals = edge.surface.map(s => table[this.aliases[s] ?? s] ?? unknown);
      return vals.length ? Math.max(...vals) : unknown;
    }

    edgeBarriers(edge, rejected, a = 0.0, b = 1.0) {
      const lo = Math.min(a, b), hi = Math.max(a, b);
      const out = {};
      for (const bi of edge.barriers) {
        const br = this.barriers[bi];
        if (rejected.has(br.id) || !(lo <= br.t && br.t <= hi)) continue;
        if (!(br.type in out) || br.conf > out[br.type].conf) out[br.type] = br;
      }
      return out;
    }

    edgeCost(edge, profile, rejected, a = 0.0, b = 1.0) {
      const walked = edge.len * Math.abs(b - a);
      let cost = walked * this.surfaceFactor(edge, profile);
      let blocked = false;
      const thr = this.defaults.forbid_threshold;
      for (const [t, br] of Object.entries(this.edgeBarriers(edge, rejected, a, b))) {
        if (profile.forbidden.includes(t)) {
          if (br.conf >= thr) blocked = true;
          else cost += this.defaults.fallback_penalty_m;
        } else {
          cost += (profile.penalty_m[t] ?? 0.0) * br.conf;
        }
      }
      if (edge.wheelchair_no && profile.osm_rules.wheelchair_no === "forbid") blocked = true;
      return [Math.max(cost, MIN_COST_FACTOR * walked), blocked];
    }

    // -- search -----------------------------------------------------------------------
    search(s, e, costFn) {
      let best = null;
      if (s.edge === e.edge) {
        const c = costFn(s.edge, s.t, e.t);
        if (c !== null) best = { cost: c, nodes: [], edges: [] };
      }
      const se = this.edges[s.edge], ee = this.edges[e.edge];
      const dist = new Map(), prev = new Map(), heap = new Heap();
      for (const [node, a, b] of [[se.u, s.t, 0.0], [se.v, s.t, 1.0]]) {
        const c = costFn(s.edge, a, b);
        if (c !== null && c < (dist.get(node) ?? Infinity)) { dist.set(node, c); prev.set(node, null); heap.push([c, node]); }
      }
      const ends = new Map();
      for (const [node, a, b] of [[ee.u, 0.0, e.t], [ee.v, 1.0, e.t]]) {
        const c = costFn(e.edge, a, b);
        if (c !== null) ends.set(node, Math.min(c, ends.get(node) ?? Infinity));
      }
      const done = new Set();
      while (heap.size) {
        const [d, n] = heap.pop();
        if (done.has(n)) continue;
        if (best !== null && d >= best.cost) break;
        done.add(n);
        if (ends.has(n) && (best === null || d + ends.get(n) < best.cost)) best = { cost: d + ends.get(n), end_node: n };
        for (const [ei, m] of this.adj[n]) {
          const w = costFn(ei, 0.0, 1.0);
          if (w === null) continue;
          const nd = d + w;
          if (nd < (dist.get(m) ?? Infinity)) { dist.set(m, nd); prev.set(m, [n, ei]); heap.push([nd, m]); }
        }
      }
      if (best === null) return null;
      if (!("end_node" in best)) return best;
      const nodes = [best.end_node], edges = [];
      while (prev.get(nodes[nodes.length - 1]) !== null) {
        const [n, ei] = prev.get(nodes[nodes.length - 1]);
        edges.push(ei); nodes.push(n);
      }
      return { cost: best.cost, nodes: nodes.reverse(), edges: edges.reverse() };
    }

    // -- describe a path --------------------------------------------------------------
    pieces(s, e, path) {
      if (!path.nodes.length) return [[s.edge, s.t, e.t]];
      const first = path.nodes[0], last = path.nodes[path.nodes.length - 1];
      const se = this.edges[s.edge], ee = this.edges[e.edge];
      const out = [[s.edge, s.t, se.u === first ? 0.0 : 1.0]];
      path.edges.forEach((ei, i) => {
        const ed = this.edges[ei];
        out.push(ed.u === path.nodes[i] ? [ei, 0.0, 1.0] : [ei, 1.0, 0.0]);
      });
      out.push([e.edge, ee.u === last ? 0.0 : 1.0, e.t]);
      return out;
    }

    subCoords(edge, a, b) {
      const pts = edge.coords.map(c => this.xy(c));
      const cum = [0.0];
      for (let i = 0; i + 1 < pts.length; i++) cum.push(cum[i] + Math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]));
      const total = cum[cum.length - 1] || 1.0;
      const at = f => {
        const x = f * total;
        for (let i = 1; i < cum.length; i++) {
          if (x <= cum[i] || i === cum.length - 1) {
            const seg = (cum[i] - cum[i - 1]) || 1.0;
            const r = Math.max(0.0, Math.min(1.0, (x - cum[i - 1]) / seg));
            const p = pts[i - 1], q = pts[i];
            return [(p[0] + r * (q[0] - p[0])) / this.kx, (p[1] + r * (q[1] - p[1])) / M_PER_DEG_LAT];
          }
        }
        return edge.coords[edge.coords.length - 1].slice();
      };
      const lo = Math.min(a, b), hi = Math.max(a, b);
      const inner = [];
      for (let i = 1; i < cum.length - 1; i++) if (lo * total < cum[i] && cum[i] < hi * total) inner.push(edge.coords[i].slice());
      const out = [at(lo), ...inner, at(hi)];
      return a <= b ? out : out.reverse();
    }

    describe(pieces, profile, rejected) {
      const coords = [];
      let length = 0.0;
      const found = [];
      const thr = this.defaults.forbid_threshold;
      const wheelchairRule = profile.osm_rules.wheelchair_no === "forbid";
      for (const [ei, a, b] of pieces) {
        const ed = this.edges[ei];
        if (a === b) continue;
        const seg = this.subCoords(ed, a, b);
        coords.push(...(coords.length ? seg.slice(1) : seg));
        length += ed.len * Math.abs(b - a);
        for (const [t, br] of Object.entries(this.edgeBarriers(ed, rejected, a, b))) {
          const forbidden = profile.forbidden.includes(t);
          found.push({ id: br.id, type: t, conf: br.conf, source: br.source, lonlat: br.lonlat,
            problem: forbidden || (profile.penalty_m[t] ?? 0.0) > 0,
            violation: forbidden && br.conf >= thr });
        }
        if (ed.wheelchair_no && wheelchairRule) {
          found.push({ id: `osm-wheelchair-no-${ei}`, type: "wheelchair_no", conf: 1.0, source: "osm",
            lonlat: seg[0], problem: true, violation: true });
        }
      }
      const counts = {};
      for (const br of found) if (br.problem) counts[br.type] = (counts[br.type] || 0) + 1;
      return { length_m: round(length, 1), coords, barriers: found, problems: counts,
        problem_count: Object.values(counts).reduce((s, x) => s + x, 0),
        violations: found.filter(br => br.violation).length,
        pieces: pieces.filter(([, a, b]) => a !== b).map(([ei, a, b]) => [ei, round(a, 4), round(b, 4)]) };
    }

    // -- public -----------------------------------------------------------------------
    route(start, end, profileName, rejected = new Set()) {
      const t0 = (typeof performance !== "undefined" ? performance : Date).now();
      if (!(profileName in this.profiles)) throw new RoutingError(`unknown profile '${profileName}'`);
      const profile = this.profiles[profileName];
      const full = this.edges.map(e => this.edgeCost(e, profile, rejected));
      const canLeave = (ei, t) => {
        const ed = this.edges[ei];
        return [[ed.u, 0.0], [ed.v, 1.0]].some(([node, end]) =>
          this.adj[node].length > 1 && !this.edgeCost(ed, profile, rejected, t, end)[1]);
      };
      const s = this.snap(start, canLeave);
      const e = this.snap(end, canLeave);
      if (s.edge === e.edge && Math.abs(s.t - e.t) * this.edges[s.edge].len < 1) {
        throw new RoutingError("start and end are the same point of the network");
      }
      const warnings = [];
      const lengthCost = (ei, a, b) => this.edges[ei].len * Math.abs(b - a);
      const profileCost = relaxed => (ei, a, b) => {
        let c, blocked;
        if ((a === 0.0 && b === 1.0) || (a === 1.0 && b === 0.0)) [c, blocked] = full[ei];
        else [c, blocked] = this.edgeCost(this.edges[ei], profile, rejected, a, b);
        if (blocked) return relaxed ? c + VIOLATION_M : null;
        return c;
      };
      const base = this.search(s, e, lengthCost);
      if (base === null) throw new RoutingError("no path between these points in the walking network");
      let acc = this.search(s, e, profileCost(false));
      if (acc === null) {
        acc = this.search(s, e, profileCost(true));
        warnings.push({ code: "no_barrier_free_path" });
      }
      const baseline = this.describe(this.pieces(s, e, base), profile, rejected);
      const accessible = this.describe(this.pieces(s, e, acc), profile, rejected);
      const ratio = baseline.length_m ? accessible.length_m / baseline.length_m - 1 : 0.0;
      if (this.maxDetour !== null && ratio > this.maxDetour + 1e-9) {
        warnings.push({ code: "detour_exceeds_limit", detour: round(ratio, 3), limit: this.maxDetour });
      }
      const accIds = new Set(accessible.barriers.map(br => br.id));
      const avoided = {};
      for (const br of baseline.barriers) if (br.problem && !accIds.has(br.id)) avoided[br.type] = (avoided[br.type] || 0) + 1;
      return {
        profile: profileName,
        from: { lonlat: s.lonlat, snap_m: round(s.snap_m, 1) },
        to: { lonlat: e.lonlat, snap_m: round(e.snap_m, 1) },
        baseline, accessible,
        same_route: JSON.stringify(baseline.pieces) === JSON.stringify(accessible.pieces),
        extra_m: round(accessible.length_m - baseline.length_m, 1),
        detour: round(ratio, 3),
        avoided, warnings,
        ms: round((typeof performance !== "undefined" ? performance : Date).now() - t0, 1),
      };
    }
  }

  AccessRouter.RoutingError = RoutingError;
  globalThis.AccessRouter = AccessRouter;
})();
