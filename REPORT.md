# access-map report — Bielefeld centre core

## Imagery (phase 2)
- Area 0.92 km²; Mapillary metadata 4,581 images → 4,345 in bbox → 3,415 ≤ 6 years →
  2,184 with consistent raw/corrected positions → 1,555 not shot from motorway/trunk roads →
  1,404 within 15 m of the walk network → 695 after thinning (8 m × 45°) → **300 selected**
  (289 regular + 11 panoramas) → **333 frames** (panoramas → 4 crops of 90°).
- Most imagery is car dashcam footage (NEXTBASE watermarks, hood visible at the bottom), so
  sidewalks appear at the image edges, often at 5–15 m. Expect weaker detection of kerbs
  on the far side of the street than in on-foot imagery.
- QA sheets: `reports/qa/frames_sample.jpg`, `reports/qa/pano_check.jpg`,
  `reports/qa/frames_map.png`.
- Found and fixed during QA: frames from the B 61 trunk road (no pedestrian space), one
  sequence shifted 48 m by SfM correction, a panorama with a doubtful heading.
- Known leftovers: a few frames inside multi-storey car parks (indoor GPS is unreliable, so
  geometry filters miss them); Gemini's `image_usable` should reject them.

## Gemini pilot (phase 3, prompt v1, 30 frames, same frames for both models)
| model | valid | usable | features | raised_curb | narrow_passage | mean conf | tokens in/out | $/call (paid) |
|---|---|---|---|---|---|---|---|---|
| gemini-3.8-flash (thinking LOW) | 30/30 | 26 | 30 | 11 | 10 | 0.84 | 1770 / 361 | 0.0027 |
| gemini-3.1-flash-lite | 30/30 | 30 | 63 | 22 | 27 | 0.89 | 1770 / 326 | 0.0009 |

Visual check of `reports/qa/pilot_compare_1..5.jpg`:
- Boxes line up with the objects for both models (cobblestones, construction fences, bollards,
  a kerb ramp at a crossing): the `[ymin, xmin, ymax, xmax]` handling is correct.
- 3.8 Flash is conservative and correctly rejects unusable frames (dashcam view blocked by the
  mirror, a garage wall, a night Christmas-market photo). Errors seen: "stairs 0.85" on a
  distant sidewalk edge, one doubtful curb_ramp under a parked car.
- 3.1 Flash-Lite hallucinates at high confidence: 8 narrow_passage boxes on a row of plaza
  bollards, "step 0.90" on a Christmas pyramid, rough_surface on a manhole cover,
  no_sidewalk on a wall and on a mirror-blocked frame; it never marks a frame unusable.
- Shared v1 weakness: ordinary curbs running along the street are reported as
  `raised_curb`, although the definition means curbs at crossing points/corners.
  Main target for prompt tuning in phase 5.
- 3.8 Flash rejects `thinking_level=MINIMAL`; the client steps up to LOW automatically.

## Gemini full run (phase 3, gemini-3.8-flash, prompt v1, 333 frames)
- 333/333 valid (100%), 302 usable, 397 features, mean confidence 0.83; 303 new calls
  (366/1,200 used in total). Paid-tier equivalent ≈ $0.0025/frame, $0.83 for the area.
- Features: raised_curb 158, narrow_passage 107, rough_surface 55, curb_ramp 37, stairs 18,
  no_sidewalk 11, step 11. Permanence: 89 of 107 narrow_passage and 5 of 11 no_sidewalk are
  `temporary` (construction fences, sign bases, bins); geolocation keeps permanent ones only.
- Per-type sheets `reports/qa/types_<type>.jpg` (random 12 per type, opened and checked):
  - Boxes line up with the objects for every type; box order handling is correct.
  - raised_curb: dominated by ordinary along-street curbs (the known v1 issue); one inside a
    car park. Phase 5 target.
  - curb_ramp: mostly real lowered kerbs at crossings; one grass verge mislabelled.
  - stairs: real, but most are building-entrance stairs next to the sidewalk, not on the
    walking route; step: mixed (entrance steps, a sign base, a planter edge).
  - rough_surface: sett/cobblestone sidewalks and gutters, plausible.
  - no_sidewalk: 4 of 11 are car-park ramp frames from a dashcam with a blocked windshield
    that Gemini still called usable; 5 are construction fences (temporary).

## Geolocation (phase 4)
- 301 permanent detections placed (300 with Gemini's distance, 1 from the box bottom) →
  249 clusters (26 seen from more than one image) → 166 snapped, 83 unsnapped (33%).
- Unsnapped: raised_curb 60 of 126, curb_ramp 8/31, rough_surface 6/44, stairs 4/17,
  no_sidewalk 3/6, step 2/11, narrow_passage 0/14. Without raised_curb: 18.7%.
- Cause: v1 reports ordinary along-street curbs as raised_curb; these lie mid-block, a median
  18.7 m from any node or crossing, so they correctly fail the 12 m curb rule. User decision:
  keep 12/10 m and fix the class in phase 5.
- `reports/qa/barriers_map.png`, `barriers_map_zoom.png`: barriers follow the camera tracks and
  sit on sidewalks/footways, not inside blocks. Visible leftovers: building-entrance stairs
  snapped ~10 m onto the nearest footway, and no_sidewalk points from car-park ramp frames
  near Jahnplatz (unsnapped, so they do not affect routing).


## Evaluation and prompt tuning (phase 5)
- OSM ground truth in view of the frames is too thin for curb metrics (8 curb_ramp,
  1 raised_curb, 6 stairs), so per the user's choice Claude labelled 60 frames at frame
  level (labels/frame_labels.json; 42 tuning, 18 hold-out never looked at while tuning);
  the user spot-checked the disagreements and kept the labels. Strict labels give few
  positives (6 rough_surface, 3 curb_ramp, 2 stairs, 1 raised_curb, 8 unusable frames):
  dashcam imagery rarely shows curbs at crossing points clearly.
- OSM precision is a lower bound (OSM is incomplete); OSM numbers use objects in view only.
- All versions: gemini-3.8-flash, the same 150-frame tuning subset, min confidence 0.5.
  User priority: precision; targets for v2: along-street raised_curb, curb_ramp away from
  crossings, entrance steps/stairs.

Tuning subset, frame level vs labels (42 frames) and OSM:

| prompt | TP | FP | precision | recall | FP raised_curb | FP curb_ramp | FP step+stairs | stairs found | curb_ramp found | barriers | unsnapped | OSM curb_ramp TP/pred |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 | 6 | 35 | 0.15 | 0.75 | 14 | 7 | 6 | 1/1 | 2/2 | 129 | 31% | 1/23 |
| v2 | 6 | 11 | 0.35 | 0.75 | 0 | 4 | 2 | 0/1 | 2/2 | 56 | 23% | 0/21 |
| v3 | 6 | 12 | 0.33 | 0.75 | 0 | 4 | 2 | 1/1 | 1/2 | 59 | 25% | 0/22 |

- v1 → v2: false alarms 35 → 11 (raised_curb 14 → 0, step 3 → 0, curb_ramp 7 → 4), recall
  unchanged. v2's "NOT stairs: … striped paving" made it miss real plaza stairs.
- v3 = v2 + stairs fix + ~15 m curb_ramp limit: stairs recovered; curb_ramp false alarms on a
  grass island and a doorstep came back despite the explicit exclusion. v2 vs v3 differ by
  1–2 frames, which is within run-to-run noise at 42 labelled frames.
- Against OSM, v2 lost curb_ramp recall: v1 had curb_ramps 0 m and 5 m from two of the 6
  OSM lowered/flush kerbs in view; v2 has none within 12 m of any of them (nearest 12–290 m).
  The stricter "clearly visible crossing point" rule trades these for fewer false alarms.

### Final prompt v3 (user choice) — full run and hold-out
- Full run: 333/333 valid, 304 usable, 183 features (v1: 397). 183 new calls; total
  849/1,200. Geolocation: 109 barriers (v1: 249), 78 snapped, 31 unsnapped (28%, below the
  phase 4 limit of 30%); 10 seen from more than one image.
- Hold-out (18 labelled frames, scored once): precision 0.43 (3 TP / 4 FP), recall 0.75,
  image_usable 18/18 correct. Consistent with tuning (0.33 / 0.75); the sample is small.
- All 60 labelled frames: precision 0.36, recall 0.75. Per type: rough_surface 0.60 / 1.00,
  stairs 0.50 / 1.00, curb_ramp 0.17 / 0.33, raised_curb 0 / 0 (1 labelled instance).
- OSM, all frames (lower bound): curb_ramp 1 TP of 32 predicted, 8 OSM kerbs in view;
  stairs 1 of 5, 9 OSM steps in view. OSM cannot confirm most detections either way.
- Map `reports/qa/barriers_map.png`: the rows of along-street raised_curb from v1 are gone;
  barriers sit on the network.

### Typical errors that remain (v3)
1. rough_surface on cobbled surfaces that are not the walking route: a parking/loading lane
   (165811579359519), a dirt tree strip (4279130162134528), a cobbled median seen from a
   dashcam in rain (1380908117482935).
2. curb_ramp away from crossings despite the explicit exclusion: a grass traffic island
   (2602540939950109_right), a gateway threshold (555108905510671).
3. Distant building-entrance stairs still reported as stairs: 1110092740012739,
   700643409383152_back.
4. narrow_passage on poles and bollards with enough clear width: an overhead-line pylon
   (1073455601811251), bollards (774001902184920).
5. image_usable=true on frames blocked by rain, a wiper or night blur: 1600907968435799,
   923316520846444, 937976785374024.


## Routing (phase 6)
- `accessmap build-graph` → `data/processed/routing_graph.json` (316 KB): 1,055 nodes, 1,356
  walkable edges, 190 barriers on edges = 78 detected (all snapped ones) + 75 OSM steps edges
  + 37 OSM kerbs; 11 edges tagged `wheelchair=no`.
- Profiles as agreed (wheelchair blocks stairs/step/raised_curb/no_sidewalk, stroller blocks
  stairs, suitcase only penalties); max detour +50%.
- `accessmap route-demo` (`reports/routing_demo.md`, map `reports/qa/routing_demo.png`):
  10 pairs × 3 profiles; the accessible route differs from the shortest in 25 of 30; no
  warnings; median 17 ms, max 68 ms per request (limit 1 s). Example: pair 1, wheelchair
  +74 m (+11%) avoids 2 staircases and 1 rough surface.
- Some "different" routes have +0% and no barriers either way: the profile prefers smoother
  surfaces (surface multipliers), so an equal-length street with paving stones loses to asphalt.
- Found and fixed while testing: snapping a click to the nearest node started wheelchair
  routes on an underground stair-only stub at the Hauptbahnhof, producing a false "no
  barrier-free path". Now clicks snap to the nearest edge point the profile can leave from.
- The browser router (web/static/router.js) matched the Python router on 60 random real
  routes (0 mismatches) and on the synthetic test cases.
- Limits: kerbs are attached to edges, not to crossing movements, so a raised curb on one
  side of a crossing also affects walking along that corner; elevators are not modelled
  (OSM `highway=elevator` nodes), so stations with lifts can look inaccessible.


## Local vision model test (user request, 2026-09-28)
- Setup: Ollama + `qwen3-vl:8b-instruct` (Q4_K_M, 6.1 GB) on an RTX 5060 8 GB, same prompt v3,
  schema and cache as Gemini (`analyze --model ollama:<tag>`).
- Speed: with an 8k context, 20% of the model spilled onto the CPU (157 s/frame). A 4k context
  and a cap of 8 features / 1,500 output tokens brought it to 16–24 s/frame. The first answer
  had looped: 26 copies of one curb, each box shifted a few pixels.
- Quality, stopped at 76/150 tuning frames: on the 14 labelled frames both models answered,
  13 false alarms against Gemini's 1 (precision 0.07 vs 0.50). It reported along-street curbs
  despite v3's rule, invented obstacles on an empty night-time pavement, gave every detection
  confidence 0.95, and placed many boxes wrongly (on a facade, in the sky).
  Sheets: `reports/qa/local_vs_gemini_partial_*.jpg`.
- Decision (user): stay with Gemini. The backend stays as an option; a larger model
  (Qwen3-VL 30B-A3B, needs ~32 GB RAM) would be the fairer next test.

## Web app (phase 7)
- Map page: barrier layer (colour = type, size = confidence, faded = not on the network),
  filters, list, detail panel with the photo and detection box, confirm/reject feedback,
  profile selector, two-click routing, shortest vs accessible comparison. EN/DE.
- Statistics page (`/stats.html`, also on the static site): barriers per type on/off the
  network, per-type accuracy on all 60 labelled frames with raw counts, the hold-out headline,
  user feedback, and a method note.
- All endpoints curl-checked (200 with data; 404 for unknown frames and barriers; 422 for bad
  input). Screenshots `reports/qa/ui_phase7_*.png`: desktop and 390 px, EN and DE, no console
  errors, no horizontal scrolling.

## Export and demo (phase 8)
- `accessmap export-osm` → `data/processed/osm_suggestions.geojson`: 5 suggestions, all
  `kerb=lowered`, from 109 barriers. Skipped: 63 with no clear OSM tag (rough surface, narrow
  passage, no sidewalk), 13 below confidence 0.8, 27 seen in only one photo, and 1 staircase
  13 m from mapped OSM steps (same stairs, so not suggested; the steps radius is 20 m).
  3 of the 5 are 65–265 m from any mapped kerb; one is 15 m from one, likely the unmapped
  other side of a crossing.
- `accessmap demo`: a ~500 m excerpt north of Jahnplatz in `tests/fixtures/demo/` (2.2 MB:
  43 barriers, 51 downscaled photos, 357-edge routing graph, the real run's metrics). It was
  run with `GEMINI_API_KEY` and `MAPILLARY_TOKEN` unset: map, photos, routes and stats work
  (`reports/qa/ui_phase8_demo.png`). Offline tests cover it.

## Final numbers
| | |
|---|---|
| Area | Bielefeld centre core, 0.92 km², 36 km of walking network |
| Frames analysed | 333 (300 Mapillary images incl. 11 panoramas), 2021-07 to 2026-08 |
| Barriers | 109 (78 on the walking network, 10 seen in 2+ photos) |
| Precision / recall, hold-out (18 frames) | 0.43 / 0.75 |
| Precision / recall, all 60 labelled frames | 0.36 / 0.75 |
| Routing | accessible ≠ shortest in 25 of 30 demo routes; ≤ 68 ms per request |
| OSM suggestions | 5 lowered curbs not in OSM |
| Gemini calls | 849 of 1,200 budget (≈ $0.0025 per frame on the paid tier; the free tier was used) |
| Local model test | 76 frames, no API cost, not adopted |

## Spec self-check (docs/SPEC.md, section by section)
- §1 Goal, profiles: done; wheelchair, stroller, suitcase (config/profiles.yaml).
- §2 Area: done; coverage reported before fetching; the area was shrunk to the core at the
  phase 0 review, as the user decided.
- §3.1 OSM: done; walking network with the listed tags; Overpass kerbs, steps, barriers.
- §3.2 Mapillary: done; 0.005° tiles, paging, `computed_*` preferred, download right after the
  metadata fetch.
- §3.3 Frame selection: done, with additions: frames from motorway/trunk roads, images whose
  raw and corrected positions differ by more than 15 m, and misoriented panoramas are dropped
  (DECISIONS.md).
- §3.4 Panoramas: done; 4 × 90° crops at 768 px (larger crops add no detail from a 2048 px
  panorama); FOV from camera parameters, 70° fallback.
- §3.5 Own photos: not chosen at kickoff; not built.
- §4 Gemini: done. Changes: 4 worker threads instead of up to 8 (the key's real limit is
  60 RPM); the cache key is a fingerprint of model, prompt, schema, image and settings
  (stale answers are kept, not deleted); descriptions are returned in EN and DE in one call.
- §5 Taxonomy: 8 permanent types active. `construction` and `parked_vehicle` were not chosen,
  so temporary barriers do not occur.
- §6 Placement: done. Changed: bearing uses the pinhole model instead of the linear
  approximation (up to 4° more accurate). DBSCAN at 6 m, confidence cap 0.97, snapping at
  12/10 m, unsnapped barriers kept in a separate layer.
- §7 Evaluation: done, with a change the user approved: OSM had only 15 observable ground
  truth objects, so 60 frames were hand-labelled (42 tuning, 18 hold-out). OSM metrics are
  still reported as a lower bound. Three prompt versions were tried; v3 was chosen and scored
  once on the hold-out. The contact sheets are in reports/qa.
- §8 Routing: done; forbid threshold 0.6 with a fallback penalty, OSM steps, kerbs and
  `wheelchair=no`, OSM surfaces as multipliers, baseline + accessible route with stats and
  warnings. Simplified: kerbs act on whole edges, and elevators are not modelled.
- §9 Web app: done; all listed endpoints plus /api/graph, /api/frames, /stats.html. Rejected
  barriers leave routing immediately, both in the Python router and in the browser.
- §10 OSM export: done (above); never edits OSM; how to use it is in the README.
- §11 Offline demo: done (above). The server needs no network; the browser still fetches
  MapLibre and the base map, so it is not fully offline in a browser without internet.
- §12 Non-functional: timeouts, retries and backoff on all external calls; idempotent stages;
  logging with tqdm; tests offline (124 pass); ruff clean. `make` is missing on this Windows
  machine; the same commands work through `uv run accessmap …`.

## Known limitations
- Detection precision is modest (about 4 in 10 on held-out frames). Curb ramps are the weakest
  type (0.17 precision, 0.33 recall on all labelled frames). Most imagery is dashcam footage,
  where curbs at crossings are small or far away.
- Evaluation samples are small: 18 hold-out frames, and 1–9 labelled examples per type.
- Only 10 of 109 barriers were seen in more than one photo, which limits cross-checking and
  the OSM export (min 2 views).
- Kerbs block whole street segments instead of crossing movements; elevators are not modelled.
- The photos are up to 6 years old.

## Next steps (proposed)
1. **More views per barrier:** raise the image budget along the main walking routes (or add
   own photos) so more barriers are confirmed by 2+ photos. That improves precision through
   clustering and feeds the OSM export.
2. **Model curbs at crossings:** attach curb ramps and raised curbs to crossing movements
   instead of whole edges, and add elevators (`highway=elevator`) so stations are routed
   correctly.
3. **Use the feedback:** feed confirmed and rejected barriers back as labelled examples,
   report precision from real users on the stats page, and use disputed ones to tune the
   next prompt.
