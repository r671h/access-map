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
