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
