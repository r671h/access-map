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
