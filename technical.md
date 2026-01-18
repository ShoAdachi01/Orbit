## Engineering PRD (2 pages): Project Multi-Cam v1 (“Orbit”)

### 0) One-line

Turn a single creator video into a **bounded parallax orbit preview** + a **high-fidelity exported MP4**, with explicit quality gates and fallbacks so the product is predictable.

---

### 1) Target user + use cases

**Primary user:** creator / editor who wants “fake multi-cam orbit” from one clip.

**Core use cases**

1. Upload clip → get an interactive orbit preview (small, believable parallax)
2. Author an orbit camera path
3. Export a polished MP4 that looks like a real camera move (within constraints)

**Non-goals (v1)**

* 90° freeview / full scene recovery
* multi-person scenes
* outdoor foliage-heavy / chaotic motion clips without capture guidance
* “perfectly accurate 3D”; we aim for *believable parallax + stable identity*

---

### 2) Product contract (hard constraints, enforced)

**Orbit bounds (default, enforced)**

* yaw ≤ ±20°
* pitch ≤ ±10°
* roll ≤ ±3°
* translation ≤ min(0.10 m, 2% median scene depth)
* clamp_to_parallax = true

**Scene bounds**

* single primary subject (largest consistent foreground)
* background mostly static outside subject mask

**Clip bounds**

* recommended 5–20s
* supported up to 60s best-effort (auto downgrade orbit/tier)

---

### 3) User experience (UX)

**Flow**

1. Upload video
2. System runs quick local checks + shows expected quality badge (Good / Risky / Likely Downgrade)
3. User sees a low-res orbit preview and draws camera path (bounded)
4. “Render” → gets:

   * **Orbit Preview Scene** (interactive)
   * **Export MP4** (final), plus a visible “mode” label if fallback occurred

**User-visible modes**

* Full Orbit (normal)
* Micro-Parallax (orbit clamps reduced)
* 2.5D Subject (subject becomes billboard-like)
* Render-Only (no interactive 3D; stabilized cinematic export only)

---

### 4) System overview (what we ship)

**Canonical output representation (OrbitScene pack)**

* bg.splat (static background)
* subject_4d.splat (dynamic subject)
* camera.json (intrinsics + reference frame)
* quality.json (pose/depth/track metrics + orbit bounds + fallback mode)
* lod/ (optional chunked assets for streaming)

**Main pipeline**

* Local (Mac): proxy + tagging + camera path UI
* Cloud:

  1. segmentation/masks
  2. background-only pose solve + QA gate
  3. depth prior + confidence
  4. background splat + subject 4D reconstruction
  5. base render + generative refinement + identity lock
  6. publish OrbitScene + MP4 + quality report

---

### 5) Success metrics (measurable)

**User-perceived**

* ≥80% of “Good” clips produce export that passes a simple human QA (“looks like a real camera move”)
* ≤5% catastrophic failures (identity swap, severe wobble, unusable output) on “Good” clips
* interactive preview starts <5s after scene assets ready (progressive streaming)

**Technical**

* Pose stability: jitter score under threshold (defined below)
* Identity stability: face embedding drift under threshold on anchor frames
* Temporal flicker: frame-to-frame diff metric under threshold (in refined output)

---

### 6) Risks + mitigation (explicit)

**Risks**

* Pose solve fails (low texture, subject dominates)
* Depth prior wrong (hair/hands/blur)
* Segmentation leakage contaminates pose and geometry
* Refinement causes identity/logo drift

**Mitigations**

* Hard QA gates + fallback modes
* Confidence-weighted priors (no pseudo-ground truth)
* Anchor frame conditioning for identity/logo lock
* “Fail closed” to base_render when refinement risks identity

---

### 7) Deliverables (v1)

* OrbitScene asset pack + WebGPU viewer
* Render worker producing base_render + refined MP4
* Quality gates + fallbacks + user-visible explanation
* API contract + progress events + output artifacts

---

## Implementation checklist: gates, metrics, fallbacks, and build plan

### A) Data contracts (must be decided early)

* **Pose convention:** T_world_from_cam, right-handed, meters, row-major 4×4
* **Intrinsics:** fx, fy, cx, cy in pixels + distortion model (brown-conrady default)
* **Masks:** lossless PNG sequence (uint8 0/255). Avoid MP4 masks in v1.
* **Camera path resampling:** slerp rotation + linear translation + linear fov to render FPS.

✅ Output contract includes: OrbitScene pack, base_render.mp4, final_render.mp4, quality.json, logs.

---

### B) Quality gates (hard checks) and what to compute

#### B1) Segmentation gate (mask sanity)

Compute per frame:

* subject area ratio (0–1)
* edge jitter (mask boundary displacement over time)
* “leak score” heuristic: high-frequency mask fragments outside main component

**Fail conditions**

* subject covers >65% frame for >30% of frames → mark “pose risk”
* severe mask jitter → require Micro-Parallax or 2.5D Subject fallback candidate

Outputs:

* `mask_quality_score` (0–1)
* per-frame stats for debugging

---

#### B2) Pose gate (background-only camera solving)

From pose solver:

* inlier ratio
* median reprojection error (px)
* track coverage: distribution across image quadrants
* pose jitter score: norm(ΔR, Δt) high-frequency energy

**Pass thresholds (starting point; tune later)**

* inlier ratio ≥ 0.35
* median reprojection error ≤ 2.0 px (scaled by resolution)
* coverage: ≥3 quadrants populated for ≥60% of frames
* jitter score ≤ threshold

**If fail → fallback selection begins**

* If pose is unstable but not catastrophic → Micro-Parallax
* If pose basically unusable → Render-Only

---

#### B3) Track gate (foreground TAPIR)

For each track:

* confidence
* lifespan (frames)
* consistency vs neighborhood motion

**Discard tracks**

* conf < 0.6 (initial)
* lifespan < 15 frames
* large residual vs local motion model

Track health summary:

* `num_tracks_kept`, `median_lifespan`, `median_conf`

If too few tracks → 2.5D Subject fallback candidate.

---

#### B4) Depth confidence (DepthCrafter as prior)

Compute:

* temporal consistency (depth change smoothness)
* edge stability around subject boundary
* agreement with multiview cues (if available) / reprojection sanity

Produce:

* `depth_conf_map` (optional) or per-frame scalar `depth_conf`

**Use rule**
Depth weight = depth_conf * global_depth_weight (tunable). Never treat depth as authority.

---

### C) Reconstruction rules (the “no hallucination as truth” part)

#### C1) Background completion prior (GaMO panorama)

* Generate background prior only after subject mask available.
* Compute hole confidence:

  * disocclusion likelihood + no track support + mask certainty

**Pixel fusion**

* Always trust real pixels weight = 1.0
* Prior weight = clamp(0, 0.2, hole_conf * depth_conf * track_support_conf)

If hole_conf is low → leave hole (and let refinement handle lightly) rather than injecting wrong background.

---

#### C2) Canonical scene generation

* Produce bg.splat using masked frames + poses
* Produce subject_4d.splat via SoM + depth prior + tracks
* Composite rule is fixed and documented (alpha + boundary stabilization)

Deliver `quality.json` with:

* selected mode (Full/Micro/2.5D/Render-only)
* pose metrics
* mask metrics
* depth metrics
* track metrics
* enforced orbit bounds

---

### D) Refinement logic (identity/logo stability)

#### D1) Anchor frame selection

Select K anchor frames:

* sharpness high
* face/logo visible
* minimal motion blur
* representative lighting

Store:

* anchor frame ids + crops + embeddings

#### D2) Identity drift detection

During refinement (or post):

* compute face embedding drift vs anchors
* compute logo/text drift heuristics (optional OCR-like score)

**Fail-closed rule**
If drift exceeds threshold:

* reduce refinement strength and re-run short pass OR
* output base_render as final (with “Render-Only / Base” label)

---

### E) Fallback mode decision tree (explicit)

1. **Pose fails hard** → Render-Only
2. **Pose borderline** → Micro-Parallax (shrink yaw/pitch/translation clamps)
3. **Pose ok but subject unstable (tracks/depth)** → 2.5D Subject
4. **Everything ok** → Full Orbit

Always surface mode to user and include it in output metadata.

---

### F) Build plan (what to implement first)

**Week 1–2: Contracts + Viewer skeleton**

* finalize pose conventions, masks, schema
* WebGPU viewer that can load OrbitScene pack (even dummy assets)
* job orchestration + progress events

**Week 3–4: Pose gate + background splat**

* SAM2 masks + mask QA metrics
* background-only pose solve + pose QA metrics
* bg.splat generation + streaming LOD

**Week 5–6: Subject layer**

* TAPIR tracks + gating
* DepthCrafter integration + confidence
* SoM subject reconstruction (initial) + 2.5D fallback implementation

**Week 7–8: Base render + refinement**

* base_render pipeline from OrbitScene along camera path
* refine with identity lock + drift detection
* fail-closed logic to base_render

**Week 9+: Tune thresholds + collect dataset**

* internal eval suite
* threshold tuning by “Good/Risky” labels
* build capture guidance prompts from failure stats

---

### G) Debugging artifacts (non-negotiable)

For every job, store (even if only for internal builds):

* mask preview mp4 (lossless-ish)
* pose reprojection overlay video
* depth preview video
* track overlay video
* base_render + refined_render
* quality.json + selected fallback + reasons

If you don’t store these, you won’t be able to iterate.

---

