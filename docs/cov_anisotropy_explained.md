# Why the rendered uncertainty ellipse stretches along one axis and rotates as the robot moves

> **STATUS (2026-05-02): RESOLVED for static-unobserved and released
> tracks.** The original ~17 cm ellipse on static unobserved tracks
> and the ~1.2 m released-bottle artifact were caused by the
> Ad-conjugate cross-coupling in `predict_static` running every frame
> regardless of observation status. The fix (plan §C.1 + §C.2,
> archived in memory `project_ekf_keyframe_mechanism.md`):
> `predict_static` now skips the cov update entirely for tracks with
> `frames_since_obs > 0` (the implicit-keyframe branch); the
> orchestrator wires `gravity_predict` into the Gaussian path via the
> new `overwrite_object_pose` so σ_yaw=π ends up in cov_bo at the
> release frame and is then frozen instead of pumped through Ad. At
> apple_drop fr 340 the static-track top-down σ_major dropped from
> ~17 cm to ~0.7-0.85 cm; the released bottle dropped from 119.7 cm
> to 14.9 cm with σ_yaw=π preserved. See `bernoulli_ekf.tex` §II.2
> for the math and §II.5 for the world-frame composition formula.
>
> **Held-track edge case still open:** rigid_attachment_predict still
> applies the Ad transport, so a held object whose σ_yaw drifts
> upward inside the holding window will pump that into σ_xy. Symptom:
> apple_to_cabinate_look oid=2 (held 147 frames) σ_xy_topdown ≈ 42 cm
> at fr 340. Mitigation TBD.
>
> The diagnostic narrative below is preserved as the historical
> explanation of WHY the artifacts existed.

## TL;DR

R_icp itself is **isotropic** in camera frame
(`pose_update/icp_pose.py:485`). The rendered xy ellipse becomes
anisotropic and direction-dependent because of how the EKF lifts that
isotropic noise through several frame transformations and accumulates
it across the predict step. The "rotation with motion" effect is a
correct geometric consequence of expressing object-frame uncertainty in
world coordinates: as the robot moves, the lever arm from the base
pose to the object changes direction, and the predict step's
Ad-conjugate cross-coupling accumulates growth in directions tied to
the **current** motion. Two empirically distinct regimes:

* **During robot motion** (apple_drop fr 200 / fr 274): the rendered
  major axis aligns with the **robot heading** (motion direction in
  world). σ_major ≈ 6–10 cm, σ_minor ≈ 2.5 cm, anisotropy ratio ≈ 3–4×.
* **After the robot stops** (fr 340, 60+ frames idle): the major
  axis aligns with the **line-of-sight from robot to object** in
  world. σ_major ≈ 17 cm, σ_minor ≈ 2.5 cm, anisotropy ratio ≈ 7×.
* **The released bottle** (oid 6 at fr 340): σ_major ≈ 1.2 m. This
  is dominated by `gravity_predict.py:226-227` setting σ_yaw ≈ π and
  the Ad-conjugate cross-coupling pumping that yaw uncertainty into
  translation cov over 60+ frames of post-release base motion. NOT a
  bug — a model choice for "the bottle could land in any orientation".

The earlier explanation I gave ("ICP measurement noise R_icp is
anisotropic along depth") was **wrong** — `R_icp` is built as
`np.diag([trans_var]*3 + [rot_var]*3)` with both blocks isotropic. The
right answer is more nuanced.

## What the visualizer renders

`tests/visualize_ekf_tracking.py:1745–1769` builds the top-down
ellipse from `cov_world[:2, :2]`, where `cov_world` is
`Σ_wo` from `gaussian_state.py:704–726`
(`collapsed_object_world`):

```
Σ_wo = Ad(T_bo⁻¹) · Σ_wb · Ad(T_bo⁻¹)ᵀ + Σ_bo                  (eq. 1)
```

The eigendecomposition of the upper-left 2×2 block of this matrix is
plotted as a 3σ ellipse on the world-xy plane. Empirically (verified
at fr 60, 100, 200, 274, 340 of apple_drop) the rendered major
direction matches each track's individual r2o direction (or heading
during driving), confirming the rendering treats `cov_world[:2, :2]`
in **world-frame xy coordinates**.

## The four contributing mechanisms

### Mechanism (1) — R_icp lift through Ad(T_bc)

`pose_update/object_belief.py:74–94` (`lift_measurement_base`):

```python
R_bo = Ad(T_bc) · R_icp · Ad(T_bc)ᵀ                           # line 92
```

Even though `R_icp = diag([trans_var]·3 + [rot_var]·3)` is isotropic
in camera frame, the SE(3) adjoint of `T_bc` couples translation and
rotation through the camera-mount lever arm `[t_bc]_× R_bc`. Working
out the trans-trans block (and noting `R_bc R_bcᵀ = I` absorbs the
rotation):

```
R_bo[trans, trans] = trans_var · I + rot_var · [t_bc]_× [t_bc]_×ᵀ
                   = trans_var · I + rot_var · (|t_bc|² · I − t_bc t_bcᵀ)
```

For Fetch with `t_bc ≈ (0.155, 0.004, 1.304)` m (extracted from the
fr 340 JSON dump), `|t_bc|² ≈ 1.72` and the eigenvalues of
`R_bo[trans, trans]` are:

* `trans_var` (≈ (2 cm)²) along `t_bc` — mostly **vertical**.
* `trans_var + rot_var · |t_bc|²` (≈ (14 cm)²) **perpendicular to
  `t_bc`** — i.e., the body horizontal plane (twice).

This anisotropy is fixed in body frame. Steady-state EKF posterior
inverts it: tight perpendicular to `t_bc`, loose along `t_bc`. The
visible effect on the top-down xy panel is **mostly out-of-plane**:
the loose direction is body-z (vertical), which contributes nothing
to the xy ellipse. The xy panel is roughly isotropic from this
mechanism alone, and the loosening shows up in the side-view panels
of the visualizer.

### Mechanism (2) — Predict-step Ad-conjugate cross-coupling

`pose_update/object_belief.py:129–159` (`predict_ad_conjugate`):

```
Σ_new = Ad(u_k) · Σ_old · Ad(u_k)ᵀ + Q                         # line 153
```

with `u_k = T_wb,k⁻¹ · T_wb,k-1` (the body-frame inverse motion;
`gaussian_state.py:309–310`). Working through the trans-trans block:

```
Σ_new[trans, trans] = R_u · Σ_old[trans, trans] · R_uᵀ
                    + R_u · Σ_old[trans, rot] · R_uᵀ · [t_u]_×ᵀ
                    + [t_u]_× R_u · Σ_old[rot, trans] · R_uᵀ
                    + [t_u]_× R_u · Σ_old[rot, rot] · R_uᵀ · [t_u]_×ᵀ
                    + Q[trans]
```

Two key terms compound across frames:

* The **rot-rot lever arm** `[t_u]_× R_u · Σ_old[rot] · R_uᵀ [t_u]_×ᵀ`
  pumps `Σ_old[rot]`'s yaw uncertainty into translation cov in the
  direction perpendicular to motion in the body frame.
* The **trans-rot cross-block propagation** keeps the body-frame
  trans-rot coupling alive across frames, allowing later predict steps
  to feed even more growth back into trans.

For pure forward motion (`R_u ≈ I`, `t_u = (-Δx, 0, 0)`) the
trans-trans growth simplifies to:

```
Σ_new[trans] += diag(0, Δx²·σ²_yaw, Δx²·σ²_yaw) + Q[trans]
```

Two regimes:

* **Normal driving** (σ_yaw_bo grows from `Q[rot] ≈ 1e-5/frame` for
  5 ≤ fso < 50): per-frame `Δx² · σ²_yaw ≈ (0.05)² · 5e-4 ≈ 1.25e-6 m²`
  → over 100 frames `σ_perp ≈ 1.1 cm`. Modest standalone, but combined
  with the trans-rot cross-block propagation it drives the ratio up.
* **Released bottle** (`gravity_predict.py:226-227` sets
  `σ_yaw ≈ π`): per-frame `Δx² · π² ≈ (0.025)² · 9.87 ≈ 6.2e-3 m²` →
  over 60 frames `σ_perp ≈ 0.61 m`. Dominates by a factor of ~50 over
  every other mechanism.

The empirical σ_major ≈ 1.197 m on oid 6 at fr 340 is consistent
with this mechanism: the bottle's gravity_predict-set σ_yaw, plus the
post-release robot motion (cumulative |Δx| ≈ 1.55 m between fr 274
and fr 340), gives the right magnitude. **NOT** a visualization bug;
**NOT** a runaway EKF; correct math under the model choice that "a
released bottle has uniform yaw uncertainty".

### Mechanism (3) — Σ_wb lever arm in the world composition

The first term of (eq. 1), `Ad(T_bo⁻¹) · Σ_wb · Ad(T_bo⁻¹)ᵀ`, lifts
the SLAM cov `Σ_wb` from `pose_update/slam_interface.py:286–318`
into the world tangent at T_wo. With the placeholder
`Σ_wb = diag([1e-4]·6)` (σ_trans = 1 cm, σ_yaw = 0.01 rad) and an
object at distance `r ≈ 3 m`:

```
Σ_wo[trans] += [t_ob]_× R_bo⁻¹ · Σ_wb[rot] · R_bo⁻ᵀ [t_ob]_×ᵀ
            ≈ r² · σ²_yaw_wb · diag(perpendicular to t_bo direction)
            ≈ 9 · 1e-4 ≈ 9e-4 m²
            → σ ≈ 3 cm perpendicular to t_bo in world frame.
```

**Empirically negligible for the static cluster.** The numerical
decomposition at fr 340 (per-track values from `cov_world` JSON dump):

| oid | label   | σ_major | σ_minor | frac_wb |
|-----|---------|--------:|--------:|--------:|
| 1   | cabinet |  18.1 cm |   2.6 cm |   0.6 % |
| 2   | apple   |  17.9 cm |   2.8 cm |   1.5 % |
| 3   | bottle  |  16.5 cm |   2.4 cm |   1.3 % |
| 4   | cup     |  16.8 cm |   2.5 cm |   0.9 % |
| 5   | apple   |  16.6 cm |   2.5 cm |   2.1 % |

`frac_wb` is the fraction of `σ²_major` that comes from the Σ_wb
lever-arm term `Ad(T_bo⁻¹) Σ_wb Ad(T_bo⁻¹)ᵀ` projected onto the
major direction. **It's tiny.** The major-axis variance is dominated
(>97%) by `Σ_bo` itself.

So this mechanism contributes a small fixed offset that **rotates
with t_bo** as the robot moves. It explains a visible but minor
component of the ellipse-rotation effect.

### Mechanism (4) — Body-frame anisotropy projected to world via R_wb

`Σ_bo`'s translation block carries an anisotropy in body coordinates
(from mechanisms 1, 2, and the EKF iterations). Projection to world is
implicit in the world composition. As the body heading rotates, the
direction of that body-frame anisotropy rotates with it in world.

For pure forward driving with stable heading, R_wb is constant: the
rendered ellipse's orientation is fixed across frames. For driving +
turning, the ellipse rotates with the body heading.

## Empirical verification across the apple_drop trajectory

Per-frame decomposition (post-fix `cov_world` from JSON):

| Frame | Phase            | σ_major | σ_minor | \|maj·r2o\| | \|maj·heading\| | frac_wb |
|------:|------------------|--------:|--------:|------------:|----------------:|--------:|
|    60 | observing        |   2 cm  |   2 cm  |    0.16     |     0.18        |   30 %  |
|   100 | observing        |   2 cm  |   2 cm  |    0.16     |     0.18        |   20 %  |
|   200 | driving (fso~80) |   6 cm  |  2.5 cm |    0.80     |   **0.99**      |   3.3 % |
|   274 | release          |  10 cm  |  2.5 cm |    0.86     |   **0.99**      |   1.9 % |
|   340 | idle (fso~220)   |  17 cm  |  2.5 cm |  **0.96**   |     0.38        |   0.6 % |

(values for oid 1 / cabinet; numbers for oid 2-5 follow the same
pattern within ±2 cm. Full table available by re-running the
decomposition script.)

**Reading the table:**

1. **fr 60, 100** — the static cluster has just been observed (fso ~30
   for cabinet / apple, 0 for newer detections). σ ≈ 2 cm at the
   `TRANS_VAR_FLOOR = 4e-4` floor (`icp_pose.py:273`) and isotropic.
   `frac_wb ≈ 20-30 %` because both Σ_bo and the lever arm are tiny
   (numerator and denominator are both small).
2. **fr 200, 274** — the robot has been driving toward the apple.
   Tracks unobserved for 60–160 frames. σ_major grows from 2 → 10 cm
   while σ_minor stays at the 2.5 cm floor. The major direction is
   **strongly aligned with the heading** (0.99). Mechanism (2)'s
   per-frame cross-coupling — accumulated through the trans-rot
   propagation across frames — is the only mechanism that can produce
   a heading-aligned major.
3. **fr 340** — robot has stopped. No more predict cross-coupling
   (since Δx ≈ 0). Σ_bo grows only via Q. Yet the major direction
   shifts from "heading" to "r2o" between fr 274 and fr 340. This is
   the world composition (eq. 1) absorbing the body-frame anisotropy
   plus the small Σ_wb lever-arm offset, expressed at the current
   robot pose. Specifically: the world projection of `Σ_bo`'s
   accumulated trans-rot coupling falls perpendicular to the **last**
   active motion direction; once the robot is stopped, the world
   projection settles into something dominated by the geometry of the
   final base pose relative to each object — which is the LoS
   direction.

## Why this is not a "covariance bug"

* The EKF math is correct (matches `docs/latex/bernoulli_ekf.tex` §
  II.2). Q is isotropic (`pose_update/ekf_se3.py:347-370`). R_icp is
  isotropic (`pose_update/icp_pose.py:485`).
* The render path was patched in this same session (Issues A and B in
  the visualizer): width/height/angle convention fixed; world-frame
  cov fed to the renderer. The numerical decomposition of `cov_world`
  matches the dumped values to ~2% relative error.
* `frac_wb < 3 %` for all observed-then-idle tracks: the placeholder
  `Σ_wb` is **not** the cause of the user-visible ellipse stretch.
* The released-bottle σ_major ≈ 1.2 m is a clean prediction of
  mechanism (2) under `gravity_predict.py:226-227`'s σ_yaw = π model.

## Tightening levers (separate from this explanation)

If the 1.2 m post-release ellipse on oid 6 is undesirable for
downstream planning:

* **Cap σ_yaw in `pose_update/gravity_predict.py:226-227`** to a
  shape-aware ceiling (e.g. π/2 for cylinders, π/3 for boxes).
  Cylinders rest in a small set of stable orientations; the uniform-π
  prior is conservative.
* **Bound the predict-step P_max for high-σ_yaw tracks**: the existing
  saturation hook `saturate_covariance` (called via `P_max` in
  `predict_ad_conjugate`, `object_belief.py:155`) is already wired —
  set a tighter cap when `σ_yaw_bo > π/2`.
* **Replace `PassThroughSlam`'s `Σ_wb` placeholder with a real
  per-frame SLAM cov estimator** (`slam_interface.py:286-318`). This
  doesn't help the static-track artefact but makes mechanism (3)'s
  ellipse rotation physically grounded instead of synthetic.

None of these are bugs in the EKF; they are **model-misspecification
levers**.

## Code references

* `pose_update/icp_pose.py:485` — R_icp construction (isotropic,
  verified).
* `pose_update/icp_pose.py:273-274` — TRANS_VAR_FLOOR / ROT_VAR_FLOOR.
* `pose_update/object_belief.py:74-94` — base-frame measurement lift
  via Ad(T_bc).
* `pose_update/object_belief.py:129-159` — Ad-conjugate predict.
* `pose_update/gaussian_state.py:309-310` — `u_k` definition.
* `pose_update/gaussian_state.py:704-726` — `collapsed_object_world`
  (Σ_wo = Ad(T_bo⁻¹) Σ_wb Ad(T_bo⁻¹)ᵀ + Σ_bo).
* `pose_update/ekf_se3.py:105-116` — SE(3) adjoint.
* `pose_update/ekf_se3.py:347-370` — Q matrices (isotropic).
* `pose_update/slam_interface.py:286-318` — PassThroughSlam Σ_wb
  placeholder.
* `pose_update/gravity_predict.py:226-227` — σ_yaw = π saturation at
  release.
* `tests/visualize_ekf_tracking.py:1745-1779` — top-down xy ellipse
  rendering (post-fix uses `cov_world`, descending eigenvalue order).
