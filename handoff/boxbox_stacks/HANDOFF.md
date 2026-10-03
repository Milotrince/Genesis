# Box stacks handoff

Branch `trinity/boxbox-stacks-handoff`, off `trinity/boxbox-fp32-tie`
([PR 3469](https://github.com/Genesis-Embodied-AI/genesis-world/pull/3469), "Fix stability issues observed in stack
of boxes"). It adds two commits on top of the PR head `19d14ff6`, plus this folder, which is scratch material and must
not be merged.

## What the two commits do

**Box-box edge/face tie tolerance** (`genesis/engine/solvers/rigid/collider/box_contact.py`). In
`func_box_box_contact`, an edge-edge separating axis now has to beat the face axis by `tol_edge / c1` instead of
`tol_edge`, where `c1` is the norm of the edge cross product. The edge-edge separation divides its size terms by `c1`,
so its rounding error grows by the same factor.

The bug: two nearly aligned boxes, such as a cube resting flat on a plate, give edge axes (cross products of two
horizontal edges) that are almost the face normal. Their separations tie the face separation up to rounding. When the
edge axis wins by a hair (63 nm against a 60 nm tolerance in the recorded case), the edge-edge branch builds its
contact from a vertical side face of the incident box and reports a phantom point 0.1445 m deep, 5 cm outside the
cube. The solver pushes that out and launches a resting stack at 1.9 m/s (+3.3 J). With the fix the threshold is
215 nm there, while a genuine edge-edge win in the same stack (51.6 um) is unaffected. `main` has the same bug: on the
same input state it emits the identical phantom contact.

**Balanced stack generator** (`tests/rigid/test_box_stacks_stability`). Each environment gets the most precarious
balanced stack out of 1000 vectorized candidates: random order, faces, yaws, and offsets of up to 0.2 x scale.
Balanced means the center of mass of all boxes above each level lies at least 3% x scale inside both footprints in
contact (the box at that level and its support, the base under the bottom level). The margin matches a polygon
clipping reference to 1e-17. Worst shape combinations accept about 2% of candidates,
hence 1000 tries. `N_ENVS` is 64.

## Open items

1. **The at-rest assertion fails at 64 envs, for both `box_box_detection` values.** It is the remote's check that
   kinetic energy / (m g scale) <= 5e-5 at the end of each 50-step phase. Failing first: pile 0 (scale 0.1), env 32
   under box-box at 1.3e-2 (clearly moving), and env 46 under MPR at 1.1e-3. Earlier traces of the MPR failures showed
   kinetic energy bouncing between 1e-6 and 1.6e-4 for the whole phase rather than decaying, so this looks like the
   small-scale jitter and slow slide-and-tip seen at scale 0.1 (a stack with a 4.8 mm margin slid and tipped from
   exact rest). Investigate whether it is a contact or friction defect at small scale, and do not loosen the
   threshold. The assertion message prints the kinetic energy of every env of the failing pile.
2. **No regression test for the tie fix.** The random stacks no longer reach the tie with the current shapes (beam
   0.2 wide, 0.05 deg tilt), and no simple configuration found reproduces it (`search_minimal_tie.py`: about 2,100
   configurations, none kicked). The tie appeared only after a stack settled for 86 substeps, so it depends on the
   exact rounding of that drifted state. The PR owner prefers a minimal configuration expressed with the test's
   conventions (Euler angles in degrees, no long hardcoded poses) over replicating the recorded state. If none can be
   found, the fallback is a `FIXME` stating the fix has no regression test.
3. **`test_convex_collision_across_geom_scales` fails locally** (lamp hull depth off by 1.2e-6 against `atol=1e-7`),
   with or without the fix. It already failed on the PR branch before these commits.

## Scripts

Run everything from the repository root with `PYTHONPATH=$PWD`, or a worktree silently imports the primary checkout's
engine. Give a worktree its own `QD_OFFLINE_CACHE_FILE_PATH` and `GS_CACHE_FILE_PATH` when running in parallel.

- `kick_state.json`: the recorded stack (base, four box sizes, start poses and the pre-kick substep-86 poses), exact
  float32.
- `repro_kick.py replay`: simulates 100 substeps (dt 0.005, one substep per step) from the recorded start. Unfixed:
  largest energy jump 3.34 J at substep 87. Fixed: no energy rise.
- `repro_kick.py detect`: one collision pass on the pre-kick state. Unfixed: a fifth box1-box3 (plate-cube) contact
  with penetration 1.4447e-01. Fixed: four corner contacts near 8.8e-05.
- `search_minimal_tie.py`: grid search for a simple configuration that triggers the tie, for open item 2.

To see which branch of `func_box_box_contact` a pair takes, kernel `print` works on the CPU backend. Printing
`i_ga, i_gb, i, j, c1, (penetration - c3) * 1e9, tol_edge * 1e9` inside `if c3 < penetration - tol_edge / c1:` gives
the edge-axis wins in nanometres. Revert such prints before committing.

An interactive replay of the stacks, the recorded kick and the fixed run (one env at a time, contacts colored by
penetration) is at https://claude.ai/artifact/L8VUFxdmRGUhnhhmy87Mh5 (private to the PR owner).
