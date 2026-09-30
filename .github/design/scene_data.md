# Scene data and core API migration

The scene exposes an immutable sequence of typed data records. Each numerical field is a `DataReference` into the
packed arrays of its owning solver. Ownership belongs to each reference: different systems can eventually own
different fields of the same geometry. Existing solver layouts and integration methods remain authoritative.

This stack covers scene data and core construction/runtime APIs. The scope excludes sensors, new physics, plugin
registration, contact rules, behaviors, topology editing, and new heterogeneous-variant features.

## Data contract

- `scene.data` enumerates geometry, link, and joint records. Domain handles expose the corresponding `.data` record.
- `DataReference.read()` returns an independent Torch snapshot. `read(copy=False)` returns a protected live view on
  backends supporting zero-copy interoperation. Explicit clones and arithmetic results are writable.
- Torch operation schemas identify writable arguments, including nested tensor lists. Aliasing results retain read
  protection. NumPy exports are snapshots. Pointer, storage, DLPack, and tensor-type exports require a clone.
- Protection covers the supported Python interface. Private attributes and arbitrary native code are outside it.
  Native solver kernels retain their private arrays. Enforcing access in a future kernel extension interface requires
  a separate compiler/binding contract.
- References select basic slices after conversion into public axis order. Scattered selections are explicit gathers
  from snapshots. State carries an environment axis even when the scene has one environment. Shared topology has none.
- Each record documents its coordinate frame and topology index space. Rigid local vertices plus a world transform
  remain available without materializing world vertices.
- A read refreshes its owner's derived values when necessary. Retained views observe subsequent writes to their bound
  storage. A new read requests a fresh derived value. Reads after scene destruction raise.
- Current scene builds keep allocations fixed. Reset and restoration update those allocations in place. Future
  reallocation and topology editing must introduce explicit generation checking before exposing those operations.
- Public observations use the existing tensor conversion semantics. Differentiable simulation outputs continue to use
  the existing queried-state/tape API until their gradient contract is migrated explicitly.

## Records

`SceneData` carries stable identity and an entity association. `GeometryData` describes the representation role and
topology. `RigidGeomData` adds its link association and position/orientation references. `DeformableGeomData` exposes
the solver's vertex or particle state and appropriate connectivity. `LinkData` and `JointData` represent relationships
at their natural scope. Joints expose configuration and DOF velocity separately, including their different dimensions.

Numerical data belongs to its solver. Record containers are immutable. Fields added by later modalities acquire their
own owners. Solver scratch, checkpoints, and the public physical scene remain distinct contracts.

## Core construction contract

Concrete entity options select structure and runtime entity type. Materials supply compatible physical properties.
The scene selects the implementation. `add_entity(options=...)` and flattened arguments use one construction path.
Supplying both forms raises before insertion. The ordinary flattened call creates a rigid entity.

Entity, link, geom, and joint settings use typed options. Selections identify targets independently of reusable option
values. Resolution preserves omitted fields, validates material families and scopes, and detects conflicting writes.
Runtime setters continue to ask the owning solver to mutate state and maintain its derived data.

Physical material names and family validation migrate together with their callers. Partial material overrides require
supplied-field tracking. They must preserve authored asset defaults and resolved mass/inertia. No silent conversion of
an incompatible material into a compatible one is permitted.

## Stack order

1. Protected tensor reads and `DataReference`, with mutation and lifetime tests.
2. Scene registry and rigid/link/joint bindings, with public handle access.
3. Typed rigid and kinematic entity construction with explicit family validation.
4. Deformable, particle, and grid bindings, including current time slots and stable particle ordering.
   Extend typed construction to those families as their shared representations become available.
5. Typed link/geom/joint options and selections, physical material migration, and updates to existing examples/tests.
6. Remaining core consumers and removal of superseded access paths.

Each PR targets the preceding branch on the fork. The first targets a fork main synchronized with upstream main.
Intermediate PRs state their coverage explicitly. A staged conversion does not imply every solver already implements
the final contract. Existing coupling behavior is retained, and no new coupling algorithm is part of this stack.

## Validation

Exercise ordinary writes, `out=`, nested tensor mutation, aliases, independent copies, and public export routes.
Verify live reads, snapshots, setters, stepping, partial reset, destruction, and checkpoint restore through public APIs.
Cover both unbatched and batched scenes and the supported array backends. Compare physical observations with existing
getters and analytical expectations. Keep native hot paths free of per-record traversal and additional state copies.

The guiding draft is [Genesis World API Update Proposals](https://app.notion.com/p/3b20c06194e180bfb735f323401c720e).
Feature proposals outside the scope above inform representation choices without adding implementation requirements.
