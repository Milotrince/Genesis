# Genesis Architecture

## Project Structure

```
Genesis/
├── genesis/                    # Main source code
│   ├── __init__.py            # Entry point, gs.init(), global state
│   ├── engine/                # Core simulation engine
│   │   ├── scene.py           # Scene class - main API entry point
│   │   ├── simulator.py       # Manages all solvers
│   │   ├── entities/          # Entity types (RigidEntity, MPMEntity, etc.)
│   │   ├── solvers/           # Physics solvers
│   │   │   ├── rigid/         # Rigid body solver
│   │   │   ├── mpm_solver.py  # Material Point Method
│   │   │   ├── sph_solver.py  # Smoothed Particle Hydrodynamics
│   │   │   ├── fem_solver.py  # Finite Element Method
│   │   │   ├── pbd_solver.py  # Position Based Dynamics
│   │   │   └── sf_solver.py   # Stable Fluid
│   │   ├── materials/         # Material definitions per solver
│   │   └── couplers/          # Inter-solver coupling
│   ├── options/               # Configuration classes (Pydantic models)
│   │   ├── morphs.py          # Shape definitions (Box, Mesh, URDF, MJCF)
│   │   ├── solvers.py         # Solver configuration options
│   │   └── surfaces.py        # Surface properties
│   ├── vis/                   # Visualization (Visualizer, Camera, Viewer)
│   ├── sensors/               # Sensor systems (camera, IMU, etc.)
│   └── utils/                 # Utilities (mesh, geometry, etc.)
├── tests/                     # Test files
├── examples/                  # Example scripts
└── genesis/assets/            # Built-in meshes, URDFs, textures
```

## Core Components Flow

```
gs.init() → Scene → Simulator → Solvers → Entities
                 ↓
            Visualizer → Viewer / Cameras
```

## Entities

Entities are physical objects in the simulation:

| Entity Type | Solver | Use Case |
|-------------|--------|----------|
| `RigidEntity` | Rigid | Robots, rigid objects |
| `MPMEntity` | MPM | Deformable solids, granular materials |
| `SPHEntity` | SPH | Liquids, fluids |
| `FEMEntity` | FEM | Finite element deformable bodies |
| `PBD2DEntity`, `PBD3DEntity` | PBD | Cloth, soft bodies |
| `DroneEntity` | Rigid | Quadcopters with aerodynamics |
| `ToolEntity` | Tool | Cutting/interaction tools |

Location: `genesis/engine/entities/`

### Descriptions

An entity is created from a description: a dataclass holding everything its solver simulates it with, defined beside the entity that consumes it (`genesis/engine/entities/rigid_entity/description.py` for rigid and kinematic entities). Creation is two steps, each owned by the class it concerns:

- the description class resolves the morphs, material and surface into a description (`RigidEntityDescription.resolve`), which is where assets are read and post-processed;
- the entity class builds itself from the description alone (`RigidEntity.__init__`), so a description loaded from a file and one resolved from assets take the same path.

`Scene.export` writes the description of every entity, obtained through `Entity.desc`, beside the scene options. `Scene.load` creates each entity from its description through the solver simulating its material (`Solver.add_entity` given `desc`), attachments included, since a description names only entities created before it. A kind of entity gains export support by defining its description class, returning it from `desc`, and having its solver's `add_entity` build from a given description, with no change to the scene. The serialization facility alone reduces every absolute path a value holds to the name of the asset, so no option or mesh redacts anything itself.

## Morphs

Morphs define geometry and initial pose (solver-agnostic):

```python
# Primitives
gs.morphs.Box(size=(1, 1, 1), pos=(0, 0, 0.5))
gs.morphs.Sphere(radius=0.5, pos=(0, 0, 1))
gs.morphs.Plane()

# Robot descriptions
gs.morphs.URDF(file="path/to/robot.urdf", fixed=True)
gs.morphs.MJCF(file="path/to/robot.xml")
```

Location: `genesis/options/morphs.py`

## Materials

Materials define physical properties and determine which solver handles the entity: each solver declares the material class it simulates (`Solver.material_cls`).

```python
gs.materials.Rigid(rho=1000)
gs.materials.MPM.Elastic(E=1e5, nu=0.3)
gs.materials.SPH.Liquid(sampler="pbs")
gs.materials.PBD.Cloth(stretch_compliance=0.0)
```

Location: `genesis/engine/materials/`

## Solvers

| Solver | Options Class | Purpose |
|--------|--------------|---------|
| Rigid | `gs.options.RigidOptions` | Articulated rigid body dynamics |
| MPM | `gs.options.MPMOptions` | Continuum mechanics |
| SPH | `gs.options.SPHOptions` | Fluid simulation |
| FEM | `gs.options.FEMOptions` | Finite element deformation |
| PBD | `gs.options.PBDOptions` | Fast soft body simulation |
| SF | `gs.options.SFOptions` | Eulerian fluid/smoke |

Location: `genesis/engine/solvers/`

### Solver data lifecycle

`Simulator` owns an internal `SolverDataArray`. Each record has a concrete type, an owning solver and an owner-local index. Solvers and couplers resolve records during build and retain references for runtime access. Public entity APIs return safe tensors and route writes through solver setters.

The build phases are:

1. `prepare()` resolves entity ranges and static configuration.
2. `describe()` returns the dimensions and inputs for shared allocation.
3. The simulator allocates every description into its data collection.
4. `bind()` stores native buffer references and registers geometry and joint records with their ranges.
5. The collection closes registration, and `build()` initializes state and private workspaces.

Rigid and kinematic solvers use `ArticulatedDescription` for their shared native arrays. `LinksData`, `JointsData` and `VisualGeomData` identify ranges within those arrays. The articulated record also owns joint configuration, rest configuration, gravity, and mean inertia; `RigidInfo` references those allocations for native kernels. A visual geometry pose is refreshed by `update_vgeoms()` before reading its state directly. Checkpoint iteration includes each shared allocation once; range records hold references to it. Scratch and adjoint workspaces remain owned by the solver's data manager.

FEM entities resolve a `FEMEntityDescription` containing simulation vertices, element topology, visual meshes and optional hydroelastic pressure. `FEMDescription` allocates the time-indexed state, material arrays and surface data. `FEMGeomData` maps each entity into that storage. The finite element solver keeps its iterative solve, rendering and constraint workspaces locally. Scene export includes the resolved geometry and serializable material options.

## Key Files Reference

| File | Purpose |
|------|---------|
| `genesis/__init__.py` | Package entry, `gs.init()`, global state |
| `genesis/engine/scene.py` | `Scene` class - main user interface |
| `genesis/engine/simulator.py` | `Simulator` - manages all solvers |
| `genesis/options/morphs.py` | Shape/geometry definitions |
| `genesis/options/solvers.py` | Solver option classes |

MPM resolves particle samples and visual meshes into `ParticleEntityDescription`, which shares the
`VerticesDescription` mixin with FEM. `MPMDescription` allocates particle history and the coupling grid;
`MPMGeomData` identifies each entity's particle range. Grid reset bookkeeping and render buffers stay
solver-owned. Serialized type identifiers use qualified names so material classes with the same short
name remain distinct when several solver modalities share a scene.

SPH uses `ParticleEntityDescription` for resolved sampling and `SPHDescription` for allocation.
`SPHData` keeps original-order and reordered particle state, material properties, and reorder maps
together. `SPHGeomData` identifies entity ranges in original order; coupling uses the reordered buffers
during substeps. Spatial hashing and render buffers remain solver-owned.

PBD cloth and elastic entities use `PBDMeshDescription` for remeshed vertices, edges, bending pairs,
and tetrahedral rest volumes. Liquids and free particles use `ParticleEntityDescription`.
`PBDDescription` allocates original-order and reordered particle buffers plus rest topology;
`PBDGeomData` records each entity's particle, edge, bending-edge, and element ranges.

Tool entities resolve normalized/scaled mesh vertices, normals, faces, and signed-distance samples into
`ToolEntityDescription`. `ToolDescription` allocates per-entity `ToolGeomData` with pose history and mesh
fields, and the solver and collision methods share those buffers. Entity constructors hold host descriptions;
native allocation occurs during scene build.

SF declares grid resolution and jet-channel count through `SFDescription`, then binds `SFData` for
velocity, pressure, and concentration fields. Pressure-projection scratch stays solver-owned. Grid
getters return copied tensors in x/y/z order. Jet configuration is fixed at build; custom jet functions
remain runtime configuration, and exporting a scene containing them raises. SF remains unbatched.
Scene-state snapshots include its velocity, pressure, concentration, and time; scene reset restores these
for active solvers even when they have no entities. Serialized checkpoints remain unsupported for SF.

Hybrid entities serialize their rigid/soft entity references and resolved particle associations in
`HybridEntityDescription`. The MPM description allocates association buffers as `HybridData`. The
simulator applies each association after its soft solver completes the post-coupling phase, in entity
order. Authoring callbacks are consumed when resolving descriptions; loaded scenes use those resolved
results. Hybrid composition currently supports rigid links and MPM particles.

### Shared writes

`SolverDataArray.bind(record)` retains read access. Passing `write="qpos"` or `write="dofs_velocity"`
for a `JointsData` record binds the owning solver's setter to its entity range. `binding.commit(value,
envs_idx=...)` applies that setter, including forward kinematics, contact invalidation, wake-up, and
state-change notifications. Unsupported operations and records from another scene raise during binding.
Bindings expire when their scene is destroyed.

Native substep coupling declares exact written field paths with `bind_coupling`. Each solver lists its
supported coupling fields. These writes target substep history or reordered buffers in the existing
Legacy, SAP, and Hybrid phases. The simulator commits them after all post-coupling work, when original
particle ordering and derived positions are available. Owners publish state-change notifications;
rigid velocity updates invalidate forward-velocity caches. Accumulated rigid forces retain their next
rigid-step consumption and wake-up behavior. IPC uses its existing owner setters for writeback.

The collection and bindings are internal developer interfaces. Raw native buffers remain trusted
solver/coupler storage; public entity getters return isolated tensors and public writes use setters.
