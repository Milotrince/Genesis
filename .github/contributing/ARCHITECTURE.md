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

Rigid and kinematic solvers use `ArticulatedDescription` for their shared native arrays. `LinksData`, `JointsData` and `VisualGeomData` identify ranges within those arrays. A visual geometry pose is refreshed by `update_vgeoms()` before reading its state directly. Checkpoint iteration includes each shared allocation once; range records hold references to it. Scratch and adjoint workspaces remain owned by the solver's data manager.

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
