# Procedural Wire Spawner

This module generates rigid-body articulation chains (cylinders connected by D6 joints) at runtime from configuration parameters. No pre-authored USD files are needed.

A "wire" is an articulation composed of `N` rigid cylinder segments laid out along the +X axis, connected by D6 joints with spring-damper angular drives that give the wire its bending stiffness.

---

## File Overview

| File | Purpose |
|---|---|
| `wire_cfg.py` | Dataclass configs: `WireCfg` (wire parameters), `WirePortCfg` (config for translation, rotation of the male ports on the end) and `WireChainSpawnerCfg` (spawner wrapper). |
| `wire_spawner.py` | Core spawner function `spawn_wire_chain()` — builds the USD prim hierarchy on the stage. |
| `wire_builder.py` | Convenience builders: `build_wire_articulation_cfg()` wraps everything into a ready-to-use `ArticulationCfg`. |
| `wire_presets.py` | Named `WireCfg` presets (`THIN_FLEXIBLE`, `THICK_RIGID`, `MEDIUM_CABLE`). |
| `wire_urdf.py` | URDF export utilities: `export_wire_urdf()` and `save_wire_urdf()`. |
| `__init__.py` | Public re-exports for all of the above. |

---

## Quick Start

### Minimal Example (use a preset)

```python
from isaaclab.sim.spawners.wires import WirePresets, build_wire_articulation_cfg

wire_art_cfg = build_wire_articulation_cfg(
    WirePresets.THIN_FLEXIBLE,
    prim_path="{ENV_REGEX_NS}/wire",
)
```

Assign `wire_art_cfg` to an `InteractiveScene` attribute and the wire is ready to simulate.

### Custom Wire

```python
from isaaclab.sim.spawners.wires import WireCfg, build_wire_articulation_cfg

my_wire = WireCfg(
    num_segments=25,
    total_length=0.35,
    radius=0.002,
    stiffness=2.0,
    damping=0.2,
    fix_base=True,
    color=(0.8, 0.1, 0.1),
)

wire_art_cfg = build_wire_articulation_cfg(my_wire, prim_path="{ENV_REGEX_NS}/wire")
```

Making custom wire pipeline:

---

## Detailed File Reference

### `wire_cfg.py`

#### `WireCfg`

High-level description of a wire. All fields have sensible defaults:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_segments` | `int` | 20 | Number of rigid cylinder segments. |
| `total_length` | `float` | 0.3 | End-to-end wire length in meters. |
| `radius` | `float` | 0.002 | Cylinder radius in meters. |
| `stiffness` | `float` | 1.0 | D6 joint angular drive stiffness (Nm/rad). |
| `damping` | `float` | 0.1 | D6 joint angular drive damping. |
| `joint_limit` | `float \| None` | `None` | Max bend angle per joint in radians. `None` = unlimited. |
| `self_collision` | `bool` | `True` | Enable self-collisions between segments. |
| `fix_base` | `bool` | `False` | Fix the root link to the world. |
| `density` | `float` | 1000.0 | Density in kg/m^3 for auto mass computation. |
| `static_friction` | `float` | 1.0 | Static friction coefficient. |
| `dynamic_friction` | `float` | 1.0 | Dynamic friction coefficient. |
| `restitution` | `float` | 0.0 | Restitution (bounciness). |
| `color` | `tuple[float,float,float]` | `(0.2, 0.2, 0.2)` | RGB visual color. |
| `solver_position_iteration_count` | `int` | 12 | PhysX solver position iterations. |
| `solver_velocity_iteration_count` | `int` | 1 | PhysX solver velocity iterations. |
| `max_depenetration_velocity` | `float` | 5.0 | Max depenetration velocity. |

#### `WireChainSpawnerCfg`

Inherits from `RigidObjectSpawnerCfg`. Wraps a `WireCfg` and optionally overrides visual/physics materials and articulation properties:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `wire_cfg` | `WireCfg` | `WireCfg()` | Wire parameters. |
| `visual_material` | `VisualMaterialCfg \| None` | `None` | Custom visual material. Auto-created from `wire_cfg.color` if `None`. |
| `physics_material` | `PhysicsMaterialCfg \| None` | `None` | Custom physics material. Auto-created from friction/restitution if `None`. |
| `articulation_props` | `ArticulationRootPropertiesCfg \| None` | `None` | Custom articulation properties. Auto-created if `None`. |

---

### `wire_spawner.py`

#### `spawn_wire_chain(prim_path, cfg, translation, orientation)`

The core spawner function (decorated with `@clone` for multi-env support). It:

1. Creates a root Xform at `prim_path`.
2. Creates shared visual and physics materials under `<root>/materials/`.
3. Creates `N` cylinder segments (`seg_0` ... `seg_{N-1}`), each as an Xform with a Cylinder child, rigid body, collision, and mass properties.
4. Creates D6 joints between consecutive segments with locked translational DOFs and spring-damper rotational DOFs.
5. Optionally creates a fixed base joint (if `fix_base=True`).
6. Applies the `ArticulationRootAPI` to the root prim.

You typically don't call this directly — use `build_wire_articulation_cfg()` instead.

**Direct usage** (advanced):

```python
from isaaclab.sim.spawners.wires import WireChainSpawnerCfg, WireCfg

cfg = WireChainSpawnerCfg(wire_cfg=WireCfg(num_segments=10))
prim = cfg.func("/World/wire", cfg, translation=(0.0, 0.0, 0.5))
```

---

### `wire_builder.py`

#### `build_wire_articulation_cfg(wire_cfg, prim_path) -> ArticulationCfg`

The recommended entry point. Converts a `WireCfg` into a complete `ArticulationCfg` with:
- A `WireChainSpawnerCfg` as the spawn config.
- An `ImplicitActuatorCfg` with the wire's stiffness/damping as passive spring-damper actuators.
- Default initial state (all joints at zero position and velocity).

```python
from isaaclab.sim.spawners.wires import WireCfg, build_wire_articulation_cfg

art_cfg = build_wire_articulation_cfg(
    WireCfg(num_segments=20, total_length=0.3),
    prim_path="{ENV_REGEX_NS}/wire",
)
```

#### `build_wire_articulation_cfg_full(wire_cfg, prim_path) -> WireBuildResult`

Same as above but returns a `WireBuildResult` dataclass with extra metadata:

| Field | Type | Description |
|---|---|---|
| `articulation_cfg` | `ArticulationCfg` | The articulation config. |
| `head_segment_name` | `str` | Name of the first segment (e.g. `"seg_0"`). |
| `tail_segment_name` | `str` | Name of the last segment (e.g. `"seg_19"`). |

Useful when you need to reference specific segments (e.g. for attaching sensors or applying forces to the tip).

---

### `wire_presets.py`

#### `WirePresets`

Named configurations for common scenarios:

| Preset | Segments | Length | Radius | Stiffness | Damping | Description |
|---|---|---|---|---|---|---|
| `THIN_FLEXIBLE` | 20 | 0.3m | 1mm | 0.5 | 0.05 | Thin floppy wire. |
| `THICK_RIGID` | 10 | 0.2m | 5mm | 50.0 | 5.0 | Thick stiff rod. |
| `MEDIUM_CABLE` | 15 | 0.4m | 3mm | 5.0 | 0.5 | General-purpose cable. |

Usage:

```python
from isaaclab.sim.spawners.wires import WirePresets, build_wire_articulation_cfg

art_cfg = build_wire_articulation_cfg(WirePresets.MEDIUM_CABLE, prim_path="{ENV_REGEX_NS}/wire")
```

---

### `wire_urdf.py`

Export a `WireCfg` to URDF format. Since URDF has no native D6/spherical joint, each connection is approximated with 3 revolute joints (X, Y, Z) and 2 intermediate massless links.

#### `export_wire_urdf(wire_cfg) -> str`

Returns a URDF XML string.

#### `save_wire_urdf(wire_cfg, path) -> str`

Writes the URDF to a file and returns the XML string.

```python
from isaaclab.sim.spawners.wires import WireCfg, save_wire_urdf

save_wire_urdf(WireCfg(num_segments=10, total_length=0.2), "/tmp/wire.urdf")
```

---

## Customization Guide

### Tuning Stiffness and Damping

- **Stiffness** controls how strongly each joint resists bending. Higher values make the wire more rigid.
- **Damping** controls energy dissipation. Higher values make the wire settle faster but can make it feel sluggish.
- Rule of thumb: start with `stiffness=1.0, damping=0.1` and adjust from there.

### Segment Count vs. Performance

More segments produce a smoother wire but cost more in simulation. Guidelines:
- 10 segments: coarse, fast, good for stiff rods.
- 15-20 segments: balanced for most cables.
- 25-40 segments: high fidelity, use for thin flexible wires where curvature matters.

### Fixing the Base

Set `fix_base=True` to anchor the first segment to the world. Useful for hanging wires or pendulum-like setups.

### Joint Limits

Set `joint_limit` (in radians) to constrain how far each joint can bend. This prevents unrealistic over-bending:

```python
import math

wire = WireCfg(joint_limit=math.radians(15))  # max 15 degrees per joint
```

### Custom Materials

Override the auto-generated materials via `WireChainSpawnerCfg`:

```python
from isaaclab.sim.spawners.materials import visual_materials_cfg, physics_materials_cfg
from isaaclab.sim.spawners.wires import WireChainSpawnerCfg, WireCfg

spawner = WireChainSpawnerCfg(
    wire_cfg=WireCfg(),
    visual_material=visual_materials_cfg.PreviewSurfaceCfg(
        diffuse_color=(0.9, 0.1, 0.1),
        metallic=0.8,
    ),
    physics_material=physics_materials_cfg.RigidBodyMaterialCfg(
        static_friction=0.5,
        dynamic_friction=0.3,
        restitution=0.1,
    ),
)
```

### USD Prim Hierarchy

After spawning, the USD stage structure looks like:

```
<prim_path>/                    # Root Xform (ArticulationRoot)
  materials/
    visual/                     # PreviewSurface material
    physics/                    # RigidBodyMaterial
  seg_0/                        # First segment Xform (RigidBody)
    geometry/
      mesh/                     # Cylinder (Collision + Visual bindings)
    fixed_base_joint/           # (only if fix_base=True)
  seg_1/
    geometry/mesh/
    joint_to_parent/            # D6 joint connecting to seg_0
  seg_2/
    geometry/mesh/
    joint_to_parent/            # D6 joint connecting to seg_1
  ...
```

---

## Full Scene Integration Example

```python
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.wires import WireCfg, build_wire_articulation_cfg

wire_cfg = WireCfg(
    num_segments=20,
    total_length=0.3,
    radius=0.002,
    stiffness=2.0,
    damping=0.2,
    fix_base=True,
    color=(0.1, 0.4, 0.9),
)

class MySceneCfg(InteractiveSceneCfg):
    wire = build_wire_articulation_cfg(wire_cfg, prim_path="{ENV_REGEX_NS}/wire")
```
