import numpy as np
from pxr import Usd, UsdGeom, Vt, Gf, PhysxSchema
from isaacsim.core.utils.stage import get_current_stage
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.utils import configclass
from isaaclab.sim.utils import clone

@clone
def spawn_procedural_deformable_wire(
    prim_path: str,
    cfg: "ProceduralDeformableWireCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    stage = get_current_stage()
    
    radius = cfg.radius  # 3mm radius (6mm diameter HDMI) by default
    height = cfg.num_segments * 0.01 if cfg.total_length is None else cfg.total_length
    if cfg.num_segments < 2 or cfg.radius <= 0 or height <= 0 or cfg.radial_subdivisions < 3:
        raise ValueError("Invalid deformable wire dimensions or mesh subdivisions.")
    radial_subdivisions = cfg.radial_subdivisions
    height_subdivisions = cfg.num_segments 
    
    vertices = []
    # Side vertices
    for i in range(height_subdivisions + 1):
        x = height * i / height_subdivisions - height / 2.0
        for j in range(radial_subdivisions):
            theta = j * 2 * np.pi / radial_subdivisions
            y = radius * np.cos(theta)
            z = radius * np.sin(theta)
            vertices.append([x, y, z])
            
    # Add center vertices for caps
    bottom_center_idx = len(vertices)
    vertices.append([-height / 2.0, 0.0, 0.0])
    top_center_idx = len(vertices)
    vertices.append([height / 2.0, 0.0, 0.0])
            
    counts = []
    indices = []
    # Side triangles
    for i in range(height_subdivisions):
        for j in range(radial_subdivisions):
            v1 = i * radial_subdivisions + j
            v2 = i * radial_subdivisions + (j + 1) % radial_subdivisions
            v3 = (i + 1) * radial_subdivisions + (j + 1) % radial_subdivisions
            v4 = (i + 1) * radial_subdivisions + j
            
            counts.extend([3, 3])
            indices.extend([v1, v2, v3])
            indices.extend([v1, v3, v4])
            
    # Bottom cap triangles (Normal -X)
    for j in range(radial_subdivisions):
        v1 = j
        v2 = (j + 1) % radial_subdivisions
        counts.append(3)
        indices.extend([v2, v1, bottom_center_idx])
        
    # Top cap triangles (Normal +X)
    for j in range(radial_subdivisions):
        v1 = height_subdivisions * radial_subdivisions + j
        v2 = height_subdivisions * radial_subdivisions + (j + 1) % radial_subdivisions
        counts.append(3)
        indices.extend([v1, v2, top_center_idx])
            
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.array(vertices, dtype=np.float32)))
    
    # Force visibility in viewport with configurable color
    mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*cfg.color)]))
    mesh.CreateDisplayOpacityAttr().Set(Vt.FloatArray([1.0]))
    
    # PhysX Soft Body Voxelization requires proper extents!
    extent = Vt.Vec3fArray([
        Gf.Vec3f(-height / 2.0, -radius, -radius),
        Gf.Vec3f(height / 2.0, radius, radius)
    ])
    mesh.GetExtentAttr().Set(extent)
    
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.array(counts, dtype=np.int32)))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(np.array(indices, dtype=np.int32)))
    
    if translation is not None:
        mesh.AddTranslateOp().Set(Gf.Vec3d(*translation))
        
    if orientation is not None:
        # IsaacLab orientation is (w,x,y,z)
        mesh.AddOrientOp().Set(Gf.Quatf(orientation[0], orientation[1], orientation[2], orientation[3]))
        
    from isaaclab.sim.schemas.schemas_cfg import DeformableBodyPropertiesCfg
    from isaaclab.sim.schemas.schemas import define_deformable_body_properties
    
    # Material API is bound separately
    deformable_mat_api = PhysxSchema.PhysxDeformableBodyMaterialAPI.Apply(mesh.GetPrim())
    deformable_mat_api.CreateYoungsModulusAttr().Set(cfg.youngs_modulus)   
    deformable_mat_api.CreatePoissonsRatioAttr().Set(cfg.poissons_ratio)        
    deformable_mat_api.CreateDynamicFrictionAttr().Set(cfg.dynamic_friction)       
    
    # Static friction if property exists, otherwise we just bump dynamic. PhysX often uses dynamic for both in some API definitions,
    # but we can try to set static if `CreateStaticFrictionAttr` exists, or just use dynamic.
    deformable_mat_api.CreateDensityAttr().Set(cfg.density)             

    deform_cfg = DeformableBodyPropertiesCfg(
        rest_offset=0.0,
        contact_offset=0.001,
        simulation_hexahedral_resolution=cfg.num_segments,    # Match resolution to subdivisions
        collision_simplification=True,           # Must be True to help PhysX voxelize the extreme aspect ratio safely
        # NOTE: max_depenetration_velocity is not supported for deformable bodies (PhysX limitation), removed to suppress warning
        solver_position_iteration_count=32,
        vertex_velocity_damping=3.0,             # Standard damping for rigid-like materials
        sleep_threshold=0.01,                    # Higher sleep priority
    )
    # Define instead of modify so that PhysxSchema.PhysxCollisionAPI is properly injected!
    define_deformable_body_properties(prim_path, deform_cfg, stage)
    
    return mesh.GetPrim()

@configclass
class ProceduralDeformableWireCfg(SpawnerCfg):
    func = spawn_procedural_deformable_wire
    total_length: float | None = None
    """Explicit length in meters; None retains the legacy 1 cm per subdivision."""
    radial_subdivisions: int = 10
    num_segments: int = 50
    radius: float = 0.003
    
    color: tuple[float, float, float] = (0.05, 0.05, 0.05)
    """RGB color for the wire visual. Defaults to dark black."""

    # Adjustable Physical Parameters for Gripping and Deformation:
    dynamic_friction: float = 2.0  # Increased default for better gripping
    static_friction: float = 2.0   
    youngs_modulus: float = 1000000.0 # Bending stiffness (higher = stiffer)
    poissons_ratio: float = 0.49 # Volume preservation (close to 0.5)
    density: float = 500.0
