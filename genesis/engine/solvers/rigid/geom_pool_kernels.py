"""Quadrants kernels for the dynamic geometry pool (Phase 3, Stage 2).

These mirror the build-time geom/vert init kernels in ``abd/misc.py`` and the geom-pose pass in
``abd/forward_kinematics.py``, but write into a pool slot's *reserved* index ranges at runtime instead of
initializing the whole scene at index 0. Payload arrays carry one uploaded object's rows; ``*_base`` args are
the slot's reserved starts, and face/edge/verts_state indices in the payloads are already absolute (offset
into the slot's ranges host-side).
"""

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu


@qd.kernel(fastcache=True)
def kernel_upload_vert_slot(
    vert_base: qd.i32,
    face_base: qd.i32,
    edge_base: qd.i32,
    verts: qd.types.ndarray(),
    normals: qd.types.ndarray(),
    init_center_pos: qd.types.ndarray(),
    verts_geom_idx: qd.types.ndarray(),
    verts_state_idx: qd.types.ndarray(),
    is_fixed: qd.types.ndarray(),
    faces: qd.types.ndarray(),  # absolute global vertex indices
    faces_geom_idx: qd.types.ndarray(),
    edges: qd.types.ndarray(),  # absolute global vertex indices
    # Quadrants variables
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    edges_info: array_class.EdgesInfo,
    static_rigid_sim_config: qd.template(),
):
    n_verts = verts.shape[0]
    n_faces = faces.shape[0]
    n_edges = edges.shape[0]

    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_ in range(n_verts):
        i_v = vert_base + i_
        for j in qd.static(range(3)):
            verts_info.init_pos[i_v][j] = verts[i_, j]
            verts_info.init_normal[i_v][j] = normals[i_, j]
            verts_info.init_center_pos[i_v][j] = init_center_pos[i_, j]
        verts_info.geom_idx[i_v] = verts_geom_idx[i_]
        verts_info.verts_state_idx[i_v] = verts_state_idx[i_]
        verts_info.is_fixed[i_v] = is_fixed[i_]

    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_ in range(n_faces):
        i_f = face_base + i_
        for j in qd.static(range(3)):
            faces_info.verts_idx[i_f][j] = faces[i_, j]
        faces_info.geom_idx[i_f] = faces_geom_idx[i_]

    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_ in range(n_edges):
        i_ed = edge_base + i_
        edges_info.v0[i_ed] = edges[i_, 0]
        edges_info.v1[i_ed] = edges[i_, 1]
        edges_info.length[i_ed] = (verts_info.init_pos[edges[i_, 0]] - verts_info.init_pos[edges[i_, 1]]).norm()


@qd.kernel(fastcache=True)
def kernel_upload_geom_slot(
    geom_base: qd.i32,
    geoms_pos: qd.types.ndarray(),
    geoms_center: qd.types.ndarray(),
    geoms_quat: qd.types.ndarray(),
    geoms_link_idx: qd.types.ndarray(),
    geoms_type: qd.types.ndarray(),
    geoms_friction: qd.types.ndarray(),
    geoms_sol_params: qd.types.ndarray(),
    geoms_vert_start: qd.types.ndarray(),  # absolute
    geoms_face_start: qd.types.ndarray(),  # absolute
    geoms_edge_start: qd.types.ndarray(),  # absolute
    geoms_verts_state_start: qd.types.ndarray(),  # absolute
    geoms_vert_end: qd.types.ndarray(),  # absolute
    geoms_face_end: qd.types.ndarray(),  # absolute
    geoms_edge_end: qd.types.ndarray(),  # absolute
    geoms_verts_state_end: qd.types.ndarray(),  # absolute
    geoms_data: qd.types.ndarray(),
    geoms_is_convex: qd.types.ndarray(),
    geoms_needs_coup: qd.types.ndarray(),
    geoms_contype: qd.types.ndarray(),
    geoms_conaffinity: qd.types.ndarray(),
    geoms_coup_softness: qd.types.ndarray(),
    geoms_coup_friction: qd.types.ndarray(),
    geoms_coup_restitution: qd.types.ndarray(),
    geoms_is_fixed: qd.types.ndarray(),
    geoms_is_decomp: qd.types.ndarray(),
    # Quadrants variables
    geoms_info: array_class.GeomsInfo,
    geoms_state: array_class.GeomsState,
    verts_info: array_class.VertsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    static_rigid_sim_config: qd.template(),
):
    n_geoms = geoms_pos.shape[0]
    _B = geoms_state.friction_ratio.shape[1]

    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_ in range(n_geoms):
        i_g = geom_base + i_
        for j in qd.static(range(3)):
            geoms_info.pos[i_g][j] = geoms_pos[i_, j]
            geoms_info.center[i_g][j] = geoms_center[i_, j]
        for j in qd.static(range(4)):
            geoms_info.quat[i_g][j] = geoms_quat[i_, j]
        for j in qd.static(range(7)):
            geoms_info.data[i_g][j] = geoms_data[i_, j]
            geoms_info.sol_params[i_g][j] = geoms_sol_params[i_, j]

        geoms_info.vert_start[i_g] = geoms_vert_start[i_]
        geoms_info.vert_end[i_g] = geoms_vert_end[i_]
        geoms_info.vert_num[i_g] = geoms_vert_end[i_] - geoms_vert_start[i_]
        geoms_info.face_start[i_g] = geoms_face_start[i_]
        geoms_info.face_end[i_g] = geoms_face_end[i_]
        geoms_info.face_num[i_g] = geoms_face_end[i_] - geoms_face_start[i_]
        geoms_info.edge_start[i_g] = geoms_edge_start[i_]
        geoms_info.edge_end[i_g] = geoms_edge_end[i_]
        geoms_info.edge_num[i_g] = geoms_edge_end[i_] - geoms_edge_start[i_]
        geoms_info.verts_state_start[i_g] = geoms_verts_state_start[i_]
        geoms_info.verts_state_end[i_g] = geoms_verts_state_end[i_]

        geoms_info.link_idx[i_g] = geoms_link_idx[i_]
        geoms_info.type[i_g] = geoms_type[i_]
        geoms_info.friction[i_g] = geoms_friction[i_]
        geoms_info.is_convex[i_g] = geoms_is_convex[i_]
        geoms_info.needs_coup[i_g] = geoms_needs_coup[i_]
        geoms_info.contype[i_g] = geoms_contype[i_]
        geoms_info.conaffinity[i_g] = geoms_conaffinity[i_]
        geoms_info.coup_softness[i_g] = geoms_coup_softness[i_]
        geoms_info.coup_friction[i_g] = geoms_coup_friction[i_]
        geoms_info.coup_restitution[i_g] = geoms_coup_restitution[i_]
        geoms_info.is_fixed[i_g] = geoms_is_fixed[i_]
        geoms_info.is_decomposed[i_g] = geoms_is_decomp[i_]

        # Init AABB: analytic for radial primitives, else min/max over the slot's uploaded verts. Corner
        # ordering matches kernel_init_geom_fields and MUST NOT be reordered.
        lower = gu.qd_vec3(qd.math.inf)
        upper = gu.qd_vec3(-qd.math.inf)
        geom_type = geoms_type[i_]
        if geom_type == gs.GEOM_TYPE.SPHERE:
            r = geoms_data[i_, 0]
            lower = qd.Vector([-r, -r, -r], dt=gs.qd_float)
            upper = qd.Vector([r, r, r], dt=gs.qd_float)
        elif geom_type == gs.GEOM_TYPE.ELLIPSOID:
            a = geoms_data[i_, 0]
            b = geoms_data[i_, 1]
            c = geoms_data[i_, 2]
            lower = qd.Vector([-a, -b, -c], dt=gs.qd_float)
            upper = qd.Vector([a, b, c], dt=gs.qd_float)
        elif geom_type == gs.GEOM_TYPE.CAPSULE:
            r = geoms_data[i_, 0]
            hl = 0.5 * geoms_data[i_, 1]
            lower = qd.Vector([-r, -r, -(hl + r)], dt=gs.qd_float)
            upper = qd.Vector([r, r, hl + r], dt=gs.qd_float)
        elif geom_type == gs.GEOM_TYPE.CYLINDER:
            r = geoms_data[i_, 0]
            hl = 0.5 * geoms_data[i_, 1]
            lower = qd.Vector([-r, -r, -hl], dt=gs.qd_float)
            upper = qd.Vector([r, r, hl], dt=gs.qd_float)
        else:
            for i_v in range(geoms_vert_start[i_], geoms_vert_end[i_]):
                lower = qd.min(lower, verts_info.init_pos[i_v])
                upper = qd.max(upper, verts_info.init_pos[i_v])
        geoms_init_AABB[i_g, 0] = qd.Vector([lower[0], lower[1], lower[2]], dt=gs.qd_float)
        geoms_init_AABB[i_g, 1] = qd.Vector([lower[0], lower[1], upper[2]], dt=gs.qd_float)
        geoms_init_AABB[i_g, 2] = qd.Vector([lower[0], upper[1], lower[2]], dt=gs.qd_float)
        geoms_init_AABB[i_g, 3] = qd.Vector([lower[0], upper[1], upper[2]], dt=gs.qd_float)
        geoms_init_AABB[i_g, 4] = qd.Vector([upper[0], lower[1], lower[2]], dt=gs.qd_float)
        geoms_init_AABB[i_g, 5] = qd.Vector([upper[0], lower[1], upper[2]], dt=gs.qd_float)
        geoms_init_AABB[i_g, 6] = qd.Vector([upper[0], upper[1], lower[2]], dt=gs.qd_float)
        geoms_init_AABB[i_g, 7] = qd.Vector([upper[0], upper[1], upper[2]], dt=gs.qd_float)

    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_, i_b in qd.ndrange(n_geoms, _B):
        i_g = geom_base + i_
        geoms_state.friction_ratio[i_g, i_b] = 1.0
        geoms_state.scale[i_g, i_b] = qd.Vector([1.0, 1.0, 1.0], dt=gs.qd_float)


@qd.kernel(fastcache=True)
def kernel_init_pool_geom_defaults(
    geom_lo: qd.i32,
    geom_hi: qd.i32,
    geoms_info: array_class.GeomsInfo,
    geoms_state: array_class.GeomsState,
    static_rigid_sim_config: qd.template(),
):
    """Seed the reserved pool block to inert-but-finite defaults at build.

    kernel_init_geom_fields only initializes the real geoms, leaving the reserved rows zero — a zero quat
    would rotate AABB corners to NaN when the pool-FK pass runs. Set identity quats and unit scale so an
    unfilled slot poses to a finite (degenerate) AABB at its link; set_active_object overwrites these on load.
    """
    _B = geoms_state.pos.shape[1]
    for i_g in range(geom_lo, geom_hi):
        geoms_info.quat[i_g] = qd.Vector([1.0, 0.0, 0.0, 0.0], dt=gs.qd_float)
    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_g, i_b in qd.ndrange((geom_lo, geom_hi), _B):
        geoms_state.quat[i_g, i_b] = qd.Vector([1.0, 0.0, 0.0, 0.0], dt=gs.qd_float)
        geoms_state.scale[i_g, i_b] = qd.Vector([1.0, 1.0, 1.0], dt=gs.qd_float)


@qd.kernel(fastcache=True)
def kernel_update_pool_geoms(
    geom_lo: qd.i32,
    geom_hi: qd.i32,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    links_state: array_class.LinksState,
    static_rigid_sim_config: qd.template(),
):
    """Pose every reserved pool geom from its owning link and refresh its world AABB.

    The reserved block lies outside every entity's contiguous geom range, so the entity-based FK pass in
    kernel_step_1 never touches it. This runs over ``[geom_lo, geom_hi)`` each substep (after kernel_step_1,
    before collision): an unfilled slot keeps link_idx 0 and an identity quat (posed harmlessly, never in a
    collision sweep since no env's link is bound to it), while a filled slot is posed from the owning entity's
    base link exactly like a normal geom. The AABB mirrors kernel_update_geom_aabbs.
    """
    _B = geoms_state.pos.shape[1]
    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_g, i_b in qd.ndrange((geom_lo, geom_hi), _B):
        i_l = geoms_info.link_idx[i_g]
        g_pos, g_quat = gu.qd_transform_pos_quat_by_trans_quat(
            geoms_info.pos[i_g], geoms_info.quat[i_g], links_state.pos[i_l, i_b], links_state.quat[i_l, i_b]
        )
        geoms_state.pos[i_g, i_b] = g_pos
        geoms_state.quat[i_g, i_b] = g_quat
        geoms_state.verts_updated[i_g, i_b] = False

        lower = gu.qd_vec3(qd.math.inf)
        upper = gu.qd_vec3(-qd.math.inf)
        for i_corner in qd.static(range(8)):
            corner = geoms_init_AABB[i_g, i_corner]
            if qd.static(static_rigid_sim_config.enable_geom_scaling):
                corner = geoms_state.scale[i_g, i_b] * corner
            corner_pos = gu.qd_transform_by_trans_quat(corner, g_pos, g_quat)
            lower = qd.min(lower, corner_pos)
            upper = qd.max(upper, corner_pos)
        geoms_state.aabb_min[i_g, i_b] = lower
        geoms_state.aabb_max[i_g, i_b] = upper
