import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

import genesis as gs

from ..utils import (
    assert_allclose,
)


def test_geom_scale(tol):
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            enable_geom_scaling=True,
            gravity=(0.0, 0.0, -10.0),
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    box = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.3),
        ),
    )
    cylinder = scene.add_entity(
        gs.morphs.Cylinder(
            radius=0.05,
            height=0.2,
            pos=(1.0, 0.0, 0.3),
        ),
    )
    # 4 envs: scale envs 0-1 non-uniformly by (2, 1, 3); leave envs 2-3 at unit scale.
    scene.build(n_envs=4)

    scene.step()
    mass0 = box.get_mass()
    box.set_scale((2.0, 1.0, 3.0), envs_idx=[0, 1])
    scene.step()

    # Mass scales by det(S) = 6 on the scaled envs and is unchanged elsewhere.
    mass1 = box.get_mass()
    assert_allclose(mass1[[0, 1]], mass0[[0, 1]] * 6.0, tol=tol)
    assert_allclose(mass1[[2, 3]], mass0[[2, 3]], tol=tol)

    # AABB scales per-axis on the scaled envs (unit box -> extents (0.2, 0.1, 0.3)); unit envs unchanged.
    ext = box.get_AABB()[:, 1] - box.get_AABB()[:, 0]
    assert_allclose(ext[[0, 1]], (0.2, 0.1, 0.3), tol=1e-3)
    assert_allclose(ext[[2, 3]], (0.1, 0.1, 0.1), tol=1e-3)
    assert_allclose(box.get_scale()[[0, 1]], (2.0, 1.0, 3.0), tol=tol)
    assert_allclose(box.get_scale()[[2, 3]], (1.0, 1.0, 1.0), tol=tol)

    # Drop: scaled boxes rest on their scaled half-height (0.15), unit boxes on 0.05.
    for _ in range(100):
        scene.step()
    z = box.get_pos()[..., 2]
    assert torch.isfinite(z).all()
    assert_allclose(z[[0, 1]], 0.15, tol=6e-3)
    assert_allclose(z[[2, 3]], 0.05, tol=6e-3)

    # A cylinder scales anisotropically (elliptic cross-section, see test_geom_scale_anisotropic); a uniform
    # radial + axial scale is likewise fine. Neither raises.
    cylinder.set_scale((2.0, 1.0, 3.0))
    cylinder.set_scale((2.0, 2.0, 3.0))


def test_geom_scale_anisotropic(tol):
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            enable_geom_scaling=True,
            gravity=(0.0, 0.0, -10.0),
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.1,
            pos=(0.0, 0.0, 0.5),
        ),
    )
    # Lying on its side (local axis Z -> world Y), so the vertical radial semi-axis is radius * scale[1].
    cylinder = scene.add_entity(
        gs.morphs.Cylinder(
            radius=0.1,
            height=0.4,
            pos=(1.0, 0.0, 0.5),
            euler=(90.0, 0.0, 0.0),
        ),
    )
    # Capsule axis along world X, so a non-uniform radial scale is anisotropic and takes the elliptic-capsule
    # support (no analytic path). scale[0] stretches the vertical (world Z) radial semi-axis.
    mjcf = ET.Element("mujoco", model="capsule")
    body = ET.SubElement(ET.SubElement(mjcf, "worldbody"), "body", pos="2 0 0.5")
    ET.SubElement(body, "joint", type="free")
    ET.SubElement(body, "geom", type="capsule", fromto="-0.15 0 0 0.15 0 0", size="0.1", mass="0.1")
    capsule = scene.add_entity(morph=gs.morphs.MJCF(file=ET.tostring(mjcf, encoding="unicode")))
    scene.build(n_envs=2)
    scene.step()

    # env 0 stays unit; env 1 stretches the vertical semi-axis of each shape to 0.2.
    sphere.set_scale(np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 2.0]]))
    cylinder.set_scale(np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0]]))
    capsule.set_scale(np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]]))
    for _ in range(150):
        scene.step()

    sz = sphere.get_pos()[..., 2]
    cz = cylinder.get_pos()[..., 2]
    pz = capsule.get_pos()[..., 2]
    assert torch.isfinite(sz).all() and torch.isfinite(cz).all() and torch.isfinite(pz).all()
    assert_allclose(sz[0], 0.1, tol=5e-3)  # unit sphere rests on its radius
    assert_allclose(sz[1], 0.2, tol=5e-3)  # z-stretched ellipsoid rests on its 2x semi-axis
    assert_allclose(cz[0], 0.1, tol=5e-3)  # unit cylinder on its side rests on its radius
    assert_allclose(cz[1], 0.2, tol=5e-3)  # elliptic cylinder rests on its 2x vertical semi-axis
    assert_allclose(pz[0], 0.1, tol=5e-3)  # unit capsule on its side rests on its radius
    assert_allclose(pz[1], 0.2, tol=5e-3)  # elliptic capsule rests on its 2x vertical radial semi-axis


def test_geom_scale_multi_link_tree(tol):
    mjcf = ET.Element("mujoco", model="two_link")
    ET.SubElement(mjcf, "option", gravity="0 0 0")
    worldbody = ET.SubElement(mjcf, "worldbody")
    link1 = ET.SubElement(worldbody, "body", name="l1", pos="0 0 1.0")
    ET.SubElement(link1, "joint", name="j1", type="hinge", axis="0 1 0")
    ET.SubElement(link1, "geom", type="capsule", fromto="0 0 0 0.4 0 0", size="0.03", mass="0.2")
    link2 = ET.SubElement(link1, "body", name="l2", pos="0.4 0 0")
    ET.SubElement(link2, "joint", name="j2", type="hinge", axis="0 1 0")
    ET.SubElement(link2, "geom", type="capsule", fromto="0 0 0 0.4 0 0", size="0.03", mass="0.2")

    scene = gs.Scene(rigid_options=gs.options.RigidOptions(enable_geom_scaling=True), show_viewer=False)
    arm = scene.add_entity(morph=gs.morphs.MJCF(file=ET.tostring(mjcf, encoding="unicode")))
    scene.build(n_envs=2)
    scene.step()

    arm.set_scale(np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))  # env 1 is twice as big
    scene.step()
    pos = arm.get_links_pos()  # (n_envs, n_links, 3), world frame; links = [base child l1, l2]
    root = arm.n_links - 2  # index of l1 (the fixed base occupies slot 0)
    off_1x = (pos[0, root + 1] - pos[0, root]).norm()
    off_2x = (pos[1, root + 1] - pos[1, root]).norm()
    assert_allclose(off_2x / off_1x, 2.0, tol=tol)  # the child link separated twice as far
    assert_allclose(pos[1, root, 2] / pos[0, root, 2], 2.0, tol=tol)  # base child rose twice as high off the root

    # The capsule geom is offset from its link origin (its center is halfway along the link), so its link-frame
    # offset must scale too: the scaled env's geom sits twice as far from its link as the unit env's.
    link1_geom = arm.geoms[0].get_pos(relative=False)  # (n_envs, 3) world frame; the capsule on link l1
    off_geom_1x = (link1_geom[0] - pos[0, root]).norm()
    off_geom_2x = (link1_geom[1] - pos[1, root]).norm()
    assert_allclose(off_geom_2x / off_geom_1x, 2.0, tol=tol)

    # Returning to unit scale restores the native tree (scaling reads from the baseline, not cumulative), so the
    # previously-2x env's child offset shrinks back to match the unit env's.
    arm.set_scale(np.ones((2, 3)))
    scene.step()
    pos = arm.get_links_pos()
    off_0 = (pos[0, root + 1] - pos[0, root]).norm()
    off_1 = (pos[1, root + 1] - pos[1, root]).norm()
    assert_allclose(off_1 / off_0, 1.0, tol=1e-4)


def test_geom_scale_requires_option():
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    box = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.3)))
    scene.build(n_envs=2)
    with pytest.raises(gs.GenesisException):
        box.set_scale((2.0, 2.0, 2.0))


def test_geom_scale_guards():
    # Anisotropic scale is rejected for a jointed (multi-link) body; an isotropic scale is accepted.
    mjcf = ET.Element("mujoco", model="two_link")
    worldbody = ET.SubElement(mjcf, "worldbody")
    link1 = ET.SubElement(worldbody, "body", name="l1", pos="0 0 1.0")
    ET.SubElement(link1, "joint", name="j1", type="hinge", axis="0 1 0")
    ET.SubElement(link1, "geom", type="capsule", fromto="0 0 0 0.4 0 0", size="0.03", mass="0.2")
    link2 = ET.SubElement(link1, "body", name="l2", pos="0.4 0 0")
    ET.SubElement(link2, "joint", name="j2", type="hinge", axis="0 1 0")
    ET.SubElement(link2, "geom", type="capsule", fromto="0 0 0 0.4 0 0", size="0.03", mass="0.2")
    scene = gs.Scene(rigid_options=gs.options.RigidOptions(enable_geom_scaling=True), show_viewer=False)
    arm = scene.add_entity(morph=gs.morphs.MJCF(file=ET.tostring(mjcf, encoding="unicode")))
    scene.build(n_envs=2)
    with pytest.raises(gs.GenesisException):
        arm.set_scale((2.0, 1.0, 1.0))
    arm.set_scale(2.0)

    # A nonconvex collision mesh in the scene blocks scaling (its distance-field contacts do not rescale yet),
    # even for a convex entity that might collide with it.
    nonconvex_scene = gs.Scene(rigid_options=gs.options.RigidOptions(enable_geom_scaling=True), show_viewer=False)
    box = nonconvex_scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.5)))
    nonconvex_scene.add_entity(gs.morphs.Mesh(file="meshes/duck.obj", scale=0.1, pos=(1.0, 0.0, 0.5), convexify=False))
    nonconvex_scene.build(n_envs=2)
    with pytest.raises(gs.GenesisException):
        box.set_scale(2.0)


@pytest.mark.parametrize("backend", [gs.gpu])
def test_geom_scale_collision(tol):
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            enable_geom_scaling=True,
            gravity=(0.0, 0.0, -10.0),
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    # Fixed platform with its top surface at z = 0.1.
    scene.add_entity(gs.morphs.Box(size=(0.6, 0.6, 0.1), pos=(0.0, 0.0, 0.05), fixed=True))
    faller = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.6)))
    sphere = scene.add_entity(gs.morphs.Sphere(radius=0.05, pos=(2.0, 0.0, 0.6)))
    # env 0 is the unscaled control; env 1 scales the faller (box-box MPR) and sphere (plane-special support).
    scene.build(n_envs=2)

    faller.set_scale((1.0, 1.0, 3.0), envs_idx=[1])
    sphere.set_scale(2.0, envs_idx=[1])
    for _ in range(150):
        scene.step()

    fz = faller.get_pos()[..., 2]
    sz = sphere.get_pos()[..., 2]
    assert torch.isfinite(fz).all() and torch.isfinite(sz).all()
    # Faller rests on the platform top (0.1) plus its half-height: 0.15 unscaled, 0.25 with z-scale 3.
    assert_allclose(fz[0], 0.15, tol=8e-3)
    assert_allclose(fz[1], 0.25, tol=8e-3)
    # Sphere rests on the plane at its radius: 0.05 unscaled, 0.10 with uniform scale 2.
    assert_allclose(sz[0], 0.05, tol=8e-3)
    assert_allclose(sz[1], 0.10, tol=8e-3)


@pytest.mark.parametrize("backend", [gs.gpu])
def test_geom_scale_collision_gjk(tol):
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            enable_geom_scaling=True,
            use_gjk_collision=True,
            gravity=(0.0, 0.0, -10.0),
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.Box(size=(0.6, 0.6, 0.1), pos=(0.0, 0.0, 0.05), fixed=True))
    faller = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.6)))
    # env 0 unscaled control; env 1 z-scaled x3 -> box-box contact resolved by GJK.
    scene.build(n_envs=2)

    faller.set_scale((1.0, 1.0, 3.0), envs_idx=[1])
    for _ in range(150):
        scene.step()

    fz = faller.get_pos()[..., 2]
    assert torch.isfinite(fz).all()
    # Rests on the platform top (0.1) plus its half-height: 0.15 unscaled, 0.25 with z-scale 3.
    assert_allclose(fz[0], 0.15, tol=1e-2)
    assert_allclose(fz[1], 0.25, tol=1e-2)


def test_link_mass_api_scaled(tol):
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(enable_geom_scaling=True),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    box = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.5)))
    scene.build(n_envs=4)
    scene.step()

    link = box.links[0]
    # With scaling enabled, per-link mass is per-env even before any scale is applied.
    base = link.get_mass()
    assert base.shape == (4,)
    assert_allclose(base, base[0], tol=tol)

    # link.get_mass reflects the per-env runtime mass after a non-uniform scale (det(S) = 6 on envs 0-1).
    box.set_scale((2.0, 1.0, 3.0), envs_idx=[0, 1])
    scene.step()
    scaled = link.get_mass()
    assert_allclose(scaled[[0, 1]], base[[0, 1]] * 6.0, tol=tol)
    assert_allclose(scaled[[2, 3]], base[[2, 3]], tol=tol)
    # Single-link entity: entity mass equals the link mass on every env.
    assert_allclose(box.get_mass(), scaled, tol=tol)

    # entity.set_mass distributes a scalar target across links; get_mass must return it on every env,
    # including the scaled ones (the ratio uses the current per-env mass, not the stale build-time value).
    box.set_mass(2.0)
    assert_allclose(box.get_mass(), 2.0, tol=tol)
    assert_allclose(link.get_mass(), 2.0, tol=tol)

    # link.set_mass accepts an explicit per-env vector.
    target = np.array([1.0, 2.0, 3.0, 4.0])
    link.set_mass(target)
    assert_allclose(link.get_mass(), target, tol=tol)


def test_potential_energy_scaled(tol):
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(enable_geom_scaling=True),
        show_viewer=False,
    )
    box = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 1.0)))
    scene.build(n_envs=4)

    # Non-uniform scale on envs 0-1 (det(S) = 6); envs 2-3 stay unit. No plane: free fall keeps every env's
    # COM at the same height, so at a common instant PE is proportional to mass.
    box.set_scale((2.0, 1.0, 3.0), envs_idx=[0, 1])
    for _ in range(5):
        scene.step()

    pe = box.get_potential_energy()
    assert pe.shape == (4,)
    assert_allclose(pe[0], pe[1], tol=tol)
    assert_allclose(pe[2], pe[3], tol=tol)
    assert_allclose(pe[0] / pe[2], 6.0, tol=tol)

    # envs_idx subset must not crash and must return the matching per-env slice (regression: the per-env mass
    # axis was previously fetched for all envs, mismatching the sliced links_pos).
    pe_subset = box.get_potential_energy(envs_idx=[0, 2])
    assert pe_subset.shape == (2,)
    assert_allclose(pe_subset, pe[[0, 2]], tol=tol)


@pytest.mark.parametrize("backend", [gs.gpu])
def test_geom_scale_visual(tol):
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(enable_geom_scaling=True),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    box = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.5)))
    scene.build(n_envs=2)
    scene.step()

    ext0 = torch.diff(box.get_vAABB(), dim=-2)[:, 0]  # (n_envs, 3) visual extents, both unit
    assert_allclose(ext0[0], ext0[1], tol=1e-3)

    box.set_scale((2.0, 1.0, 3.0), envs_idx=[1])
    scene.step()

    # Visual AABB: env 0 unchanged, env 1 scaled per axis.
    ext1 = torch.diff(box.get_vAABB(), dim=-2)[:, 0]
    assert_allclose(ext1[0], ext0[0], tol=1e-3)
    assert_allclose(ext1[1] / ext0[1], (2.0, 1.0, 3.0), tol=2e-2)

    # get_vverts reflects the same scale (env 1 world-vert bbox is the per-axis-scaled env 0 bbox).
    vv = box.get_vverts()
    vext = vv.max(dim=-2).values - vv.min(dim=-2).values
    assert_allclose(vext[1] / vext[0], (2.0, 1.0, 3.0), tol=2e-2)

    # Visual and collision geometry stay consistent under scale.
    assert_allclose(torch.diff(box.get_AABB(), dim=-2)[1, 0], ext1[1], tol=1e-2)


@pytest.mark.parametrize("backend", [gs.gpu])
def test_geom_scale_analytic_primitive_contacts(tol):
    # Scaled sphere-box, sphere-capsule and capsule-capsule pairs previously took the analytic narrowphase,
    # which reads unscaled geom dimensions. They now defer to the support-based path, so the scaled shape rests
    # at the scaled height. env 0 is the unscaled control; env 1 scales each faller. Every support is a fixed
    # geom whose top surface sits at z = 0.2, so every faller rests at 0.2 + its (possibly scaled) bottom radius.
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            enable_geom_scaling=True,
            gravity=(0.0, 0.0, -10.0),
        ),
        show_viewer=False,
    )
    scene.add_entity(
        gs.morphs.Box(
            size=(0.4, 0.4, 0.2),
            pos=(0.0, 0.0, 0.1),
            fixed=True,
        ),
    )
    sphere_on_box = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.1,
            pos=(0.0, 0.0, 0.45),
        ),
    )

    fixed_capsule_a = ET.Element("mujoco")
    body_a = ET.SubElement(ET.SubElement(fixed_capsule_a, "worldbody"), "body", pos="2 0 0")
    ET.SubElement(body_a, "geom", type="capsule", fromto="-0.15 0 0.1 0.15 0 0.1", size="0.1", mass="1.0")
    scene.add_entity(morph=gs.morphs.MJCF(file=ET.tostring(fixed_capsule_a, encoding="unicode")))
    sphere_on_capsule = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.1,
            pos=(2.0, 0.0, 0.45),
        ),
    )

    fixed_capsule_b = ET.Element("mujoco")
    body_b = ET.SubElement(ET.SubElement(fixed_capsule_b, "worldbody"), "body", pos="4 0 0")
    ET.SubElement(body_b, "geom", type="capsule", fromto="-0.15 0 0.1 0.15 0 0.1", size="0.1", mass="1.0")
    scene.add_entity(morph=gs.morphs.MJCF(file=ET.tostring(fixed_capsule_b, encoding="unicode")))
    faller_capsule = ET.Element("mujoco")
    body_c = ET.SubElement(ET.SubElement(faller_capsule, "worldbody"), "body", pos="4 0 0.45")
    ET.SubElement(body_c, "joint", type="free")
    ET.SubElement(body_c, "geom", type="capsule", fromto="0 -0.15 0 0 0.15 0", size="0.1", mass="0.5")
    capsule_on_capsule = scene.add_entity(morph=gs.morphs.MJCF(file=ET.tostring(faller_capsule, encoding="unicode")))

    scene.build(n_envs=2)

    # env 1: anisotropic sphere (-> ellipsoid, vertical semi-axis 0.2) and a radius-doubled capsule (radial
    # scale stays isotropic). Both have a 0.2 bottom radius, matching the scaled spheres.
    sphere_on_box.set_scale((1.0, 1.0, 2.0), envs_idx=[1])
    sphere_on_capsule.set_scale((1.0, 1.0, 2.0), envs_idx=[1])
    capsule_on_capsule.set_scale((2.0, 2.0, 1.0), envs_idx=[1])
    for _ in range(100):
        scene.step()

    for faller in (sphere_on_box, sphere_on_capsule, capsule_on_capsule):
        z = faller.get_pos()[..., 2]
        assert torch.isfinite(z).all()
        assert_allclose(z[0], 0.30, tol=1.5e-2)  # unscaled: support top 0.2 + 0.1 radius
        assert_allclose(z[1], 0.40, tol=1.5e-2)  # scaled: support top 0.2 + 0.2 scaled radius / semi-axis


@pytest.mark.parametrize("backend", [gs.gpu])
def test_geom_scale_box_box_detection(tol):
    # The analytic box-box specialization (box_box_detection=True) reads box half-extents directly; per-env
    # scale must scale them so a scaled box rests at the scaled height. env 0 unscaled control, env 1 z-scaled.
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            enable_geom_scaling=True,
            box_box_detection=True,
            gravity=(0.0, 0.0, -10.0),
        ),
        show_viewer=False,
    )
    scene.add_entity(
        gs.morphs.Box(
            size=(0.6, 0.6, 0.2),
            pos=(0.0, 0.0, 0.1),
            fixed=True,
        ),
    )
    faller = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.45),
        ),
    )
    scene.build(n_envs=2)

    faller.set_scale((1.0, 1.0, 3.0), envs_idx=[1])
    for _ in range(80):
        scene.step()

    z = faller.get_pos()[..., 2]
    assert torch.isfinite(z).all()
    assert_allclose(z[0], 0.25, tol=1e-2)  # rests on platform top 0.2 + half-height 0.05
    assert_allclose(z[1], 0.35, tol=1e-2)  # z-scale 3 -> half-height 0.15
