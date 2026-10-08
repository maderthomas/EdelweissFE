"""Embedded beams: Marmot Euler-Bernoulli beam elements (B23, B33) with the beam section, embedded in host
continuum elements by the embedded region constraint (perfect bond, with or without tying the rotations) and by the
embedded bond generator (bond-slip), verified against analytical solutions.

Needs the Marmot modules BeamElement, TrussElement, EmbeddedBondElement and LinearElasticBondSlip.
"""

import math

import numpy as np
import pytest

pytest.importorskip("edelweissfe.elements.marmotelement.element")

from edelweissfe.drivers.inputfiledrivensimulation import (  # noqa: E402
    finiteElementSimulation,
)
from edelweissfe.utils.inputfileparser import parseInputFile  # noqa: E402


def _marmotHas(elementType: str) -> bool:
    from edelweissfe.elements.marmotelement.element import MarmotElementWrapper

    try:
        MarmotElementWrapper(elementType, 1)
        return True
    except NotImplementedError:
        return False


pytestmark = pytest.mark.skipif(
    not (_marmotHas("B23") and _marmotHas("B33") and _marmotHas("TR2D2") and _marmotHas("EB2D2Q4")),
    reason="Marmot is built without the BeamElement / TrussElement / EmbeddedBondElement modules",
)

E_STEEL, NU_STEEL, E_CONCRETE, NU_CONCRETE = 200000.0, 0.3, 30000.0, 0.2
G_STEEL = E_STEEL / (2 * (1 + NU_STEEL))


def _run(tmp_path, deck: str, *results):
    path = tmp_path / "test.inp"
    path.write_text(deck)
    model, fieldOutputController = finiteElementSimulation(parseInputFile(str(path)), suppressPlots=True)
    return model, [fieldOutputController.fieldOutputs[r] for r in results]


def _line(firstLabel: int, start, end, nElements: int, elType: str):
    """*node and *element blocks of a straight line of 2-node elements, and its first and last node labels."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    nodes = "\n".join(
        f"{firstLabel + i}, " + ", ".join(f"{c:.12g}" for c in start + (end - start) * i / nElements)
        for i in range(nElements + 1)
    )
    elements = "\n".join(f"{firstLabel + i}, {firstLabel + i}, {firstLabel + i + 1}" for i in range(nElements))
    return f"*node\n{nodes}\n\n*element, type={elType}\n{elements}\n", firstLabel, firstLabel + nElements


def _beamDeck(elType, start, end, nElements, section, domain, steps, extra=""):
    line, first, last = _line(1, start, end, nElements, elType)
    return f"""*material, name=linearelastic, id=steel, provider=marmot
{E_STEEL}, {NU_STEEL}

{line}
*elSet, elSet=beam, generate=True
{first}, {last - 1}, 1
*nSet, nSet=root
{first}
*nSet, nSet=tip
{last}

*section, name=s, type=beam, {section}, material=steel
beam
{extra}
*job, name=job, domain={domain}
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perNode, name=u, nSet=tip, field=displacement, result=U, saveHistory=True,
>>perNode, name=r, nSet=tip, field=rotation, result=U, saveHistory=True,

*step, solver=theSolver
maxInc=1.0, minInc=1e-8, maxNumInc=10, maxIter=10, stepLength=1
{steps}
"""


def test_beam_cantilever_2d(tmp_path):
    """an inclined 2D cantilever (rectangle 10 x 20) under a tip force normal to it and a tip moment: exact tip
    deflection and rotation"""
    b, h, L, P, M = 10.0, 20.0, 500.0, 7.0, 1500.0
    I = b * h**3 / 12  # noqa: E741
    t, n = np.array([0.6, 0.8]), np.array([-0.8, 0.6])
    deck = _beamDeck(
        "B23",
        (0, 0),
        L * t,
        4,
        f"profile=rectangle, b={b}, h={h}",
        "2d",
        f""">>dirichlet, name=root, nSet=root, field=displacement, 1=0, 2=0
>>dirichlet, name=rootR, nSet=root, field=rotation, 1=0
>>nodeforces, name=P, nSet=tip, field=displacement, 1={P * n[0]}, 2={P * n[1]}
>>nodeforces, name=M, nSet=tip, field=rotation, 1={M}
""",
    )
    _, (u, r) = _run(tmp_path, deck, "u", "r")
    u, r = u.getLastResult().ravel(), r.getLastResult().item()
    EI = E_STEEL * I
    np.testing.assert_allclose(u @ n, P * L**3 / (3 * EI) + M * L**2 / (2 * EI), rtol=1e-10)
    np.testing.assert_allclose(r, P * L**2 / (2 * EI) + M * L / EI, rtol=1e-10)
    assert abs(u @ t) < 1e-12


def test_beam_cantilever_3d(tmp_path):
    """a 3D cantilever (circle, d = 20) along (1, 2, 2) / 3 under a tip force and a torque: exact deflection and
    twist (G J of the circle)"""
    d, L, P, T = 20.0, 300.0, 3.0, 500.0
    I, J = math.pi * d**4 / 64, math.pi * d**4 / 32  # noqa: E741
    t = np.array([1.0, 2.0, 2.0]) / 3
    force = P * np.array([2.0, -1.0, 0.0]) / math.sqrt(5)  # normal to the axis
    torque = T * t
    deck = _beamDeck(
        "B33",
        (0, 0, 0),
        L * t,
        3,
        f"profile=circle, d={d}, nFibers=4",
        "3d",
        f""">>dirichlet, name=root, nSet=root, field=displacement, 1=0, 2=0, 3=0
>>dirichlet, name=rootR, nSet=root, field=rotation, 1=0, 2=0, 3=0
>>nodeforces, name=P, nSet=tip, field=displacement, 1={force[0]}, 2={force[1]}, 3={force[2]}
>>nodeforces, name=T, nSet=tip, field=rotation, 1={torque[0]}, 2={torque[1]}, 3={torque[2]}
""",
    )
    _, (u, r) = _run(tmp_path, deck, "u", "r")
    u, r = u.getLastResult().ravel(), r.getLastResult().ravel()
    np.testing.assert_allclose(u, force * L**3 / (3 * E_STEEL * I), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(
        r, np.cross(t, force) * L**2 / (2 * E_STEEL * I) + torque * L / (G_STEEL * J), rtol=1e-9, atol=1e-12
    )


def _strip(nX=8, nY=2, elType="CPE4", E=E_CONCRETE, lX=4.0, lY=1.0):
    return f"""*material, name=linearelastic, id=concrete, provider=marmot
{E}, {NU_CONCRETE}

*material, name=linearelastic, id=steel, provider=marmot
{E_STEEL}, {NU_STEEL}

*modelGenerator, generator=planeRectQuad, name=gen
l={lX}
h={lY}
nX={nX}
nY={nY}
elType={elType}

*section, name=concreteSection, thickness=1.0, material=concrete, type=plane
gen_all
"""


@pytest.mark.parametrize("rotations", ["free", "axis", "host"])
def test_embedded_beam_follows_uniform_host_strain(tmp_path, rotations):
    """an inclined beam embedded in a strip in uniaxial stress (plane strain): the beam takes the host strain along
    its axis; it stays straight (zero curvature) with free rotations or rotations tied to the host material line
    along the axis, but is bent if tied to the host rotation (zero here), as the uniform strain rotates the inclined
    material line"""
    start, end = np.array([0.3, 0.2]), np.array([3.7, 0.8])
    line, first, last = _line(100001, start, end, 5, "B23")
    deck = (
        _strip()
        + f"""{line}
*elSet, elSet=beams, generate=True
{first}, {last - 1}, 1

*section, name=beamSection, type=beam, area=1e-8, I=1e-8, material=steel
beams

*constraint, type=embeddedRegion, name=embedded
embeddedElSet=beams, hostElSet=gen_all, rotations={rotations}

*job, name=job, domain=2d
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perElement, name=strain, elSet=beams, result=axial strain, quadraturePoint=0, saveHistory=True,
>>perElement, name=kappa, elSet=beams, result=curvature, quadraturePoint=0, saveHistory=True,

*step, solver=theSolver
maxInc=1.0, minInc=1e-8, maxNumInc=10, maxIter=10, stepLength=1
>>dirichlet, name=left, nSet=gen_left, field=displacement, 1=0
>>dirichlet, name=corner, nSet=gen_leftBottom, field=displacement, 2=0
>>dirichlet, name=right, nSet=gen_right, field=displacement, 1=0.004
"""
    )
    _, (strain, kappa) = _run(tmp_path, deck, "strain", "kappa")
    t = (end - start) / np.linalg.norm(end - start)
    # the beam is negligibly stiff (E A / (E_c A_c) ~ 7e-5): the host stays in the uniform strain of the plain strip
    epsXX = 1e-3
    epsYY = -NU_CONCRETE / (1 - NU_CONCRETE) * epsXX
    curvature = np.asarray(kappa.getLastResult()).ravel()
    # the material line rotates by theta = t x (eps t); a beam element of length l with the chord rotated by theta,
    # but both nodes at zero rotation (rotations=host), has the curvature 6 theta / l at its ends (here at the
    # first quadrature point, (6 - 12 s) theta / l with s = (1 - sqrt(3/5)) / 2)
    theta = t[0] * t[1] * (epsYY - epsXX)
    elementLength = np.linalg.norm(end - start) / 5
    s = (1 - math.sqrt(0.6)) / 2
    curvatureOfHostRotation = (6 - 12 * s) * theta / elementLength
    if rotations == "host":
        np.testing.assert_allclose(curvature, curvatureOfHostRotation, rtol=1e-3)
        return
    np.testing.assert_allclose(strain.getLastResult(), t[0] ** 2 * epsXX + t[1] ** 2 * epsYY, rtol=1e-4)
    assert np.abs(curvature).max() < 1e-4 * np.abs(curvatureOfHostRotation)


@pytest.mark.parametrize("rotations", ["axis", "free"])
def test_embedded_cantilever_clamped_in_stiff_block(tmp_path, rotations):
    """a cantilever embedded with its root in a (practically rigid) block: with the rotations tied to the host, the
    free part deflects as a cantilever clamped at the block face, P L^3 / (3 E I); with free rotations the root is
    not clamped (only the nodal displacements are tied) and the beam deflects more"""
    b, h, L, P = 10.0, 10.0, 200.0, 1.0
    I = b * h**3 / 12  # noqa: E741
    embedded, _, _ = _line(100001, (50.0, 50.0), (100.0, 50.0), 2, "B23")
    free, _, tip = _line(100003, (100.0, 50.0), (100.0 + L, 50.0), 8, "B23")
    free = free.replace("100003, 100, 50\n", "")  # the face node is the last node of the embedded part
    deck = (
        _strip(nX=4, nY=4, E=E_STEEL * 1e4, lX=100.0, lY=100.0)
        + f"""{embedded}
{free}
*elSet, elSet=beams, generate=True
100001, 100010, 1

*nSet, nSet=tip
{tip}

*section, name=beamSection, type=beam, profile=rectangle, b={b}, h={h}, material=steel
beams

*constraint, type=embeddedRegion, name=embedded
embeddedElSet=beams, hostElSet=gen_all, rotations={rotations}, allowUnembedded=True

*job, name=job, domain=2d
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perNode, name=u, nSet=tip, field=displacement, result=U, saveHistory=True,

*step, solver=theSolver
maxInc=1.0, minInc=1e-8, maxNumInc=10, maxIter=10, stepLength=1
>>dirichlet, name=left, nSet=gen_left, field=displacement, 1=0, 2=0
>>nodeforces, name=P, nSet=tip, field=displacement, 2={P}
"""
    )
    _, (u,) = _run(tmp_path, deck, "u")
    deflection = u.getLastResult().ravel()[1]
    clamped = P * L**3 / (3 * E_STEEL * I)
    if rotations == "axis":
        np.testing.assert_allclose(deflection, clamped, rtol=1e-5)
    else:
        assert deflection > 1.05 * clamped


def test_embedded_cantilever_clamped_in_stiff_block_3d(tmp_path):
    """3D: a cantilever (B33) embedded in a stiff C3D8 block with tied rotations: biaxial bending and torsion of the
    free part as clamped at the block face"""
    b, h, L = 10.0, 20.0, 200.0
    Iz, Iy = b * h**3 / 12, h * b**3 / 12  # h along the local y axis (orientation (0, 1, 0)), b along z
    J = h * b**3 * (1 / 3 - 0.21 * b / h * (1 - b**4 / (12 * h**4)))
    Py, Pz, T = 1.0, 0.5, 20.0
    embedded, _, _ = _line(100001, (50.0, 50.0, 50.0), (100.0, 50.0, 50.0), 2, "B33")
    free, _, tip = _line(100003, (100.0, 50.0, 50.0), (100.0 + L, 50.0, 50.0), 6, "B33")
    free = free.replace("100003, 100, 50, 50\n", "")
    deck = f"""*material, name=linearelastic, id=block, provider=marmot
{E_STEEL * 1e4}, 0.2

*material, name=linearelastic, id=steel, provider=marmot
{E_STEEL}, {NU_STEEL}

*modelGenerator, generator=boxGen, name=gen
lX=100
lY=100
lZ=100
nX=2
nY=2
nZ=2
elType=C3D8

*section, name=blockSection, material=block, type=solid
gen_all

{embedded}
{free}
*elSet, elSet=beams, generate=True
100001, 100008, 1

*nSet, nSet=tip
{tip}

*section, name=beamSection, type=beam, profile=rectangle, b={b}, h={h}, n1x=0, n1y=1, n1z=0, nFibers=4, material=steel
beams

*constraint, type=embeddedRegion, name=embedded
embeddedElSet=beams, hostElSet=gen_all, rotations=axis, allowUnembedded=True

*job, name=job, domain=3d
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perNode, name=u, nSet=tip, field=displacement, result=U, saveHistory=True,
>>perNode, name=r, nSet=tip, field=rotation, result=U, saveHistory=True,

*step, solver=theSolver
maxInc=1.0, minInc=1e-8, maxNumInc=10, maxIter=10, stepLength=1
>>dirichlet, name=left, nSet=gen_left, field=displacement, 1=0, 2=0, 3=0
>>nodeforces, name=P, nSet=tip, field=displacement, 2={Py}, 3={Pz}
>>nodeforces, name=T, nSet=tip, field=rotation, 1={T}
"""
    _, (u, r) = _run(tmp_path, deck, "u", "r")
    u, r = u.getLastResult().ravel(), r.getLastResult().ravel()
    np.testing.assert_allclose(u[1], Py * L**3 / (3 * E_STEEL * Iz), rtol=1e-5)
    np.testing.assert_allclose(u[2], Pz * L**3 / (3 * E_STEEL * Iy), rtol=1e-5)
    np.testing.assert_allclose(r[0], T * L / (G_STEEL * J), rtol=1e-5)


def _pullOutDeck(barSection: str, elType: str, bond: str, generatorOptions: str = ""):
    """a bar along y = 0.37 through a rigid strip (x = 0..4), 10 elements, pulled at x = 4"""
    bar, first, last = _line(100001, (0.0, 0.37), (4.0, 0.37), 10, elType)
    return (
        _strip(nX=5)
        + f"""{bar}
*elSet, elSet=rebars, generate=True
{first}, {last - 1}, 1

*nSet, nSet=loadedEnd
{last}

*section, name=rebarSection, {barSection}, material=steel
rebars

*material, name=LinearElasticBondSlip, id=bond, provider=marmot
{bond}

*modelGenerator, generator=embeddedBond, name=rebarBond, executeAfterManualGeneration=True
rebarElSet=rebars
hostElSet=gen_all
material=bond
perimeter=0.3545
{generatorOptions}

*job, name=job, domain=2d
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perNode, name=F, nSet=loadedEnd, field=displacement, result=P, f(x)='sum(x[:,0])', saveHistory=True,

*step, solver=theSolver
maxInc=0.25, minInc=1e-8, maxNumInc=100, maxIter=10, stepLength=1
>>dirichlet, name=host, nSet=gen_all, field=displacement, 1=0, 2=0
>>dirichlet, name=pull, nSet=loadedEnd, field=displacement, 1=0.01
"""
    )


@pytest.mark.parametrize("generatorOptions", ["", "largeSlip=True\nmaxSlip=0.5"])
def test_bond_slip_pull_out_beam_equals_truss(tmp_path, generatorOptions):
    """axial pull-out of a beam with bond-slip (small and large slip) gives the force of a truss of the same area
    and the analytical F = E A beta tanh(beta L) u; the beam rotations stay zero"""
    A, Kt = 0.01, 1000.0
    _, (FTruss,) = _run(tmp_path, _pullOutDeck(f"type=truss, area={A}", "TR2D2", f"{Kt}, 1e6", generatorOptions), "F")
    _, (FBeam,) = _run(
        tmp_path, _pullOutDeck(f"type=beam, area={A}, I=1e-5", "B23", f"{Kt}, 1e6", generatorOptions), "F"
    )
    np.testing.assert_allclose(FBeam.getLastResult(), FTruss.getLastResult(), rtol=1e-9)
    EA = E_STEEL * A
    beta = math.sqrt(Kt * 0.3545 / EA)
    np.testing.assert_allclose(FBeam.getLastResult(), EA * beta * math.tanh(beta * 4.0) * 0.01, rtol=5e-3)


def test_bond_slip_split_beams_match_perfect_bond(tmp_path):
    """an inclined beam in a strip bent by a transverse end displacement: with splitBars=True, the beam is re-meshed
    with B23 elements at the host crossings, and a stiff bond converges to perfect bond (displacements tied, rotations
    free) of the split beam"""

    def deck(bondStiffness, perfectBond):
        line, first, last = _line(100001, (0.3, 0.2), (3.7, 0.8), 3, "B23")
        coupling = (
            "*constraint, type=embeddedRegion, name=embedded\nembeddedElSet=beams, hostElSet=gen_all"
            if perfectBond
            else ""
        )
        return (
            _strip()
            + f"""{line}
*elSet, elSet=beams, generate=True
{first}, {last - 1}, 1

*section, name=beamSection, type=beam, area=0.01, I=1e-4, material=steel
beams

*material, name=LinearElasticBondSlip, id=bond, provider=marmot
{bondStiffness}, {bondStiffness}

*modelGenerator, generator=embeddedBond, name=beamBond, executeAfterManualGeneration=True
rebarElSet=beams
hostElSet=gen_all
material=bond
perimeter=0.3545
splitBars=True

{coupling}

*job, name=job, domain=2d
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perNode, name=uBeam, elSet=beams, field=displacement, result=U, saveHistory=True,
>>perNode, name=rBeam, elSet=beams, field=rotation, result=U, saveHistory=True,
>>perNode, name=uHost, elSet=gen_all, field=displacement, result=U, saveHistory=True,

*step, solver=theSolver
maxInc=1.0, minInc=1e-8, maxNumInc=10, maxIter=10, stepLength=1
>>dirichlet, name=left, nSet=gen_left, field=displacement, 1=0, 2=0
>>dirichlet, name=right, nSet=gen_right, field=displacement, 2=0.01
"""
        )

    model, (uPerfect, rPerfect, hostPerfect) = _run(tmp_path, deck(0.0, True), "uBeam", "rBeam", "uHost")
    model, (uBond, rBond, hostBond) = _run(tmp_path, deck(1e10, False), "uBeam", "rBeam", "uHost")
    beams = model.elementSets["beams"]
    assert len(beams) > 3 and all(el.elType == "B23" for el in beams)
    np.testing.assert_allclose(uBond.getLastResult(), uPerfect.getLastResult(), atol=1e-6 * 0.01)
    np.testing.assert_allclose(hostBond.getLastResult(), hostPerfect.getLastResult(), atol=1e-6 * 0.01)
    # the rotations follow from the bending of the beam between the bond points; the finite bond stiffness shows most
    rScale = np.abs(rPerfect.getLastResult()).max()
    assert rScale > 1e-4
    np.testing.assert_allclose(rBond.getLastResult(), rPerfect.getLastResult(), atol=1e-4 * rScale)


@pytest.mark.parametrize("shape, nDim", [("quad4", 2), ("quad8", 2), ("hexa8", 3), ("hexa20", 3)])
def test_rotation_operator(shape, nDim):
    """the rotation operators of the embedded region constraint: a rigid rotation gives the rotation itself (both
    modes); a uniform strain leaves the axis mode at t x (eps t) and the host mode at zero"""
    from types import SimpleNamespace

    from edelweissfe.constraints.embeddedregion import rotationOperator
    from edelweissfe.utils.embedding import HostPoint
    from edelweissfe.utils.isoparametricshapes import NODE_PARAMETRIC_COORDINATES

    rng = np.random.default_rng(1)
    xiNodes = NODE_PARAMETRIC_COORDINATES[shape]
    A = np.eye(nDim) * 2.0 + 0.2 * rng.random((nDim, nDim))  # an affinely distorted element
    X = xiNodes @ A.T + 1.0
    element = SimpleNamespace(ensightType=shape, nodes=[SimpleNamespace(coordinates=x) for x in X])
    point = HostPoint(element, 0.3 * np.ones(nDim) - 0.1)
    t = np.ones(nDim) / math.sqrt(nDim)

    if nDim == 2:
        w = 1e-3
        W = np.array([[0.0, -w], [w, 0.0]])
        rotation = np.array([w])
    else:
        rotation = np.array([1e-3, -2e-3, 0.5e-3])
        W = np.array(
            [[0.0, -rotation[2], rotation[1]], [rotation[2], 0.0, -rotation[0]], [-rotation[1], rotation[0], 0.0]]
        )
    eps = 1e-3 * np.diag(np.arange(1, nDim + 1, dtype=float))

    for mode in ("axis", "host"):
        C = rotationOperator(point, mode, [t])
        np.testing.assert_allclose(np.einsum("rai,ai->r", C, X @ W.T), rotation, atol=1e-15)
        theta = np.einsum("rai,ai->r", C, X @ eps.T)
        if mode == "host":
            np.testing.assert_allclose(theta, 0.0, atol=1e-15)
        elif nDim == 2:
            np.testing.assert_allclose(theta, [t[0] * (eps @ t)[1] - t[1] * (eps @ t)[0]], atol=1e-15)
        else:
            np.testing.assert_allclose(theta, np.cross(t, eps @ t), atol=1e-15)
