"""Co-rotational (finite rotation) beams: the Marmot element BE2D2CR with the beam section, in large rotation
benchmarks (roll-up of a cantilever into a full circle, elastica), the section multiplicity, and with the embedded
bond generator (small and large slip).

Needs the Marmot modules BeamElement (with BE2D2CR), TrussElement, EmbeddedBondElement and LinearElasticBondSlip.
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
    not (_marmotHas("BE2D2CR") and _marmotHas("TR2D2") and _marmotHas("EB2D2Q4")),
    reason="Marmot is built without BE2D2CR / TrussElement / EmbeddedBondElement",
)

E_STEEL = 200000.0


def _run(tmp_path, deck: str, *results):
    path = tmp_path / "test.inp"
    path.write_text(deck)
    model, fieldOutputController = finiteElementSimulation(parseInputFile(str(path)), suppressPlots=True)
    return model, [fieldOutputController.fieldOutputs[r] for r in results]


def _cantilever(elType, L, nElements, section, load, maxInc, extraOutput=""):
    nodes = "\n".join(f"{i + 1}, {L * i / nElements:.12g}, 0" for i in range(nElements + 1))
    elements = "\n".join(f"{i + 1}, {i + 1}, {i + 2}" for i in range(nElements))
    return f"""*material, name=linearelastic, id=steel, provider=marmot
{E_STEEL}, 0.3

*node
{nodes}
*element, type={elType}
{elements}
*elSet, elSet=beam, generate=True
1, {nElements}, 1
*nSet, nSet=root
1
*nSet, nSet=tip
{nElements + 1}

*section, name=s, type=beam, {section}, material=steel
beam

*job, name=job, domain=2d
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perNode, name=u, nSet=tip, field=displacement, result=U, saveHistory=True,
>>perNode, name=r, nSet=tip, field=rotation, result=U, saveHistory=True,
{extraOutput}
*step, solver=theSolver
maxInc={maxInc}, minInc=1e-8, maxNumInc=1000, maxIter=15, stepLength=1
>>dirichlet, name=root, nSet=root, field=displacement, 1=0, 2=0
>>dirichlet, name=rootR, nSet=root, field=rotation, 1=0
{load}
"""


def test_roll_up_full_circle(tmp_path):
    """an end moment M = 2 pi EI / L rolls the cantilever up into a full circle: the tip returns to the root with the
    rotation 2 pi; on the way the tip follows x = R sin(L/R), y = R (1 - cos(L/R)), R = EI / M"""
    L, b, h, n = 100.0, 1.0, 2.0, 20
    EI = E_STEEL * b * h**3 / 12
    M = 2 * math.pi * EI / L
    deck = _cantilever(
        "BE2D2CR",
        L,
        n,
        f"profile=rectangle, b={b}, h={h}, nY=2",
        f">>nodeforces, name=M, nSet=tip, field=rotation, 1={M}",
        0.05,
    )
    _, (u, r) = _run(tmp_path, deck, "u", "r")
    U, R_ = np.asarray(u.getResultHistory()).reshape(-1, 2), np.asarray(r.getResultHistory()).ravel()
    t = np.asarray(u.getTimeHistory())
    np.testing.assert_allclose(R_, 2 * math.pi * t, rtol=1e-8)  # tip rotation = M L / EI, exact
    m = t > 0
    Rad = L / (2 * math.pi * t[m])
    x, y = L + U[m, 0], U[m, 1]
    err = np.hypot(x - Rad * np.sin(L / Rad), y - Rad * (1 - np.cos(L / Rad))) / L
    assert err.max() < 0.5 * (math.pi / n) ** 2  # chords of the circle: O(h^2)
    assert abs(x[-1]) < 1e-6 * L and abs(y[-1]) < 1e-6 * L  # full circle: the tip at the root


@pytest.mark.parametrize("lam, ref", [(2.0, (0.49346, 0.16064, 0.78175)), (10.0, (0.81061, 0.55500, 1.43029))])
def test_elastica_tip_force(tmp_path, lam, ref):
    """cantilever with a dead tip force P normal to the axis, lambda = P L^2 / EI: tip deflection v/L, shortening
    (L - x)/L and rotation vs the elastica (Mattiasson; rotation by the elliptic integral)"""
    L, b, h, n = 100.0, 1.0, 1.0, 40
    EI = E_STEEL * b * h**3 / 12
    deck = _cantilever(
        "BE2D2CR",
        L,
        n,
        f"profile=rectangle, b={b}, h={h}, nY=2",
        f">>nodeforces, name=P, nSet=tip, field=displacement, 2={lam * EI / L**2}",
        0.1,
    )
    _, (u, r) = _run(tmp_path, deck, "u", "r")
    U, R_ = u.getLastResult().ravel(), r.getLastResult().ravel()
    np.testing.assert_allclose([U[1] / L, -U[0] / L, R_[0]], ref, atol=1e-3)


def test_small_load_equals_BE2D2(tmp_path):
    """small tip load: BE2D2CR = BE2D2 (geometrically linear) up to the second order effects"""
    L, b, h, n, P = 100.0, 2.0, 4.0, 4, 1e-3
    results = []
    for elType in ("BE2D2", "BE2D2CR"):
        deck = _cantilever(
            elType,
            L,
            n,
            f"profile=rectangle, b={b}, h={h}",
            f">>nodeforces, name=P, nSet=tip, field=displacement, 2={P}",
            1.0,
        )
        _, (u,) = _run(tmp_path, deck, "u")
        results.append(u.getLastResult().ravel()[1])
    np.testing.assert_allclose(results[1], results[0], rtol=1e-8)
    np.testing.assert_allclose(results[0], P * L**3 / (3 * E_STEEL * b * h**3 / 12), rtol=1e-10)


def test_section_multiplicity(tmp_path):
    """multiplicity = k: a beam of k identical members, k times stiffer (A, I scaled, coordinates not)"""
    L, d, n, P = 50.0, 1.0, 4, 1e-4
    tips = []
    for k in (1.0, 3.0):
        deck = _cantilever(
            "BE2D2CR",
            L,
            n,
            f"profile=circle, d={d}, nRings=2, nSectors=8, multiplicity={k}",
            f">>nodeforces, name=P, nSet=tip, field=displacement, 2={P}",
            1.0,
        )
        _, (u,) = _run(tmp_path, deck, "u")
        tips.append(u.getLastResult().ravel()[1])
    np.testing.assert_allclose(tips[0] / tips[1], 3.0, rtol=1e-9)
    np.testing.assert_allclose(tips[0], P * L**3 / (3 * E_STEEL * math.pi * d**4 / 64), rtol=1e-6)


def _strip(nX):
    """a rigid (fixed) CPE4 strip 4 x 1 (x = 0..4)"""
    nodes, els = [], []
    for j in range(2):
        for i in range(nX + 1):
            nodes.append(f"{j * (nX + 1) + i + 1}, {4.0 * i / nX:.12g}, {float(j)}")
    for i in range(nX):
        els.append(f"{i + 1}, {i + 1}, {i + 2}, {nX + 1 + i + 2}, {nX + 1 + i + 1}")
    return (
        "*material, name=linearelastic, id=concrete, provider=marmot\n30000, 0.2\n"
        "*material, name=linearelastic, id=steel, provider=marmot\n200000, 0.3\n"
        "*node\n" + "\n".join(nodes) + "\n*element, type=CPE4\n" + "\n".join(els) + "\n"
        f"*elSet, elSet=gen_all, generate=True\n1, {nX}, 1\n"
        f"*nSet, nSet=hostNodes, generate=True\n1, {2 * (nX + 1)}, 1\n"
        "*section, name=host, thickness=1, material=concrete, type=plane\ngen_all\n"
    )


@pytest.mark.parametrize("generatorOptions", ["", "largeSlip=True\nmaxSlip=0.5"])
def test_bond_slip_pull_out_corotational_beam_equals_truss(tmp_path, generatorOptions):
    """axial pull-out of a BE2D2CR beam with bond-slip (small and large slip) in a fixed strip: the force of a TR2D2
    truss of the same area (the difference is the second order axial strain term, ~ strain)"""
    A, Kt, p = 0.01, 1000.0, 0.3545

    def deck(barSection, elType):
        nodes = "\n".join(f"{100001 + i}, {4.0 * i / 10:.12g}, 0.37" for i in range(11))
        els = "\n".join(f"{100001 + i}, {100001 + i}, {100002 + i}" for i in range(10))
        return _strip(5) + f"""*node
{nodes}
*element, type={elType}
{els}
*elSet, elSet=rebars, generate=True
100001, 100010, 1
*nSet, nSet=loadedEnd
100011
*section, name=rebarSection, {barSection}, material=steel
rebars
*material, name=LinearElasticBondSlip, id=bond, provider=marmot
{Kt}, 1e6
*modelGenerator, generator=embeddedBond, name=rebarBond, executeAfterManualGeneration=True
rebarElSet=rebars
hostElSet=gen_all
material=bond
perimeter={p}
{generatorOptions}

*job, name=job, domain=2d
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perNode, name=F, nSet=loadedEnd, field=displacement, result=P, f(x)='sum(x[:,0])', saveHistory=True,

*step, solver=theSolver
maxInc=0.25, minInc=1e-8, maxNumInc=100, maxIter=10, stepLength=1
>>dirichlet, name=host, nSet=hostNodes, field=displacement, 1=0, 2=0
>>dirichlet, name=pull, nSet=loadedEnd, field=displacement, 1=0.01
"""

    _, (FTruss,) = _run(tmp_path, deck(f"type=truss, area={A}", "TR2D2"), "F")
    _, (FBeam,) = _run(tmp_path, deck(f"type=beam, profile=generic, area={A}, I=1e-5", "BE2D2CR"), "F")
    np.testing.assert_allclose(FBeam.getLastResult(), FTruss.getLastResult(), rtol=1e-4)
