"""The ``tangentOnly`` option of the directional spring penalty: stiffness in the tangent only (no force), a numerical
stabilization that must not change the solution."""

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


def _run(tmp_path, deck: str):
    path = tmp_path / "test.inp"
    path.write_text(deck)
    model, fo = finiteElementSimulation(parseInputFile(str(path)), suppressPlots=True)
    return model, fo


def _plate(spring: str) -> str:
    return f"""*material, name=linearelastic, id=mat, provider=marmot
1000.0, 0.3

*modelGenerator, generator=planeRectQuad, name=gen
l=1
h=1
nX=1
nY=1
elType=CPE4

*section, name=sec, thickness=1.0, material=mat, type=plane
gen_all

{spring}
*job, name=job, domain=2d
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perNode, name=U, nSet=gen_right, field=displacement, result=U, saveHistory=True,
>>perNode, name=RF, nSet=gen_right, field=displacement, result=P, f(x)='sum(x[:,0])', saveHistory=True,

*step, solver=theSolver
maxInc=0.5, minInc=1e-8, maxNumInc=100, maxIter=20, stepLength=1
>>dirichlet, name=left, nSet=gen_left, field=displacement, 1=0, 2=0
>>dirichlet, name=pull, nSet=gen_right, field=displacement, 1=0.01
"""


def _spring(tangentOnly: bool, penalty: float) -> str:
    return f"""*constraint, type=directionalspringpenalty, name=s
nSet=gen_right
field=displacement
component=1
penalty={penalty}
tangentOnly={tangentOnly}
"""


@pytest.mark.skipif(not _marmotHas("CPE4"), reason="Marmot is built without CPE4")
def test_tangent_only_spring_does_not_change_the_solution(tmp_path):
    _, foFree = _run(tmp_path, _plate(""))
    # the stabilization must be small compared with the stiffness of the structure (modified Newton: convergence
    # rate ~ penalty / (penalty + stiffness)); a real spring of the same size restrains the lateral contraction
    _, foTangent = _run(tmp_path, _plate(_spring(True, 10.0)))
    _, foSpring = _run(tmp_path, _plate(_spring(False, 1e4)))
    UFree = np.asarray(foFree.fieldOutputs["U"].getLastResult())
    UTangent = np.asarray(foTangent.fieldOutputs["U"].getLastResult())
    USpring = np.asarray(foSpring.fieldOutputs["U"].getLastResult())
    RFree = float(foFree.fieldOutputs["RF"].getLastResult())
    RTangent = float(foTangent.fieldOutputs["RF"].getLastResult())
    # the free plate contracts laterally at its right face; a real spring restrains it, a tangent-only one does not
    assert np.abs(UFree[:, 1]).max() > 1e-4
    assert np.allclose(UTangent[:, 0], 0.01, rtol=0, atol=1e-14)  # the step was completed
    assert np.allclose(UTangent, UFree, rtol=0, atol=1e-9)
    assert RTangent == pytest.approx(RFree, rel=1e-7)  # (modified Newton: equal to the tolerance)
    assert np.abs(USpring[:, 1]).max() < 0.1 * np.abs(UFree[:, 1]).max()


@pytest.mark.skipif(not _marmotHas("TR2D2"), reason="Marmot is built without the TrussElement module")
def test_tangent_only_spring_stabilizes_a_free_transverse_dof(tmp_path):
    """A truss has no transverse stiffness: pulled along its axis with the transverse dof of its end node free, the
    system is singular; the tangent-only spring makes it solvable, with the exact axial solution and no transverse
    displacement."""
    deck = """*material, name=linearelastic, id=steel, provider=marmot
200000.0, 0.3
*node
1, 0.0, 0.0
2, 2.0, 0.0
*element, type=TR2D2
1, 1, 2
*elSet, elSet=bar
1
*nSet, nSet=n1
1
*nSet, nSet=n2
2
*section, name=sec, area=3.0, material=steel, type=truss
bar
*constraint, type=directionalspringpenalty, name=s
nSet=n2
field=displacement
component=1
penalty=1.0
tangentOnly=True
*job, name=job, domain=2d
*solver, solver=NIST, name=theSolver
*fieldOutput
>>perNode, name=U, nSet=n2, field=displacement, result=U
>>perNode, name=RF, nSet=n1, field=displacement, result=P, f(x)='sum(x[:,0])'
*step, solver=theSolver
maxInc=1, minInc=1e-8, maxNumInc=10, maxIter=10, stepLength=1
>>dirichlet, name=fix, nSet=n1, field=displacement, 1=0, 2=0
>>dirichlet, name=pull, nSet=n2, field=displacement, 1=0.004
"""
    _, fo = _run(tmp_path, deck)
    U = np.asarray(fo.fieldOutputs["U"].getLastResult()).ravel()
    assert U[1] == pytest.approx(0.0, abs=1e-14)
    assert float(fo.fieldOutputs["RF"].getLastResult()) == pytest.approx(-200000.0 * 3.0 * 0.004 / 2.0, rel=1e-10)
