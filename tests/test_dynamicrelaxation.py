#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         _____ _____
# | ____|__| | ___| |_      _____(_)___ ___|  ___| ____|
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |_  |  _|
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \  _| | |___
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|   |_____|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2017 - today
#
#  Matthias Neuner matthias.neuner@uibk.ac.at
#
#  This file is part of EdelweissFE.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------
"""The matrix-free dynamic relaxation solver (``DynamicRelaxation``, Sutcliffe et al., IJNME 2026).

A bar of CPE4 elements with Poisson's ratio 0 and all vertical DOFs fixed is a 1D bar: under an end load its
displacement is linear, under a uniform body load (consistent nodal forces) quadratic, and in both cases the nodal
values of linear elements are exact. The bar is solved with and without a multi-point constraint (an interior node
slaved to its neighbours), against the analytical solution and against NIST, for the damping and mass variants; and a
plastic plate given by an input file is solved by DR and by NIST.
"""

import numpy as np
import pytest

from edelweissfe.config.configurator import loadConfiguration
from edelweissfe.config.elementlibrary import getElementClass
from edelweissfe.config.materiallibrary import getMaterialClass
from edelweissfe.constraints.base.multipointconstraintbase import MultiPointConstraintBase
from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.sections.plane import PlaneSectionSchema, Section
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.solvers.dynamicrelaxation import DynamicRelaxation, RowSumBlocks
from edelweissfe.solvers.nonlinearimplicitstatic import NIST
from edelweissfe.stepactions.dirichlet import StepAction as DirichletStepAction
from edelweissfe.stepactions.nodeforces import StepAction as NodeForcesStepAction
from edelweissfe.steps.adaptivestep import AdaptiveStep
from edelweissfe.steps.stepmanager import StepActionCollection
from edelweissfe.utils.fieldoutput import FieldOutputController
from edelweissfe.utils.inputfileparser import parseInputFile

E = 100.0
L = 8.0
N = 8  # elements along the bar, of size 1 x 1


class SlaveToNeighbours(MultiPointConstraintBase):
    """u_x of the bottom node at x = xs is the mean of u_x of its two neighbours on the bottom row."""

    def __init__(self, name, model, xs):
        self.name = name
        nodes = {tuple(n.coordinates): n for n in model.nodes.values()}
        self.slave = nodes[(xs, 0.0)]
        self.masters = [nodes[(xs - 1.0, 0.0)], nodes[(xs + 1.0, 0.0)]]
        self._records = [(self.slave,)]

    def getMultiPointConstraints(self, numbering):
        idx = numbering.idcsOfFieldVariablesInDofVector
        s = idx[self.slave.fields["displacement"]][0]
        return [(s, [(idx[m.fields["displacement"]][0], 0.5) for m in self.masters])]

    @property
    def nodes(self):
        return [self.slave] + self.masters

    @property
    def fields(self):
        return [["displacement"]] * 3


def bar(load: str, solverClass, mpc: bool = False, nIncrements: int = 1, **options):
    """The bar under an end load F = 1 ("end") or a body load q = 1 per unit length ("body"); returns the nodal u_x of
    the bottom row and the solver."""

    nodes, label = {}, 1
    for j in range(2):
        for i in range(N + 1):
            nodes[label] = Node(label, np.array([float(i), float(j)]))
            label += 1
    ElementClass = getElementClass("CPE4", "edelweiss")
    elements = {}
    for i in range(N):
        el = ElementClass("CPE4", i + 1)
        el.setNodes([nodes[i + 1], nodes[i + 2], nodes[N + i + 3], nodes[N + i + 2]])
        elements[i + 1] = el
    model = FEModel(2)
    model.nodes.update(nodes)
    model.elements.update(elements)
    allNodes = list(nodes.values())
    model.elementSets["all"] = ElementSet("all", list(elements.values()))
    model.nodeSets["all"] = NodeSet("all", allNodes)
    model.nodeSets["left"] = NodeSet("left", [n for n in allNodes if n.coordinates[0] == 0.0])
    model.nodeSets["right"] = NodeSet("right", [n for n in allNodes if n.coordinates[0] == L])
    model.nodeSets["inner"] = NodeSet("inner", [n for n in allNodes if 0.0 < n.coordinates[0] < L])
    material = getMaterialClass("linearelastic", "edelweiss")(np.array([E, 0.0]))
    model.materials["linearelastic"] = material
    section = Section(
        "section1", model, material, [model.elementSets["all"]], configuration=PlaneSectionSchema(thickness=1.0)
    )
    model.sections["section1"] = section
    for el in elements.values():
        section.assignSectionPropertiesToElement(el)
    if mpc:
        model.multiPointConstraints["slave"] = SlaveToNeighbours("slave", model, 3.0)

    journal = Journal(verbose=False)
    model.prepareYourself(journal)
    model.advanceToTime(0.0)
    jobInfo = loadConfiguration(dict())
    model._linkFieldVariableObjects(model.nodeSets["all"])
    fieldOutputController = FieldOutputController(model, journal)
    model.fieldOutputController = fieldOutputController
    fieldOutputController.initializeJob()

    actions = StepActionCollection()
    actions["dirichlet"]["left"] = DirichletStepAction(
        "left", model.nodeSets["left"], "displacement", {0: 0.0}, model, journal
    )
    actions["dirichlet"]["y"] = DirichletStepAction(
        "y", model.nodeSets["all"], "displacement", {1: 0.0}, model, journal
    )
    if load == "end":
        actions["nodeforces"]["end"] = NodeForcesStepAction(
            "end", model.nodeSets["right"], "displacement", np.array([0.5, 0.0]), model, journal
        )
    else:  # consistent nodal forces of q = 1 per unit length, a quarter per node of an element
        actions["nodeforces"]["inner"] = NodeForcesStepAction(
            "inner", model.nodeSets["inner"], "displacement", np.array([0.5, 0.0]), model, journal
        )
        actions["nodeforces"]["end"] = NodeForcesStepAction(
            "end", model.nodeSets["right"], "displacement", np.array([0.25, 0.0]), model, journal
        )
    solver = solverClass(jobInfo, journal, **options)
    step = AdaptiveStep(
        0, model, fieldOutputController, journal, jobInfo, solver, [], actions,
        stepLength=1.0, startInc=1.0 / nIncrements, maxInc=1.0 / nIncrements, maxNumInc=10 * nIncrements,
    )  # fmt: skip
    step.solve()
    fieldOutputController.finalizeJob()
    bottom = sorted((n for n in allNodes if n.coordinates[1] == 0.0), key=lambda n: n.coordinates[0])
    U = model.nodeFields["displacement"]["U"]
    idx = [model.nodeFields["displacement"].nodes.index(n) for n in bottom]
    ux = np.array([U[i][0] for i in idx])
    return ux, solver


def analytical(load):
    x = np.arange(N + 1, dtype=float)
    return x / E if load == "end" else (L * x - 0.5 * x**2) / E


@pytest.mark.parametrize("load", ["end", "body"])
def test_the_bar_matches_the_analytical_solution(load):
    ux, solver = bar(load, DynamicRelaxation)
    np.testing.assert_allclose(ux, analytical(load), rtol=0.0, atol=1e-7 * np.abs(analytical(load)).max())
    record = solver.history[-1]
    assert record["converged"] and record["residuals"][-1] < 1e-9
    assert record["iterations"] > 10  # it is an iteration, not a direct solve


@pytest.mark.parametrize(
    "options",
    [
        dict(damping="kinetic", stabilityFactor=2.0),
        dict(dampingFactor=0.9),
        dict(massUpdate="increment"),
        dict(pseudoTimeStep=0.1),
    ],
)
def test_the_damping_and_mass_variants_reach_the_same_solution(options):
    ux, solver = bar("body", DynamicRelaxation, **options)
    np.testing.assert_allclose(ux, analytical("body"), rtol=0.0, atol=1e-7 * analytical("body").max())
    if options.get("damping") == "kinetic":
        assert solver.history[-1]["nResets"] > 0
        assert all(a == 0.0 for a in solver.history[-1]["alphas"])


def test_the_pseudo_time_step_does_not_change_the_iteration():
    # eq. 34: the mass scales with dT^2, so dT has no effect on the iterates
    _, a = bar("body", DynamicRelaxation)
    _, b = bar("body", DynamicRelaxation, pseudoTimeStep=0.1)
    np.testing.assert_allclose(a.history[-1]["residuals"], b.history[-1]["residuals"], rtol=1e-6)


def test_a_multi_point_constraint_with_a_linear_field_is_exact():
    ux, _ = bar("end", DynamicRelaxation, mpc=True)
    np.testing.assert_allclose(ux, analytical("end"), rtol=0.0, atol=1e-7 * analytical("end").max())


@pytest.mark.parametrize("mpcMass", ["consistent", "lumped"])
def test_a_multi_point_constraint_gives_the_solution_of_nist(mpcMass):
    uxNIST, _ = bar("body", NIST, mpc=True)
    ux, _ = bar("body", DynamicRelaxation, mpc=True, mpcMass=mpcMass)
    # the constraint changes the solution (the quadratic field is not linear across the slave) ...
    assert abs(uxNIST[3] - analytical("body")[3]) > 1e-3 * analytical("body").max()
    # ... and DR finds the constrained one; the slave is the mean of its masters
    np.testing.assert_allclose(ux, uxNIST, rtol=0.0, atol=1e-7 * uxNIST.max())
    assert ux[3] == pytest.approx(0.5 * (ux[2] + ux[4]), abs=1e-14)


def test_load_steps():
    ux, solver = bar("body", DynamicRelaxation, nIncrements=4)
    np.testing.assert_allclose(ux, analytical("body"), rtol=0.0, atol=1e-7 * analytical("body").max())
    assert len([r for r in solver.history if r["converged"] and r["stepProgress"] > 0.0]) == 4


def test_unknown_options_are_refused():
    for options in (dict(damping="viscous"), dict(fictitiousMass="pwave"), dict(mpcMass="full")):
        with pytest.raises(ValueError, match="Unknown"):
            bar("end", DynamicRelaxation, **options)
    with pytest.raises(ValueError, match="fictitiousMassProvider"):
        bar("end", DynamicRelaxation, fictitiousMass="provider")


def test_a_fictitious_mass_provider_is_used():
    class Provider:
        calls = 0

        def fictitiousMass(self, solver, dT):
            Provider.calls += 1
            # the Gerschgorin bound of a bar of E = 100, h = 1: row sums <= 4 E per DOF, times a safety factor
            return np.full(len(solver._U), 2.0 * dT**2 / 4.0 * 4.0 * E)

    provider = Provider()
    ux, _ = bar("body", DynamicRelaxation, fictitiousMass="provider", fictitiousMassProvider=provider)
    np.testing.assert_allclose(ux, analytical("body"), rtol=0.0, atol=1e-7 * analytical("body").max())
    assert Provider.calls > 1


def test_the_row_sums_of_dense_and_listed_blocks():
    class Dense:
        nDof = 2
        couplesAllItsDofs = True

        def shapeBlock(self, values):
            return values.reshape((2, 2), order="F")

    class Listed:
        nDof = 3
        couplesAllItsDofs = False

        def localEntriesOfBlock(self):
            return np.array([0, 0, 2], dtype=np.intc), np.array([0, 2, 1], dtype=np.intc)

        def shapeBlock(self, values):
            return values

    dense, listed = Dense(), Listed()
    locations = {dense: np.array([0, 1]), listed: np.array([1, 2, 3])}
    K = RowSumBlocks(locations, 4)
    K.beginAssembly()
    K.target = K.rowSumsOfElements
    K[dense][:] = [[1.0, -2.0], [3.0, 4.0]]
    K.flush()
    K.target = K.rowSumsOfOthers
    K[listed][:] = [-1.0, 5.0, -7.0]
    K.flush()
    np.testing.assert_allclose(K.rowSumsOfElements, [3.0, 7.0, 0.0, 0.0])
    np.testing.assert_allclose(K.rowSumsOfOthers, [0.0, 6.0, 0.0, 7.0])


def plateDeck(solver: str, options: str = "") -> str:
    """The plastic plate of the backtracking tests (von Mises with hardening, stretched beyond yield)."""
    return f"""
*material, name=VonMises, id=plasticity, provider=edelweiss
210000, 0.3, 550, 1000, 200, 1400

*section, name=section1, material=plasticity, type=plane, thickness=1
all

*job, name=job, domain=2d
*solver, solver={solver}, name=theSolver
{options}

*modelGenerator, generator=planeRectQuad, name=gen
l=10
h=10
nX=2
nY=2
elType=CPE4
elProvider=edelweiss

*step, solver=theSolver
stepLength=1, startInc=0.25, maxInc=0.25, minInc=1e-2, maxNumInc=100, maxIter=50, criticalIter=100000
>>dirichlet, name=left, nSet=gen_left, field=displacement, 1=0
>>dirichlet, name=bottom, nSet=gen_leftBottom, field=displacement, 2=0
>>dirichlet, name=right, nSet=gen_right, field=displacement, 1=0.08
"""


def test_a_plastic_plate_gives_the_solution_of_nist(tmp_path):
    def run(text):
        path = tmp_path / "test.inp"
        path.write_text(text)
        model, _ = finiteElementSimulation(parseInputFile(str(path)), verbose=False, suppressPlots=True)
        return model

    reference = run(plateDeck("NIST"))
    relaxed = run(plateDeck("DR"))
    assert relaxed.time == pytest.approx(1.0)
    U = relaxed.nodeFields["displacement"]["U"]
    UReference = reference.nodeFields["displacement"]["U"]
    np.testing.assert_allclose(U, UReference, rtol=0.0, atol=1e-6 * np.max(np.abs(UReference)))
