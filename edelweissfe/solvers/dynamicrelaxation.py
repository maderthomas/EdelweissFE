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
"""A matrix-free quasi-static solver by dynamic relaxation (DR), after Sutcliffe, Coombs, Zhang, Duddu, Pretti and
Augarde, "A Matrix-Free, Scalable, and Robust Implicit Material Point Method", IJNME 127 (2026) e70384, Sec. 3.

Every increment is solved as NIST solves it -- the same model evaluation (:meth:`NIST.assembleResidualAndTangent`),
the same loads, constraints and multi-point-constraint condensation -- but the equilibrium
:math:`\\boldsymbol{R}(\\Delta\\boldsymbol{u}) = \\boldsymbol{f}_\\mathrm{ext} - \\boldsymbol{f}_\\mathrm{int} = 0` is
found by integrating the damped pseudo-dynamic system (paper eq. 29)

.. math::

    \\tilde{\\boldsymbol{M}} \\, \\ddot{\\boldsymbol{u}} + \\alpha \\tilde{\\boldsymbol{M}} \\, \\dot{\\boldsymbol{u}} =
    \\boldsymbol{R}(\\boldsymbol{u})

in a fictitious time with the explicit midpoint rule (eqs. 32/33),

.. math::

    \\boldsymbol{v}_{i+1/2} = \\frac{2 - \\alpha\\Delta\\mathcal{T}}{2 + \\alpha\\Delta\\mathcal{T}}
    \\boldsymbol{v}_{i-1/2} + \\frac{2\\Delta\\mathcal{T}}{2 + \\alpha\\Delta\\mathcal{T}}
    \\tilde{\\boldsymbol{M}}^{-1} \\boldsymbol{R}_i, \\qquad
    \\Delta\\boldsymbol{u}_{i+1} = \\Delta\\boldsymbol{u}_i + \\Delta\\mathcal{T} \\, \\boldsymbol{v}_{i+1/2},

from :math:`\\Delta\\boldsymbol{u}_0 = 0` (plus the prescribed Dirichlet increment) until
:math:`\\|\\boldsymbol{R}\\|_2 / \\|\\boldsymbol{f}_\\mathrm{ext}\\|_2 <` ``drTolerance`` (Algorithm 1, default
:math:`10^{-9}` as in the paper). No tangent is assembled into a global matrix, and no linear system is solved: the
entities' blocks of the tangent, which they compute anyway, are only summed in absolute value per row for the
fictitious mass (below), into a vector.

**Fictitious mass** (diagonal, Sec. 3.2.1), ``fictitiousMass``:

* ``gerschgorin`` (default; physics-agnostic): eq. 34, :math:`\\tilde M_{ii} = \\zeta \\, \\Delta\\mathcal{T}^2/4 \\,
  \\sum_j |K_{ij}|`, the Gerschgorin bound of the largest eigenvalue. The row sums are taken of the entities' blocks,
  :math:`\\sum_e \\sum_j |K^e_{ij}|` -- an upper bound of the row sum of the assembled matrix, so still a stable
  mass, and matrix free. ``massUpdate=iteration`` reassembles it in every iteration (the paper's *K updated*),
  ``increment`` once per increment (*K constant*). ``stabilityFactor`` :math:`\\zeta` (paper: 1).
* a provider object (constructor argument ``fictitiousMassProvider``), with a method
  ``fictitiousMass(solver, pseudoTimeStep) -> np.ndarray`` returning the diagonal (full size) for the elements, e.g.
  the P-wave estimator of the paper's eq. 35 (:class:`edelweissmeshfree.solvers.pwavemass.PWaveFictitiousMass`),
  which knows the material. The blocks of everything else (constraints, e.g. a ghost penalty, and loads with a
  tangent) still enter by their Gerschgorin row sums (scaled with ``stabilityFactor``): they are matrices the
  provider does not know of.

**Damping** (Sec. 3.2.2), ``damping``: ``critical`` (default): mass-proportional, :math:`\\alpha = 2 \\omega_0` from
the Rayleigh quotient of the change of the residual (eq. 40 with :math:`\\boldsymbol{R}` for :math:`\\boldsymbol{f}`,
as the paper suggests for loads that depend on the displacement), times ``dampingFactor`` (0.9 = the paper's
*under-damped*, eq. 52), zero if the quotient is negative, :math:`\\alpha\\Delta\\mathcal{T}` capped at
``maxDampingTimesPseudoTimeStep`` (1.9). ``kinetic``: no viscous damping; the velocity is reset whenever the kinetic
energy :math:`\\tfrac{1}{2}\\boldsymbol{v}^T\\tilde{\\boldsymbol{M}}\\boldsymbol{v}` decreases (eq. 41). The paper sets
:math:`\\boldsymbol{v}_i = 0`; here the iteration is restarted at :math:`\\boldsymbol{v}_i = 0` with the midpoint starter,
:math:`\\boldsymbol{v}_{i+1/2} = \\tfrac{\\Delta\\mathcal{T}}{2}\\tilde{\\boldsymbol{M}}^{-1}\\boldsymbol{R}_i`, which is
what the next update from :math:`\\boldsymbol{v}_i = 0` gives (no wasted model evaluation).

**Starter**: :math:`\\boldsymbol{v}_{-1/2} = -\\tfrac{\\Delta\\mathcal{T}}{2} \\tilde{\\boldsymbol{M}}^{-1}
\\boldsymbol{R}_0` with :math:`\\boldsymbol{R}_0` the out-of-balance force of the start state, i.e.
:math:`\\boldsymbol{v}_0 = 0` (the paper writes :math:`\\boldsymbol{f}^\\mathrm{ext}_n` in Algorithm 1, which is
:math:`\\boldsymbol{R}_0` for a body in equilibrium under the previous load with :math:`\\boldsymbol{f}^\\mathrm{int}`
taken at the start of the step; the residual is what drives the fictitious motion in general).

**Multi-point constraints** (Sec. 3.3, eqs. 42-47, e.g. AgMPM): with the extension operator :math:`T` of the
condensation (:attr:`~edelweissfe.numerics.mpctransformation.MultiPointConstraintTransformation.extensionMatrix`), the
iteration runs on the independent DOFs: :math:`\\boldsymbol{R}^a = T^T \\boldsymbol{R}`, :math:`\\Delta\\boldsymbol{u} = T
\\Delta\\boldsymbol{u}^\\mathrm{int}`, and the mass is ``mpcMass``: ``consistent`` (default) :math:`\\tilde{M}^a = T^T
\\tilde{M} T` (eq. 42; sparse, factorized once per mass assembly), or ``lumped``, its absolute row sum
:math:`\\mathrm{diag}(|T|^T \\tilde{M} |T| \\boldsymbol{1})`, which bounds :math:`\\tilde{M}^a` from above (diagonal
dominance) and therefore keeps the iteration stable. :math:`T^T\\tilde{M}T` is stable for :math:`T^T K T` whenever
:math:`\\tilde{M}` is for :math:`K`: :math:`x^T T^T K T x \\le \\lambda_\\mathrm{max}(\\tilde M^{-1} K) \\, x^T T^T \\tilde
M T x`.

**Dirichlet** DOFs take their prescribed increment from the start and do not move in the iteration; their residual
(the reaction) is excluded from the convergence check. The reference :math:`\\|\\boldsymbol{f}_\\mathrm{ext}\\|_2` is the
norm of the loads of the step at the end of the increment (all DOFs); without loads (displacement control) the norm of
the internal forces, reactions included, is taken instead. A residual below ``drAbsoluteTolerance`` (:math:`10^{-12}`,
in the units of the forces) is converged, too: in a state without any load (e.g. the zero increment that starts a
step) the ratio is round-off over round-off.

Every iteration evaluates the model once. ``drMaxIterations`` bounds an increment (a ``ReachedMaxIterations`` cuts it
back like a Newton failure); a non-finite residual, or one that grows beyond ``drDivergenceRatio`` times its first
value, is a ``DivergingSolution``. The iteration counter reported to the step is the number of DR iterations, so give
an adaptive step a ``criticalIter`` that fits.

The history of every attempted increment is kept in :attr:`DynamicRelaxation.history` (iterations, the relative
residual per iteration, damping and resets).
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu

import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.numerics.dofvector import DofVector
from edelweissfe.numerics.systemmatrixassembly import SystemMatrixBlocks
from edelweissfe.numerics.systemmatrixcontributor import countEntriesOfBlock
from edelweissfe.solvers.nonlinearimplicitstatic import NIST
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.exceptions import DivergingSolution, ReachedMaxIterations
from edelweissfe.utils.schema import schemaField


@dataclass(frozen=True)
class DynamicRelaxationSchema(NIST.schema):
    """NIST's options (of which the dynamic relaxation uses none of the Newton ones), plus those of the dynamic
    relaxation."""

    drTolerance: float | None = schemaField(
        description="Convergence: ||R||_2 / ||f_ext||_2 below it (paper: 1e-9).", dtype=float, default=1e-9
    )
    drAbsoluteTolerance: float | None = schemaField(
        description="Convergence also if ||R||_2 is below it (a state without load, where the ratio is 0/0).",
        dtype=float,
        default=1e-12,
    )
    drMaxIterations: int | None = schemaField(
        description="The maximum number of dynamic relaxation iterations of an increment.", dtype=int, default=200000
    )
    drDivergenceRatio: float | None = schemaField(
        description="An increment diverges once ||R|| exceeds this multiple of its first value.",
        dtype=float,
        default=1e8,
    )
    fictitiousMass: str | None = schemaField(
        description="gerschgorin (eq. 34, row sums of the entities' tangent blocks) or provider (a "
        "fictitiousMassProvider object given to the constructor, e.g. the P-wave estimator of eq. 35).",
        dtype=str,
        default="gerschgorin",
    )
    stabilityFactor: float | None = schemaField(
        description="zeta of eq. 34 (paper: 1), for the Gerschgorin row sums.", dtype=float, default=1.0
    )
    massUpdate: str | None = schemaField(
        description="iteration (reassemble the fictitious mass in every iteration, 'K updated') or increment (once "
        "per increment, 'K constant').",
        dtype=str,
        default="iteration",
    )
    damping: str | None = schemaField(
        description="critical (eq. 39/40, Rayleigh quotient) or kinetic (velocity reset at a peak of the kinetic "
        "energy, eq. 41).",
        dtype=str,
        default="critical",
    )
    dampingFactor: float | None = schemaField(
        description="Factor on the critical damping (0.9: the paper's under-damped variant, eq. 52).",
        dtype=float,
        default=1.0,
    )
    maxDampingTimesPseudoTimeStep: float | None = schemaField(
        description="Cap of alpha * dT of the midpoint rule (paper: 1.9).", dtype=float, default=1.9
    )
    mpcMass: str | None = schemaField(
        description="The mass on the independent DOFs of multi-point constraints: consistent (T^T M T, eq. 42) or "
        "lumped (its absolute row sum).",
        dtype=str,
        default="consistent",
    )
    pseudoTimeStep: float | None = schemaField(
        description="The fictitious time step dT (has no effect on the iteration, paper: 1).", dtype=float, default=1.0
    )
    drReportEvery: int | None = schemaField(
        description="Journal the relative residual every this many iterations (0: only at the end).",
        dtype=int,
        default=0,
    )


class RowSumBlocks(SystemMatrixBlocks):
    """The system matrix blocks of a matrix-free solver: the blocks are never assembled into a matrix; only their
    absolute row sums, :math:`\\sum_j |K^e_{ij}|`, are summed per DOF of the discrete system, into
    :attr:`rowSumsOfElements` or :attr:`rowSumsOfOthers`, whichever :attr:`target` names when the block is handed
    out (``None``: nowhere, the block goes into a scratch array). Dense blocks and blocks of an entity's own layout
    (``localEntriesOfBlock``) alike.

    The blocks of an evaluation lie one after the other in one buffer, in the order they are handed out, which is the
    same in every evaluation (the layout is kept, and redone from the first entity that differs); :meth:`finalize`
    sums them per row, vectorised over the dense blocks of equal size. An entity may get several blocks in one
    evaluation (e.g. a cell, and a body load on it).

    Parameters
    ----------
    locations
        The locations of the entities in the discrete system.
    nDof
        The size of the discrete system.
    """

    def __init__(self, locations, nDof: int):
        self.locations = locations
        self.nDof = nDof
        self._rowSums = np.zeros((2, nDof))
        self._views = (self._rowSums[0], self._rowSums[1])
        #: Where the row sums of the blocks handed out next go: :attr:`rowSumsOfElements`, :attr:`rowSumsOfOthers`, or
        #: None.
        self.target = None
        self._buffer = np.zeros(0)
        self._scratch = np.zeros(0)
        self._layout = []
        self._cursor = 0
        self._groups = None

    @property
    def rowSumsOfElements(self) -> np.ndarray:
        """The row sums of the blocks of the elements (and the cells and particles of a meshfree model)."""
        return self._views[0]

    @property
    def rowSumsOfOthers(self) -> np.ndarray:
        """The row sums of all other blocks: loads, constraints."""
        return self._views[1]

    def beginAssembly(self):
        """Start a new evaluation."""
        self._cursor = 0

    def __getitem__(self, entity) -> np.ndarray:
        target = self.target
        kind = None if target is None else (0 if target is self._views[0] else 1)
        if kind is None:
            count = countEntriesOfBlock(entity)
            if count > self._scratch.shape[0]:
                self._scratch = np.zeros(count)
            values = self._scratch[:count]
            values[:] = 0.0
            return entity.shapeBlock(values)
        i = self._cursor
        layout = self._layout
        if i < len(layout) and layout[i][0] is entity and layout[i][3] == kind:
            _, offset, count, _ = layout[i]
        else:
            del layout[i:]
            offset = layout[-1][1] + layout[-1][2] if layout else 0
            count = countEntriesOfBlock(entity)
            layout.append((entity, offset, count, kind))
            self._groups = None
            if offset + count > self._buffer.shape[0]:
                grown = np.zeros(max(2 * self._buffer.shape[0], offset + count))
                grown[:offset] = self._buffer[:offset]
                self._buffer = grown
        self._cursor = i + 1
        values = self._buffer[offset : offset + count]
        values[:] = 0.0
        return entity.shapeBlock(values)

    def flush(self):
        """Nothing to commit: the blocks stay in the buffer until :meth:`finalize`."""

    def _buildGroups(self):
        dense, listed = {}, []
        for entity, offset, count, kind in self._layout:
            locations = np.asarray(self.locations[entity])
            if getattr(entity, "couplesAllItsDofs", True):
                dense.setdefault((kind, entity.nDof), []).append((offset, locations))
            else:
                listed.append((kind, offset, count, np.asarray(entity.localEntriesOfBlock()[0]), locations))
        groups = []
        for (kind, n), members in dense.items():
            offsets = np.array([o for o, _ in members])
            groups.append(
                (kind, n, offsets[:, None] + np.arange(n * n)[None, :], np.array([loc for _, loc in members]).ravel())
            )
        self._groups = (groups, listed)

    def finalize(self):
        """Sum the blocks of the evaluation just done per row, into :attr:`rowSumsOfElements` and
        :attr:`rowSumsOfOthers`."""
        if self._cursor < len(self._layout):
            del self._layout[self._cursor :]
            self._groups = None
        if self._groups is None:
            self._buildGroups()
        groups, listed = self._groups
        self._rowSums[:] = 0.0
        for kind, n, index, locations in groups:
            # column-major blocks: values[a + n b] = K[a, b]; as (k, b, a), the row sums are the sums over b
            sums = np.abs(self._buffer[index]).reshape(-1, n, n).sum(axis=1).ravel()
            self._rowSums[kind] += np.bincount(locations, weights=sums, minlength=self.nDof)
        for kind, offset, count, rows, locations in listed:
            sums = np.bincount(rows, weights=np.abs(self._buffer[offset : offset + count]), minlength=len(locations))
            np.add.at(self._rowSums[kind], locations, sums)


class DynamicRelaxation(NIST):
    """The matrix-free quasi-static solver by dynamic relaxation of Sutcliffe et al. (IJNME 2026); see the module
    documentation.

    Parameters
    ----------
    jobInfo
        A dictionary containing the job information.
    journal
        The journal instance for logging.
    fictitiousMassProvider
        For ``fictitiousMass=provider``: an object whose ``fictitiousMass(solver, pseudoTimeStep)`` returns the
        diagonal fictitious mass of the elements, a full-size vector of the discrete system.
    """

    identification = "DynamicRelaxation"

    supportsMPC = True

    schema = DynamicRelaxationSchema

    SolverSpecificOptions = NIST.SolverSpecificOptions | {
        "drTolerance": 1e-9,
        "drAbsoluteTolerance": 1e-12,
        "drMaxIterations": 200000,
        "drDivergenceRatio": 1e8,
        "fictitiousMass": "gerschgorin",
        "stabilityFactor": 1.0,
        "massUpdate": "iteration",
        "damping": "critical",
        "dampingFactor": 1.0,
        "maxDampingTimesPseudoTimeStep": 1.9,
        "mpcMass": "consistent",
        "pseudoTimeStep": 1.0,
        "drReportEvery": 0,
    }

    def __init__(self, jobInfo, journal, fictitiousMassProvider=None, **kwargs):
        #: The provider of the fictitious mass of the elements (``fictitiousMass=provider``).
        self.fictitiousMassProvider = fictitiousMassProvider
        #: One record per attempted increment: see :meth:`solveIncrement`.
        self.history = []
        super().__init__(jobInfo, journal, **kwargs)

    # --- no linear solver, no system matrix -----------------------------------------------------------------------

    def selectLinearSolverForStep(self) -> None:
        """No linear solver: nothing is solved."""
        self.linSolver = None

    def onNewNumbering(self):
        """Build the vectors of the new discrete system and its multi-point-constraint condensation; no linear
        solver."""

        system = self.system
        self._U = system.locations.constructDofVector()
        self._P = system.locations.constructDofVector()
        self._dU = system.locations.constructDofVector()
        self.readStateFromModel(self._U)
        self.buildAndCheckMPCTransformation()

    def onNewLocations(self):
        """Rewrap the vectors, and make the blocks the entities write their tangents into: :class:`RowSumBlocks`,
        never assembled."""

        locations = self.system.locations
        self._U = self._U.onLocations(locations)
        self._dU = self._dU.onLocations(locations)
        self._P = self._P.onLocations(locations)
        self._K = RowSumBlocks(locations, len(self._U))

    def beginStep(self, step, model, fieldOutputController, outputmanagers):
        o = self.options
        for name, allowed in (
            ("fictitiousMass", ("gerschgorin", "provider")),
            ("massUpdate", ("iteration", "increment")),
            ("damping", ("critical", "kinetic")),
            ("mpcMass", ("consistent", "lumped")),
        ):
            if o[name] not in allowed:
                raise ValueError("Unknown {:} '{:}', use {:}".format(name, o[name], " or ".join(allowed)))
        if o["fictitiousMass"] == "provider" and self.fictitiousMassProvider is None:
            raise ValueError("fictitiousMass=provider needs a fictitiousMassProvider")
        if o["drTolerance"] <= 0.0 or o["pseudoTimeStep"] <= 0.0 or o["stabilityFactor"] <= 0.0:
            raise ValueError("drTolerance, pseudoTimeStep and stabilityFactor must be positive")
        super().beginStep(step, model, fieldOutputController, outputmanagers)

    # --- the model evaluation: NIST's, with the loads kept apart ----------------------------------------------------

    def assembleInternalForces(self, U_np, dU, P, K, F, timeStep):
        if isinstance(K, RowSumBlocks):
            K.target = K.rowSumsOfElements if self._needRowSums else None
        P, K, F = super().assembleInternalForces(U_np, dU, P, K, F, timeStep)
        if isinstance(K, RowSumBlocks):
            K.target = K.rowSumsOfOthers if self._needRowSums else None
        return P, K, F

    def assembleLoadsAtIncrementEnd(self, stepActions, U_np, dU, P, K, timeStep):
        """NIST's, and the external load is kept (the reference of the convergence check): the loads are summed first
        into the zeroed vector of the loads and constraints."""
        P, K = super().assembleLoadsAtIncrementEnd(stepActions, U_np, dU, P, K, timeStep)
        self._externalLoad = -np.asarray(P).copy()
        return P, K

    # --- the dynamic relaxation -------------------------------------------------------------------------------------

    def solveIncrement(
        self,
        U_n: DofVector,
        dU: DofVector,
        P: DofVector,
        K,
        stepActions: dict,
        model,
        timeStep: TimeStep,
        prevTimeStep: TimeStep,
        extrapolation: str,
        maxIter: int,
        maxGrowingIter: int,
    ):
        """Solve the increment by dynamic relaxation (Algorithms 1 and 2 of the paper). ``prevTimeStep``,
        ``extrapolation``, ``maxIter`` and ``maxGrowingIter`` of the Newton loop are not used: every increment starts
        from zero (the prescribed increment at the Dirichlet DOFs), and ``drMaxIterations`` bounds it.

        Returns
        -------
        tuple
            The solution, its increment, the internal forces, the number of iterations and an (empty) residual
            history per field, as NIST's.
        """

        o = self.options
        tol = o["drTolerance"]
        maxIterations = o["drMaxIterations"]
        dT = o["pseudoTimeStep"]
        massEveryIteration = o["massUpdate"] == "iteration"
        kinetic = o["damping"] == "kinetic"
        reportEvery = o["drReportEvery"]
        tic = perf_counter()

        system = self.system
        locations = system.locations
        R = locations.constructDofVector()
        F = locations.constructDofVector()
        FConstraints = locations.constructDofVector()
        PLoadsAndConstraints = locations.constructDofVector()
        U_np = locations.constructDofVector()
        nDof = len(R)

        dirichlets = list(stepActions["dirichlet"].values())
        with performancetiming.timeit("step actions at increment start"):
            self.locateConstrainedDofs(dirichlets)
            self.applyStepActionsAtIncrementStart(model, timeStep, stepActions)
        self.initializeIncrement(U_n, stepActions, model, timeStep)

        mpc = self.mpcTransformation
        fixed = np.zeros(nDof, dtype=bool)
        dU[:] = 0.0
        for dirichlet in dirichlets:
            fixed[dirichlet.constrainedDofIndices] = True
            dU[dirichlet.constrainedDofIndices] = dirichlet.getPrescribedIncrementOfConstrainedDofs(timeStep)
        independent = np.ones(nDof, dtype=bool)
        if mpc is not None:
            independent[mpc.slaveDofIndices] = False
            mpc.applySlaveKinematics(dU)
        free = np.flatnonzero(independent & ~fixed)
        T = mpc.extensionMatrix if mpc is not None else None
        if T is not None:
            TFree = T[:, free].tocsc()
            absT = abs(T)
            absTRowSums = np.asarray(absT.sum(axis=1)).ravel()

        record = dict(
            increment=timeStep.number,
            stepProgress=timeStep.stepProgress,
            nFree=int(len(free)),
            residuals=[],
            alphas=[],
            nResets=0,
            converged=False,
            reason="",
        )
        self.history.append(record)

        def evaluate():
            R_ = R
            P_, _, _ = self.assembleResidualAndTangent(
                U_n, dU, U_np, R_, P, PLoadsAndConstraints, F, FConstraints, K, stepActions, model, timeStep
            )
            if self._needRowSums:
                K.finalize()
            return P_

        def assembleMass():
            """The diagonal fictitious mass (full size), from the row sums of the evaluation just done."""
            scale = o["stabilityFactor"] * dT**2 / 4.0
            if o["fictitiousMass"] == "gerschgorin":
                m = scale * (K.rowSumsOfElements + K.rowSumsOfOthers)
            else:
                m = np.asarray(self.fictitiousMassProvider.fictitiousMass(self, dT), dtype=float).copy()
                m += scale * K.rowSumsOfOthers
            return m

        def massOperator(m):
            """The mass on the free DOFs: (apply, solve) closures."""
            if T is None:
                mf = m[free]
                mf = np.maximum(mf, 1e-14 * max(mf.max(initial=0.0), 1e-300))
                return (lambda x: mf * x), (lambda r: r / mf)
            if o["mpcMass"] == "lumped":
                mf = (absT.T @ (m * absTRowSums))[free]
                mf = np.maximum(mf, 1e-14 * max(mf.max(initial=0.0), 1e-300))
                return (lambda x: mf * x), (lambda r: r / mf)
            MA = (TFree.T @ diags(m) @ TFree).tocsc()
            floor = 1e-14 * max(abs(MA.diagonal()).max(initial=0.0), 1e-300)
            MA = MA + diags(np.full(MA.shape[0], floor))
            lu = splu(MA.tocsc())
            return (lambda x: MA @ x), lu.solve

        # --- iteration 0 ---
        self._needRowSums = True
        evaluate()
        P0 = np.asarray(P)
        mApply, mSolve = massOperator(assembleMass())
        self._needRowSums = massEveryIteration

        def residualOnFree():
            return np.asarray(R)[free]

        def reference():
            ref = float(np.linalg.norm(self._externalLoad))
            if ref == 0.0:
                ref = float(np.linalg.norm(P0))
            return ref if ref > 0.0 else 1.0

        r = residualOnFree()
        ref = reference()
        rho = float(np.linalg.norm(r)) / ref
        rho0 = max(rho, 1e-300)
        record["residuals"].append(rho)

        # midpoint starter: v_{-1/2} = -dT/2 M^-1 R_0, i.e. v_0 = 0
        v = -0.5 * dT * mSolve(r)
        rPrev = None
        kePrev = 0.0
        iteration = 0

        while True:
            if not np.isfinite(rho):
                record["reason"] = "non-finite residual"
                raise DivergingSolution("dynamic relaxation: non-finite residual in iteration {:}".format(iteration))
            if rho < tol or float(np.linalg.norm(r)) <= o["drAbsoluteTolerance"]:
                record["converged"] = True
                break
            if rho > o["drDivergenceRatio"] * rho0:
                record["reason"] = "diverging"
                raise DivergingSolution(
                    "dynamic relaxation: residual grew to {:.3e} x its first value".format(rho / rho0)
                )
            if iteration >= maxIterations:
                record["reason"] = "max iterations"
                raise ReachedMaxIterations(
                    "dynamic relaxation: {:} iterations, ||R||/||f_ext|| = {:.3e}".format(iteration, rho)
                )
            if reportEvery and iteration % reportEvery == 0:
                self.journal.message(
                    "DR iteration {:7d}: ||R||/||f_ext|| = {:.3e}".format(iteration, rho), self.identification, 2
                )

            # damping
            if kinetic or rPrev is None:
                alpha = 0.0
            else:
                Mv = mApply(v)
                vMv = float(v @ Mv)
                num = float(v @ (rPrev - r))  # v^T (f_int,i - f_int,i-1)
                alpha = 0.0
                if vMv > 0.0 and num > 0.0:
                    alpha = o["dampingFactor"] * 2.0 * np.sqrt(num / (dT * vMv))
                    alpha = min(alpha, o["maxDampingTimesPseudoTimeStep"] / dT)
            record["alphas"].append(alpha)

            # velocity, eq. 32 / 44
            a = mSolve(r)
            c = 2.0 + alpha * dT
            v = (2.0 - alpha * dT) / c * v + 2.0 * dT / c * a

            if kinetic:
                ke = 0.5 * float(v @ mApply(v))
                if ke < kePrev:
                    # a peak of the kinetic energy has been passed: restart from rest, v_i = 0
                    v = 0.5 * dT * a
                    ke = 0.5 * float(v @ mApply(v))
                    record["nResets"] += 1
                kePrev = ke

            # displacement, eq. 33 / 45-46
            dU[free] += dT * v
            if mpc is not None:
                mpc.applySlaveKinematics(dU)

            rPrev = r
            iteration += 1
            evaluate()
            if massEveryIteration:
                mApply, mSolve = massOperator(assembleMass())
            r = residualOnFree()
            rho = float(np.linalg.norm(r)) / reference()
            record["residuals"].append(rho)

        record["iterations"] = iteration
        record["wall"] = perf_counter() - tic
        self.journal.message(
            "dynamic relaxation: {:} iterations, ||R||/||f_ext|| = {:.3e} ({:.1f} s{:})".format(
                iteration,
                rho,
                record["wall"],
                ", {:} kinetic resets".format(record["nResets"]) if kinetic else "",
            ),
            self.identification,
            1,
        )
        self.finalizeIncrement(model)
        return U_np, dU, P, iteration, {}

    _needRowSums = True
    _externalLoad = None
