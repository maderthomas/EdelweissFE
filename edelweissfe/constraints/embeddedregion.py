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
"""
Embedded region constraint: perfect bond of embedded elements (e.g., reinforcement bars) to host continuum
elements, enforced by master-slave DOF elimination (multi-point constraint), like Abaqus' ``*EMBEDDED ELEMENT``.

Every node of the embedded elements is tied to the host element it lies in,

.. math::
    \\boldsymbol{u}_s = \\sum_a N_a(\\boldsymbol{\\xi}_s) \\, \\boldsymbol{u}_a ,

with the parametric coordinates :math:`\\boldsymbol{\\xi}_s` of the node in the host element, found once in the
reference configuration. The embedded elements are meshed independently of the host mesh; they keep their own
stiffness, which is added to the host's (the volume of the embedded material is not subtracted from the host).

For bond-slip instead of perfect bond, see the ``embeddedBond`` generator
(:mod:`edelweissfe.generators.embeddedbond`).

.. code-block:: edelweiss
    :caption: Example:

    *constraint, type=embeddedRegion, name=rebarsInConcrete, embeddedElSet=rebars, hostElSet=concrete
"""

from dataclasses import dataclass

from edelweissfe.constraints.base.multipointconstraintbase import (
    MultiPointConstraintBase,
)
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.utils.embedding import HostElementLocator
from edelweissfe.utils.schema import buildSchemaFromOptions, schemaField


@dataclass(frozen=True)
class EmbeddedRegionSchema:
    """The options this constraint accepts, owned by this module and never mutated from outside it."""

    embeddedElSet: str | None = schemaField(
        description="The element set of the embedded elements (e.g., truss elements); all their nodes are tied to "
        "the host elements.",
        dtype=str,
        default=None,
        required=True,
    )
    hostElSet: str | None = schemaField(
        description="The element set of the host (continuum) elements.",
        dtype=str,
        default=None,
        required=True,
    )
    field: str = schemaField(description="The field to be constrained.", dtype=str, default="displacement")
    allowUnembedded: bool = schemaField(
        description="Whether embedded nodes lying outside of all host elements are left free (with a warning) "
        "instead of raising an error, e.g., for bars sticking out of the host.",
        dtype=bool,
        default=False,
    )
    tolerance: float = schemaField(
        description="Tolerance of the parametric coordinates of an embedded node on the boundary of a host " "element.",
        dtype=float,
        default=1e-6,
    )


class Constraint(MultiPointConstraintBase):
    """Embedded region multi-point constraint enforced by DOF elimination.

    Parameters
    ----------
    name
        The name of the constraint.
    model
        The model tree.
    journal
        The journal.
    configuration
        The options this constraint accepts.
    """

    #: Option schema for this constraint, per OptionSchemaProvider.
    schema = EmbeddedRegionSchema

    def __init__(
        self,
        name: str,
        model: FEModel,
        journal: Journal = None,
        *,
        configuration: EmbeddedRegionSchema = EmbeddedRegionSchema(),
    ):
        self._name = name
        self._model = model
        self._field = configuration.field

        embeddedElements = list(model.elementSets[configuration.embeddedElSet])
        hostElements = list(model.elementSets[configuration.hostElSet])
        hostNodes = {n for el in hostElements for n in el.nodes}

        locator = HostElementLocator(hostElements, tolerance=configuration.tolerance)

        # (slaveNode, [(masterNode, weight), ...]); slave node first, see claimedSlaveNodes()
        self._records = []
        #: The embedded nodes outside of all host elements, left free.
        self.unembeddedNodes = []

        embeddedNodes = list(dict.fromkeys(n for el in embeddedElements for n in el.nodes))
        for node in embeddedNodes:
            if node in hostNodes:
                continue  # shared with the host: nothing to tie
            hostPoint = locator.locate(node.coordinates)
            if hostPoint is None:
                self.unembeddedNodes.append(node)
                continue
            N = hostPoint.shapeFunctions()
            self._records.append((node, [(m, w) for m, w in zip(hostPoint.element.nodes, N) if w != 0.0]))

        if self.unembeddedNodes:
            labels = [n.label for n in self.unembeddedNodes]
            message = f"{name}: {len(labels)} embedded node(s) lie outside of all host elements: {labels[:20]}"
            if not configuration.allowUnembedded:
                raise ValueError(message + " (use allowUnembedded=True to leave them free)")
            if journal:
                journal.message(message + "; they are left free", name)

        if journal:
            journal.message(f"{len(self._records)} node(s) embedded in {len(hostElements)} host element(s)", name)

    @classmethod
    def fromConstraintDefinition(
        cls, name: str, definition: dict, model: FEModel, journal: "Journal" = None
    ) -> "Constraint":
        """Build this constraint from a parsed ``*constraint`` definition."""
        configuration = buildSchemaFromOptions(cls.schema, definition)
        return cls(name, model, journal, configuration=configuration)

    def getMultiPointConstraints(self, dofManager) -> list:
        fieldVariableIndices = dofManager.idcsOfFieldVariablesInDofVector
        records = []
        field = self._field
        for slaveNode, masters in self._records:
            if field not in slaveNode.fields:
                continue
            slaveDofs = fieldVariableIndices[slaveNode.fields[field]]
            masterDofs = [(fieldVariableIndices[mNode.fields[field]], w) for mNode, w in masters]
            for component in range(len(slaveDofs)):
                records.append((slaveDofs[component], [(mDofs[component], w) for mDofs, w in masterDofs]))
        return records
