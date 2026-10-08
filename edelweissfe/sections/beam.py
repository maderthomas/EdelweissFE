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
A section for beam elements, e.g., the Marmot Euler-Bernoulli beams ``B23`` (2D) and ``B33`` (3D), assigning the
cross section and the fiber layout of the section integration.

The cross section is given either by its profile dimensions (``profile=rectangle`` with ``b`` and ``h``,
``profile=circle`` with ``d``), or by its section constants (``profile=generic`` with ``area``, ``I`` (2D) or
``Iy``, ``Iz``, ``J`` (3D)); constants given explicitly override those computed from the dimensions. The local
:math:`y` axis of the section lies in the plane of a 2D beam; in 3D it is the part of the orientation vector
``(n1x, n1y, n1z)`` normal to the beam axis, and :math:`z = x \\times y`. ``h`` is the height along :math:`y`, ``b``
the width along :math:`z` (2D: out of plane), :math:`I_z = \\int y^2 \\mathrm{d}A` (bending in the
:math:`x`-:math:`y` plane, the in-plane bending of a 2D beam), :math:`I_y = \\int z^2 \\mathrm{d}A`.

The section is integrated with fibers of the beam's material (any Marmot hypoelastic material), which reproduce
the section constants exactly: ``generic`` uses the fewest fibers (2 in 2D, 4 in 3D; an idealized sandwich section
for inelastic materials), ``rectangle`` a grid of ``nFibers`` (x ``nFibers`` in 3D) fibers and ``circle`` rings and
sectors (see the Marmot documentation of the beam element).

Without an orientation vector in 3D, :math:`(0, 0, 1)` is used, or :math:`(0, 1, 0)` for beams (almost) parallel
to the global :math:`z` axis.

.. code-block:: edelweiss
    :caption: Example:

    *material, name=VonMises, id=steel, provider=marmot
        200000, 0.3, 500, 1000, 0, 1, 7.85e-9

    *section, name=pileSection, type=beam, profile=circle, d=300, nFibers=8, material=steel
        piles
"""

from dataclasses import dataclass

import numpy as np

from edelweissfe.sections.base.sectionbase import MaterialParameterFromFieldSchema
from edelweissfe.sections.base.sectionbase import Section as SectionBase
from edelweissfe.sections.base.sectionbase import WriteMaterialPropertiesToFileSchema
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.utils.schema import datalineField, schemaField, subKeywordField

#: The profile codes of the Marmot beam elements.
PROFILE_CODES = {"generic": 0, "rectangle": 1, "circle": 2}


def rectangleTorsionConstant(b: float, h: float) -> float:
    """The torsion constant of a solid rectangle (Roark's approximation, error below 0.5 %)."""
    a, c = max(b, h), min(b, h)
    return a * c**3 * (1.0 / 3.0 - 0.21 * c / a * (1.0 - c**4 / (12.0 * a**4)))


@dataclass(frozen=True)
class BeamSectionSchema:
    """The options this section accepts, owned by this module and never mutated from outside it."""

    profile: str = schemaField(
        description="The cross section profile: 'generic' (section constants only), 'rectangle' or 'circle'.",
        dtype=str,
        default="generic",
    )
    b: float | None = schemaField(description="rectangle: width along the local z axis", dtype=float, default=None)
    h: float | None = schemaField(description="rectangle: height along the local y axis", dtype=float, default=None)
    d: float | None = schemaField(description="circle: diameter", dtype=float, default=None)
    area: float | None = schemaField(description="cross section area", dtype=float, default=None)
    I: float | None = schemaField(  # noqa: E741
        description="2D: second moment of area for bending in the plane (synonym of Iz)", dtype=float, default=None
    )
    Iy: float | None = schemaField(description="3D: second moment of area int z^2 dA", dtype=float, default=None)
    Iz: float | None = schemaField(description="second moment of area int y^2 dA", dtype=float, default=None)
    J: float | None = schemaField(description="3D: torsion constant", dtype=float, default=None)
    n1x: float | None = schemaField(
        description="3D: orientation vector of the local y axis, x", dtype=float, default=None
    )
    n1y: float | None = schemaField(
        description="3D: orientation vector of the local y axis, y", dtype=float, default=None
    )
    n1z: float | None = schemaField(
        description="3D: orientation vector of the local y axis, z", dtype=float, default=None
    )
    nFibers: int = schemaField(
        description="rectangle and circle: the number of fibers per direction", dtype=int, default=8
    )
    materialParameterFromField: tuple[MaterialParameterFromFieldSchema, ...] = subKeywordField(
        description="use material properties given by an analytical field",
        schema=MaterialParameterFromFieldSchema,
    )
    writeMaterialPropertiesToFile: tuple[WriteMaterialPropertiesToFileSchema, ...] = subKeywordField(
        description="export material properties to file",
        schema=WriteMaterialPropertiesToFileSchema,
    )
    elementSets: str | None = datalineField(
        description="elementSets as comma separated list of element sets for this section", required=True
    )


def sectionConstants(configuration: BeamSectionSchema) -> dict:
    """The section constants ``area``, ``Iy``, ``Iz``, ``J`` of a beam section definition.

    Raises
    ------
    ValueError
        If the profile is unknown or a dimension or constant is missing.
    """
    profile = configuration.profile.lower()
    if profile not in PROFILE_CODES:
        raise ValueError(f"Unknown beam profile '{configuration.profile}', use one of {list(PROFILE_CODES)}.")

    constants = {"area": None, "Iy": None, "Iz": None, "J": None}
    if profile == "rectangle":
        b, h = configuration.b, configuration.h
        if b is None or h is None:
            raise ValueError("A rectangle beam section requires b and h.")
        constants = {"area": b * h, "Iy": h * b**3 / 12, "Iz": b * h**3 / 12, "J": rectangleTorsionConstant(b, h)}
    elif profile == "circle":
        d = configuration.d
        if d is None:
            raise ValueError("A circle beam section requires d.")
        constants = {"area": np.pi * d**2 / 4, "Iy": np.pi * d**4 / 64, "Iz": np.pi * d**4 / 64, "J": np.pi * d**4 / 32}

    given = {
        "area": configuration.area,
        "Iy": configuration.Iy,
        "Iz": configuration.Iz if configuration.Iz is not None else configuration.I,
        "J": configuration.J,
    }
    constants.update({k: v for k, v in given.items() if v is not None})
    return constants


class Section(SectionBase):
    """A section for beam elements, assigning the cross section constants, the profile and the orientation."""

    #: Option schema for this section, per OptionSchemaProvider.
    schema = BeamSectionSchema

    def __init__(
        self,
        name,
        model,
        material: dict,
        elementSets: list[ElementSet],
        *,
        configuration: BeamSectionSchema = BeamSectionSchema(),
    ):
        """Constructible standalone, with no parser involvement.

        Parameters
        ----------
        name
            The name of this section.
        model
            The model tree.
        material
            The (Marmot) material assigned to this section.
        elementSets
            The element sets this section is applied to.
        configuration
            The options this section accepts, see :class:`BeamSectionSchema`.
        """
        super().__init__(
            name,
            model,
            material,
            elementSets,
            configuration.materialParameterFromField,
            configuration.writeMaterialPropertiesToFile,
        )
        self.name = name
        self.nDim = model.domainSize
        self.profileCode = PROFILE_CODES.get(configuration.profile.lower())
        self.constants = sectionConstants(configuration)
        self.nFibers = configuration.nFibers

        required = ["area", "Iz"] if self.nDim == 2 else ["area", "Iy", "Iz", "J"]
        missing = [k for k in required if self.constants[k] is None]
        if missing:
            raise ValueError(f"Beam section {name}: missing section constants {missing}.")
        if any(self.constants[k] <= 0 for k in required):
            raise ValueError(f"Beam section {name}: the section constants must be positive.")

        n1 = (configuration.n1x, configuration.n1y, configuration.n1z)
        if any(c is not None for c in n1):
            self.orientation = np.array([c or 0.0 for c in n1], dtype=float)
            if np.linalg.norm(self.orientation) == 0:
                raise ValueError(f"Beam section {name}: the orientation vector must not be zero.")
        else:
            self.orientation = None

    def elementProperties(self, element) -> np.ndarray:
        """The Marmot element properties of a beam element of this section."""
        c = self.constants
        if self.nDim == 2:
            return np.array([c["area"], c["Iz"], self.profileCode, self.nFibers], dtype=float)

        n1 = self.orientation
        if n1 is None:
            coordinates = np.array([n.coordinates for n in element.nodes[:2]], dtype=float)
            axis = coordinates[1] - coordinates[0]
            axis /= np.linalg.norm(axis)
            n1 = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
        return np.array([c["area"], c["Iy"], c["Iz"], c["J"], *n1, self.profileCode, self.nFibers], dtype=float)

    def assignSectionPropertiesToElement(self, element, **kwargs):
        material = kwargs.get("material", self.material)

        if element.ensightType != "bar2" or "rotation" not in element.fields[0]:
            raise Exception(f"Beam section is incompatible with element {element.elNumber} ({element.elType}).")

        element.setProperties(self.elementProperties(element))
        element.initializeElement()
        if not isinstance(material, dict):
            raise Exception(f"Beam section {self.name}: beam elements require a Marmot material.")
        element.setMaterial(material["name"], material["properties"])
