import warnings

import brian2.only as b2
import cleo
import cleo.light
import numpy as np
from attrs import define, field


@define(eq=False)
class GroupedLight(cleo.light.Light):
    """How to implement?
    - inheritance or composition? inheritance. it's simpler.

    Just overwriting a few methods
    """

    groups: list[list[int]] = field(factory=list)
    """n_groups-length list of lists of indices of light sources.
    
    Must assign every light source to exactly one group.
    """

    @groups.validator
    def _validate_groups(self, attr, value):
        assert sum(len(g) for g in value) == len(self.coords), value
        assert all(len(g) > 0 for g in value)
        assert np.all(
            sorted(i for g in value for i in g) == np.arange(len(self.coords))
        )

    @property
    def n_groups(self):
        return len(self.groups)

    def update(self, value) -> None:
        """Set the light power or irradiance. The units of the input value
        determine which.

        Parameters
        ----------
        value : Quantity
            The light irradiance (mW/mm^2) or power (mW). If a scalar, it will be
            broadcast to all sources. If an array, it must be the same shape
            as the number of groups.
        """
        value = self._preprocess_value(value)
        if np.any(value < 0):
            warnings.warn(f"{self.name}: negative light value clipped to 0")
            value[value < 0] = 0
        if self.max_value is not None:
            max_val = self._val_same_unit_as(self.max_value, value)
            value[value > max_val] = max_val

        assert cleo.light.light._is_irr(value), "only doing irradiance---1P opto"
        irr0_grouped = value

        # values are stored as irradiance with mW/mm2 stripped
        super(cleo.light.Light, self).update(irr0_grouped / (b2.mwatt / b2.mm2))
        irr0_ungrouped = np.zeros(len(self.coords))
        for i, group in enumerate(self.groups):
            irr0_ungrouped[group] = irr0_grouped[i]
        if self.source is not None:
            self.source.Irr0 = irr0_ungrouped
