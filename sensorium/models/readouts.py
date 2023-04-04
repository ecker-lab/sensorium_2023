from neuralpredictors.layers.readouts import (
    MultiReadoutSharedParametersBase,
    FullGaussian2d,
    FullFactorized2d,
)


class MultipleFullGaussian2d(MultiReadoutSharedParametersBase):
    _base_readout = FullGaussian2d


class MultipleFullFactorized2d(MultiReadoutSharedParametersBase):
    _base_readout = FullFactorized2d
