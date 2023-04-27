from neuralpredictors.layers.readouts import (FullFactorized2d, FullGaussian2d,
                                              MultiReadoutSharedParametersBase)


class MultipleFullGaussian2d(MultiReadoutSharedParametersBase):
    _base_readout = FullGaussian2d


class MultipleFullFactorized2d(MultiReadoutSharedParametersBase):
    _base_readout = FullFactorized2d
