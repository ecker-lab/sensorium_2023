class VideoFiringRateEncoder(nn.Module):
    def __init__(
        self,
        core,
        readout,
        *,
        shifter=None,
        modulator=None,
        elu_offset=0.0,
        nonlinearity_type="elu",
        nonlinearity_config=None,
        use_gru=False,
        gru_module=None,
        twoD_core=False,
    ):
        """
        An Encoder that wraps the core, readout and optionally a shifter amd modulator into one model.
        The output is one positive value that can be interpreted as a firing rate, for example for a Poisson distribution.
        Args:
            core (nn.Module): Core model. Refer to neuralpredictors.layers.cores
            readout (nn.ModuleDict): MultiReadout model. Refer to neuralpredictors.layers.readouts
            elu_offset (float): Offset value in the final elu non-linearity. Defaults to 0.
            shifter (optional[nn.ModuleDict]): Shifter network. Refer to neuralpredictors.layers.shifters. Defaults to None.
            modulator (optional[nn.ModuleDict]): Modulator network. Modulator networks are not implemented atm (24/06/2021). Defaults to None.
            nonlinearity (str): Non-linearity type to use. Defaults to 'elu'.
            nonlinearity_config (optional[dict]): Non-linearity configuration. Defaults to None.
            use_gru (boolean) : specifies if there is some module, which should be called between core and readouts 
            gru_module (nn.Module) : the module, which should be called between core and readouts 
            twoD_core (boolean) : specifies if the core is 2 or 3 dimensinal to change the input respectively
        """
        super().__init__()
        self.core = core
        self.readout = readout
        self.shifter = shifter
        self.modulator = modulator
        self.offset = elu_offset
        self.use_gru = use_gru
        self.gru_module = gru_module

        if nonlinearity_type != "elu" and not np.isclose(elu_offset, 0.0):
            warnings.warn("If `nonlinearity_type` is not 'elu', `elu_offset` will be ignored")
        if nonlinearity_type == "elu":
            self.nonlinearity_fn = nn.ELU()
        elif nonlinearity_type == "identity":
            self.nonlinearity_fn = nn.Identity()
        else:
            self.nonlinearity_fn = activations.__dict__[nonlinearity_type](
                **nonlinearity_config if nonlinearity_config else {}
            )
        self.nonlinearity_type = nonlinearity_type
        self.twoD_core = twoD_core

    def forward(
        self,
        inputs,
        *args,
        targets=None,
        data_key=None,
        behavior=None,
        pupil_center=None,
        trial_idx=None,
        shift=None,
        detach_core=False,
        **kwargs
    ):
        if self.twoD_core:
            batch_size = inputs.shape[0]
            time_points = inputs.shape[1]
            inputs = torch.transpose(inputs, 1, 2)
            inputs = inputs.reshape(((-1,) + inputs.size()[2:]))
            
        x = self.core(inputs)
        if detach_core:
            x = x.detach()
            
        if self.use_gru:
            if self.twoD_core:
                x = x.reshape(((batch_size, -1) + x.size()[1:]))
                x = torch.transpose(x, 1, 2)
            x = self.gru_module(x)
            if isinstance(x, list):
                x = x[-1]
                

        x = torch.transpose(x, 1, 2)
        batch_size = x.shape[0]
        time_points = x.shape[1] 
        
        if self.shifter:
            if pupil_center is None:
                raise ValueError("pupil_center is not given")
            pupil_center = pupil_center[:, :, -time_points:]
            pupil_center = torch.transpose(pupil_center, 1, 2)
            pupil_center = pupil_center.reshape(((-1,) + pupil_center.size()[2:]))
            shift = self.shifter[data_key](pupil_center, trial_idx)
            
        
        x = x.reshape(((-1,) + x.size()[2:]))
        x = self.readout(x, data_key=data_key, shift=shift, **kwargs)

        if self.modulator:
            if behavior is None:
                raise ValueError("behavior is not given")
            x = self.modulator[data_key](x, behavior=behavior)

        if self.nonlinearity_type == "elu":
            x = self.nonlinearity_fn(x + self.offset) + 1
        else:
            x = self.nonlinearity_fn(x)
            
        x = x.reshape(((batch_size, time_points) + x.size()[1:]))
        return x

    def regularizer(self, data_key=None, reduction="sum", average=None, detach_core=False):
        reg = self.core.regularizer().detach() if detach_core else self.core.regularizer()
        reg = reg + self.readout.regularizer(data_key=data_key, reduction=reduction, average=average)
        if self.shifter:
            reg += self.shifter.regularizer(data_key=data_key)
        if self.modulator:
            reg += self.modulator.regularizer(data_key=data_key)
        return reg
