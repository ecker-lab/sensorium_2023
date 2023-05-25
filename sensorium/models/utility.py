import copy

def get_io_dims(data_loader, deeplake_ds):
    """
    ## adopted from nnfabrik - https://github.com/sinzlab/nnfabrik/blob/master/nnfabrik/utility/nn_helpers.py
    Returns the shape of the dataset for each item within an entry returned by the `data_loader`
    The DataLoader object must return either a namedtuple, dictionary or a plain tuple.
    If `data_loader` entry is a namedtuple or a dictionary, a dictionary with the same keys as the
    namedtuple/dict item is returned, where values are the shape of the entry. Otherwise, a tuple of
    shape information is returned.

    Note that the first dimension is always the batch dim with size depending on the data_loader configuration.

    Args:
        data_loader (torch.DataLoader): is expected to be a pytorch Dataloader object returning
            either a namedtuple, dictionary, or a plain tuple.
    Returns:
        dict or tuple: If data_loader element is either namedtuple or dictionary, a ditionary
            of shape information, keyed for each entry of dataset is returned. Otherwise, a tuple
            of shape information is returned. The first dimension is always the batch dim
            with size depending on the data_loader configuration.
    """
    items = next(iter(data_loader))
    if hasattr(items, "_asdict"):  # if it's a named tuple
        items = items._asdict()
    if deeplake_ds:
        res = {}
        for k in items.keys():
            if k not in ['index', 'id', 'group', 'hash']:
                res[k] = items[k].shape
        return res
    else:
        if hasattr(items, "items"):  # if dict like
            return {k: v.shape for k, v in items.items()}
        else:
            return (v.shape for v in items)


def get_dims_for_loader_dict(dataloaders, deeplake_ds):
    """
    ## adopted from nnfabrik - https://github.com/sinzlab/nnfabrik/blob/master/nnfabrik/utility/nn_helpers.py
    Given a dictionary of DataLoaders, returns a dictionary with same keys as the
    input and shape information (as returned by `get_io_dims`) on each keyed DataLoader.

    Args:
        dataloaders (dict of DataLoader): Dictionary of dataloaders.

    Returns:
        dict: A dict containing the result of calling `get_io_dims` for each entry of the input dict
    """     
    return {k: get_io_dims(v, deeplake_ds) for k, v in dataloaders.items()}

def prepare_grid(grid_mean_predictor, dataloaders, deeplake_ds):
    """
    Utility function for using the neurons cortical coordinates
    to guide the readout locations in image space.
    Args:
        grid_mean_predictor (dict): config dictionary, for example:
          {'type': 'cortex',
           'input_dimensions': 2,
           'hidden_layers': 1,
           'hidden_features': 30,
           'final_tanh': True}
        dataloaders: a dictionary of dataloaders, one PyTorch DataLoader per session
            in the format {'data_key': dataloader object, .. }
    Returns:
        grid_mean_predictor (dict): config dictionary
        grid_mean_predictor_type (str): type of the information that is being used for
            the grid positition estimator
        source_grids (dict): a grid of points for each data_key
    """
    if grid_mean_predictor is None:
        grid_mean_predictor_type = None
        source_grids = None
    else:
        grid_mean_predictor = copy.deepcopy(grid_mean_predictor)
        grid_mean_predictor_type = grid_mean_predictor.pop("type")

        if grid_mean_predictor_type == "cortex":
            input_dim = grid_mean_predictor.pop("input_dimensions", 2)
            if deeplake_ds:
                import deeplake
                source_grids = {
                    k: deeplake.load(f'hub://sinzlab/sensorium2023_{k}_train').info['cell_motor_coordinates'][:, :input_dim]
                    for k, _ in dataloaders.items()
                }
            else:
                source_grids = {
                    k: v.dataset.neurons.cell_motor_coordinates[:, :input_dim]
                    for k, v in dataloaders.items()
                }
    return grid_mean_predictor, grid_mean_predictor_type, source_grids
