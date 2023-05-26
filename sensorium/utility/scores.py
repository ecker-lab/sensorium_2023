import warnings

import numpy as np
import torch
from neuralpredictors.measures.np_functions import corr
from neuralpredictors.training import device_state


def model_predictions(
    model, dataloader, data_key, device="cpu", skip=50, deeplake_ds=False
):
    """
    computes model predictions for a given dataloader and a model
    Returns:
        target: ground truth, i.e. neuronal firing rates of the neurons
        output: responses as predicted by the network
    """

    target, output = [], []
    for batch in dataloader:

        batch_kwargs = batch._asdict() if not isinstance(batch, dict) else batch
        if deeplake_ds:
            for k in batch_kwargs.keys():
                if k not in ["id", "index"]:
                    batch_kwargs[k] = torch.Tensor(np.asarray(batch_kwargs[k])).to(
                        device
                    )
            images = batch_kwargs["videos"]
            responses = batch_kwargs["responses"]
        else:
            images, responses = (
                batch[:2]
                if not isinstance(batch, dict)
                else (batch["videos"], batch["responses"])
            )
            
        with torch.no_grad():
            resp = responses.detach().cpu()[:, :, skip:]
            target = target + list(resp)
            with device_state(model, device):
                out = (
                    model(images.to(device), data_key=data_key, **batch_kwargs)
                    .detach()
                    .cpu()[:, -resp.shape[-1] :, :]
                )
                assert (
                    out.shape[1] == resp.shape[-1]
                ), f"model prediction is too short ({out.shape[1]} vs {resp.shape[-1]})"
                output = output + list(out.permute(0, 2, 1))

    return target, output


def get_correlations(
    model,
    dataloaders,
    tier=None,
    device="cpu",
    as_dict=False,
    per_neuron=True,
    deeplake_ds=False,
    **kwargs,
):
    """
    Computes single-trial correlation between model prediction and true responses
    Args:
        model (torch.nn.Module): Model used to predict responses.
        dataloaders (dict): dict of test set torch dataloaders.
        tier(str): the data-tier (train/test/val). If tier is None, then it is assumed that the the tier-key is not present.
        device (str, optional): device to compute on. Defaults to "cpu".
        as_dict (bool, optional): whether to return the results per data_key. Defaults to False.
        per_neuron (bool, optional): whether to return the results per neuron or averaged across neurons. Defaults to True.
    Returns:
        dict or np.ndarray: contains the correlation values.
    """
    correlations = {}
    dl = dataloaders[tier] if tier is not None else dataloaders
    for k, v in dl.items():
        target, output = model_predictions(
            dataloader=v,
            model=model,
            data_key=k,
            device=device,
            deeplake_ds=deeplake_ds,
        )
        target = np.concatenate(target, axis=1).T
        output = np.concatenate(output, axis=1).T
        correlations[k] = corr(target, output, axis=0)

        if np.any(np.isnan(correlations[k])):
            warnings.warn(
                "{}% NaNs , NaNs will be set to Zero.".format(
                    np.isnan(correlations[k]).mean() * 100
                )
            )
        correlations[k][np.isnan(correlations[k])] = 0

    if not as_dict:
        correlations = (
            np.hstack([v for v in correlations.values()])
            if per_neuron
            else np.mean(np.hstack([v for v in correlations.values()]))
        )
    return correlations


def get_poisson_loss(
    model,
    dataloaders,
    device="cpu",
    as_dict=False,
    avg=False,
    per_neuron=True,
    eps=1e-12,
):
    poisson_loss = {}
    for k, v in dataloaders.items():
        target, output = model_predictions(
            dataloader=v, model=model, data_key=k, device=device
        )
        loss = output - target * np.log(output + eps)
        poisson_loss[k] = np.mean(loss, axis=0) if avg else np.sum(loss, axis=0)
    if as_dict:
        return poisson_loss
    else:
        if per_neuron:
            return np.hstack([v for v in poisson_loss.values()])
        else:
            return (
                np.mean(np.hstack([v for v in poisson_loss.values()]))
                if avg
                else np.sum(np.hstack([v for v in poisson_loss.values()]))
            )
