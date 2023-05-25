import numpy as np
from neuralpredictors.data.datasets import MovieFileTreeDataset
from neuralpredictors.data.samplers import SubsetSequentialSampler
from neuralpredictors.data.transforms import (AddBehaviorAsChannels,
                                              AddPupilCenterAsChannels,
                                              ChangeChannelsOrder, CutVideos,
                                              ExpandChannels, NeuroNormalizer,
                                              ScaleInputs, SelectInputChannel,
                                              Subsample, Subsequence, ToTensor)
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def mouse_video_loader(
    paths,
    batch_size,
    normalize=True,
    exclude: str = None,
    cuda: bool = False,
    max_frame=None,
    frames=50,
    offset=-1,
    inputs_mean=None,
    inputs_std=None,
    include_behavior=True,
    include_pupil_centers=True,
    include_pupil_centers_as_channels=False,
    scale=1,
    to_cut=True,
    exclude_beh_channels=None,
):
    """
    Symplified version of the sensorium mouse_loaders.py
     Returns a dictionary of dataloaders (i.e., trainloaders, valloaders, and testloaders) for >= 1 dataset(s).
    Args:
        paths (list): list of paths for the datasets
        batch_size (int): batch size.
        frames (int, optional): how many frames ot take per video
        max_frame (int, optional): which is the maximal frame that could be taken per video
        offset (int, optional): Offset to start the subsequence from. Defaults to -1, corresponding to random but valid offset at each iteration.
        cuda (bool, optional): whether to place the data on gpu or not.
        normalize (bool, optional): whether to normalize the data (see also exclude)
        exclude (str, optional): data to exclude from data-normalization. Only relevant if normalize=True. Defaults to 'images'
        include_behavior (bool, optional): whether to include behavioral data
        include_pupil_centers (bool, optional): whether to include pupil center data
        include_pupil_centers_as_channels(bool, optional): whether to include pupil center data as channels
        scale(float, optional): scalar factor for the image resolution.
            scale = 1: full iamge resolution (144 x 256)
            scale = 0.25: resolution used for model training (36 x 64)
        float64 (bool, optional):  whether to use float64 in MovieFileTreeDataset
    Returns:
        dict: dictionary of dictionaries where the first level keys are 'train', 'validation', and 'test', and second level keys are data_keys.
    """

    data_keys = [
        "videos",
        "responses",
    ]
    if include_behavior:
        data_keys.append("behavior")
    if include_pupil_centers:
        data_keys.append("pupil_center")

    #     dataloaders_combined = {"validation": {}, "train": {}, "test": {}}
    dataloaders_combined = {}

    for path in paths:
        dat2 = MovieFileTreeDataset(path, *data_keys)

        conds = np.ones(len(dat2.neurons.cell_motor_coordinates), dtype=bool)
        idx = np.where(conds)[0]

        more_transforms = [
            Subsample(idx, target_index=0),
            CutVideos(
                max_frame=max_frame,
                frame_axis={data_key: -1 for data_key in data_keys},
                target_groups=data_keys,
            ),
            ChangeChannelsOrder((2, 0, 1), in_name="videos"),
            ChangeChannelsOrder((1, 0), in_name="responses"),
            ChangeChannelsOrder((1, 0), in_name="behavior"),
            ChangeChannelsOrder((1, 0), in_name="pupil_center"),
        ]
        if to_cut:
            more_transforms.append(
                Subsequence(frames=frames, channel_first=(), offset=offset)
            )
        more_transforms = more_transforms + [
            ChangeChannelsOrder((1, 0), in_name="responses"),
            ChangeChannelsOrder((1, 0), in_name="behavior"),
            ChangeChannelsOrder((1, 0), in_name="pupil_center"),
            ExpandChannels("videos"),
        ]

        if include_behavior:
            more_transforms.append(
                AddBehaviorAsChannels(
                    "videos", exclude_beh_channels=exclude_beh_channels
                )
            )
        if include_pupil_centers and include_pupil_centers_as_channels:
            more_transforms.append(AddPupilCenterAsChannels("videos"))

        more_transforms.append(ToTensor(cuda))
        more_transforms.insert(
            0, ScaleInputs(scale=scale, in_name="videos", channel_axis=-1)
        )
        if normalize:
            try:
                more_transforms.insert(
                    0,
                    NeuroNormalizer(
                        dat2,
                        exclude=exclude,
                        inputs_mean=inputs_mean,
                        inputs_std=inputs_std,
                        in_name="videos",
                    ),
                )
            except:
                more_transforms.insert(
                    0, NeuroNormalizer(dat2, exclude=exclude, in_name="videos")
                )

        dat2.transforms.extend(more_transforms)

        # subsample images
        tier = None
        dataloaders = {}
        keys = [tier] if tier else list(set(list(dat2.trial_info.tiers)))
        tier_array = dat2.trial_info.tiers

        for tier in keys:
            subset_idx = np.where(tier_array == tier)[0]

            sampler = (
                SubsetRandomSampler(subset_idx)
                if tier == "train"
                else SubsetSequentialSampler(subset_idx)
            )
            dataloaders[tier] = DataLoader(
                dat2,
                sampler=sampler,
                batch_size=batch_size,
            )

        dataset_name = path.split("/")[-2]
        for k, v in dataloaders.items():
            if k not in dataloaders_combined.keys():
                dataloaders_combined[k] = {}
            dataloaders_combined[k][dataset_name] = v

    return dataloaders_combined
