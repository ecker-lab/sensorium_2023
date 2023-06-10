import os
import numpy as np
import pandas as pd
import torch
from neuralpredictors.training import device_state


def generate_submission(dataloader, model, deeplake_ds=False, path=None, tier=None, track='main', skip=50, device='cpu'):    
    assert track == 'main' or track == 'ood', 'Track should be "main" or "ood"'
    
    mice = list(dataloader[list(dataloader.keys())[0]].keys())
    
    if tier is None:
        tier_list = ['live_test', 'final_test']
    else:
        if isinstance(tier, list): 
            assert all(x in ['live_test', 'final_test'] for x in tier),\
                    'tier is either ["live_test"] or ["final_test"] or ["live_test", "final_test"]'
        else:     
            assert tier in ['live_test', 'final_test'],\
                    'tier is either ["live_test"] or ["final_test"] or ["live_test", "final_test"]'
            tier_list = [tier]
                    
    for tier in tier_list:
        tier = f'{tier}_{track}'
        dataframes_pred = []
        for m in mice:
            test_predictions = []
            trial_indices = []
            ds = dataloader[tier][m].dataset
            if deeplake_ds:
                import deeplake
                
                neuron_ids = deeplake.load(f'hub://sinzlab/Sensorium_2023_{m}_train').info['neuron_ids']
                tiers = [tier] * len(dataloader[tier][m].dataset)
                ds = iter(dataloader[tier][m])
            else:
                neuron_ids = np.asarray(ds.neurons.unit_ids.tolist()).astype(np.uint32)
                tiers = ds.trial_info.tiers
            for idx in range(len(tiers)):
                if tiers[idx] == tier:
                    if deeplake_ds:
                        batch_kwargs = next(ds)
                        length = batch_kwargs['videos'].shape[2] - skip
                        for k in batch_kwargs.keys():
                            if k not in ['id', 'index']:
                                batch_kwargs[k] = batch_kwargs[k].to(device)
                    else:
                        batch = ds.__getitem__(idx)
                        batch_kwargs = batch._asdict() if not isinstance(batch, dict) else batch
                        for bk, bv in batch_kwargs.items():
                            batch_kwargs[bk] = torch.unsqueeze(bv, 0)

                        length = batch[0].shape[2] - skip
                        
                    if deeplake_ds:
                        trial_indices = trial_indices + batch_kwargs['id']
                    else:
                        trial_indices.append(idx)                        

                    videos = batch_kwargs['videos']
                    with torch.no_grad():
                        with device_state(model, device):
                            out = (model(videos.to(device), data_key=m, **batch_kwargs)
                                        .detach()
                                        .cpu()[:, -length:, :])
                            assert out.shape[1] == length, \
                                f'model prediction is too short ({out.shape[1]} vs {length})'
                            pred = out.permute(0, 2, 1).squeeze().numpy().astype(np.float32).tolist()
                            test_predictions.append(pred)
            df = pd.DataFrame(
                {
                    "mouse": [m] * len(test_predictions),
                    "trial_indices": trial_indices,
                    "prediction": test_predictions,
                    "neuron_ids": [neuron_ids] * len(test_predictions),
                }
            )
            dataframes_pred.append(df) 

        #save file
        df = pd.concat(dataframes_pred, ignore_index=True)
        submission_filename = f"predictions_file_{tier}_{track}_track.parquet.brotli"
        save_path = os.path.join(path, submission_filename) if path is not None else submission_filename
        df.to_parquet(save_path, compression='brotli', engine='pyarrow', index=False)
        print(f"Submission file saved for tier: {tier}, track {track}. Saved in: {save_path}")              
        