<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![hub](https://img.shields.io/badge/powered%20by-hub%20-ff5a1f.svg)](https://github.com/activeloopai/Hub)

# SENSORIUM 2023 Competition

![plot](figures/competition.png)
SENSORIUM is a competition on predicting large scale mouse primary visual cortex activity. We will provide large scale datasets of neuronal activity in the visual cortex of mice. Participants will train models on pairs of natural stimuli and recorded neuronal responses, and submit the predicted responses to a set of test images for which responses are withheld. 

In 2022 we used images as stimuli, in 2023 we add the time component and use videos.

Join our challenge and compete for the best neural predictive model!

For more information about the competition, vist our [website](http://sensorium-competition.net/).

<!-- Have a look at our [White paper on arXiv](https://arxiv.org/abs/2206.08666), which describes the dataset and competition in detail. -->

The White paper, which describes the dataset and competition in detail will be pubished later./

# Important Dates
**June 12, 2023**: Start of the competition and data release.
<br>**Sep 15, 2023**: Submission deadline.
<br>**Sep 29, 2023**: Validation of all submitted scores completed. Preliminary winners are announced. Rank 1-3 in both competition tracks are contacted to provide the code for their submission.
<br>**Oct 13, 2023**: Deadline for top-ranked entries to provide the code for their submission.
<br>**Oct 23, 2023**: Winners contacted to contribute to the competition summary write-up.

# Starter-kit

This is a starter kit repo. The links for data would be published upon the competition start. Participants would have access to zip archieves or to the Deep Lake dataloaders, not to store the datasets locally.

<!-- Below we provide a step-by-step guide for getting started with the competition. -->
<!-- 
## 1. Pre-requisites
- install [**docker**](https://docs.docker.com/get-docker/) and [**docker-compose**](https://docs.docker.com/compose/install/)
- install git
- clone the repo via `git clone https://github.com/sinzlab/sensorium.git`

## 2. Download neural data

You can download the data from [https://gin.g-node.org/cajal/Sensorium2022](https://gin.g-node.org/cajal/Sensorium2022) and place it in `sensorium/notebooks/data`.
**Note:** Downloading the files all at once as a directory does lead to unfortunate errors. Thus, all datastes have to be downloaded individually.

## 3. Run the example notebooks

### **Start Jupyterlab environment**
```
cd sensorium/
docker-compose run -d -p 10101:8888 jupyterlab
```
now, type in `localhost:10101` in your favorite browser, and you are ready to go!
 -->

<!-- ## **Competition example notebooks** -->
We provide notebooks that illustrate the structure of our data, our baselines models, and how to make a submission to the competition.
<br>[**Dataset tutorial**](notebooks/load_data_demo.ipynb): Shows the structure of the data and how to turn it into a PyTorch DataLoader.
<br>[**Model tutorial**](notebooks/models_demo.ipynb): How to train and evaluate our baseline models and even more models.
<!-- <br>[**Submission tutorial**](notebooks/submission_tutorial/): Use our API to make a submission to our competition. -->
**To download data as .zip, click [here](https://gin.g-node.org/pollytur/sensorium_2023_data)**

## Submission comments

Participants should submit a zip file with 2 files in it : `predictions_live_main.parquet.brotli` and `predictions_final_main.parquet.brotli` for the live and final test correspondingly. For the bonus track please replace `main` with `ood` like this : `predictions_final_ood.parquet.brotli`. 

If you do not want to use API for the competition submission,here are some guides. Each file should contains 4 columns: `mouse` with the session name, `trial_indices` (like '1.npy'), `prediction`, where each entity in the predictions column is a list of lists with shape = (number or neurons, n_frames), where n_frames is the last n_frames frames for the video, excluding the first 50 frames (like for 300 frames in video we need predictions only for the last 250), and  `neuron_ids`, which is simple the order of the neurons in the predictions.

If you use pandas, here is a toy example how to save the data:

```
data = [
    ('mouse1', '1.npy', np.random.random((8000, 250)).tolist(), [1,2,3]),
    ('mouse1', '2.npy', np.random.random((8000, 250)).tolist(), [1,2,3]),
    ('mouse1', '3.npy', np.random.random((8000, 250)).tolist(), [1,2,3]),
]


df = pd.DataFrame.from_records(data, columns =['mouse', 'trial_indices', 'prediction', 'neuron_ids'])
df = pd.concat(dataframes_pred, ignore_index=True)
submission_filename = f"predictions_file_{tier}_{track}_track.parquet.brotli"
save_path = os.path.join(path, submission_filename) if path is not None else submission_filename
df.to_parquet(save_path, compression='brotli', engine='pyarrow', index=False)
```

If you have any questions, feel free to reach out to us (Contact section on our [website](http://sensorium-competition.net/)), or raise an issue here on GitHub!
