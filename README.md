# TEACh
[Task-driven Embodied Agents that Chat](https://arxiv.org/abs/2110.00534)

Aishwarya Padmakumar*, Jesse Thomason*, Ayush Shrivastava, Patrick Lange, Anjali Narayan-Chen, Spandana Gella, Robinson Piramuthu, Gokhan Tur, Dilek Hakkani-Tur

TEACh is a dataset of human-human interactive dialogues to complete tasks in a simulated household environment. 
The code and model weights are licensed under the MIT License (see SOFTWARELICENSE), images are licensed under Apache 2.0 
(see IMAGESLICENSE) and other data files are licensed under CDLA-Sharing 1.0 (see DATALICENSE).
Please include appropriate licensing and attribution when using our data and code, and please cite our paper.

Citation:
```buildoutcfg
@inproceedings{teach,
  title={{TEACh: Task-driven Embodied Agents that Chat}},
  author={Padmakumar, Aishwarya and Thomason, Jesse and Shrivastava, Ayush and Lange, Patrick and Narayan-Chen, Anjali and Gella, Spandana and Piramuthu, Robinson and Tur, Gokhan and Hakkani-Tur, Dilek},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={2},
  pages={2017--2025},
  year={2022}
}
```


As of 09/07/2022, the dataset has been updated to include dialog acts annotated in the paper

[Dialog Acts for Task-Driven Embodied Agents](https://assets.amazon.science/9c/af/d18d00b44a129e10f1f29de9861a/dialog-acts-for-task-driven-embodied-agents.pdf)

Spandana Gella*, Aishwarya Padmakumar*, Patrick Lange, Dilek Hakkani-Tur

If using the dialog acts in your work, please cite the following paper:
```buildoutcfg
@inproceedings{teachda,
  title={{Dialog Acts for Task-Driven Embodied Agents}},
  author={Gella, Spandana and Padmakumar, Aishwarya and Lange, Patrick and Hakkani-Tur, Dilek},
  booktitle={Proceedings of the 23nd Annual Meeting of the Special Interest Group on Discourse and Dialogue (SIGDial)},
  year={2022},
  pages={111-123}
}
```

Interactions in the games, EDH instances and TfD instances that are utterances now have an additional field `da_metadata` containing the dialog act annotations.
See the [data exploration notebook](https://github.com/alexa/teach/blob/main/src/teach/analysis/teach_data_exploration.ipynb) for sample code to view dialog acts.

## Prerequisites
- python3 `>=3.7,<=3.8`
- python3.x-dev, example: `sudo apt install python3.8-dev`
- tmux, example: `sudo apt install tmux`
- xorg, example: `sudo apt install xorg openbox`
- ffmpeg, example: `sudo apt install ffmpeg`

## Installation
```
pip install -r requirements.txt
pip install -e .
```
## Downloading the dataset
Run the following script:
```
teach_download 
```
This will download and extract the archive files (`experiment_games.tar.gz`, `all_games.tar.gz`, 
`images_and_states.tar.gz`, `edh_instances.tar.gz` & `tfd_instances.tar.gz`) in the default 
directory (`/tmp/teach-dataset`).  
**Optional arguments:**
- `-d`/`directory`: The location to store the dataset into. Default=`/tmp/teach-dataset`.
- `-se`/`--skip-extract`: If set, skip extracting archive files.
- `-sd`/`--skip-download`: If set, skip downloading archive files.
- `-f`/`--file`: Specify the file name to be retrieved from S3 bucket.

File changes (12/28/2022): 
We have modified EDH instances so that the state changes checked for to evaluate success are only those that contribute towards task success in the main task of the gameplay session the EDH instance is created from. 
We have removed EDH instances that had no state changes meeting these requirements. 
Additionally, two game files, and their corresponding EDH and TfD instances were deleted from the `valid_unseen` split due to issues in the game files. 
Version 3 of our paper on Arxiv, which will be public on Dec 30, 2022 contains the updated dataset size and experimental results.  

## Remote Server Setup
If running on a remote server without a display, the following setup will be needed to run episode replay, model inference of any model training that invokes the simulator (student forcing / RL). 

Start an X-server 
```
tmux
sudo python ./bin/startx.py
```
Exit the `tmux` session (`CTRL+B, D`). Any other commands should be run in the main terminal / different sessions. 


## Replaying episodes
Most users should not need to do this since we provide this output in `images_and_states.tar.gz`.

The following steps can be used to read a `.json` file of a gameplay session, play it in the AI2-THOR simulator, and at each time step save egocentric observations of the `Commander` and `Driver` (`Follower` in the paper). It also saves the target object panel and mask seen by the `Commander`, and the difference between current and initial state.     

Replaying a single episode locally, or in a new `tmux` session / main terminal of remote headless server:
```
teach_replay \
--game_fn /path/to/game/file \
--write_frames_dir /path/to/desired/output/images/dir \
--write_frames \
--write_states \
--status-out-fn /path/to/desired/output/status/file.json
```
Note that `--status-out-fn` must end in `.json`
Also note that the script will by default not replay sessions for which an output subdirectory already exists under `--write-frames-dir`
Additionally, if the file passed to `--status-out-fn` already exists, the script will try to resume files not marked as replayed in that file. It will error out if there is a mismatch between the status file and output directories on which sessions have been previously played. 
It is recommended to use a new `--write-frames-dir` and new `--status-out-fn` for additional runs that are not intended to resume from a previous one.

Replay all episodes in a folder locally, or in a new `tmux` session / main terminal of remote headless server:
```
teach_replay \
--game_dir /path/to/dir/containing/.game.json/files \
--write_frames_dir /path/to/desired/output/images/dir \
--write_frames \
--write_states \
--num_processes 50 \
--status-out-fn /path/to/desired/output/status/file.json
```

To generate a video, additionally specify `--create_video`. Note that for images to be saved, `--write_images` must be specified and `--write-frames-dir` must be provided. For state changes to be saved, `--write_states` must be specified and `--write_frames_dir` must be provided.

## Evaluation

We include sample scripts for inference and calculation of metrics. `teach_inference` and `teach_eval`. 
`teach_inference` is a wrapper that implements loading EDH instance, interacting with the simulator as well as writing the game
file and predicted action sequence as JSON files after each inference run. It dynamically loads the model based on the `--model_module`
and `--model_class` arguments. Your model has to implement `teach.inference.teach_model.TeachModel`. See `teach.inference.sample_model.SampleModel`
for an example implementation which takes random actions at every time step. 

After running `teach_inference`, you use `teach_eval` to compute the metrics based output data produced by `teach_inference`.


Sample run:
```
export DATA_DIR=/path/to/data/with/games/and/edh_instances/as/subdirs (Default in Downloading is /tmp/teach-dataset)
export OUTPUT_DIR=/path/to/output/folder/for/split
export METRICS_FILE=/path/to/output/metrics/file_without_extension

teach_inference \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --split valid_seen \
    --metrics_file $METRICS_FILE \
    --model_module teach.inference.sample_model \
    --model_class SampleModel

teach_eval \
    --data_dir $DATA_DIR \
    --inference_output_dir $OUTPUT_DIR \
    --split valid_seen \
    --metrics_file $METRICS_FILE
```    
To run TfD inference instead of EDH inference add `--benchmark tfd` to the inference command.

## TEACh Benchmark Challenge

For participation in the challenge, you will need to submit a docker image container your code and model.
Docker containers using your image will serve your model as HTTP API following the [TEACh API Specification](#TEACh API Specification).
For your convenience, we included the `teach_api` command which implements this API and is compatible with models implementing `teach.inference.teach_model.TeachModel` also used by `teach_inference`.

We have also included two sample Docker images using `teach.inference.sample_model.SampleModel` and `teach.inference.et_model.ETModel` respectively in 
[`docker/`](./docker).

When evaluating a submissions, the submitted container will be started with access to a single GPU and no internet access. For details see [Step 3 - Start your container](#step-3---start-your-container).

The main evaluation code invoking your submission will also be run as Docker container. It reuses the `teach_inference` CLI command together with `teach.inference.remote_model.RemoteModel` to call the HTTP API running in your container. For details on how to start it locally see [Step 4 - Start the evaluation](#step-4---start-the-evaluation).

Please note that TfD inference is not currently supported via Docker image. 

### Testing Locally

Assuming you have [downloaded the data](#downloading-the-dataset) to `/home/ubuntu/teach-dataset` and followed [Prerequisites](#prerequisites) and [Remote Server Setup](#remote-server-setup).


#### Step 0 - Setup Environment

```buildoutcfg
export HOST_DATA_DIR=/home/ubuntu/teach-dataset
export HOST_IMAGES_DIR=/home/ubuntu/images
export HOST_OUTPUT_DIR=/home/ubuntu/output
export API_PORT=5000
export SUBMISSION_PK=168888
export INFERENCE_GPUS='"device=0"'
export API_GPUS='"device=1"'
export SPLIT=valid_seen
export DOCKER_NETWORK=no-internet

mkdir -p $HOST_IMAGES_DIR $HOST_OUTPUT_DIR
docker network create --driver=bridge --internal $DOCKER_NETWORK
```
Note: If you run on a machine that only has a single GPU, set `API_GPUS='"device=0"'`.

#### Step 1 - Build the `remote-inference-runner` container

```buildoutcfg
docker build -t remote-inference-runner -f docker/Dockerfile.RemoteInferenceRunner .
```

#### Step 2 - Build your container

Note: When customizing the images for your own usage, do not edit the following or your submission will fail:
- `teach_api` options: `--data_dir /data --images_dir /images --split $SPLIT`
- `EXPOSE 5000` and don't change the port the flask API listens on

For the `SampleModel` example, the corresponding command is:

```buildoutcfg
docker build -t teach-model-api-samplemodel -f docker/Dockerfile.TEAChAPI-SampleModel .
```

For the `baseline models`, follow the corresponding command replacing `MODEL_VARIANT=et` with 
the desired variant e.g. `et_plus_a`.

```buildoutcfg
mkdir -p ./models
mv $HOST_DATA_DIR/baseline_models ./models/
mv $HOST_DATA_DIR/et_pretrained_models ./models/
docker build --build-arg MODEL_VARIANT=et -t teach-model-api-etmodel -f docker/Dockerfile.TEAChAPI-ETModel .
```

#### Step 3 - Start your container

For the `SampleModel` example, the corresponding command is:

```buildoutcfg
docker run -d --rm \
    --gpus $API_GPUS \
    --name TeachModelAPI \
    --network $DOCKER_NETWORK \
    -e SPLIT=$SPLIT \
    -v $HOST_DATA_DIR:/data:ro \
    -v $HOST_IMAGES_DIR/$SUBMISSION_PK:/images:ro \
    -t teach-model-api-samplemodel    
```

For the baseline models, just replace the image name e.g. if you followed the commands above

```buildoutcfg
docker run -d --rm \
    --gpus $API_GPUS \
    --name TeachModelAPI \
    --network $DOCKER_NETWORK \
    -e SPLIT=$SPLIT \
    -v $HOST_DATA_DIR:/data:ro \
    -v $HOST_IMAGES_DIR/$SUBMISSION_PK:/images:ro \
    -t teach-model-api-etmodel    
```

Verify the API is running with

```buildoutcfg
docker exec TeachModelAPI curl @TeachModelAPI:5000/ping

Output:
{"action":"Look Up","obj_relative_coord":[0.1,0.2]}
```

#### Step 4 - Start the evaluation

```buildoutcfg
docker run --rm \
    --privileged \
    -e DISPLAY=:0 \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    --name RemoteInferenceRunner \
    --network $DOCKER_NETWORK \
    --gpus $INFERENCE_GPUS \
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
    -v $HOST_DATA_DIR:/data:ro \
    -v $HOST_IMAGES_DIR/$SUBMISSION_PK:/images \
    -v $HOST_OUTPUT_DIR/$SUBMISSION_PK:/output \
    remote-inference-runner teach_inference \
        --data_dir /data \
        --output_dir /output \
        --images_dir /images \
        --split $SPLIT \
        --metrics_file /output/metrics_file \
        --model_module teach.inference.remote_model \
        --model_class RemoteModel \
        --model_api_host_and_port "@TeachModelAPI:$API_PORT"
```

#### Step 5 - Results

The evaluation metrics will be in `$HOST_OUTPUT_DIR/$SUBMISSION_PK/metrics_file`.
Images for each episode will be in `$HOST_IMAGES_DIR/$SUBMISSION_PK`.

### Running without docker

You may want to test your implementation without rebuilding Docker images. You can test your model by directly calling the `teach_api` CLI command e.g.

Using the `teach.inference.sample_model.SampleModel`:

```buildoutcfg
export DATA_DIR=/home/ubuntu/teach-dataset
export IMAGE_DIR=/tmp/images

teach_api \
    --data_dir $DATA_DIR \
    --images_dir $IMAGE_DIR
```

Using the `teach.inference.et_model.ETModel` assuming you already moved the models from the teach-dataset location to 
`./models` following instructions in [Step 2 - Build your container](#step-2---build-your-container).

```buildoutcfg
export DATA_DIR=/home/ubuntu/teach-dataset
export IMAGE_DIR=/tmp/images

teach_api \
    --data_dir $DATA_DIR \
    --images_dir $IMAGE_DIR \
    --split valid_seen \
    --model_module teach.inference.et_model \
    --model_class ETModel \
    --model_dir ./models/baseline_models/et \
    --visual_checkpoint ./models/et_pretrained_models/fasterrcnn_model.pth
    --object_predictor ./models/et_pretrained_models/maskrcnn_model.pth \
    --seed 4 
```

The corresponding command for running `teach_inference` against such an API
without container uses `teach.inference.remote_model.RemoteModel`.

```buildoutcfg
export DATA_DIR=/home/ubuntu/teach-dataset
export OUTPUT_DIR=/home/ubuntu/output/valid_seen
export METRICS_FILE=/home/ubuntu/output/valid_seen/metrics
export IMAGE_DIR=/tmp/images

teach_inference \
    --data_dir $DATA_DIR  \
    --output_dir $OUTPUT_DIR \    
    --split valid_seen \
    --metrics_file $METRICS_FILE \    
    --model_module teach.inference.remote_model \
    --model_class RemoteModel \        
    --model_api_host_and_port 'localhost:5000' \
    --images_dir $IMAGE_DIR
    
```

### Smaller split

It may be useful for faster turn around time to locally create a smaller split in `$DATA_DIR/edh_instances/test_seen` 
with a handful of files from `$DATA_DIR/edh_instances/valid_seen` for faster turn around times. 

### Runtime Checks

The TEACh Benchmark Challenge places a maximum time limit of 36 hours when using all GPUs of a `p3.16xlarge` instance.
The best way to verify that your code is likely to satisfy this requirement would be to use a script to run two Docker evaluation processes in sequence on a `p3.16xlarge` EC2 instance, one for the `valid_seen` split and one for the `valid_unseen` split.
Note that you will need to specify `export API_GPUS='"device=1,2,3,4,5,6,7"'` (we reserve GPU 0 for `ai2thor` in our runs) to use all GPUs and your model code will need to place different instances of the model on different GPUs for this test (see the use of `process_index` in `ETModel.set_up_model()` for an example).
Also note that while the test splits are close in size to the validation splits, they are not identical so your runtime estimate will necessarily be an approximation. 

### TEACh API Specification

As mentioned above, `teach_api` already implements this API and it is usually not necessary to implement this yourself. During evaluations of submissions, edh_instances without ground truth and images corresponding to the edh_instances' histories will be available in `/data`. `/images` will contain images produced during inference at runtime. `teach_api` already handles loading and passes them to your implementation of `teach.inference.teach_model.TeachModel`.

#### Start EDH Instance

This endpoint will be called once at the start of processing a new EDH instance. Currently, we ensure that the API processes only a single EDH instance from start to finish i.e. once called it can be assumed that the previous EDH instance has completed.

URL : `/start_new_edh_instance`  
Method : `POST`  
Payload:  

```json
{
    "edh_name": "[name of the EDH instance file]"
}
```

Responses:

Status Code: `200`  
Response: `success`

Status Code: `500`  
Response: `[error message]`


#### Get next action

This endpoint will be called at each timestep during inference to get the next predicted action from the model.

URL : `/get_next_action`  
Method : `POST`  
Payload:  

```json
{
    "edh_name": "[name of the EDH instance file]",
    "img_name": "[name of the image taken in the simulator after the previous action]",
    "prev_action": "[JSON string representation of previous action]", // this is optional
}
```

Responses:

Status Code: `200`  

```json
{
    "action": "[An action name from all_agent_actions]",
    "obj_relative_coord": [0.1, 0.5] // see teach.inference.teach_model.TeachModel.get_next_action
}
```

Status Code: `500`  
Response: `[error message]`

## TEACh EDH Offline Evaluation

While the leaderboard for the TEACh EDH benchmark is not active, we recommend that researchers follow the following protocol for evaluation. 
A split of the existing TEACh validation splits has been provided in the `src/teach/meta_data_files/divided_split` directory. 
For your experiments, please use the `divided_val_seen` and `divided_val_unseen` splits for validation and `divided_test_seen` and `divided_test_unseen` for testing.
Note that the TEACh code has not been modified at the moment to directly support use of these splits, so you will need to locally reorganize your data directory so that games, EDH instances and image folders are reorganized according to the divided split.
Some additional notes:

1. If you have previously tuned hyperparameters using the full TEACh validation split, you will need to re-tune hyperparameters on just the `divided_val_seen` or `divided_val_unseen` splits for fair comparison to other papers.
2. The divided test splits are likely to be easier than the original TEACh test split as the floorplans used in the `divided_val_unseen` and `divided_test_unseen` splits are identical.
3. Please do not incorporate the `divided_val_seen` or `divided_val_unseen` splits into your training set and retrain after hyperparameter tuning if using this protocol, as the `divided_test_unseen` split will then no longer be unseen. 
4. We have observed that the ET model can show some variance when being retrained on ALFRED or TEACh even when changing only the random seeds, and as such we expect some performance differences between the full TEACh validation splits, TEACh test splits and divided splits. 
5. Alexa Prize SimBot Challenge Participants please refer to challenge rules regarding publications.


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

The code is licensed under the MIT License (see SOFTWARELICENSE), images are licensed under Apache 2.0 
(see IMAGESLICENSE) and other data files are licensed under CDLA-Sharing 1.0 (see DATALICENSE).


