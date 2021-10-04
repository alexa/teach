# TEACh
[Task-driven Embodied Agents that Chat](https://arxiv.org/abs/2110.00534)

Aishwarya Padmakumar*, Jesse Thomason*, Ayush Shrivastava, Patrick Lange, Anjali Narayan-Chen, Spandana Gella, Robinson Piramuthu, Gokhan Tur, Dilek Hakkani-Tur

TEACh is a dataset of human-human interactive dialogues to complete tasks in a simulated household environment. 
The code is licensed under the MIT License (see SOFTWARELICENSE), images are licensed under Apache 2.0 
(see IMAGESLICENSE) and other data files are licensed under CDLA-Sharing 1.0 (see DATALICENSE).
Please include appropriate licensing and attribution when using our data and code, and please cite our paper.

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

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

The code is licensed under the MIT License (see SOFTWARELICENSE), images are licensed under Apache 2.0 
(see IMAGESLICENSE) and other data files are licensed under CDLA-Sharing 1.0 (see DATALICENSE).
