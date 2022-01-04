# Files Structure

```
/data
    create_lmdb.py       (script to create an LMDB dataset out of trajectory files)
    preprocessor.py      (class to preprocess trajectories annotations and actions)
    process_tests.py     (script to process test splits for leaderboard evaluation)
    zoo/base.py          (base class for LMDB dataset loading using multiple threads)
    zoo/alfred.py        (class to load an LMDB dataset for an E.T. training)
    zoo/speaker.py       (class to load an LMDB dataset for a translation pretraining)
/env
    reward.py            (rewards definitions)
    tasks.py             (tasks definitions)
    thor_env.py          (interface between AI2Thor and E.T. code)
/eval
    eval_agent.py        (script to evaluate an agent on full tasks or subgoals)
    eval_master.py       (class for multi-process evaluation)
    eval_subgoals.py     (functions for subgoal evaluation)
    eval_task.py         (functions for full task evaluation)
    leaderboard.py       (script to evaluate an agent on test splits)
/gen
    constants.py         (list of constants)
    generate_trajs.py    (script to generate new trajectories)
    goal_library.py      (library defining goals using PDDL)
    render_trajs.py      (script to render existing trajectories)
/model
    train.py             (script for models training)
    base.py              (base class for E.T. and translator models)
    learned.py           (class with main train routines)
    speaker.py           (translator model)
    transformer.py       (E.T. model)
/nn
    attention.py         (basic attention mechanisms)
    dec_object.py        (object decoder class)
    enc_lang.py          (language encoder class)
    enc_visual.py        (visual observations encoder class)
    enc_vl.py            (multimodal encoder class)
    encodings.py         (positional and temporal encodings)
    transforms.py        (visual observations transformations)
/utils
    data_util.py         (data handling utils)
    eval_util.py         (evaluation utils)
    helper_util.py       (help utils)
    metric_util.py       (utils to compute scores)
    model_util.py        (utils for E.T. and translation models)
```
