from sacred import Ingredient
from sacred.settings import SETTINGS

exp_ingredient = Ingredient("exp")
train_ingredient = Ingredient("train")
eval_ingredient = Ingredient("eval")
dagger_ingredient = Ingredient("dagger")

SETTINGS.CONFIG.READ_ONLY_CONFIG = False


@exp_ingredient.config
def cfg_exp():
    # HIGH-LEVEL MODEL SETTINGS
    # where to save model and/or logs
    name = "default"
    # model to use
    model = "transformer"
    # which device to use
    device = "cuda"
    # number of data loading workers or evaluation processes (0 for main thread)
    num_workers = 12
    # we can fine-tune a pre-trained model
    pretrained_path = None
    # run the code on a small chunk of data
    fast_epoch = False

    # Set this to 1 if running on a Mac and to large numbers like 250 if running on EC2
    lmdb_max_readers = 1

    # DATA SETTINGS
    data = {
        # dataset name(s) for training and validation
        "train": None,
        # additional dataset name(s) can be specified for validation only
        "valid": "",
        # specify the length of each dataset
        "length": 30000,
        # what to use as annotations: {'lang', 'lang_frames', 'frames'}
        "ann_type": "lang",
        # Train dataloader type - sample or shuffle ("sample" results in sampling length points per epoch with
        # replacement and "shuffle" results in iterating through the train dataset in random order per epoch
        "train_load_type": "shuffle",
    }

    lang_pretrain_over_history_subgoals = False


@eval_ingredient.config
def cfg_eval():
    # which experiment to evaluate (required)
    exp = None
    # which checkpoint to load ('latest.pth', 'model_**.pth')
    checkpoint = "latest.pth"
    # which split to use ('train', 'valid_seen', 'valid_unseen')
    split = "valid_seen"
    use_sample_for_train = True
    use_random_actions = False
    no_lang = False
    no_vision = False

    # shuffle the trajectories
    shuffle = False
    # max steps before episode termination
    max_steps = 1000
    # max API execution failures before episode termination
    max_fails = 10
    # subgoals to evaluate independently, eg:all or GotoLocation,PickupObject or 0,1
    subgoals = ""
    # smooth nav actions (might be required based on training data)
    smooth_nav = False
    # forward model with expert actions (only for subgoals)
    no_model_unroll = False
    # no teacher forcing with expert (only for subgoals)
    no_teacher_force = False
    # run in the debug mode
    debug = False
    # X server number
    x_display = "0"
    # range of checkpoints to evaluate, (9, 20, 2) means epochs 9, 11, 13, 15, 17, 19
    # if None, only 'latest.pth' will be evaluated
    eval_range = (9, 20, 1)
    # object predictor path
    object_predictor = None

    # Is this evaluation for EDH instances or TFD instances?
    eval_type = "edh"

    # Set this to 1 if running on a Mac and to large numbers like 250 if running on EC2
    # lmdb_max_readers = 1

    # Set this to true if the model was trained (and should for inference try to get a wide view)
    wide_view = False

    force_retry = False


@train_ingredient.config
def cfg_train():
    # GENERAL TRANING SETTINGS
    # random seed
    seed = 1
    # load a checkpoint from a previous epoch (if available)
    resume = True
    # whether to print execution time for different parts of the code
    profile = False

    # For ablations
    no_lang = False
    no_vision = False

    # HYPER PARAMETERS
    # batch size
    batch = 8
    # number of epochs
    epochs = 20
    # optimizer type, must be in ('adam', 'adamw')
    optimizer = "adamw"
    # L2 regularization weight
    weight_decay = 0.33
    # learning rate settings
    lr = {
        # learning rate initial value
        "init": 1e-4,
        # lr scheduler type: {'linear', 'cosine', 'triangular', 'triangular2'}
        "profile": "linear",
        # (LINEAR PROFILE) num epoch to adjust learning rate
        "decay_epoch": 10,
        # (LINEAR PROFILE) scaling multiplier at each milestone
        "decay_scale": 0.1,
        # (COSINE & TRIANGULAR PROFILE) learning rate final value
        "final": 1e-5,
        # (TRIANGULAR PROFILE) period of the cycle to increase the learning rate
        "cycle_epoch_up": 0,
        # (TRIANGULAR PROFILE) period of the cycle to decrease the learning rate
        "cycle_epoch_down": 0,
        # warm up period length in epochs
        "warmup_epoch": 0,
        # initial learning rate will be divided by this value
        "warmup_scale": 1,
    }
    # weight of action loss
    action_loss_wt = 1.0
    # weight of object loss
    object_loss_wt = 1.0
    # weight of subgoal completion predictor
    # subgoal_aux_loss_wt = 0.1
    subgoal_aux_loss_wt = 0
    # weight of progress monitor
    # progress_aux_loss_wt = 0.1
    progress_aux_loss_wt = 0
    # maximizing entropy loss (by default it is off)
    entropy_wt = 0.0

    # Should train loss be computed over history actions? (default False)
    compute_train_loss_over_history = False

    # TRANSFORMER settings
    # size of transformer embeddings
    demb = 768
    # number of heads in multi-head attention
    encoder_heads = 12
    # number of layers in transformer encoder
    encoder_layers = 2
    # how many previous actions to use as input
    num_input_actions = 1
    # which encoder to use for language encoder (by default no encoder)
    encoder_lang = {
        "shared": True,
        "layers": 2,
        "pos_enc": True,
        "instr_enc": False,
    }
    # which decoder to use for the speaker model
    decoder_lang = {
        "layers": 2,
        "heads": 12,
        "demb": 768,
        "dropout": 0.1,
        "pos_enc": True,
    }
    # do not propagate gradients to the look-up table and the language encoder
    detach_lang_emb = False

    # DROPOUTS
    dropout = {
        # dropout rate for language (goal + instr)
        "lang": 0.0,
        # dropout rate for Resnet feats
        "vis": 0.3,
        # dropout rate for processed lang and visual embeddings
        "emb": 0.0,
        # transformer model specific dropouts
        "transformer": {
            # dropout for transformer encoder
            "encoder": 0.1,
            # remove previous actions
            "action": 0.0,
        },
    }

    # ENCODINGS
    enc = {
        # use positional encoding
        "pos": True,
        # use learned positional encoding
        "pos_learn": False,
        # use learned token ([WORD] or [IMG]) encoding
        "token": False,
        # dataset id learned encoding
        "dataset": False,
    }

    use_alfred_weights = False
