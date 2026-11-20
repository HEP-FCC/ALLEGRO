
from lightning.pytorch.callbacks import (
    TQDMProgressBar,
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.profilers import AdvancedProfiler
from src.layers.utils_training import FreezeClustering

# callbacks for training step
def get_callbacks(args):
    checkpoint_callback = ModelCheckpoint(
                dirpath=args.model_prefix,  # checkpoints_path, # <--- specify this on the trainer itself for version control
                filename="_{epoch}_{step}",
                every_n_epochs=1,
                #every_n_train_steps=500,
                save_top_k=-1,  # <--- this is important!
                save_weights_only=True,
            )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks = [
        TQDMProgressBar(refresh_rate=10),
        checkpoint_callback,
        lr_monitor,
    ]
    if args.freeze_clustering:
            callbacks.append(FreezeClustering())
    return callbacks

# trainer options:
#profiler = AdvancedProfiler(dirpath=".", filename="perf_logs_train_23042024_2")
#profiler=profiler,
# accumulate_grad_batches=1,
# resume_from_checkpoint=args.load_model_weig


#profiler = AdvancedProfiler(dirpath="/eos/home-g/gkrzmanc/profiler/", fgatr_pf_eilename="profiler_eval_0705")
#print("USING PROFILER")

#callbacks for evaluation step
def get_callbacks_eval(args):
    callbacks=[TQDMProgressBar(refresh_rate=1)]
    return callbacks