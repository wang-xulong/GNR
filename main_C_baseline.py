import argparse
import torch
import wandb
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.benchmarks.classic import SplitCIFAR100
from model import MTAlexnet
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from torch.optim.lr_scheduler import ReduceLROnPlateau
from avalanche.training.supervised import Naive
from avalanche.evaluation.metrics import (forgetting_metrics, accuracy_metrics, bwt_metrics,
                                          loss_metrics, forward_transfer_metrics, disk_usage_metrics)
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from plugins.customise_plugin import GetLr, MyEarlyStoppingPlugin
from plugins.customise_logger import MyWandBLogger
from plugins.gnr import GNRPlugin

# test!!
def run(args):
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu")
    print(f'Emptying CUDA cache...')
    torch.cuda.empty_cache()

    # initiate model
    model = MTAlexnet(
        drop1=args.drop1, drop2=args.drop2)
    # Benchmark
    benchmark = SplitCIFAR100(n_experiences=args.n_experiences, seed=args.seed, return_task_id=True,
                              class_ids_from_zero_in_each_exp=True)
    train_stream = benchmark.train_stream
    test_stream = benchmark.test_stream

    # optimizer and criterion
    optimizer = SGD(model.parameters(), lr=args.init_lr)
    criterion = CrossEntropyLoss()

    # lr scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=1.0 / 3,
        patience=0,
        min_lr=args.min_lr,
    )
    lr_scheduler_plugin = LRSchedulerPlugin(scheduler, metric="val_loss")
    interactive_logger = InteractiveLogger()
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True, trained_experience=True),
        loss_metrics(experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        loggers=[
            interactive_logger,
        ],
        collect_all=True
    )
    gnr_plugin = GNRPlugin(alpha=args.alpha, r=args.r)
    get_lr = GetLr()
    strategy = Naive(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=args.batch_size,
        train_epochs=400,
        eval_mb_size=args.batch_size,
        device=device,
        evaluator=eval_plugin,
        plugins=[
            gnr_plugin,
            lr_scheduler_plugin,
            get_lr,
            MyEarlyStoppingPlugin(args.patience, 'test_stream', verbose=True),
        ],
        eval_every=args.eval_every,
    )
    print(f"Starting training with seed {args.seed}.")
    strategy.train(train_stream, eval_streams=[test_stream])
    print(f"Starting evaluation with seed {args.seed}.")
    strategy.eval(test_stream)


def assign_plugin_hyperparameters(args):
    """
    Assigns hyperparameters based on the selected plugin in args.
    """
    if args.plugin == 'gnr':
        args.alpha = 0.3
        args.r = 0.02
    else:
        raise ValueError(f"Unknown plugin specified in args.plugin: {args.plugin}")


def main(args):
    for plugin in ['gnr']:
        seeds = [1, 2, 3, 4, 5]  # 默认随机种子列表
        for seed in seeds:
            print(f"Running experiment with seed {seed}...")
            # loading hyperparameters
            args.seed = seed
            args.n_experiences = 10
            args.drop1 = 0.2
            args.drop2 = 0.2
            args.patience = 10
            args.eval_every = 5
            args.batch_size = 64
            args.init_lr = 0.1
            args.min_lr = 1e-3
            args.plugin = plugin
            assign_plugin_hyperparameters(args)

            args.project_name = f'TAI-baselines-C{args.n_experiences}'
            args.run_name = f"{args.plugin} seed_{seed}"

            print(args)
            run(args)
            wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    parsed_args = parser.parse_args()

    main(parsed_args)
