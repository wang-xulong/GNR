from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.benchmarks import benchmark_with_validation_stream
from avalanche.training.supervised import EWC
from avalanche.evaluation.metrics import forgetting_metrics, \
    accuracy_metrics, loss_metrics, timing_metrics, cpu_usage_metrics, \
    confusion_matrix_metrics, disk_usage_metrics, StreamBWT
from avalanche.logging import InteractiveLogger, WandBLogger
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin, EarlyStoppingPlugin

from alexnet import MTAlexnet
from utils import set_seed, set_seed_pt
from utils import BColors, myprint as print

import wandb

WANDB = False

def main(args):
    # Config
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )
    print(f"device: {device}", bcolor=BColors.OKBLUE)
    set_seed(1)
    set_seed_pt(42)
    appr_args = {
        'lr': 0.1,
        'lr_factor': 3,
        'lr_min': 1e-3,
        'epochs_max': 500,
        'patience_max': 30
    }

    model = MTAlexnet(num_classes=10, hidden_size=2048, drop1=0.321, drop2=0.4657, input_size=[3, 32, 32])

    # CL Benchmark Creation
    mean_3ch = [0.485, 0.456, 0.406]
    std_3ch = [0.229, 0.224, 0.225]
    _cifar100_train_transform = transforms.Compose(
        [
            transforms.Resize((32, 32), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean_3ch, std_3ch),
        ]
    )
    _cifar100_eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean_3ch, std_3ch)
        ]
    )
    scenario = SplitCIFAR100(
        n_experiences=10,  # 10 incremental experiences
        return_task_id=True,  # add task labels
        seed=1,
        shuffle=True,
        train_transform=_cifar100_train_transform,
        eval_transform=_cifar100_eval_transform
    )
    train_stream = scenario.train_stream
    test_stream = scenario.test_stream
    # print(f'train sample: {len(train_stream[0].dataset)}')
    # print(f'test sample: {len(test_stream[0].dataset)}')
    # Prepare for training & testing
    optimizer = SGD(model.parameters(), lr=appr_args['lr'])
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                               mode='min',
                                               factor=1.0 / appr_args['lr_factor'],
                                               patience=5,
                                               min_lr=appr_args['lr_min'],
                                               verbose=True
                                               )
    criterion = CrossEntropyLoss()

    # choose some loggers method
    loggers = list()
    loggers.append(InteractiveLogger())
    if WANDB:
        loggers.append(WandBLogger(project_name="avalanche C-10", run_name="ewc"))
    # choose some metrics and evaluation method
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=False, experience=True, stream=True),
        loss_metrics(minibatch=False, epoch=False, experience=True, stream=True),
        forgetting_metrics(experience=False, stream=True),
        # confusion_matrix_metrics(num_classes=scenario.n_classes, save_image=True,stream=True),
        StreamBWT(),
        loggers=loggers,
    )
    early_stopping_p = EarlyStoppingPlugin(appr_args['patience_max'], "test_stream")
    lr_scheduler_p = LRSchedulerPlugin(scheduler, metric="val_loss")

    # Choose a CL strategy
    strategy = EWC(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=64,
        train_epochs=appr_args['epochs_max'],
        eval_mb_size=64,
        device=device,
        evaluator=eval_plugin,
        ewc_lambda=0.4,
        eval_every=1,
        plugins=[early_stopping_p, lr_scheduler_p]
    )

    # train and test loop
    results = []
    for train_task, test_task in zip(train_stream, test_stream):
        print("Current Classes: ", train_task.classes_in_this_experience)
        # print(f"validation samples = {len(test_task.dataset)}")
        strategy.train(train_task, eval_streams=[test_task])
        print("eval the result")
        results.append(strategy.eval(test_stream))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    args = parser.parse_args()
    main(args)
    if WANDB:
        wandb.finish()
