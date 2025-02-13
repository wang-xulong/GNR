import torch
from avalanche.core import Template
from avalanche.training.plugins import SupervisedPlugin
from torch import nn
from plugins.other_tasks_loss import OtherTasksLoss
from plugins.spg import SPG
from torch.nn.utils import clip_grad_norm_
from avalanche.models.dynamic_modules import MultiHeadClassifier


def wrap_feature_extraction_layers(model):
    """
    Obtain the target modules (Conv2d and Linear) of the feature extraction layer in the model, exclude the
    classification layer, and replace these layers with the SPG-packaged version.
    """
    conv_and_linear_layers = []

    for name_module, module in model.named_modules():
        #  Filter criteria: Conv2d or Linear and paths in conv_features or fc_features
        if isinstance(module, (nn.Conv2d, nn.Linear)) and \
                ('conv_features' in name_module or 'fc_features' in name_module):

            # Gets the name of the parent and current module
            parent_name = '.'.join(name_module.split('.')[:-1])
            child_name = name_module.split('.')[-1]

            # Get parent module
            parent_module = model
            if parent_name:
                for sub_name in parent_name.split('.'):
                    parent_module = getattr(parent_module, sub_name)

            wrapped_module = SPG(module)
            setattr(parent_module, child_name, wrapped_module)

            conv_and_linear_layers.append((name_module, getattr(parent_module, child_name)))
            print(f"Replaced {name_module} with SPG")

    return conv_and_linear_layers


class GNRPlugin(SupervisedPlugin):
    def __init__(
            self,
            alpha: float = 0.8,
            r: float = 0.02,
    ):
        super().__init__()
        self.alpha = alpha
        self.r = r
        self.feature_extraction_spg = None
        self.history_params = {}

    def before_training_exp(self, strategy, **kwargs):

        model = strategy.model

        idx_task = strategy.experience.current_experience

        if idx_task == 0:
            print("Initializing softmask for the first task...")
            self.feature_extraction_spg = wrap_feature_extraction_layers(model)

    def after_backward(self, strategy: Template, *args, **kwargs):
        alpha = self.alpha
        r = self.r
        epsilon = 1e-6
        model = strategy.model
        idx_task = strategy.experience.current_experience
        x = strategy.mb_x
        target = strategy.mb_y

        def get_target_layers():

            target_layers = []

            for name, module in model.named_modules():

                if isinstance(module, (nn.Conv2d, nn.Linear)) and \
                        ('conv_features' in name or 'fc_features' in name):
                    target_layers.append(module)
                elif name == 'classifier' and isinstance(module, MultiHeadClassifier) and idx_task is not None:
                    classifier = module.classifiers[str(idx_task)].classifier
                    target_layers.append(classifier)
                    return target_layers

            return target_layers

        original_grads = []
        with torch.no_grad():
            target_layers = get_target_layers()
            for module in target_layers:
                for param in module.parameters():
                    if param.grad is not None:
                        original_grads.append(param.grad.clone().detach())

        count = 0
        with torch.no_grad():
            target_layers = get_target_layers()
            for module in target_layers:
                for param in module.parameters():
                    if count < len(original_grads):
                        grad = original_grads[count]
                        param.add_(r * grad / (torch.norm(grad) + epsilon))
                        count += 1

        model.zero_grad()
        perturbed_logits = model(x=x, task_labels=idx_task)
        lossfunc = nn.CrossEntropyLoss()
        perturbed_loss = lossfunc(perturbed_logits, target)
        perturbed_loss.backward()

        final_grads = []
        count = 0
        with torch.no_grad():
            target_layers = get_target_layers()
            for module in target_layers:
                for param in module.parameters():
                    if count < len(original_grads):
                        perturbed_grad = param.grad.clone().detach()
                        orig_grad = original_grads[count]
                        interpolated_grad = (1 - alpha) * orig_grad + alpha * perturbed_grad
                        final_grads.append(interpolated_grad)
                        param.sub_(r * orig_grad / (torch.norm(orig_grad) + epsilon))
                        count += 1

        count = 0
        target_layers = get_target_layers()
        for module in target_layers:
            for param in module.parameters():
                if count < len(final_grads):
                    param.grad = final_grads[count]
                    count += 1

        return final_grads

    def before_update(self, strategy, **kwargs):

        idx_task = strategy.experience.current_experience
        model = strategy.model
        if idx_task == 0:
            return None

        for name_module, spg_module in model.named_modules():
            if isinstance(spg_module, SPG):
                spg_module.softmask(idx_task)

        max_grad_norm = 10000
        total_norm = clip_grad_norm_(strategy.model.parameters(), max_grad_norm)
        if total_norm > max_grad_norm:
            print(f"Gradient norm clipped to {max_grad_norm}")

    def after_training_exp(self, strategy, **kwargs):
        print("[gnr] protect current task's parameters")

        idx_task = strategy.experience.current_experience

        dl = strategy.dataloader

        range_tasks = range(idx_task + 1)

        strategy.model.train()

        for t in range_tasks:
            strategy.model.zero_grad()

            for x, y, tid in dl:
                x = x.to(strategy.device)
                y = y.to(strategy.device)

                out = strategy.model.__call__(x=x, task_labels=t)

                if t == idx_task:
                    lossfunc = nn.CrossEntropyLoss()
                else:
                    lossfunc = OtherTasksLoss()

                loss = lossfunc(out, y)
                loss.backward()
            # endfor

            self.spg_register_grad(idx_task=idx_task, t=t, model=strategy.model)
        # endfor

        self.spg_compute_mask(idx_task=idx_task, model=strategy.model)

        current_task = strategy.experience.current_experience
        self.history_params[current_task] = {
            name: param.clone().detach()
            for name, param in strategy.model.named_parameters()
        }

    def spg_register_grad(self, idx_task, t, model):

        for name_module, module in model.named_modules():
            if isinstance(module, SPG):
                grads = {}
                for name_param, param in module.target_module.named_parameters():
                    grad = param.grad.data.clone().cpu() if param.grad is not None else 0
                    grads[name_param] = grads.get(name_param, 0) + grad

                module.register_grad(idx_task=idx_task, t=t, grads=grads)
    def spg_compute_mask(self, idx_task, model):

        for name_module, module in model.named_modules():
            if isinstance(module, SPG):
                module.compute_mask(idx_task=idx_task)
        # endfor
    # enddef
