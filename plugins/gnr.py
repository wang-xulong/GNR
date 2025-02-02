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
    获取模型中特征提取层的目标模块（Conv2d 和 Linear），排除分类层，
    并将这些层替换为 SPG 包装的版本。
    """
    conv_and_linear_layers = []

    # 遍历所有模块
    for name_module, module in model.named_modules():
        # 过滤条件：Conv2d 或 Linear 且路径在 conv_features 或 fc_features 内
        if isinstance(module, (nn.Conv2d, nn.Linear)) and \
                ('conv_features' in name_module or 'fc_features' in name_module):

            # 获取父模块和当前模块的名称
            parent_name = '.'.join(name_module.split('.')[:-1])
            child_name = name_module.split('.')[-1]

            # 获取父模块
            parent_module = model
            if parent_name:  # 如果有父模块（非根节点）
                for sub_name in parent_name.split('.'):
                    parent_module = getattr(parent_module, sub_name)

            # 将当前模块替换为 SPG 包装的模块
            wrapped_module = SPG(module)
            setattr(parent_module, child_name, wrapped_module)

            # 保存替换后的模块信息，包括模块路径和模型引用
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
        self.softmask_head = None  # SoftMaskHead 实例
        self.history_params = {}  # 保存历史任务的参数

    def before_training_exp(self, strategy, **kwargs):
        """
        在每次新任务开始训练前：
        - 包装特征提取层（仅第一个任务）。
        - 动态包装分类头。
        """
        model = strategy.model  # 获取当前模型

        idx_task = strategy.experience.current_experience  # 当前任务索引

        # 1. 包装特征提取层（仅在第一个任务中执行）
        if idx_task == 0:
            print("Initializing SPG for the first task...")
            self.feature_extraction_spg = wrap_feature_extraction_layers(model)  # 替换 Conv2d 和 Linear 层为 SPG 包装的版本

        # # 2. 包装分类头（动态增长）
        # if not hasattr(model, 'classifier') or not isinstance(model.classifier, MultiHeadClassifier):
        #     raise ValueError("The model does not have a MultiHeadClassifier.")
        #
        # # 提取分类头（ModuleDict）和特征提取层的 SPG 信息
        # list__spg = [spg_layer for _, spg_layer in self.feature_extraction_spg]

        # 实例化 SoftMaskHead
        # if self.softmask_head is None:
        #     self.softmask_head = SoftMaskHead(list__spg)
        #     print("Initialized SoftMaskHead for all tasks.")

    # # 版本5 增加 \delta w,   \delta w 从梯度变化改为参数变化
    # def after_backward(self, strategy: Template, *args, **kwargs):
    #     r = 0.02
    #     alpha = 0.8
    #     lambda_ = r * alpha
    #     epsilon = 1e-6  # 防止除零的小常量
    #     model = strategy.model
    #     idx_task = strategy.experience.current_experience  # 获取当前任务索引
    #     x = strategy.mb_x  # 当前 mini-batch 输入
    #     target = strategy.mb_y  # 当前 mini-batch 目标
    #     lr = strategy.optimizer.state_dict()['param_groups'][0]['lr']
    #
    #     # 内嵌目标层提取函数
    #     def get_target_layers():
    #         """
    #         获取所有目标层：Conv2d 和 Linear，以及当前任务的 MultiHeadClassifier 分类头。
    #         返回一个列表，其中包含所有目标层。
    #         """
    #         target_layers = {}
    #
    #         for name, module in model.named_modules():
    #             # 判断是否属于 conv_features 或 fc_features
    #             if isinstance(module, (nn.Conv2d, nn.Linear)) and \
    #                     ('conv_features' in name or 'fc_features' in name):
    #                 target_layers[name] = module
    #             # 判断是否为 MultiHeadClassifier 中的分类头
    #             elif name == 'classifier' and isinstance(module, MultiHeadClassifier) and idx_task is not None:
    #                 classifier = module.classifiers[str(idx_task)].classifier
    #                 target_layers[f'classifier.classifiers.{idx_task}.classifier'] = classifier
    #                 return target_layers
    #
    #         return target_layers
    #
    #     # Step 1: 提取原始梯度
    #     original_grads = []
    #     with torch.no_grad():
    #         target_layers = get_target_layers().items()
    #         for (_, module) in target_layers:
    #             for param in module.parameters():
    #                 if param.grad is not None:
    #                     original_grads.append(param.grad.clone().detach())
    #
    #     # Step 2: 扰动模型参数
    #     count = 0
    #     with torch.no_grad():
    #         target_layers = get_target_layers().items()
    #         for _, module in target_layers:
    #             for param in module.parameters():
    #                 if count < len(original_grads):
    #                     grad = original_grads[count]
    #                     param.add_(r * grad / (torch.norm(grad) + epsilon))
    #                     count += 1
    #
    #     # Step 3: 前向传播和反向传播（扰动后的模型）
    #     model.zero_grad()
    #     perturbed_logits = model(x=x, task_labels=idx_task)
    #     lossfunc = nn.CrossEntropyLoss()
    #     perturbed_loss = lossfunc(perturbed_logits, target)
    #     perturbed_loss.backward()
    #
    #     # Step 4: 计算插值梯度并还原参数
    #     final_grads = []
    #     count = 0
    #
    #     #  add new
    #     prev_task = idx_task - 1
    #     history_params = self.history_params.get(prev_task, {})
    #
    #     with torch.no_grad():
    #         target_layers = get_target_layers()
    #         for module_name, module in target_layers.items():
    #             for param_name, param in module.named_parameters():
    #                 if count < len(original_grads):
    #                     perturbed_grad = param.grad.clone().detach()
    #                     orig_grad = original_grads[count]
    #                     # # add new
    #                     # 首任务，无历史参数
    #                     if idx_task == 0:
    #                         interpolated_grad = (1 - alpha) * orig_grad + alpha * perturbed_grad
    #                     else:
    #                         if 'conv_features' in module_name or 'fc_features' in module_name:
    #                             # 特征层，获取历史参数
    #                             key = f"{module_name}.{param_name}"
    #                         elif 'classifier.classifiers' in module_name:
    #                             # 分类层，动态生成旧任务分类头的键
    #                             key = f"classifier.classifiers.{prev_task}.classifier.{param_name}"
    #                         else:
    #                             raise ValueError(f"Unexpected module name: {module_name}")
    #
    #                         # 调用函数计算插值梯度
    #                         delta_w = param - history_params[key]
    #                         interpolated_grad = (1 - alpha * torch.pow(delta_w, 2)) * orig_grad + \
    #                                             alpha * perturbed_grad * torch.pow(delta_w, 2) \
    #                                             + 2 * lambda_ * torch.norm(orig_grad) * delta_w
    #                     # # add new
    #                     final_grads.append(interpolated_grad)
    #                     # 恢复参数到原始状态
    #                     param.sub_(r * orig_grad / (torch.norm(orig_grad) + epsilon))
    #                     count += 1
    #
    #     # Step 5: 应用插值后的梯度
    #     count = 0
    #     target_layers = get_target_layers()
    #     for _, module in target_layers.items():
    #         for param in module.parameters():
    #             if count < len(final_grads):
    #                 param.grad = final_grads[count]
    #                 count += 1
    #
    #     return final_grads

    # # 版本3 增加 \delta w,   \delta w = -( lr * grad_w )
    # def after_backward(self, strategy: Template, *args, **kwargs):
    #     r = 0.02
    #     alpha = 0.8
    #     lambda_ = r * alpha
    #     epsilon = 1e-6  # 防止除零的小常量
    #     model = strategy.model
    #     idx_task = strategy.experience.current_experience  # 获取当前任务索引
    #     x = strategy.mb_x  # 当前 mini-batch 输入
    #     target = strategy.mb_y  # 当前 mini-batch 目标
    #     lr = strategy.optimizer.state_dict()['param_groups'][0]['lr']
    #
    #     # 内嵌目标层提取函数
    #     def get_target_layers():
    #         """
    #         获取所有目标层：Conv2d 和 Linear，以及当前任务的 MultiHeadClassifier 分类头。
    #         返回一个列表，其中包含所有目标层。
    #         """
    #         target_layers = []
    #
    #         for name, module in model.named_modules():
    #             # 判断是否属于 conv_features 或 fc_features
    #             if isinstance(module, (nn.Conv2d, nn.Linear)) and \
    #                     ('conv_features' in name or 'fc_features' in name):
    #                 target_layers.append(module)
    #             # 判断是否为 MultiHeadClassifier 中的分类头
    #             elif name == 'classifier' and isinstance(module, MultiHeadClassifier) and idx_task is not None:
    #                 classifier = module.classifiers[str(idx_task)].classifier
    #                 target_layers.append(classifier)
    #                 return target_layers
    #
    #         return target_layers
    #
    #     # Step 1: 提取原始梯度
    #     original_grads = []
    #     with torch.no_grad():
    #         target_layers = get_target_layers()
    #         for module in target_layers:
    #             for param in module.parameters():
    #                 if param.grad is not None:
    #                     original_grads.append(param.grad.clone().detach())
    #
    #     # Step 2: 扰动模型参数
    #     count = 0
    #     with torch.no_grad():
    #         target_layers = get_target_layers()
    #         for module in target_layers:
    #             for param in module.parameters():
    #                 if count < len(original_grads):
    #                     grad = original_grads[count]
    #                     param.add_(r * grad / (torch.norm(grad) + epsilon))
    #                     count += 1
    #
    #     # Step 3: 前向传播和反向传播（扰动后的模型）
    #     model.zero_grad()
    #     perturbed_logits = model(x=x, task_labels=idx_task)
    #     lossfunc = nn.CrossEntropyLoss()
    #     perturbed_loss = lossfunc(perturbed_logits, target)
    #     perturbed_loss.backward()
    #
    #     # Step 4: 计算插值梯度并还原参数
    #     final_grads = []
    #     count = 0
    #     with torch.no_grad():
    #         target_layers = get_target_layers()
    #         for module in target_layers:
    #             for param in module.parameters():
    #                 if count < len(original_grads):
    #                     perturbed_grad = param.grad.clone().detach()
    #                     orig_grad = original_grads[count]
    #                     # add new
    #                     delta_w = - lr * orig_grad
    #                     interpolated_grad = (1 - alpha * torch.pow(delta_w, 2)) * orig_grad + \
    #                                         alpha * perturbed_grad * torch.pow(delta_w, 2) \
    #                                         + 2 * lambda_ * torch.norm(orig_grad) * delta_w
    #                     final_grads.append(interpolated_grad)
    #                     # 恢复参数到原始状态
    #                     param.sub_(r * orig_grad / (torch.norm(orig_grad) + epsilon))
    #                     count += 1
    #
    #     # Step 5: 应用插值后的梯度
    #     count = 0
    #     target_layers = get_target_layers()
    #     for module in target_layers:
    #         for param in module.parameters():
    #             if count < len(final_grads):
    #                 param.grad = final_grads[count]
    #                 count += 1
    #
    #     return final_grads

    # # 版本2 增加 \delta w,   w_prep= rv
    # def after_backward(self, strategy: Template, *args, **kwargs):
    #     r = 0.02
    #     alpha = 0.8
    #     lambda_ = r * alpha
    #     epsilon = 1e-6  # 防止除零的小常量
    #     model = strategy.model
    #     idx_task = strategy.experience.current_experience  # 获取当前任务索引
    #     x = strategy.mb_x  # 当前 mini-batch 输入
    #     target = strategy.mb_y  # 当前 mini-batch 目标
    #
    #     # 内嵌目标层提取函数
    #     def get_target_layers():
    #         """
    #         获取所有目标层：Conv2d 和 Linear，以及当前任务的 MultiHeadClassifier 分类头。
    #         返回一个列表，其中包含所有目标层。
    #         """
    #         target_layers = []
    #
    #         for name, module in model.named_modules():
    #             # 判断是否属于 conv_features 或 fc_features
    #             if isinstance(module, (nn.Conv2d, nn.Linear)) and \
    #                     ('conv_features' in name or 'fc_features' in name):
    #                 target_layers.append(module)
    #             # 判断是否为 MultiHeadClassifier 中的分类头
    #             elif name == 'classifier' and isinstance(module, MultiHeadClassifier) and idx_task is not None:
    #                 classifier = module.classifiers[str(idx_task)].classifier
    #                 target_layers.append(classifier)
    #                 return target_layers
    #
    #         return target_layers
    #
    #     # Step 1: 提取原始梯度
    #     original_grads = []
    #     with torch.no_grad():
    #         target_layers = get_target_layers()
    #         for module in target_layers:
    #             for param in module.parameters():
    #                 if param.grad is not None:
    #                     original_grads.append(param.grad.clone().detach())
    #
    #     # Step 2: 扰动模型参数
    #     count = 0
    #     with torch.no_grad():
    #         target_layers = get_target_layers()
    #         for module in target_layers:
    #             for param in module.parameters():
    #                 if count < len(original_grads):
    #                     grad = original_grads[count]
    #                     param.add_(r * grad / (torch.norm(grad) + epsilon))
    #                     count += 1
    #
    #     # Step 3: 前向传播和反向传播（扰动后的模型）
    #     model.zero_grad()
    #     perturbed_logits = model(x=x, task_labels=idx_task)
    #     lossfunc = nn.CrossEntropyLoss()
    #     perturbed_loss = lossfunc(perturbed_logits, target)
    #     perturbed_loss.backward()
    #
    #     # Step 4: 计算插值梯度并还原参数
    #     final_grads = []
    #     count = 0
    #     with torch.no_grad():
    #         target_layers = get_target_layers()
    #         for module in target_layers:
    #             for param in module.parameters():
    #                 if count < len(original_grads):
    #                     perturbed_grad = param.grad.clone().detach()
    #                     orig_grad = original_grads[count]
    #                     # add new
    #                     delta_w = r * orig_grad / (torch.norm(orig_grad) + epsilon)
    #                     interpolated_grad = (1 - alpha * torch.pow(delta_w, 2)) * orig_grad + \
    #                                         alpha * perturbed_grad * torch.pow(delta_w, 2) \
    #                                         + 2 * lambda_ * torch.norm(orig_grad) * delta_w
    #                     final_grads.append(interpolated_grad)
    #                     # 恢复参数到原始状态
    #                     param.sub_(r * orig_grad / (torch.norm(orig_grad) + epsilon))
    #                     count += 1
    #
    #     # Step 5: 应用插值后的梯度
    #     count = 0
    #     target_layers = get_target_layers()
    #     for module in target_layers:
    #         for param in module.parameters():
    #             if count < len(final_grads):
    #                 param.grad = final_grads[count]
    #                 count += 1
    #
    #     return final_grads

    # 版本1 ||g||
    def after_backward(self, strategy: Template, *args, **kwargs):
        alpha = self.alpha
        r = self.r
        epsilon = 1e-6  # 防止除零的小常量
        model = strategy.model
        idx_task = strategy.experience.current_experience  # 获取当前任务索引
        x = strategy.mb_x  # 当前 mini-batch 输入
        target = strategy.mb_y  # 当前 mini-batch 目标

        # 内嵌目标层提取函数
        def get_target_layers():
            """
            获取所有目标层：Conv2d 和 Linear，以及当前任务的 MultiHeadClassifier 分类头。
            返回一个列表，其中包含所有目标层。
            """
            target_layers = []

            for name, module in model.named_modules():
                # 判断是否属于 conv_features 或 fc_features
                if isinstance(module, (nn.Conv2d, nn.Linear)) and \
                        ('conv_features' in name or 'fc_features' in name):
                    target_layers.append(module)
                # 判断是否为 MultiHeadClassifier 中的分类头
                elif name == 'classifier' and isinstance(module, MultiHeadClassifier) and idx_task is not None:
                    classifier = module.classifiers[str(idx_task)].classifier
                    target_layers.append(classifier)
                    return target_layers

            return target_layers

        # Step 1: 提取原始梯度
        original_grads = []
        with torch.no_grad():
            target_layers = get_target_layers()
            for module in target_layers:
                for param in module.parameters():
                    if param.grad is not None:
                        original_grads.append(param.grad.clone().detach())

        # Step 2: 扰动模型参数
        count = 0
        with torch.no_grad():
            target_layers = get_target_layers()
            for module in target_layers:
                for param in module.parameters():
                    if count < len(original_grads):
                        grad = original_grads[count]
                        param.add_(r * grad / (torch.norm(grad) + epsilon))
                        count += 1

        # Step 3: 前向传播和反向传播（扰动后的模型）
        model.zero_grad()
        perturbed_logits = model(x=x, task_labels=idx_task)
        lossfunc = nn.CrossEntropyLoss()
        perturbed_loss = lossfunc(perturbed_logits, target)
        perturbed_loss.backward()

        # Step 4: 计算插值梯度并还原参数
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
                        # 恢复参数到原始状态
                        param.sub_(r * orig_grad / (torch.norm(orig_grad) + epsilon))
                        count += 1

        # Step 5: 应用插值后的梯度
        count = 0
        target_layers = get_target_layers()
        for module in target_layers:
            for param in module.parameters():
                if count < len(final_grads):
                    param.grad = final_grads[count]
                    count += 1

        return final_grads

    def before_update(self, strategy, **kwargs):
        """
        在每次参数更新之前，调用 softmask 方法，确保掩码修改 strategy.model 的梯度。
        """
        idx_task = strategy.experience.current_experience  # 获取当前任务索引
        model = strategy.model  # 获取当前模型
        if idx_task == 0:
            return None  # 第一个任务不用做梯度修改

        # 1. 对特征提取层调用 softmask
        for name_module, spg_module in model.named_modules():
            if isinstance(spg_module, SPG):
                spg_module.softmask(idx_task)  # 调用 SPG 模块的 softmask 方法

        # 0. 检查梯度范数是否过大，并裁剪
        max_grad_norm = 10000  # 设定最大梯度范数
        total_norm = clip_grad_norm_(strategy.model.parameters(), max_grad_norm)
        # print(f"Gradient norm before softmask: {total_norm:.2f}")
        if total_norm > max_grad_norm:
            print(f"Gradient norm clipped to {max_grad_norm}")

        # 2. 对分类层调用 softmask
        # 注意，去除掩码分类层的梯度
        # if self.softmask_head is None:
        #     raise ValueError("SoftMaskHead is not initialized. Please check before_training_exp.")
        #
        # for name_module, classifier_head in strategy.model.classifier.classifiers.items():
        #     if isinstance(classifier_head, IncrementalClassifier):
        #         # print(f"Applying softmask to classifier head: {name_module}")
        #         self.softmask_head.softmask(idx_task, classifier_head.classifier)  # 调用 SoftMaskHead 的 softmask 方法

    def after_training_exp(self, strategy, **kwargs):
        """
        每个任务训练完成后，计算参数重要性。
        """
        # 遍历模型参数，计算重要性（示例基于梯度的绝对值）
        print("[gnr] protect current task's parameters")
        # 获取当前任务索引和数据加载器
        idx_task = strategy.experience.current_experience

        dl = strategy.dataloader

        range_tasks = range(idx_task + 1)

        # 确保模型处于训练模式
        strategy.model.train()

        # 遍历任务范围计算重要性
        for t in range_tasks:
            strategy.model.zero_grad()

            # 遍历数据加载器中的批次
            for x, y, tid in dl:
                x = x.to(strategy.device)
                y = y.to(strategy.device)

                # 前向传播
                out = strategy.model.__call__(x=x, task_labels=t)

                # 根据当前任务选择不同的损失函数
                if t == idx_task:
                    lossfunc = nn.CrossEntropyLoss()
                else:
                    lossfunc = OtherTasksLoss()

                # 计算损失并反向传播
                loss = lossfunc(out, y)
                loss.backward()
            # endfor

            # 调用 SPG 模块注册梯度
            self.spg_register_grad(idx_task=idx_task, t=t, model=strategy.model)
        # endfor

        # 调用 SPG 模块计算掩码
        self.spg_compute_mask(idx_task=idx_task, model=strategy.model)

        # 在每个任务完成后保存参数
        current_task = strategy.experience.current_experience
        self.history_params[current_task] = {
            name: param.clone().detach()
            for name, param in strategy.model.named_parameters()
        }

    def spg_register_grad(self, idx_task, t, model):
        """
        遍历模型的所有 SPG 包装的模块，注册梯度。
        """
        # for name_module, module in self.target_model:
        for name_module, module in model.named_modules():
            if isinstance(module, SPG):
                grads = {}
                for name_param, param in module.target_module.named_parameters():
                    grad = param.grad.data.clone().cpu() if param.grad is not None else 0
                    grads[name_param] = grads.get(name_param, 0) + grad

                # 将聚合的梯度注册到 SPG 模块
                module.register_grad(idx_task=idx_task, t=t, grads=grads)

    def spg_compute_mask(self, idx_task, model):
        """
        调用 SPG 模块以计算重要性掩码。
        """
        for name_module, module in model.named_modules():
            if isinstance(module, SPG):
                module.compute_mask(idx_task=idx_task)
                # # 更新掩码（可选逻辑：保留重要性高于阈值的部分）
                # threshold = importance * 0.1
                # self.masks[name] = (torch.abs(param.grad) >= threshold).float()
        # endfor
    # enddef
