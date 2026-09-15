# -*- coding: UTF-8 -*-
"""
DERS (Dynamically Expandable Representation with Subject-based evaluation) Agent.

Implements the DER algorithm from:
"DER: Dynamically Expandable Representation for Class Incremental Learning"
https://openaccess.thecvf.com/content/CVPR2021/papers/Shekhar_DER_Dynamically_Expandable_Representation_for_Class_Incremental_Learning_CVPR_2021_paper.pdf

DERS is a standalone implementation that:
1. Supports dynamic branch expansion for each task
2. Concatenates features from all branches
3. Uses auxiliary classifier for new classes (optional)
4. Freezes old branches to prevent forgetting
5. Provides subject-aware evaluation for BCI cross-subject protocol

Key components (from DER paper):
- Dynamic Network Expansion: adds new branches for each task
- Feature Concatenation: concatenates features from all branches
- Auxiliary Classifier: classifies only new classes using latest branch features
- Old Branch Freezing: freezes historical branches during training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from agents.base import BaseLearner
from utils.setup_elements import n_classes
from utils.utils import BinaryCrossEntropy
from utils.data import Dataloader_from_numpy


class DERS(BaseLearner):
    """
    DERS: Dynamically Expandable Representation with Subject-based evaluation.

    A standalone implementation of DER with explicit support for BCI cross-subject evaluation.

    Usage:
        python main_config_bci.py --agent DERS --data mi --encoder MI_EEGNet \
            --der_dynamic True --der_aux True --epochs 50
    """

    def __init__(self, model, args):
        super(DERS, self).__init__(model, args)

        # DER/DERS-specific parameters
        self.der_dynamic = getattr(args, 'der_dynamic', False)
        self.der_aux = getattr(args, 'der_aux', True)
        self.aux_classifier = None

        # Track increments for evaluation (following original DER)
        self._increments = []  # Number of classes per task
        self._task_classes = []  # Actual class labels for each task

        # Subject-aware tracking (DERS extension)
        self.subject_accuracies = {}

        # Initialize confusion matrix attributes
        if not hasattr(self, 'y_pred_cf'):
            self.y_pred_cf = []
        if not hasattr(self, 'y_true_cf'):
            self.y_true_cf = []

        print('DERS Mode: Dynamic={}, Aux Classifier={}'.format(
            self.der_dynamic, self.der_aux))

        # Print configuration summary
        if self.der_dynamic:
            print('=> DERS dynamic expansion enabled: branches will be added for each new task')
            print('=> Old branches will be frozen, only new branch + head will be optimized')
        if self.der_aux:
            print('=> DERS auxiliary classifier enabled: auxiliary loss for new classes')

    def before_task(self, y_train):
        """
        Prepare for a new task:
        1. Update task_now and classes_in_task
        2. Track class increments for evaluation
        3. If dynamic expansion: add new branch for task 1+ and update head
        4. Freeze old branches and update optimizer
        5. Initialize auxiliary classifier if enabled
        """
        if self.der_dynamic:
            # Manual setup: update task_now and classes_in_task
            self.task_now += 1
            # Use sorted() to ensure consistent class order
            self.classes_in_task = sorted(list(set(y_train.tolist())))
            n_new_classes = len(self.classes_in_task)
            assert n_new_classes >= 1, "A task must contain more than 1 class"

            # Track class increments
            self._increments.append(n_new_classes)
            self._task_classes.append(self.classes_in_task)

            if self.args.verbose:
                print('\n--> Task {}: {} classes in total'.format(
                    self.task_now, len(self.learned_classes + self.classes_in_task)))

            # Add new branch and update head for task 1+
            if self.task_now > 0:
                print('[DERS] Task {}: Adding new branch...'.format(self.task_now))
                self._add_new_branch()
                print('[DERS] Task {}: Total branches now: {}'.format(
                    self.task_now, self.get_n_branches()))

                # Update head for expanded features and new classes
                if hasattr(self.model, 'update_head_for_der_branch_and_classes'):
                    self.model.update_head_for_der_branch_and_classes(n_new_classes, self.task_now)
                    print('[DERS] Task {}: Updated head for expanded features and {} new classes'.format(
                        self.task_now, n_new_classes))
                else:
                    # Fallback: expand features then expand classes
                    if hasattr(self.model, 'update_head_for_der_branch'):
                        self.model.update_head_for_der_branch()
                        print('[DERS] Task {}: Updated head for expanded features'.format(self.task_now))
                    self.model.update_head(n_new=n_new_classes, task_now=self.task_now)
                    print('[DERS] Task {}: Expanded head for {} new classes'.format(
                        self.task_now, n_new_classes))

                # Freeze old branches
                self._freeze_old_branches()
                print('[DERS] Task {}: Frozen old branches'.format(self.task_now))

                # Update optimizer to only optimize new branch + head
                self._update_optimizer()
                print('[DERS] Task {}: Updated optimizer (new branch + head only)'.format(self.task_now))

            # Initialize criterion
            if self.args.criterion == 'BCE':
                self.criterion = BinaryCrossEntropy(dim=self.model.head.out_features, device=self.device)
            else:
                self.criterion = torch.nn.CrossEntropyLoss()

            # Initialize auxiliary classifier for new classes (task 1+)
            if self.der_aux and self.task_now > 0:
                print('[DERS] Task {}: Initializing auxiliary classifier for {} new classes'.format(
                    self.task_now, n_new_classes))
                self._init_aux_classifier()
        else:
            # Non-dynamic mode: use standard parent behavior
            super().before_task(y_train)
            self._increments.append(len(self.classes_in_task))

    def _add_new_branch(self):
        """
        Add a new branch to the network.
        For DER_Net wrapper, call its add_branch method.
        """
        if hasattr(self.model, 'add_branch'):
            self.model.add_branch()
        elif hasattr(self.model.encoder, 'branches'):
            self.model.encoder.add_branch()
        else:
            print('Warning: Model does not support dynamic branch addition')

    def get_n_branches(self):
        """Get the current number of branches."""
        if hasattr(self.model, 'n_branches'):
            return self.model.n_branches
        elif hasattr(self.model.encoder, 'n_branches'):
            return self.model.encoder.n_branches
        return 1

    def _freeze_old_branches(self):
        """
        Freeze all old branches, keeping only the last branch trainable.
        """
        if hasattr(self.model, 'freeze_branches'):
            self.model.freeze_branches(exclude_last=True)
        elif hasattr(self.model.encoder, 'freeze_branches'):
            self.model.encoder.freeze_branches(exclude_last=True)
        elif hasattr(self.model.encoder, 'branches'):
            for i in range(len(self.model.encoder.branches) - 1):
                for p in self.model.encoder.branches[i].parameters():
                    p.requires_grad = False
        else:
            print('Warning: Model does not support branch freezing')

    def _update_optimizer(self):
        """
        Update optimizer to only optimize the new branch and classification head.
        """
        from utils.optimizer import set_optimizer
        params_to_optimize = []

        if hasattr(self.model, 'branches'):
            params_to_optimize.extend(self.model.branches[-1].parameters())
        elif hasattr(self.model.encoder, 'branches'):
            params_to_optimize.extend(self.model.encoder.branches[-1].parameters())
        else:
            params_to_optimize.extend(self.model.parameters())

        # Always optimize the classification head
        params_to_optimize.extend(self.model.head.parameters())

        # Include auxiliary classifier if enabled
        if self.der_aux and self.aux_classifier:
            params_to_optimize.extend(self.aux_classifier.parameters())

        self.optimizer = set_optimizer(self.model, self.args, params_to_optimize=params_to_optimize)

    def train(self):
        """
        Set model to training mode with explicit branch mode control.

        Following the reference implementation (DER/inclearn/models/incmodel.py:93-101):
        - Set model to train mode
        - Only the last (newest) branch is in train mode
        - All old branches are in eval mode
        """
        if self.der_dynamic:
            self.model.train()
            if hasattr(self.model, 'branches'):
                if len(self.model.branches) > 1:
                    for i in range(len(self.model.branches) - 1):
                        self.model.branches[i].eval()
                    self.model.branches[-1].train()
            elif hasattr(self.model.encoder, 'branches'):
                if len(self.model.encoder.branches) > 1:
                    for i in range(len(self.model.encoder.branches) - 1):
                        self.model.encoder.branches[i].eval()
                    self.model.encoder.branches[-1].train()
        else:
            self.model.train()

    def _init_aux_classifier(self):
        """
        Initialize auxiliary classifier that only classifies new classes.
        Uses features from the latest branch only.

        The auxiliary classifier has n_new_classes + 1 outputs:
        - Class 0: background class (for old classes)
        - Class 1 to n_new_classes: new classes (relative indices)
        """
        # Determine input features (single branch feature dim)
        if self.der_dynamic and hasattr(self.model.encoder, 'get_single_branch_feature_dim'):
            aux_in_features = self.model.encoder.get_single_branch_feature_dim()
        elif hasattr(self.model, 'feature_dim'):
            aux_in_features = self.model.feature_dim
        elif hasattr(self.model.encoder, 'feature_dim'):
            aux_in_features = self.model.encoder.feature_dim
        else:
            aux_in_features = self.args.feature_dim

        n_new_classes = len(self.classes_in_task)

        # Auxiliary classifier: branch features -> n_new_classes + 1
        self.aux_classifier = nn.Linear(aux_in_features, n_new_classes + 1)
        self.aux_classifier.to(self.device)
        # Initialize weights
        nn.init.xavier_uniform_(self.aux_classifier.weight)
        nn.init.zeros_(self.aux_classifier.bias)

    def train_epoch(self, dataloader, epoch):
        """
        DERS training loop (NO Experience Replay):
        1. Standard classification loss on current task data
        2. Auxiliary classification loss (if enabled) on new classes

        No buffer retrieval, no logits distillation, no DER++ losses.
        """
        total = 0
        correct = 0
        epoch_loss = 0

        self.train()

        for batch_id, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            total += y.size(0)

            if y.size == 1:
                y.unsqueeze()

            self.optimizer.zero_grad()
            loss = 0

            # Main classification loss on current samples
            outputs = self.model(x)
            loss += self.criterion(outputs, y)

            # Auxiliary classifier loss (only for new classes)
            if self.der_aux and self.aux_classifier and self.task_now > 0:
                # Get features from the last branch only
                if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'branches'):
                    last_branch_feature = self.model.encoder.branches[-1](x)
                elif hasattr(self.model, 'branches'):
                    last_branch_feature = self.model.branches[-1](x)
                else:
                    last_branch_feature = self.model.encoder(x)

                aux_outputs = self.aux_classifier(last_branch_feature)

                # Create auxiliary targets: old classes -> background class (index 0)
                # new classes -> relative index [1, n_new_classes]
                old_n_classes = sum(self._increments[:-1])
                aux_targets = y.clone()
                mask_old = y < old_n_classes
                mask_new = ~mask_old
                aux_targets[mask_old] = 0
                n_new_classes = len(self.classes_in_task)
                aux_targets[mask_new] = (y[mask_new] - old_n_classes + 1)

                loss += F.cross_entropy(aux_outputs, aux_targets)

            loss.backward()
            self.optimizer_step(epoch=epoch)

            epoch_loss += loss
            prediction = torch.argmax(outputs, dim=1)
            correct += prediction.eq(y).sum().item()

        epoch_acc = 100. * (correct / total)
        epoch_loss /= (batch_id + 1)

        return epoch_loss, epoch_acc

    def after_task(self, x_train, y_train):
        """
        DERS post-task processing (NO buffer update):
        1. Save model checkpoint
        2. Compute NCM classifier means if enabled
        3. Deep copy model as teacher if using KD

        Note: Unlike base class, we don't reload checkpoint because:
        - DER's head expands with each task, making old checkpoints incompatible
        - We need to keep the current model state for all learned tasks
        """
        self.learned_classes.extend(self.classes_in_task)
        # Save current model state - checkpoint for NEXT task
        torch.save(self.model.state_dict(), self.ckpt_path)

        # NCM classifier (if enabled)
        if self.ncm_classifier:
            self.means_of_exemplars = self._compute_ncm_means(x_train, y_train)

        # Store teacher model for knowledge distillation
        if self.use_kd:
            self.teacher = copy.deepcopy(self.model)
            if not self.args.teacher_eval:
                self.teacher.train()

    def _compute_ncm_means(self, x_train, y_train):
        """
        Compute class means for NCM classifier from training data.
        Simplified version that doesn't use buffer.
        """
        self.model.eval()

        class_features = {}
        class_counts = {}

        dataloader = Dataloader_from_numpy(x_train, y_train, self.batch_size, shuffle=False)

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                features = self.model.feature(x).cpu()

                for i, label in enumerate(y.tolist()):
                    if label not in class_features:
                        class_features[label] = []
                        class_counts[label] = 0
                    class_features[label].append(features[i])
                    class_counts[label] += 1

        # Compute mean for each class
        n_total_classes = len(self.learned_classes)
        means = torch.zeros(n_total_classes, features.shape[-1])

        for label in class_features:
            class_features[label] = torch.stack(class_features[label])
            means[label] = class_features[label].mean(dim=0)

        return means.to(self.device)

    @torch.no_grad()
    def evaluate(self, task_stream, path=None):
        """
        Evaluate on the test sets of all the learned tasks (<= task_now).

        Following the original DER implementation (DER/inclearn/models/incmodel.py):
        1. Use ALL logits for evaluation (no truncation)
        2. Compute accuracy per task based on increments
        3. Report both per-task accuracy and overall accuracy

        Key difference from previous implementation:
        - Do NOT truncate logits to n_classes_so_far
        - Use the full classification head output
        - Accuracy is computed by grouping samples according to their task increments
        """
        # Get num_tasks and create Accuracy Matrix
        if self.task_now == 0:
            self.num_tasks = task_stream.n_tasks
            self.Acc_tasks = {
                'valid': np.zeros((self.num_tasks, self.num_tasks + 1)),
                'test': np.zeros((self.num_tasks, self.num_tasks + 1))
            }

        # Set model to eval mode (DON'T reload checkpoint - head has expanded)
        self.model.eval()

        eval_modes = ['valid', 'test']
        for mode in eval_modes:
            if self.verbose:
                print('\n ======== Evaluate on {} set ========'.format(mode))
            x_eval_all_data, y_eval_all_data = [], []
            for i in range(self.task_now + 1):
                (x_eval, y_eval) = task_stream.tasks[i][1] if mode == 'valid' else task_stream.tasks[i][2]
                eval_dataloader_i = Dataloader_from_numpy(x_eval, y_eval, self.batch_size, shuffle=False)
                x_eval_all_data.append(x_eval)
                y_eval_all_data.append(y_eval)
                if self.cf_matrix and self.task_now+1 == self.num_tasks and mode == 'test':
                    eval_loss_i, eval_acc_i = self.test_for_cf_matrix(eval_dataloader_i)
                else:
                    # Use ALL logits for evaluation (consistent with original DER)
                    eval_loss_i, eval_acc_i = self.cross_entropy_epoch_run_ders(
                        eval_dataloader_i,
                        mode='test',
                        increments=self._increments[:i+1]
                    )

                if self.verbose:
                    print('Task {}: Accuracy == {}, Test CE Loss == {} ;'.format(i, eval_acc_i, eval_loss_i))
                self.Acc_tasks[mode][self.task_now][i] = np.around(eval_acc_i, decimals=4)

            # Mean accuracy over all tasks
            eval_dataloader_all_i = Dataloader_from_numpy(
                np.concatenate(x_eval_all_data),
                np.concatenate(y_eval_all_data),
                self.batch_size,
                shuffle=False
            )

            total_classes = sum(self._increments[:self.task_now + 1])
            print('total_classes == {}'.format(total_classes))
            eval_loss_i, eval_acc_i = self.test_for_cf_matrix(eval_dataloader_all_i)
            print('Mean Accuracy == {}, Test CE Loss == {} ;'.format(eval_acc_i, eval_loss_i))
            self.Acc_tasks[mode][self.task_now][-1] = np.around(eval_acc_i, decimals=4)

            if self.task_now + 1 == self.num_tasks and self.verbose:
                with np.printoptions(suppress=True):
                    print('Accuracy matrix of all tasks:')
                    print(self.Acc_tasks[mode])

        # TSNE visualization
        if self.tsne and not self.args.tune:
            tsne_path = path + 't{}'.format(self.task_now)
            self.feature_space_tsne_visualization(task_stream, path=tsne_path)

        if self.args.tsne_g and self.args.agent == 'GR' and not self.args.tune:
            tsne_path = path + 't{}_g'.format(self.task_now)
            self.feature_space_tsne_visualization(task_stream, path=tsne_path, view_generator=True)

        # Log subject-specific information
        if self.verbose and hasattr(self.args, 'cross_subject_id'):
            subject_id = self.args.cross_subject_id
            self.subject_accuracies[subject_id] = eval_acc_i
            print('[DERS] Subject {}: Accuracy = {:.2f}%'.format(subject_id + 1, eval_acc_i))

        return eval_acc_i

    def cross_entropy_epoch_run_ders(self, dataloader, mode='test', increments=None):
        """
        DERS evaluation using ALL logits (consistent with original DER).

        Key insight from DER/inclearn/tools/utils.py::compute_accuracy:
        - Model outputs logits for ALL learned classes
        - argmax(logits) gives predicted class ID
        - Ground truth labels determine which task group the sample belongs to
        - No need to know the incremental stage beforehand

        How it works:
        - Class IDs are globally unique: 0 to n_total_classes-1
        - Task 0: labels in [0, increments[0])
        - Task 1: labels in [increments[0], increments[0]+increments[1])
        - etc.
        - Model predicts among ALL learned classes
        - Prediction is correct if argmax(logits) == ground truth label

        Args:
            dataloader: dataloader for evaluation
            mode: 'test' or 'val'
            increments: list of class counts per task, used for per-task breakdown
        """
        self.model.eval()

        all_logits = []
        all_targets = []
        epoch_loss = 0
        batch_count = 0

        # Collect all predictions and targets
        with torch.no_grad():
            for batch_id, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)

                # Use ALL logits - model outputs for all learned classes
                loss = F.cross_entropy(outputs, y)
                epoch_loss += loss.item()
                batch_count += 1

                all_logits.append(outputs.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        all_logits = np.concatenate(all_logits, axis=0)  # [N, n_classes]
        all_targets = np.concatenate(all_targets, axis=0)  # [N]

        # Get predictions by argmax over ALL logits
        all_preds = all_logits.argmax(axis=1)  # [N]

        # Compute overall accuracy (all samples)
        overall_acc = (all_preds == all_targets).sum() / len(all_targets) * 100

        # If increments provided, compute per-task accuracy by grouping samples
        # based on their ground truth label ranges
        if increments is not None and self.verbose:
            start, end = 0, 0
            for task_idx, inc in enumerate(increments):
                if inc <= 0:
                    continue
                start = end
                end += inc
                # Find samples whose ground truth labels are in [start, end)
                idxes = np.where(np.logical_and(all_targets >= start, all_targets < end))[0]
                if len(idxes) > 0:
                    # Compute accuracy for this task's samples
                    task_acc = (all_preds[idxes] == all_targets[idxes]).sum() / len(idxes) * 100
                    print(f'  Task {task_idx} (classes {start}-{end-1}): Accuracy = {task_acc:.2f}%')
                    print(f'    Samples: {len(idxes)}, Correct: {(all_preds[idxes] == all_targets[idxes]).sum()}')

        epoch_acc = overall_acc
        epoch_loss /= batch_count

        return epoch_loss, epoch_acc

    def get_branch_contributions(self, x):
        """
        Analyze the contribution of each branch to the final prediction.

        This is useful for understanding which task's representation
        is most important for a given input.

        Args:
            x: Input tensor

        Returns:
            contributions: List of contribution scores for each branch
        """
        if not self.der_dynamic:
            return None

        # Get features from each branch separately
        if hasattr(self.model.encoder, 'branches'):
            branches = self.model.encoder.branches
        else:
            return None

        branch_features = []
        for branch in branches:
            branch.eval()
            with torch.no_grad():
                feat = branch(x)
                branch_features.append(feat)

        # Simple contribution metric: feature magnitude per branch
        contributions = [feat.norm(dim=1).mean().item() for feat in branch_features]

        # Normalize to sum to 1
        total = sum(contributions)
        if total > 0:
            contributions = [c / total for c in contributions]

        return contributions
