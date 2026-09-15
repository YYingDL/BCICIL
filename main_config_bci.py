import argparse
import sys
import os
import torch
import time
from experiment.exp import experiment_multiple_runs
from utils.utils import Logger, boolean_string
import numpy as np


def auto_configure(args):
    """
    根据数据集和 encoder 自动配置参数

    自动设置:
    - feature_dim: 不同 encoder 的特征维度
    - adr_buffer_size: 不同数据集的 buffer 大小
    - head: IL2A 使用 IL2AHead (虚拟类别)
    """
    # feature_dim 默认值配置

    # Fallback: use comment suggestions from original args
    if args.encoder in ['Emotion_Net_ln2', 'MI_EEGNet']:
        args.feature_dim = 248
    elif args.encoder == 'MI_IFNet':
        args.feature_dim = 512
    elif args.encoder == 'MI_ADFCNN':
        args.feature_dim = 32
    elif args.encoder == 'MI_EISATC':
        args.feature_dim = 64
    elif args.encoder == 'SSVEPFormer':
        args.feature_dim = 128
    else:
        args.feature_dim = 128  # Default fallback
    print(f'[Auto-Configure] feature_dim set to {args.feature_dim} (fallback for {args.encoder})')

    # ADR buffer_size 自动配置

    buffer_size_defaults = {
        'mi': 64,
        'benchmark': 800,
        'seed3': 256,
    }
    if args.data in buffer_size_defaults:
        args.adr_buffer_size = buffer_size_defaults[args.data]
        print(f'[Auto-Configure] adr_buffer_size set to {args.adr_buffer_size} for {args.data}')

    # IL2A agent auto-configuration
    if args.agent == 'IL2A':
        # Use IL2A specialized head with virtual classes
        args.head = 'IL2AHead'
        print('[Auto-Configure] IL2A agent: head set to IL2AHead (virtual classes enabled)')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the continual learning agent on task sequence')

    # #################### Main setting for the experiment ####################
    parser.add_argument('--agent', dest='agent', default='DERS', type=str,
                        choices=['SFT', 'Offline',
                                 'LwF', 'EWC', 'PASS', 'IL2A',
                                 'ADR', 'ER', 'DER', 'DERS', 'ASER',  'CLOPS', 'FastICARL'],
                        help='Continual learning agent')

    parser.add_argument('--scenario', type=str, default='class',
                        choices=['class', 'domain'],
                        help='Scenario of the task steam. Current codes only include class-il')

    parser.add_argument('--stream_split', type=str, default='exp',
                        choices=['val', 'exp', 'all'],
                        help='The split of the tasks stream: val tasks, exp tasks or all the tasks')

    parser.add_argument('--data', dest='data', default='seed3', type=str,
                        choices=['seed3','mi','benchmark'])

    # Backbone
    parser.add_argument('--encoder', dest='encoder', default='Emotion_Net_ln2', type=str,
                        choices=['Emotion_Net_ln2','MI_EEGNet','MI_IFNet', 'MI_ADFCNN','MI_EISATC', 'SSVEPFormer'])

    # Classifier
    parser.add_argument('--head', dest='head', default='Linear', type=str,
                        choices=['Linear', 'CosineLinear', 'SplitCosineLinear', 'IL2AHead'])
    parser.add_argument('--criterion', dest='criterion', default='CE', type=str,
                        choices=['CE', 'BCE'])  # Main classification loss and activation of head
    parser.add_argument('--ncm_classifier', dest='ncm_classifier', type=boolean_string, default=False,
                        help='Use NCM classifier or not. Only work for ER-based methods.')

    # Normalizaton layers
    parser.add_argument('--norm', dest='norm', default='BN', type=str,
                        choices=['BN', 'LN', 'IN', 'BIN', 'SwitchNorm', 'StochNorm'])

    parser.add_argument('--input_norm', dest='input_norm', default='none', type=str,
                        choices=['LN', 'IN', 'ZScore', 'none'])  # ZScore is only applicable for Offline

    """ General params """
    parser.add_argument('--runs', dest='runs', default=1, type=int,
                        help='Number of runs')
    parser.add_argument('--epochs', dest='epochs', default=50, type=int,
                        help='Number of epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', dest='lr', default=1e-3, type=float,
                        help='Learning rate')
    parser.add_argument('--lradj', type=str, default='step15')
    parser.add_argument('--early_stop', type=boolean_string, default=True)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--weight_decay', dest='weight_decay', default=0, type=float,
                        help='weight decay')
    parser.add_argument('--dropout', dest='dropout', default=0, type=float)
    parser.add_argument('--feature_dim', dest='feature_dim', type=int, default=None,
                        help='eegnet: 248, IFNet:512, ADFCNN:32, EISATC:64, SSVEPFormer: 128, Emotion_Net_ln2: 248')
    parser.add_argument('--n_layers', dest='n_layers', type=int, default=4)

    # #################### Nuisance variables  ####################
    parser.add_argument('--tune', type=boolean_string, default=False, help='flag of tuning')
    parser.add_argument('--debug', type=boolean_string, default=True)  # save the results in a 'debug' folder
    parser.add_argument('--seed', dest='seed', default=1, type=int)
    parser.add_argument('--device', dest='device', default='cuda', type=str)
    parser.add_argument('--verbose', type=boolean_string, default=True)
    parser.add_argument('--exp_start_time', dest='exp_start_time', type=str)
    parser.add_argument('--fix_order', type=boolean_string, default=True,
                        help='Fix the class order for different runs')
    parser.add_argument('--cf_matrix', type=boolean_string, default=True,
                        help='Plot confusion matrix or not')
    parser.add_argument('--tsne', type=boolean_string, default=False,
                        help='Visualize the feature space of learner with TSNE')
    parser.add_argument('--tsne_g', type=boolean_string, default=True,
                        help='Visualize the feature space of generator with TSNE')

    # ######################## Methods-related params ###########################
    # Experience Replay
    parser.add_argument('--er_mode', type=str, default='task', choices=['online', 'task'],
                        help='Collect mem samples online or after_task')
    parser.add_argument('--mem_budget', type=float, default=0.05, help='Percentage of mem_budget/# train data')
    parser.add_argument('--buffer_tracker', type=boolean_string, default=False)  # Never be ture
    parser.add_argument('--er_sub_type', type=str, default='part', choices=['balanced', 'part'],
                        help='2 variant of ER to utilize subject labels.')
    parser.add_argument('--der_plus', type=boolean_string, default=False)  # DER++

    # Dynamically Expandable Representation
    parser.add_argument('--der_dynamic', type=boolean_string, default=False,
                        help='Enable DER dynamic representation expansion')
    parser.add_argument('--der_aux', type=boolean_string, default=True,
                        help='Enable DER auxiliary classifier for new classes')

    # KD: LwF / DT2W
    parser.add_argument('--teacher_eval', type=boolean_string, default=False)  # As Online CL survey and Avalanche
    parser.add_argument('--lambda_kd_lwf', dest='lambda_kd_lwf', default=100, type=float)
    parser.add_argument('--lambda_kd_fmap', dest='lambda_kd_fmap', default=1e-2, type=float,
                        help='lambda for KD loss on feature map')
    parser.add_argument('--fmap_kd_metric', dest='fmap_kd_metric', default="dtw", type=str,
                        choices=['dtw', 'euclidean', 'pod_temporal', 'pod_variate'],
                        help='KD metric for temporal feature map')
    parser.add_argument('--lambda_protoAug', dest='lambda_protoAug', default=100, type=float,
                        help='lambda for protoAug, no implementing this technique if lambda=0')
    parser.add_argument('--adaptive_weight',  type=boolean_string, default=False,
                        help='use linear adaptive lambda or not')

    # EWC / MAS / SI
    parser.add_argument('--lambda_impt', dest='lambda_impt', default=10000, type=float)
    parser.add_argument('--ewc_mode', dest='ewc_mode', default="separate", type=str,
                        choices=['separate', 'online'],
                        help='Mode for EWC, "separate" or "online"')

    # ASER
    parser.add_argument('--aser_k', dest='aser_k', default=3,
                        type=int,
                        help='Number of nearest neighbors (K) to perform ASER (default: %(default)s)')

    parser.add_argument('--aser_type', dest='aser_type', default="asvm", type=str, choices=['neg_sv', 'asv', 'asvm'],
                        help='Type of ASER: '
                             '"neg_sv" - Use negative SV only,'
                             ' "asv" - Use extremal values of Adversarial SV and Cooperative SV,'
                             ' "asvm" - Use mean values of Adversarial SV and Cooperative SV')

    parser.add_argument('--aser_n_smp_cls', dest='aser_n_smp_cls', default=4,
                        type=float,
                        help='Maximum number of samples per class for random sampling (default: %(default)s)')

    # CLOPS
    parser.add_argument('--mc_retrieve',  type=boolean_string, default=False,
                        help='use mc dropout retrieve strategy or not')
    parser.add_argument('--beta_lr', dest='beta_lr', default=1e-4, type=float)
    parser.add_argument('--lambda_beta', dest='lambda_beta', default=1, type=float)

    # Generative Replay
    parser.add_argument('--epochs_g', type=float, default=500)
    parser.add_argument('--lr_g', type=float, default=1e-3)
    parser.add_argument('--recon_wt', type=float, default=0.1)

    # Mnemonics
    parser.add_argument('--mnemonics_epochs', default=1, type=int)
    parser.add_argument('--mnemonics_lr', type=float, default=1e-5)

    # ADR (analytic RLS with score-based target relaxation)
    parser.add_argument('--adr_buffer_size', type=int, default=800,
                        help='Random buffer size for ADR feature expansion: MI 64, SSVEP 800')
    parser.add_argument('--adr_gamma', type=float, default=1e-3,
                        help='Regularization parameter for ADR RLS (larger = more regularization)')
    parser.add_argument('--adr_overconf_threshold', type=float, default=3.0,
                        help='ADR target relaxation threshold for over-confident correct-class scores')
    parser.add_argument('--adr_relax_margin', type=float, default=0.3,
                        help='ADR safe margin for relaxing non-target class scores')

    # PASS (Prototype Augmentation and Self-Supervision)
    parser.add_argument('--protoAug_weight', type=float, default=20.0,
                        help='Weight for prototype augmentation loss in PASS')
    parser.add_argument('--kd_weight', type=float, default=10.0,
                        help='Weight for knowledge distillation loss in PASS')
    parser.add_argument('--pass_buffer_size', type=int, default=100,
                        help='Number of samples to generate per class for prototype augmentation')
    parser.add_argument('--pass_temperature', type=float, default=2.0,
                        help='Temperature parameter for PASS knowledge distillation')

    # IL2A (Dual Augmentation)
    parser.add_argument('--seman_weight', type=float, default=10.0,
                        help='Weight for semantic augmentation loss in IL2A (original: 10.0)')
    parser.add_argument('--il2a_kd_weight', type=float, default=100.0,
                        help='Weight for knowledge distillation loss in IL2A (original: 10.0)')
    parser.add_argument('--il2a_temperature', type=float, default=0.1,
                        help='Temperature parameter for IL2A (original: 0.1)')
    parser.add_argument('--mix_alpha', type=float, default=20.0,
                        help='Alpha parameter for Mixup in IL2A ClassAug (original: 20.0)')
    parser.add_argument('--mix_times', type=int, default=1,
                        help='Number of mixup iterations in IL2A ClassAug (original: 4)')
    parser.add_argument('--aug_ratio', type=float, default=2.5,
                        help='Ratio for semantic augmentation in IL2A (original: 2.5)')
    parser.add_argument('--il2a_kd_temperature', type=float, default=2.0,
                        help='Temperature parameter for IL2A logits-level KD (original: 2.0)')

    # Contrastive Learning (NT-Xent + ProjectionHead)
    parser.add_argument('--use_contrastive', type=boolean_string, default=False,
                        help='Use contrastive learning for discriminative feature representation')
    parser.add_argument('--lambda_contrast', type=float, default=0.1,
                        help='Weight for contrastive learning loss')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Temperature parameter for NT-Xent loss')
    parser.add_argument('--projection_dim', type=int, default=64,
                        help='Output dimension of projection head')
    parser.add_argument('--projection_hidden', type=int, default=256,
                        help='Hidden dimension of projection head')
    parser.add_argument('--augment_types', type=str, nargs='+', default=[ 'channel_drop', 'time_warp'],
                        choices=['noise', 'channel_drop','time_warp'],
                        help='Types of EEG data augmentation for contrastive learning')

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Auto-configure parameters based on dataset and encoder
    auto_configure(args)

    # Set directories
    exp_start_time = time.strftime("%b-%d-%H-%M-%S", time.localtime())
    exp_path_0 = './result/exp/' if not args.debug else './result/exp/debug'
    exp_path_1 = args.encoder + '_' + args.data
    exp_path_2 = args.agent + '_' + args.norm + '_' + exp_start_time
    exp_path = os.path.join(exp_path_0, exp_path_1, exp_path_2)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    log_dir = exp_path + '/' + 'log.txt'
    sys.stdout = Logger(log_dir)
    args.exp_path = exp_path  # One experiment with multiple runs

    subject_nums = {'seed3': 15,'mi': 9, 'benchmark': 35}
    subject_acc_all =[]
    subject_cf_matrices = []  # 存储原始计数混淆矩阵
    for i in range(subject_nums[args.data]):
        args.cross_subject_id = i
        print(args)
        sub_acc, sub_cf_matrix = experiment_multiple_runs(args)
        subject_acc_all.append(sub_acc)
        if sub_cf_matrix is not None:
            subject_cf_matrices.append(sub_cf_matrix)
    subject_acc_all = np.array(subject_acc_all)
    if args.agent != 'Offline':
        mean_acc_list = subject_acc_all[:,0,:,-1]
        mean_acc = np.mean(subject_acc_all[:,0,:,-1],axis=0)
    else:
        mean_acc = np.mean(subject_acc_all,axis=0)
    print('Mean Accuracy is {}'.format(mean_acc))

    # 平均混淆矩阵：先平均再归一化
    if len(subject_cf_matrices) > 0:
        from utils.metrics import plot_confusion_matrix_direct
        subject_cf_matrices = np.array(subject_cf_matrices)
        mean_cf_matrix_raw = np.mean(subject_cf_matrices, axis=0)  # 先平均
        # 归一化
        mean_cf_matrix = mean_cf_matrix_raw / np.sum(mean_cf_matrix_raw, axis=1, keepdims=True)

        # 打印并保存平均混淆矩阵
        print('Mean Confusion Matrix:')
        print(mean_cf_matrix)
        np.save(exp_path + '/mean_cf_matrix.npy', mean_cf_matrix)

        # 绘制平均混淆矩阵热力图
        n_classes = mean_cf_matrix.shape[0]
        plot_confusion_matrix_direct(
            cf_matrix=mean_cf_matrix,
            path=exp_path + '/mean_cf_matrix',
            classes=np.arange(n_classes),
            title=f'Mean Confusion Matrix - {args.data} ({args.agent})'
        )
