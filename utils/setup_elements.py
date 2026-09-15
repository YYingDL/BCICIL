# -*- coding: UTF-8 -*-
import os

content_root = os.path.abspath('.')

data_path = {
    'seed3': content_root + '/data/saved/Seed/',
    'mi': content_root + '/data/saved/mi/',
    'benchmark':content_root + '/data/saved/benchmark/',
    }

input_size_match = {
    'seed3': [62, 5],  # [seq_len, input_channels] for Emotion_Net_ln2 (62 channels, 310 frames)
    'mi':[1001, 22],     # [seq_len, input_channels] for MI_EEGNet (22 channels, 1001 samples)
    'benchmark':[250, 9]  # [seq_len, input_channels] for SSVEPFormer (9 channels, 250 samples)
    }


# Number of training sample per class
n_smp_per_cls = {
    'seed3': 1000*14,
    'mi': 72*8,
    'benchmark': 6*34
     }

n_classes = {
    'seed3': 3,
    'mi': 4,
    'benchmark': 40
    }

# Number of tasks in the entire task sequence
n_tasks = {
    'seed3': 2,
    'mi': 3,
    'benchmark': 5
    }

n_classes_init_task = {
    'seed3': 2,
    'mi': 2,
    'benchmark':20
}

n_classes_per_task = {
    'seed3': 1,
    'mi': 1,
    'benchmark':5
}

preset_orders = {
    'seed3': [0, 1, 2],
    'mi': list(range(4)),
    'benchmark': list(range(40)),
    }


def get_num_classes(args):
    if args.stream_split == 'all':
        return n_classes[args.data]
    elif args.stream_split == 'val':
        return n_classes_init_task[args.data]
    else:
        return n_classes_init_task[args.data]


def get_buffer_size(args):
    n_exemplar_per_cls = int(args.mem_budget * n_smp_per_cls[args.data])
    if args.stream_split == 'all':
        mem_size = n_exemplar_per_cls * n_tasks[args.data]
    elif args.stream_split == 'val':
        mem_size = n_exemplar_per_cls * n_classes_init_task[args.data]
    else:
        mem_size = n_exemplar_per_cls * n_classes[args.data]

    return mem_size

