import torch


def get_label2samples(dataset, num_classes):
    label2samples = [[] for _ in range(num_classes)]

    for sample, label in dataset:
        label2samples[label].append(sample)
    for label, sample in enumerate(label2samples):
        label2samples[label] = torch.stack(sample)

    return label2samples
