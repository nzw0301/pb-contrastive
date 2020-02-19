import glob
import itertools
import json

import numpy as np
import torch


def compute_mean_W_tensor(model, class_id2samples, device: torch.device, samples: int, num_trials: int) -> torch.Tensor:
    """
    Compute tensor contains several mean classifier's weights.

    :param model: Neural network instance.
    :param class_id2samples: List contains images per index. The index is corresponding to label id.
    :param device: PyTorch's device instance
    :param samples: the number of samples to compute mean classifier
    :param num_trials: The number of mean classifiers.

    :return: FloatTensor. shape is  (num_trails, num_classes, dim_embeddings)
    """
    W = []
    for _ in range(num_trials):
        W.append(compute_mean_W(model, class_id2samples, device, samples))
    return torch.stack(W)  # (num-trails, num-classes, dim-embeddings)


def compute_mean_W(model, class_id2samples, device: torch.device, num_sub_samples=None) -> torch.Tensor:
    """
    Compute mean classifier's weight.

    :param model: Neural network instance.
    :param class_id2samples: List contains images per index. The index is corresponding to label id.
    :param device: PyTorch's device instance
    :param num_sub_samples: The number of samples to compute mean classifier

    :return: 2D FloatTensor. The shape is (num-classes, dim-embeddings).
    """
    w_mu = []
    for samples in class_id2samples:
        if num_sub_samples is not None:
            random_ids = torch.randperm(len(samples))[:num_sub_samples]
            samples = samples[random_ids]

        samples = samples.to(device)
        w_mu.append(torch.mean(model(samples), dim=0).cpu())
    return torch.stack(w_mu)  # (num-classes, dim-embeddings)


def compute_test_fx(model, test_data_loader, device: torch.device) -> torch.Tensor:
    """
    Cache feature representation on test data.

    :param model: Instance of neural networks
    :param test_data_loader: test data loader
    :param device: PyTorch's device instance.

    :return: FloatTensor contains feature representation on test data.
    """
    test_fx = []
    for images, _ in test_data_loader:
        test_fx.append(model(images.to(device)).cpu())
    return torch.cat(test_fx)


def tasks_generator(num_all_classes=100, num_avg_classes=2):
    """
    Generator to create average tasks; (k sub classification tasks)

    :param num_all_classes: The number of classes
    :param num_avg_classes: The number of k.

    :return: return tuple contains k labels
    """
    tasks_iter = itertools.combinations(np.arange(num_all_classes), num_avg_classes)
    for task in tasks_iter:
        yield task


def get_best_model_name(args, criterion_key_in_json='lowest_val_loss') -> str:
    """
    Select the best model in json files.

    :param args: Parsed args.
    :param criterion_key_in_json: the filed name to store parameter selection metric.

    :return: str. Best PyTorch's file name.
    """
    if not args.model_name_dir:
        return args.model_name

    if args.model_name_dir[-1] == '/':
        args.model_name_dir = args.model_name_dir[:-1]
    fnames = glob.glob('{}/*.json'.format(args.model_name_dir))

    lowest_val_loss = np.finfo(np.float(0.)).max

    best_model_fname = ''
    for fname in fnames:
        result = json.load(open(fname))
        if criterion_key_in_json not in result:
            print('{} field is not found in {}'.format(criterion_key_in_json, fname))
            continue

        print(fname, result[criterion_key_in_json])
        if result[criterion_key_in_json] < lowest_val_loss:
            lowest_val_loss = result[criterion_key_in_json]
            best_model_fname = fname

    return best_model_fname.replace('json', 'pt')


def dataset_to_list_of_samples_per_class(dataset, num_classes=100) -> list:
    """
    :param dataset: CIFAR100/Auslan class's instance.
    :param num_classes: the number of supervised labels
    :return: list: label id -> list of torch tensor.
    """
    class_id2samples = [[] for _ in range(num_classes)]

    for image, label in dataset:
        class_id2samples[label].append(image)

    for label, images in enumerate(class_id2samples):
        class_id2samples[label] = torch.stack(images)

    return class_id2samples


def calculate_average_accuracy(test_loader, device, W) -> float:
    """

    :param test_loader: DataLoader
    :param device: PyTorch's device instance.
    :param W: Precomputed weight tensor.

    :return: Accuracy of average task.
    """

    if W.dim() not in (2, 3):
        raise ValueError('`W.dim()` must be either 2 or 3.')

    if W.dim() == 2:
        top_1_correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred = torch.argmax(torch.mm(data, W.t()), dim=1)

            top_1_correct += pred.eq(target.view_as(pred)).sum().item()

        top1_acc = top_1_correct / len(test_loader.dataset)

    else:  # W.dim() == 3
        top_1_correct = np.zeros(W.size()[0])
        W_vec = W.view((np.prod(W.size()[:2]), W.size()[-1]))

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            pred = torch.argmax(
                torch.mm(data, W_vec.t()).t().view((W.size()[0], W.size()[1], len(target))),
                dim=1
            )  # (num-trails, samples)

            top_1_correct += pred.eq(target).sum(dim=1).cpu().numpy()  # num_trails

        top1_acc = top_1_correct / len(test_loader.dataset)

    return top1_acc


def calculate_top1_and_topk_accuracy(test_loader, device, model, W, top_k=5) -> tuple:
    """
    Calculate top-k accuracy.

    :param test_loader: DataLoader.
    :param device: PyTorch's device instance.
    :param model: Trained model.
    :param W: Precomputed weight tensor.
    :param top_k: The number of top-k

    :return: Tuple. It contains top1 and top-k accuracies.
    """
    top_1_correct = 0
    top_k_correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        features = model(data)  # mini-batch, hidden
        pred_top_k = torch.topk(torch.mm(features, W.t()), dim=1, k=top_k)[1]

        pred_top_1 = pred_top_k[:, 0]

        top_1_correct += pred_top_1.eq(target.view_as(pred_top_1)).sum().item()
        if top_k > 1:
            top_k_correct += (pred_top_k == target.view(len(target), 1)).sum().item()

    top1_acc = top_1_correct / len(test_loader.dataset)
    if top_k > 1:
        topk_acc = top_k_correct / len(test_loader.dataset)
    else:
        topk_acc = top1_acc

    return top1_acc, topk_acc


def pb_parameter_selection(
        logger,
        json_fname='pb_bound_values.json'
) -> str:
    """
    Find the best lambda such that it minimizes the PAC-Bayes bound.

    :param logger: Logger instance.
    :param json_fname: JSON file name that stores PAC-Bayes bounds' ingredients

    :return: The best file name.
    """
    data = json.load(open(json_fname))

    fname2bound = {}
    for fname, terms in data.items():
        num_train_data = terms['m']

        def catoni_bound(catoni_lambda_array: np.ndarray) -> np.ndarray:
            inner_exp = catoni_lambda_array / num_train_data * terms['train-zero-one-loss'] + \
                        (terms['complexity'] + np.log(2. * np.sqrt(num_train_data))) / num_train_data

            pb_bound = (1. - np.exp(-inner_exp)) / (1. - np.exp(- catoni_lambda_array / num_train_data))

            return pb_bound

        x = np.arange(10, 10 ** 7, 1)
        bounds = catoni_bound(x)
        pb_bound_train = min(bounds)
        logger.info('PAC-Bayes bound: {}, Optimal lambda: {}\n'.format(pb_bound_train, x[np.argmin(bounds)]))
        fname2bound[fname] = pb_bound_train

    return min(fname2bound, key=fname2bound.get)


def non_iid_bound(terms) -> float:
    """
    Helper function to compute non-iid PAC-Bayes bound to use data stored in a json file.
    Calculate non-iid PAC-Bayes bound.

    :param terms: dict instance created by using `precompute_bound.py`.
    :return: Float. PAC-Bayes bound.

    """
    num_train_data = terms['m']
    empirical_risk = terms['train-zero-one-loss']
    union_bound_term = terms['union']  # pi * j
    chi_square = terms['chi_square']
    T = terms['T']
    delta = terms['delta']

    complexity_term = union_bound_term * np.sqrt(
        (1. + 4. * T) / (24 * num_train_data * delta) * (chi_square + 1.)
    )

    return empirical_risk + complexity_term


def non_iid_pb_parameter_selection(
        logger,
        json_fname='pb_bound_values.json'
) -> str:
    """
    Find the best model such that its the PAC-Bayes bound is lowest.

    :param logger: Logger instance.
    :param json_fname: string. JSON file name that stores PAC-Bayes bounds' ingredients.

    :return: The best file name.
    """
    data = json.load(open(json_fname))

    fname2bound = {}
    for fname, terms in data.items():
        pb_bound = non_iid_bound(terms)
        logger.info('PAC-Bayes bound: {}, \n'.format(pb_bound))
        fname2bound[fname] = pb_bound

    return min(fname2bound, key=fname2bound.get)
