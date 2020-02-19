import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import calculate_top1_and_topk_accuracy
from .common import compute_mean_W
from .common import dataset_to_list_of_samples_per_class
from .common import get_best_model_name
from ..args import common_parser, check_args
from ..datasets.auslan import get_train_val_test_datasets as get_auslan_train_val_test_datasets
from ..datasets.cifar100 import get_train_val_test_datasets as get_cifar100_train_val_test_datasets
from ..models.cnn import CNN
from ..models.mlp import MLP
from ..utils.logger import get_logger


def main():
    logger = get_logger()
    parser = common_parser(train=False)
    parser.add_argument('--top-k', type=int, default=5,
                        help='the size of top k (default: 5)')

    args = parser.parse_args()
    check_args(args, train=False)
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device('cuda' if use_cuda else 'cpu')

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    rnd = np.random.RandomState(args.seed)

    result = {}

    if not args.mlp:
        # CNN: load CFIAR100 train and test data sets and load CNN
        train_set, _, test_set = get_cifar100_train_val_test_datasets(
            rnd, validation_ratio=args.validation_ratio
        )
        model = CNN(rnd=rnd, init_weights=False, supervised=args.supervised)
        num_classes = 100
    else:
        # MLP: load AUSLAN train test data sets and load MLP
        train_set, _, test_set = get_auslan_train_val_test_datasets(
            rnd=rnd, root=args.root, validation_ratio=args.validation_ratio, squash_time=True
        )
        model = MLP(rnd=rnd, init_weights=False, supervised=args.supervised)
        num_classes = train_set.num_classes

    class_id2samples = dataset_to_list_of_samples_per_class(train_set, num_classes=num_classes)

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )

    model_name = get_best_model_name(args)
    logger.info('best result: {}\n'.format(model_name))

    model.load_state_dict(torch.load(model_name, map_location=device))
    model = model.to(device)

    model.eval()

    # all mean vectors
    with torch.no_grad():
        W = compute_mean_W(model, class_id2samples, device).to(device)
        top1, topk = calculate_top1_and_topk_accuracy(test_loader, device, model, W, top_k=args.top_k)

    logger.info('Unsupervised μ top1 {:.1f}\n'.format(top1 * 100.))
    logger.info('Unsupervised μ top{} {:.1f}\n'.format(args.top_k, topk * 100.))
    result['top1'] = top1 * 100.
    result['top{}'.format(args.top_k)] = topk * 100.

    # mu averaged over random samples per class
    num_trials = args.num_trials
    with torch.no_grad():
        top1_accuracies = []
        topk_accuracies = []
        for _ in range(num_trials):
            W = compute_mean_W(model, class_id2samples, device, num_sub_samples=5).to(device)

            top1, topk = calculate_top1_and_topk_accuracy(test_loader, device, model, W, top_k=args.top_k)
            top1_accuracies.append(top1 * 100.)
            topk_accuracies.append(topk * 100.)

    top1_accuracies = np.array(top1_accuracies)
    topk_accuracies = np.array(topk_accuracies)
    logger.info('Unsupervised μ-5 top1 {:.1f} std: {:.1f}\n'.format(top1_accuracies.mean(), top1_accuracies.std()))
    logger.info('Unsupervised μ-5 top{} {:.1f} std: {:.1f}\n'.format(
        args.top_k, topk_accuracies.mean(), topk_accuracies.std())
    )
    result['mu5-top1'] = top1_accuracies.mean()
    result['mu5-top{}'.format(args.top_k)] = topk_accuracies.mean()

    with open(args.output_json_fname, 'w') as log_file:
        json.dump(result, log_file)


if __name__ == '__main__':
    main()
