import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import calculate_top1_and_topk_accuracy
from .common import compute_mean_W
from .common import dataset_to_list_of_samples_per_class
from .common import get_best_model_name
from .common import non_iid_pb_parameter_selection
from .common import pb_parameter_selection
from ..args import common_parser
from ..datasets.auslan import get_train_val_test_datasets as get_auslan_train_val_test_datasets
from ..datasets.cifar100 import get_train_val_test_datasets as get_cifar100_train_val_test_datasets
from ..models.pb_models import StochasticCNN
from ..models.pb_models import StochasticMLP
from ..utils.logger import get_logger


def main():
    logger = get_logger()
    parser = common_parser(pac_bayes=True, train=False)
    parser.add_argument('--top-k', type=int, default=5,
                        help='The size of top k (default: 5)')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    is_catoni_bound = not args.non_iid

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    rnd = np.random.RandomState(args.seed)

    result = {}

    if not args.mlp:
        # CNN: load CFIAR100 train and test data sets and load CNN
        train_set, _, test_set = get_cifar100_train_val_test_datasets(rnd, validation_ratio=args.validation_ratio)
        model = StochasticCNN(num_training_samples=0, rnd=rnd, num_last_units=args.dim_h, init_weights=False)
        num_classes = 100
    else:
        # MLP: load AUSLAN train/test data sets, and initialize PAC-Bayesian MLP.
        train_ids = tuple(range(9))
        test_ids = (9,)

        # The validation data set is not used during evaluation
        train_set, _, test_set = get_auslan_train_val_test_datasets(
            rnd=rnd,
            root=args.root,
            train_ids=train_ids,
            test_ids=test_ids,
            validation_ratio=args.validation_ratio,
            squash_time=True
        )
        model = StochasticMLP(num_training_samples=0, rnd=rnd, num_last_units=args.dim_h, init_weights=False)
        num_classes = train_set.num_classes

    class_id2samples = dataset_to_list_of_samples_per_class(train_set, num_classes=num_classes)

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )

    if args.criterion == 'pb':
        if not args.model_name_dir:
            model_name = args.model_name
        else:
            if is_catoni_bound:
                model_name = pb_parameter_selection(logger, json_fname=args.json_fname)
            else:
                model_name = non_iid_pb_parameter_selection(logger, json_fname=args.json_fname)
    else:
        model_name = get_best_model_name(args)

    logger.info('The lowest model name {}\n'.format(model_name))

    model.load_state_dict(torch.load(model_name, map_location=device))
    model = model.to(device)

    model.eval()
    top1_accuracies = []
    top5_accuracies = []

    if args.deterministic:
        num_snn = 1
    else:
        num_snn = args.num_snn

    for _ in range(num_snn):
        with torch.no_grad():
            if args.deterministic:
                model.deterministic()
            else:
                model.sample_noise()

            W = compute_mean_W(model, class_id2samples, device).to(device)

            top1, topk = calculate_top1_and_topk_accuracy(test_loader, device, model, W, top_k=args.top_k)
            top1_accuracies.append(top1 * 100.)
            top5_accuracies.append(topk * 100.)

    top1_accuracies = np.array(top1_accuracies)
    top5_accuracies = np.array(top5_accuracies)
    logger.info(
        'Unsupervised μ top1 acc: {:.1f} snn std: {:.1f}\n'.format(top1_accuracies.mean(), top1_accuracies.std()))
    logger.info(
        'Unsupervised μ top5 acc: {:.1f} snn std: {:.1f}\n'.format(top5_accuracies.mean(), top5_accuracies.std()))
    result['top1'] = top1_accuracies.mean()
    result['top{}'.format(args.top_k)] = top5_accuracies.mean()

    # mu averaged over random samples per class
    top1_accuracies = np.zeros((num_snn, args.num_trials))
    topk_accuracies = np.zeros((num_snn, args.num_trials))

    for snn_index in range(num_snn):

        with torch.no_grad():
            if args.deterministic:
                model.deterministic()
            else:
                model.sample_noise()

            for trial_id in range(args.num_trials):
                W = compute_mean_W(model, class_id2samples, device, num_sub_samples=5).to(device)
                top1, topk = calculate_top1_and_topk_accuracy(test_loader, device, model, W, top_k=args.top_k)
                top1_accuracies[snn_index, trial_id] = top1 * 100
                topk_accuracies[snn_index, trial_id] = topk * 100

    # summarise the snn per trials
    top1_accuracies = np.mean(top1_accuracies, axis=0)
    topk_accuracies = np.mean(topk_accuracies, axis=0)

    logger.info(
        'Unsupervised μ-5 top1 acc: {:.1f} trial std: {:.1f}\n'.format(
            top1_accuracies.mean(), top1_accuracies.std()
        )
    )
    logger.info(
        'Unsupervised μ-5 top{} acc: {:.1f} trial std: {:.1f}\n'.format(
            args.top_k, topk_accuracies.mean(), topk_accuracies.std()
        )
    )
    result['mu5-top1'] = top1_accuracies.mean()
    result['mu5-top{}'.format(args.top_k)] = topk_accuracies.mean()

    with open(args.output_json_fname, 'w') as log_file:
        json.dump(result, log_file)


if __name__ == '__main__':
    main()
