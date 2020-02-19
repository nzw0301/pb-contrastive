import json

import numpy as np
import torch
from scipy.special import comb
from torch.utils.data import DataLoader

from .common import calculate_average_accuracy
from .common import compute_mean_W, compute_mean_W_tensor
from .common import compute_test_fx
from .common import dataset_to_list_of_samples_per_class
from .common import get_best_model_name
from .common import non_iid_pb_parameter_selection
from .common import pb_parameter_selection
from .common import tasks_generator
from ..args import common_parser
from ..datasets.auslan import get_train_val_test_datasets as get_auslan_train_val_test_datasets
from ..datasets.average import AverageDataset
from ..datasets.cifar100 import get_train_val_test_datasets as get_cifar100_train_val_test_datasets
from ..models.pb_models import StochasticCNN
from ..models.pb_models import StochasticMLP
from ..utils.logger import get_logger


def main():
    logger = get_logger()
    parser = common_parser(pac_bayes=True, train=False)

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
        model = StochasticCNN(num_last_units=args.dim_h, num_training_samples=0, rnd=rnd, init_weights=False)
        num_classes = 100
    else:
        # MLP: load australian train test data sets and load MLP
        train_ids = tuple(range(1, 9))
        test_ids = (9,)

        train_set, _, test_set = get_auslan_train_val_test_datasets(
            rnd=rnd,
            train_ids=train_ids,
            test_ids=test_ids,
            root=args.root,
            validation_ratio=args.validation_ratio,
            squash_time=True
        )
        model = StochasticMLP(rnd=rnd, num_training_samples=0, num_last_units=args.dim_h, init_weights=False)
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

    num_tasks = comb(num_classes, args.num_avg_classes)

    if args.deterministic:
        num_snn = 1
    else:
        num_snn = args.num_snn

    model.eval()

    # all mean vectors
    accuracy_per_snn = []
    with torch.no_grad():
        for snn_index in range(1, num_snn + 1):
            average_accuracy = []

            if args.deterministic:
                model.deterministic()
            else:
                model.sample_noise()

            W = compute_mean_W(model, class_id2samples, device).to(device)
            fx_test = compute_test_fx(model, test_loader, device)
            for task_id, task in enumerate(tasks_generator(
                    num_all_classes=num_classes, num_avg_classes=args.num_avg_classes
            )):
                sub_W = W[[task]]

                test_avg_dataset = AverageDataset(fx_test, test_set.targets, target_ids=task)
                test_avg_data_loader = DataLoader(
                    test_avg_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=2
                )

                avg_acc = calculate_average_accuracy(test_avg_data_loader, device, sub_W)
                avg_acc *= 100.
                average_accuracy.append(avg_acc)
                logger.info('\r{:.2f}% task: {} avg-{} acc. {:.1f}'.format(
                    task_id / num_tasks * 100., task, args.num_avg_classes, avg_acc
                ))

            average_accuracy = np.array(average_accuracy).mean()
            logger.info(
                '\n{}th snn, avg-{} test acc: {:.1f}\n'.format(snn_index, args.num_avg_classes, average_accuracy))
            accuracy_per_snn.append(average_accuracy)

    average_acc_over_snn = np.array(accuracy_per_snn)
    logger.info('Unsupervised avg-{} acc. {:.1f} std: {:.1f}\n'.format(args.num_avg_classes,
                                                                       average_acc_over_snn.mean(),
                                                                       average_acc_over_snn.std()))
    result['avg{}'.format(args.num_avg_classes)] = average_acc_over_snn.mean()

    # mu averaged over random samples per class
    avg_accuracies = np.zeros((num_snn, args.num_trials))
    with torch.no_grad():
        for snn_index in range(1, num_snn + 1):
            if args.deterministic:
                model.deterministic()
            else:
                model.sample_noise()

            W = compute_mean_W_tensor(model, class_id2samples, device, samples=5, num_trials=args.num_trials)
            fx_test = compute_test_fx(model, test_loader, device)

            for task_id, task in enumerate(tasks_generator(
                    num_all_classes=num_classes, num_avg_classes=args.num_avg_classes
            )):
                sub_W = W[:, task].to(device)
                test_avg_dataset = AverageDataset(fx_test, test_set.targets, target_ids=task)

                test_avg_data_loader = DataLoader(
                    test_avg_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=2
                )

                avg_acc_per_trials = calculate_average_accuracy(test_avg_data_loader, device, sub_W)
                avg_acc_per_trials *= 100.
                avg_accuracies[snn_index - 1, :] += avg_acc_per_trials

                progress = task_id / num_tasks * 100.
                logger.info('\r{:.2f}% task: {} μ-5 avg-{} acc. {:.1f}'.format(
                    progress, task, args.num_avg_classes, avg_acc_per_trials.mean()
                ))
            avg_accuracies[snn_index - 1, :] /= num_tasks
            logger.info('\r{}th snn, μ-5 avg-{} acc: {}\n'.format(
                snn_index, args.num_avg_classes, avg_accuracies[snn_index - 1, :])
            )

    avg_accuracies = np.mean(avg_accuracies, axis=0)
    logger.info('Avg. accuracy per trial {}\n'.format(avg_accuracies))
    logger.info('Unsupervised μ-5 avg-{} acc. {:.1f} std: {:.1f}\n'.format(
        args.num_avg_classes, avg_accuracies.mean(), avg_accuracies.std())
    )

    result['mu5-avg{}'.format(args.num_avg_classes)] = avg_accuracies.mean()
    with open(args.output_json_fname, 'w') as log_file:
        json.dump(result, log_file)


if __name__ == '__main__':
    main()
