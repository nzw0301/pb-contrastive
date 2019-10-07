import json

import numpy as np
import torch
from scipy.special import comb
from torch.utils.data import DataLoader

from .common import calculate_average_accuracy
from .common import compute_mean_W
from .common import compute_mean_W_tensor
from .common import compute_test_fx
from .common import dataset_to_list_of_samples_per_class
from .common import get_best_model_name
from .common import tasks_generator
from ..args import common_parser, check_args
from ..datasets.australian import get_train_val_test_datasets as get_australian_train_val_test_datasets
from ..datasets.average import AverageDataset
from ..datasets.cifar100 import get_train_val_test_datasets as get_cifar100_train_val_test_datasets
from ..models.cnn import CNN
from ..models.mlp import MLP
from ..utils.logger import get_logger


def main():
    logger = get_logger()
    parser = common_parser(train=False)

    args = parser.parse_args()
    check_args(args, train=False)
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device('cuda' if use_cuda else 'cpu')

    torch.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    rnd = np.random.RandomState(args.seed)
    result = {}

    if not args.mlp:
        # iid case: load CFIAR100 train and test data sets and load CNN
        train_set, _, test_set = get_cifar100_train_val_test_datasets(
            rnd, validation_ratio=args.validation_ratio, root=args.root
        )
        model = CNN(rnd=rnd, init_weights=False, supervised=args.supervised, num_last_units=args.dim_h)
        num_classes = 100
    else:
        # mlp case: load australian train test data sets, and load MLP
        train_set, _, test_set = get_australian_train_val_test_datasets(
            root=args.root,
            to_tensor=False
        )
        model = MLP(rnd=rnd, init_weights=False, supervised=args.supervised, num_last_units=args.dim_h)
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

    num_tasks = comb(num_classes, args.num_avg_classes)

    # all mean vectors
    average_accuracy = []
    model.eval()
    with torch.no_grad():
        W = compute_mean_W(model, class_id2samples, device)

        # cache all fx representation for test data
        fx_test = compute_test_fx(model, test_loader, device)

        for task_id, task in enumerate(tasks_generator(
                num_all_classes=num_classes, num_avg_classes=args.num_avg_classes
        )):
            sub_W = W[[task]].to(device)

            test_avg_dataset = AverageDataset(fx_test, test_set.targets, target_ids=task)
            test_avg_data_loader = DataLoader(
                test_avg_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=2
            )

            avg_acc = calculate_average_accuracy(test_avg_data_loader, device, sub_W)
            avg_acc = avg_acc * 100.
            average_accuracy.append(avg_acc)
            logger.info('\r{:.2f}% task: {} avg-{} acc. {:.1f}'.format(
                task_id / num_tasks * 100., task, args.num_avg_classes, avg_acc
            ))

    average_accuracy = np.array(average_accuracy)
    logger.info('\navg-{} acc: {:.1f}\n'.format(args.num_avg_classes, average_accuracy.mean()))
    result['avg{}'.format(args.num_avg_classes)] = average_accuracy.mean()

    # mu averaged over random samples per class
    num_trials = args.num_trials
    accuracies = np.zeros(num_trials)
    with torch.no_grad():
        W = compute_mean_W_tensor(model, class_id2samples, device, samples=5, num_trials=num_trials)
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

            avg_acc = calculate_average_accuracy(test_avg_data_loader, device, sub_W)
            avg_acc = avg_acc * 100.
            accuracies += avg_acc
            logger.info('\r{:.2f}% task: {} μ-5 avg-{} acc. {:.5f}'.format(
                (task_id / num_tasks * 100.), task, args.num_avg_classes, avg_acc.mean()
            ))

    accuracies /= num_tasks
    logger.info('\rall trial accuracies: {}\n'.format(accuracies))
    logger.info('avg-{} μ-5 acc. {:.1f} std: {:.1f}\n'.format(
        args.num_avg_classes, accuracies.mean(), accuracies.std())
    )
    result['mu5-avg{}'.format(args.num_avg_classes)] = accuracies.mean()
    with open(args.output_json_fname, 'w') as log_file:
        json.dump(result, log_file)


if __name__ == '__main__':
    main()
