import json

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from ..args import common_parser, check_args
from ..datasets.auslan import get_train_val_test_datasets
from ..loss import SupervisedLoss
from ..models.mlp import MLP
from ..utils.earlystopping import EarlyStopping
from ..utils.logger import get_logger


def train(
        args, model: MLP, supervised_loss: SupervisedLoss, device, train_loader, optimizer, epoch: int, logger
) -> None:
    """
    Update weights per epoch.

    :param args: Parsed args
    :param model: MLP model instance.
    :param supervised_loss: Instance of `SupervisedLoss` class.
    :param device: PyTorch's device instance.
    :param train_loader: DataLoader of training data.
    :param optimizer: PyTorch's optimizer instance.
    :param epoch: The current epoch for logger
    :param logger: Logger instance.

    :return:  None
    """
    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = supervised_loss(model.g(data), targets)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            logger.info('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item())
            )


def test(
        args, model: MLP, supervised_loss: SupervisedLoss, device, test_loader, logger, data_type='val'
) -> tuple:
    """
    Evaluation function. Return loss and accuracy.

    :param args: Parsed args
    :param model: MLP model instance.
    :param supervised_loss: Instance of `SupervisedLoss` class.
    :param device: PyTorch's device instance.
    :param test_loader: DataLoader of test data.
    :param logger: Logger instance.
    :param data_type: Str. The type of data for logger. Either `val` or `test`.

    :return: Loss and accuracy.
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            output = model.g(data)
            test_loss += supervised_loss(output, targets, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logger.info(' {} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        data_type,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))
    )
    return test_loss, correct / len(test_loader.dataset)


def get_data_loaders(
        rnd: np.random.RandomState,
        validation_ratio: float,
        root: str,
        batch_size: int,
        kwargs,
        train_ids=tuple(range(1, 9)),
        test_ids=(9,),
) -> tuple:
    """
    :param rnd: `np.random.RandomState` instance.
    :param validation_ratio: ratio of validation. float.
    :param root: root to store the dataset. Note it must be absolute path.
    :param batch_size: The size of mini batch.
    :param kwargs: kwargs for Data loaders.
    :param train_ids: training samples' ids in [1, 9].
    :param test_ids: testing samples' ids in [1, 9].

    :return: Tuple of `DataLoaders: `(train_data_loader, validation_data_loader, test_data_loader)`.
    """
    train_set, val_set, test_set = get_train_val_test_datasets(
        rnd=rnd,
        root=root,
        train_ids=train_ids,
        test_ids=test_ids,
        validation_ratio=validation_ratio,
        squash_time=True
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        **kwargs
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        **kwargs
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    logger = get_logger()
    parser = common_parser()
    args = parser.parse_args()
    check_args(args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # for reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    rnd = np.random.RandomState(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    iid = not args.non_iid

    train_loader, val_loader, test_loader = get_data_loaders(
        rnd=rnd, validation_ratio=args.validation_ratio,
        root=args.root, batch_size=args.batch_size,
        kwargs=kwargs
    )

    logger.info(
        'train_data: {}, val_data:{}, test_data: {}\n'.format(
            len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset)
        )
    )
    supervised_loss = SupervisedLoss(num_last_units=95, loss=args.loss, device=device)
    model = MLP(num_input_units=22, num_last_units=50, num_hidden_units=50, rnd=rnd, supervised=True).to(device)

    optimizer_name = args.optim.lower()
    if optimizer_name == 'adam':
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(params=model.parameters(), lr=args.lr)
    else:
        raise ValueError('Optimizer must be adam, sgd, or rmsprop. Not {}'.format(optimizer_name))

    scheduler = MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gamma)

    early_stopping = EarlyStopping(mode='min', patience=args.patience)
    learning_history = {'val_loss': [], 'val_acc': []}

    save_fname = 'lr-{}_{}_{}_{}'.format(args.lr, optimizer_name, args.loss, args.output_model_name)
    for epoch in range(1, args.epoch + 1):
        train(args, model, supervised_loss, device, train_loader, optimizer, epoch, logger)
        scheduler.step()

        val_loss, val_acc = test(args, model, supervised_loss, device, val_loader, logger, 'validation')

        learning_history['val_loss'].append(val_loss)
        learning_history['val_acc'].append(val_acc)

        if early_stopping.is_stopped_and_save(val_loss, model, save_name=save_fname):
            break

    # logging_file
    save_json_fname = save_fname.replace('.pt', '.json')
    with open(save_json_fname, 'w') as log_file:
        learning_history['lowest_val_loss'] = early_stopping.best
        json.dump(learning_history, log_file)

    # report test accuracy of the best model during the training.
    model.load_state_dict(torch.load(save_fname, map_location=device))
    model = model.to(device)
    test(args, model, supervised_loss, device, test_loader, logger, 'test')
