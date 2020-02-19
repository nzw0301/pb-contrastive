import json

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from .args import common_parser, check_args
from .datasets.auslan import get_shape_for_contrastive_learning
from .datasets.contrastive import get_contrastive_data_loaders
from .loss import ContrastiveLoss
from .models.mlp import MLP
from .utils.earlystopping import EarlyStopping
from .utils.logger import get_logger


def train(
        args, model: MLP, device: torch.device, train_loader: torch.utils.data.dataloader.DataLoader,
        optimizer, epoch: int,
        contrastive_loss: ContrastiveLoss,
        logger
) -> None:
    """
    Update model weights per epoch.

    :param args: arg parser.
    :param model: Instance of `MLP`.
    :param device: PyTorch's device instance.
    :param train_loader: Training data loader.
    :param optimizer: PyTorch's optimizer instance.
    :param epoch: The number of epochs.
    :param contrastive_loss: the instance of ContrastiveLoss class.
    :param logger: logger.

    """
    model.train()
    for batch_idx, (samples, pos, negs) in enumerate(train_loader):
        optimizer.zero_grad()

        (batch2input_shape_pos, batch2input_shape_neg, output2emb_shape_pos, output2emb_shape_neg) \
            = get_shape_for_contrastive_learning(len(samples), args.block_size, args.neg_size, args.dim_h)

        # reshape
        pos = pos.view(batch2input_shape_pos)
        negs = negs.view(batch2input_shape_neg)
        num_samples = len(samples)
        data = torch.cat([samples, pos, negs], dim=0).to(device)

        features = model(data)

        data_features = features[:num_samples]
        pos_features = features[num_samples:num_samples + len(pos)].view(output2emb_shape_pos)
        neg_features = features[num_samples + len(pos):].view(output2emb_shape_neg)

        loss = contrastive_loss(feature=data_features, positive_feature=pos_features, negative_features=neg_features)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            logger.info('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.7f}'.format(
                epoch, batch_idx * len(samples), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item())
            )


def validation(
        args,
        model: MLP, device: torch.device, val_loader: torch.utils.data.dataloader.DataLoader,
        contrastive_loss: ContrastiveLoss, logger
) -> torch.FloatTensor:
    """
    Calculate validation loss.

    :param args: arg parser.
    :param model: Instance of `MLP`.
    :param device: PyTorch's device instance.
    :param val_loader: validation data loader.
    :param contrastive_loss: the instance of ContrastiveLoss class.
    :param logger: logger.

    :return: Validation loss. Float.
   """

    model.eval()
    sum_loss = 0.
    with torch.no_grad():
        for batch_idx, (samples, pos, negs) in enumerate(val_loader):
            (batch2input_shape_pos, batch2input_shape_neg, output2emb_shape_pos, output2emb_shape_neg) \
                = get_shape_for_contrastive_learning(len(samples), args.block_size, args.neg_size, args.dim_h)

            # reshape
            pos = pos.view(batch2input_shape_pos)
            negs = negs.view(batch2input_shape_neg)
            num_samples = len(samples)
            data = torch.cat([samples, pos, negs], dim=0).to(device)

            features = model(data)

            data_features = features[:num_samples]
            pos_features = features[num_samples:num_samples + len(pos)].view(output2emb_shape_pos)
            neg_features = features[num_samples + len(pos):].view(output2emb_shape_neg)

            loss = contrastive_loss(feature=data_features, positive_feature=pos_features,
                                    negative_features=neg_features, reduction='sum')
            sum_loss += loss.item()

    avg_contrastive_loss = sum_loss / len(val_loader.dataset)
    logger.info(' Validation loss: {:.7f}\n'.format(avg_contrastive_loss))

    return avg_contrastive_loss


def main():
    logger = get_logger()
    parser = common_parser()
    args = parser.parse_args()
    check_args(args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    iid = not args.non_iid

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    rnd = np.random.RandomState(args.seed)

    device = torch.device('cuda' if use_cuda else 'cpu')
    logger.info('loss: {}\n'.format(args.loss))

    contrastive_loss = ContrastiveLoss(loss_name=args.loss, device=device)

    train_loader, val_loader = get_contrastive_data_loaders(
        rnd=rnd,
        data_name='auslan',
        validation_ratio=args.validation_ratio,
        mini_batch_size=args.batch_size,
        num_blocks_per_class=45 * 24,
        block_size=args.block_size,
        neg_size=args.neg_size,
        root=args.root,
        iid=iid
    )

    num_training_samples = len(train_loader.dataset)
    if val_loader is None:
        num_val_samples = 0
    else:
        num_val_samples = len(val_loader.dataset)

    logger.info('# training samples: {} # val samples: {}\n'.format(num_training_samples, num_val_samples))

    model = MLP(rnd=rnd, num_last_units=args.dim_h, supervised=False).to(device)

    optimizer_name = args.optim.lower()
    if optimizer_name == 'adam':
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(params=model.parameters(), lr=args.lr)
    else:
        raise ValueError('Optimizer must be adam, sgd, or rmsprop. Not {}'.format(optimizer_name))

    logger.info('optimizer: {}\n'.format(optimizer_name))

    scheduler = MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gamma)
    early_stopping = EarlyStopping(mode='min', patience=args.patience)
    learning_history = {'val_loss': []}

    save_name = 'lr-{}_{}_{}'.format(args.lr, optimizer_name, args.output_model_name)
    for epoch in range(1, args.epoch + 1):
        train(
            args, model, device, train_loader, optimizer, epoch, contrastive_loss,
            logger
        )
        scheduler.step()

        val_loss = validation(
            args, model, device, val_loader, contrastive_loss, logger
        )

        learning_history['val_loss'].append(val_loss)

        if early_stopping.is_stopped_and_save(
                val_loss, model, save_name=save_name
        ):
            break

    # logging_file
    json_fname = save_name.replace('.pt', '.json')
    with open(json_fname, 'w') as log_file:
        learning_history['lowest_val_loss'] = early_stopping.best
        json.dump(learning_history, log_file)


if __name__ == '__main__':
    main()
