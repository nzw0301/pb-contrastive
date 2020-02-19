import json

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from .args import common_parser, check_args
from .datasets.auslan import get_shape_for_contrastive_learning
from .datasets.contrastive import get_contrastive_data_loaders
from .loss import ContrastiveLoss
from .models.pb_models import StochasticMLP
from .utils.earlystopping import EarlyStopping
from .utils.logger import get_logger


def train(
        args,
        model: StochasticMLP,
        device: torch.device,
        train_loader: torch.utils.data.dataloader.DataLoader,
        optimizer,
        epoch: int,
        contrastive_loss: ContrastiveLoss,
        T: int,
        logger,
) -> float:
    """
    Update model weights per epoch.

    :param args: Argument parser.
    :param model: Instance of `StochasticMLP`.
    :param device: PyTorch's device instance.
    :param train_loader: Training data loader.
    :param optimizer: PyTorch's optimizer instance.
    :param epoch: The number of epochs.
    :param contrastive_loss: the instance of ContrastiveLoss class.
    :param T: the length of dependency (== the size of blocks for non-iid contrastive data)
    :param logger: logger.

    :return: Averaged PAC-Bayes bound value.
    """
    model.train()

    average_objective = 0.
    for batch_idx, (images, pos, negs) in enumerate(train_loader):

        (batch2input_shape_pos, batch2input_shape_neg, output2emb_shape_pos,
         output2emb_shape_neg) = get_shape_for_contrastive_learning(
            len(images), args.block_size, args.neg_size,
            args.dim_h
        )

        optimizer.zero_grad()

        # reshape
        pos = pos.view(batch2input_shape_pos)
        negs = negs.view(batch2input_shape_neg)
        num_images = len(images)
        data = torch.cat([images, pos, negs], dim=0).to(device)

        model.sample_noise()

        features = model(data)

        data_features = features[:num_images]
        pos_features = features[num_images:num_images + len(pos)].view(output2emb_shape_pos)
        neg_features = features[num_images + len(pos):].view(output2emb_shape_neg)

        loss = contrastive_loss(feature=data_features, positive_feature=pos_features, negative_features=neg_features)

        objective, chi_square, complexity = model.non_iid_pac_bayes_objective(loss, T)

        objective.backward()
        optimizer.step()

        model.constraint()  # for union bound's constraint
        model.non_iid_posterior_constraint()  # for chi-square divergence's constraint

        average_objective += objective.item()

        if batch_idx % args.log_interval == 0:
            logger.info(
                '\rTrain Epoch: {} [{}/{} ({:.0f}%)] PAC-Bayes objective: {:.2f} '
                'cont. loss: {:.4f} complexity: {:.2f}, chi^2: {:.2f}, prior log std : {:.2f}'.format(
                    epoch, batch_idx * len(images), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                    objective.item(), loss.item(), complexity.item(), chi_square.item(), model.prior_log_std.item()
                )
            )

    # Note: `average_objective` is not accurate average's objective value.
    return average_objective / (batch_idx + 1)


def validation_loss(
        args,
        model: StochasticMLP,
        device: torch.device,
        val_loader: torch.utils.data.dataloader.DataLoader,
        contrastive_loss: ContrastiveLoss,
        logger,
        num_snn: int,
        deterministic=False
) -> float:
    """
    Calculate validation loss.

    :param args: arg parser.
    :param model: Instance of `StochasticMLP`.
    :param device: PyTorch's device instance.
    :param val_loader: validation data loader
    :param contrastive_loss: the instance of ContrastiveLoss class.
    :param logger: logger.
    :param num_snn: The number of samples from posterior to compute the loss value. If `deterministic` is True,
        this value is ignored.
    :param deterministic: Boolean flag for deterministic.

    :return: Validation loss. Float.
    """

    if deterministic:
        num_snn = 1

    model.eval()
    with torch.no_grad():
        validation_loss_per_realised_snn = []
        for _ in range(num_snn):

            if deterministic:
                model.deterministic()
            else:
                model.sample_noise()

            sum_loss = 0.
            for batch_idx, (images, pos, negs) in enumerate(val_loader):
                (batch2input_shape_pos, batch2input_shape_neg, output2emb_shape_pos,
                 output2emb_shape_neg) = get_shape_for_contrastive_learning(
                    len(images), args.block_size, args.neg_size,
                    args.dim_h
                )

                # reshape
                pos = pos.view(batch2input_shape_pos)
                negs = negs.view(batch2input_shape_neg)
                num_images = len(images)
                data = torch.cat([images, pos, negs], dim=0).to(device)

                features = model(data)

                data_features = features[:num_images]
                pos_features = features[num_images:num_images + len(pos)].view(output2emb_shape_pos)
                neg_features = features[num_images + len(pos):].view(output2emb_shape_neg)

                loss = contrastive_loss(feature=data_features, positive_feature=pos_features,
                                        negative_features=neg_features, reduction='sum')
                sum_loss += loss.item()
            validation_loss_per_realised_snn.append(sum_loss / len(val_loader.dataset))

    validation_loss = np.array(validation_loss_per_realised_snn).mean()

    if deterministic:
        prefix = 'deterministic'
    else:
        prefix = 'stochastic'

    logger.info(' {} val. cont. loss: {:.4f}\n'.format(prefix, validation_loss))

    return validation_loss


def main():
    logger = get_logger()

    parser = common_parser(pac_bayes=True)
    args = parser.parse_args()
    check_args(args, pac_bayes=True)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    is_criterion_val_loss = args.criterion == 'loss'

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    rnd = np.random.RandomState(args.seed)
    iid = not args.non_iid

    device = torch.device('cuda' if use_cuda else 'cpu')

    contrastive_loss = ContrastiveLoss(loss_name=args.loss, device=device)

    # `num_blocks_per_class` is ignored due to non-iid setting.
    train_loader, val_loader = get_contrastive_data_loaders(
        rnd=rnd,
        data_name='auslan',
        validation_ratio=args.validation_ratio,
        mini_batch_size=args.batch_size,
        num_blocks_per_class=45 * 24,  # this value is ignored when iid is False.
        block_size=args.block_size,
        neg_size=args.neg_size,
        root=args.root,
        include_test=False,
        iid=iid
    )

    num_training_samples = len(train_loader.dataset)
    if val_loader is None:
        num_val_samples = 0
    else:
        num_val_samples = len(val_loader.dataset)
        if args.criterion == 'pb':
            logger.warn('You can pass 0. to `validation-ratio` argument. It could make performance better.')

    logger.info('# training samples: {} # val samples: {}\n'.format(num_training_samples, num_val_samples))
    logger.info('PAC-Bayes parameters: b: {}, c: {}, δ: {}, prior log std: {}\n'.format(
        args.b, args.c, args.delta, args.prior_log_std
    ))

    model = StochasticMLP(
        num_training_samples=num_training_samples,
        rnd=rnd,
        num_last_units=args.dim_h,
        catoni_lambda=args.catoni_lambda, b=args.b, c=args.c, delta=args.delta,
        prior_log_std=args.prior_log_std,
    ).to(device)

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

    if is_criterion_val_loss:
        early_stoppings = {
            'stochastic': EarlyStopping(mode='min', patience=args.patience),
            'deterministic': EarlyStopping(mode='min', patience=args.patience),
        }

        learning_histories = {
            'stochastic': {
                'val_loss': []
            },
            'deterministic': {
                'val_loss': []
            }
        }
    else:
        learning_history = {args.criterion: []}

    save_name = 'lr-{}_{}_{}_{}'.format(args.lr, optimizer_name, args.criterion, args.output_model_name)
    if is_criterion_val_loss:
        save_names = dict()

        save_names['stochastic'] = 'lr-{}_{}_{}_stochastic_{}'.format(
            args.lr, optimizer_name, args.criterion, args.output_model_name
        )
        save_names['deterministic'] = 'lr-{}_{}_{}_deterministic_{}'.format(
            args.lr, optimizer_name, args.criterion, args.output_model_name
        )

    if iid:
        T = 0.
    else:
        T = args.block_size

    for epoch in range(1, args.epoch + 1):
        average_objective = train(
            args, model, device, train_loader, optimizer, epoch, contrastive_loss, T, logger
        )
        scheduler.step()

        # calculate criterion value for early-stopping
        if is_criterion_val_loss:
            delete_keys = []
            for eval_type, early_stopping in early_stoppings.items():
                is_deterministic = eval_type == 'deterministic'

                val_loss = validation_loss(
                    args, model, device, val_loader, contrastive_loss,
                    logger, args.num_snn, deterministic=is_deterministic
                )

                learning_histories[eval_type]['val_loss'].append(val_loss)

                # check early_stopping
                is_stopped = early_stopping.is_stopped_and_save(
                    val_loss, model,
                    save_name=save_names[eval_type]
                )

                if is_stopped:
                    delete_keys.append(eval_type)
                    learning_histories[eval_type]['lowest_val_loss'] = early_stopping.best

            for delete_key in delete_keys:
                logger.info('Remove {} evaluation\n'.format(delete_key))
                del early_stoppings[delete_key]

            # if early stopping dict becomes empty, stop the training
            if not early_stoppings:
                break

        else:
            learning_history[args.criterion].append(average_objective)

    # save learning history to json
    if is_criterion_val_loss:
        # store the lowest validation loss
        for eval_type, early_stopping in early_stoppings.items():
            filed_name = 'lowest_val_loss'
            learning_histories[eval_type][filed_name] = early_stopping.best

        for eval_type, fname in save_names.items():
            json_fname = fname.replace('.pt', '.json')
            with open(json_fname, 'w') as log_file:
                json.dump(learning_histories[eval_type], log_file)

    else:
        torch.save(model.state_dict(), save_name)
        json_fname = save_name.replace('.pt', '.json')
        with open(json_fname, 'w') as log_file:
            json.dump(learning_history, log_file)


if __name__ == '__main__':
    main()
