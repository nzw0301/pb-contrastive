import glob
import json

import numpy as np
import torch

from ..args import common_parser
from ..datasets.auslan import get_shape_for_contrastive_learning as auslan_get_shape_for_contrastive_learning
from ..datasets.cifar100 import get_shape_for_contrastive_learning as cifar100_get_shape_for_contrastive_learning
from ..datasets.contrastive import get_contrastive_data_loaders
from ..loss import ContrastiveLoss
from ..models.pb_models import StochasticCNN, StochasticMLP
from ..utils.logger import get_logger


def report_eval(
        args,
        device: torch.device,
        contrastive_data_loader: torch.utils.data.DataLoader,
        loss_names: list,
        model,
        get_shape_for_contrastive_learning
) -> dict:
    """
    Calculate losses for parameter-selection.

    :param args: Parsed args.
    :param device: PyTorch's device instance.
    :param contrastive_data_loader: Data loader of contrastive data.
    :param loss_names: list of loss function names.
    :param model: Trained model instance
    :param get_shape_for_contrastive_learning: Function to get shape of contrastive mini-batch.

    :return: Dict contains losses that specified `loss_names`.
    """

    M = len(contrastive_data_loader.dataset)

    loss_function_dict = {}
    empirical_losses_per_realised_snn = {}
    for loss_name in loss_names:
        loss_function_dict[loss_name] = ContrastiveLoss(loss_name=loss_name, device=device)
        empirical_losses_per_realised_snn[loss_name] = []

    if args.deterministic:
        num_snn = 1
    else:
        num_snn = args.num_snn

    model.eval()
    with torch.no_grad():
        for snn_index in range(1, num_snn + 1):
            if args.deterministic:
                model.deterministic()
            else:
                model.sample_noise()

            loss_name_to_sum_loss = {}
            for loss_name in loss_names:
                loss_name_to_sum_loss[loss_name] = 0.

            for batch_idx, (images, pos, negs) in enumerate(contrastive_data_loader):

                (batch2input_shape_pos, batch2input_shape_neg, output2emb_shape_pos, output2emb_shape_neg) = \
                    get_shape_for_contrastive_learning(
                        len(images), args.block_size, args.neg_size, args.dim_h
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

                for loss_name, contrastive_loss in loss_function_dict.items():
                    loss = contrastive_loss(
                        feature=data_features, positive_feature=pos_features,
                        negative_features=neg_features, reduction='sum'
                    )
                    loss_name_to_sum_loss[loss_name] += loss.item()

            for loss_name, sum_loss in loss_name_to_sum_loss.items():
                empirical_losses_per_realised_snn[loss_name].append(sum_loss / M)

    results = {
        loss_name: sum(snn_loss) / len(snn_loss)
        for loss_name, snn_loss in empirical_losses_per_realised_snn.items()
    }

    return results


def main():
    logger = get_logger()
    parser = common_parser(pac_bayes=True, train=False)

    parser.add_argument('--non-iid-bound', action='store_true', default=False,
                        help='Non-iid bound mode. (default: False)')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    is_iid_data = not args.non_iid
    is_iid_bound = not args.non_iid_bound

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    rnd = np.random.RandomState(args.seed)

    if args.mlp:
        train_loader, val_loader, test_loader = get_contrastive_data_loaders(
            rnd,
            data_name='auslan',
            validation_ratio=args.validation_ratio,
            mini_batch_size=args.batch_size,
            block_size=args.block_size,
            num_blocks_per_class=45 * 24,
            neg_size=args.neg_size,
            root=args.root,
            include_test=True,
            iid=is_iid_data
        )
        get_shape_for_contrastive_learning = auslan_get_shape_for_contrastive_learning
        m = len(train_loader.dataset)
        model = StochasticMLP(
            num_training_samples=m, rnd=rnd, num_last_units=args.dim_h, init_weights=True
        )
    else:
        train_loader, val_loader, test_loader = get_contrastive_data_loaders(
            rnd=rnd,
            data_name='cifar',
            validation_ratio=args.validation_ratio,
            mini_batch_size=args.batch_size,
            block_size=args.block_size,
            num_blocks_per_class=args.num_blocks_per_class,
            neg_size=args.neg_size,
            include_test=True
        )
        get_shape_for_contrastive_learning = cifar100_get_shape_for_contrastive_learning
        m = len(train_loader.dataset)
        model = StochasticCNN(
            num_training_samples=m, rnd=rnd, num_last_units=args.dim_h, init_weights=True
        )

    if args.criterion == 'pb' and val_loader is not None:
        logger.warn('You can pass 0. to `validation-ratio` argument. It can make the performance better.')

    if args.model_name_dir[-1] == '/':
        args.model_name_dir = args.model_name_dir[:-1]

    fnames = glob.glob('{}/*.pt'.format(args.model_name_dir))

    pb_bound_ingredients = {}
    loss_names = ['logistic', 'zero-one']

    for i, fname in enumerate(fnames, start=1):
        pb_bound_ingredients[fname] = {}

        model.load_state_dict(torch.load(fname, map_location=device))
        model = model.to(device)

        assert m == model.num_training_samples

        # train
        results = report_eval(
            args, device, train_loader, loss_names, model, get_shape_for_contrastive_learning
        )

        for loss_name, loss_value in results.items():
            pb_bound_ingredients[fname]['{}-{}-loss'.format('train', loss_name)] = loss_value

        # test
        results = report_eval(
            args, device, test_loader, loss_names, model, get_shape_for_contrastive_learning
        )

        for loss_name, loss_value in results.items():
            pb_bound_ingredients[fname]['{}-{}-loss'.format('test', loss_name)] = loss_value

        progress = i / len(fnames) * 100.
        logger.info('\r{:.1f}% {}'.format(progress, fname))

        # Compute empirical risk and a part of bounds.
        if is_iid_bound:
            # KL divergence based PAC-Bayesian bound.
            # NOTE: this complexity term does not contain `np.log(2. * np.sqrt(M))` yet.

            kl, union_bound_value = model.compute_complexity_terms_with_discretized_prior_variance()
            complexity_term = kl + union_bound_value

            pb_bound_ingredients[fname]['kl'] = kl
            pb_bound_ingredients[fname]['lambda'] = model.catoni_lambda.item()
            pb_bound_ingredients[fname]['complexity'] = complexity_term
        else:
            # Chi-square divergence based PAC-Bayesian bound.

            chi_square, union_bound_value = model.compute_non_iid_complexity_terms_with_discretized_prior_variance()

            pb_bound_ingredients[fname]['chi_square'] = chi_square
            pb_bound_ingredients[fname]['delta'] = model.delta.item()
            if is_iid_data:
                pb_bound_ingredients[fname]['T'] = 0
            else:
                pb_bound_ingredients[fname]['T'] = args.block_size

        pb_bound_ingredients[fname]['union'] = union_bound_value
        pb_bound_ingredients[fname]['m'] = model.num_training_samples.item()

    with open(args.json_fname, 'w') as log_file:
        json.dump(pb_bound_ingredients, log_file)


if __name__ == '__main__':
    main()
