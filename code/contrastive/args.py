import argparse


def check_args(args, pac_bayes=False, train=True) -> None:
    """
    Check the argument values

    :param args: Parsed args.
    :param pac_bayes: PAC-Bayes model's flag.
    :param train: training model flag.

    :return: None
    """

    assert args.lr > 0.
    assert args.epoch > 0
    assert args.batch_size > 1
    assert args.optim.lower() in ['adam', 'sgd', 'rmsprop']
    assert 0. <= args.validation_ratio < 1.
    assert args.patience >= 1
    assert args.momentum >= 0.
    assert args.gamma > 0.
    assert args.dim_h > 0
    assert args.block_size > 0
    assert args.neg_size > 0
    assert args.num_blocks_per_class > 0
    assert args.loss in ['hinge', 'logistic', 'zero-one']
    assert '.json' not in args.output_model_name

    if not train:
        assert args.num_avg_classes > 1
        if pac_bayes:
            assert args.num_trials > 0

    if pac_bayes:
        assert args.catoni_lambda > 0.
        assert args.b > 0.
        assert args.c > 0.
        assert args.delta > 0.
        assert args.num_snn > 0
        assert args.criterion in ['pb', 'loss']


def common_parser(pac_bayes=False, train=True):
    """

    :param pac_bayes: PAC-Bayes model's flag.
    :param train: training model flag.

    :return: Parser
    """

    parser = argparse.ArgumentParser(description='Experiment\'s args')

    # common learning parameters
    parser.add_argument('--lr', type=float, default=0.001, metavar='L',
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--epoch', type=int, default=500, metavar='E',
                        help='Input batch size for training (default: 500)')
    parser.add_argument('--batch-size', type=int, default=100, metavar='M',
                        help='Input batch size for training (default: 100)')
    parser.add_argument('--optim', type=str, default='RMSProp',
                        help='Optimizer name [SGD, Adam, RMSProp] (default: RMSProp)')
    parser.add_argument('--patience', type=int, default=20,
                        help='The number of epochs for earlystopping (default: 20)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Momentum parameter for SGD')
    parser.add_argument('--schedule', type=int, nargs='+', default=[375],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Learning rate is multiplied by gamma on schedule.')

    # contrastive learning parameters
    parser.add_argument('--block-size', type=int, default=2, metavar='B',
                        help='The size of block size per a sample (default: 2)')
    parser.add_argument('--neg-size', type=int, default=4, metavar='K',
                        help='The number of negative samples per a sample (default: 4)')
    parser.add_argument('--num-blocks-per-class', type=int, default=500, metavar='K',
                        help='The number of blocks pre a class (default: 500)')
    parser.add_argument('--loss', type=str, default='logistic', metavar='L',
                        help='Type of loss function. hinge or logistic (default: logistic)')

    parser.add_argument('--seed', type=int, default=7, metavar='S',
                        help='Random seed (default: 7)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='How many batches to wait before logging training status (default: 100)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--output-model-name', type=str, default='weights.pt', metavar='O',
                        help='Path and filename to save model weights.'
                             'Note `.json` and `.pt` are used for evaluation code, '
                             'so please do not use these sub-string to save model weights.'
                        )
    parser.add_argument('--model-name', type=str, default=None, metavar='M',
                        help='Path and filename of saved model weights')
    parser.add_argument('--mlp', action='store_true', default=False,
                        help='shallow model (MLP) setting. (default: False)')
    parser.add_argument('--supervised', action='store_true', default=False,
                        help='Load supervised model')
    parser.add_argument('--output-json-fname', type=str, default='./results.json',
                        help='json path to save training results (default: results.json)')
    parser.add_argument('--validation-ratio', type=float, default=0.05,
                        help='The ratio of validation data. Validation set is a part of training set. '
                             '(default: 0.05)')
    parser.add_argument('--dim-h', type=int, default=100,
                        help='The dimensionality of embedding space (default: 100)')
    parser.add_argument('--root', type=str, default='~/data',
                        help='Absolute data path (default: ~/data)')
    parser.add_argument('--non-iid', action='store_true', default=False,
                        help='Non-iid mode. (default: False)')

    # evaluation part
    if not train:
        parser.add_argument('--model-name-dir', type=str, default='',
                            help='Model weights and training results dir.'
                                 'If this value is empty, just use `model-name`s file (default:\'\')')
        parser.add_argument('--num-avg-classes', type=int, default=2,
                            help='Parameter for average loss (default: 2)')
        parser.add_argument('--num-trials', type=int, default=5,
                            help='The number of trials for average-k mean classifier (default: 5)')

        if pac_bayes:
            parser.add_argument('--deterministic', action='store_true', default=False,
                                help='Deterministic evaluation mode. (default: False)')
            parser.add_argument('--json-fname', type=str, default='pb_bound_values.json',
                                help='json file name that contains pre-computed terms to compute PAC-Bayes bound')

    # for PAC-bayesian setting
    if pac_bayes:
        parser.add_argument('--prior-log-std', type=float, default=-4.0, metavar='P',
                            help='Initial value of prior\'s log standard deviation (default: -4.0)')

        parser.add_argument('--catoni-lambda', type=float, default=1.0, metavar='C',
                            help='Lambda value in Catoni\'s bound (default: 1.0)')
        parser.add_argument('--b', type=int, default=100, metavar='B',
                            help='Prior\'s precision parameter. (default: 100)')
        parser.add_argument('--c', type=float, default=0.1, metavar='C',
                            help='Prior\'s upper bound parameter. (default: 0.1)')

        parser.add_argument('--delta', type=float, default=0.05,
                            help='Confidence parameter. (default: 0.05)')
        parser.add_argument('--num-snn', type=int, default=10, metavar='S',
                            help='The number of samplings for stochastic NN per each computing eval/validation '
                                 '(default: 10)')
        parser.add_argument('--criterion', type=str, default='loss', metavar='C',
                            help='Criterion of early-stopping. Valid argument is either `loss` or `pb` (default: loss)')

    return parser
