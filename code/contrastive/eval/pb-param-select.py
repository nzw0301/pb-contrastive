from .common import pb_parameter_selection
from ..args import common_parser
from ..utils.logger import get_logger


def main():
    logger = get_logger()
    parser = common_parser(pac_bayes=True, train=False)

    args = parser.parse_args()

    model_name = pb_parameter_selection(logger, json_fname=args.json_fname, num_train_data=args.num_train)
    logger.info('Best model with respect to PAC-Bayes bound:\n{}'.format(model_name))


if __name__ == '__main__':
    main()
