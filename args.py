import argparse


def str2bool(str):
    return True if str.lower() == 'true' else False


def arg_parse():
    parser = argparse.ArgumentParser(description='Dynamic Time Series Graph.')
    # General parameters
    parser.add_argument('--dataset', dest='dataset', type=str, help='Dataset: 10a, 10b')
    parser.add_argument('--lr',
                        help='the lr of the current model for training; the lr of the former model for fine-tuning',
                        type=float, default=0.001)
    parser.add_argument('--batch-size', dest='batch_size', type=int, help='Batch size.')
    parser.add_argument('--epochs', dest='epochs', type=int, help='Number of epochs to train.')
    parser.add_argument('--num_workers', dest='num_workers', type=int, help='Number of workers to load data.')

    # Parameters in our method
    parser.add_argument('--num_filter', help='The hyperparameter m in the paper', type=int)

    # default parameters
    parser.set_defaults(lr=0.001, batch_size=128, epochs=50, num_workers=8, dataset='10a', num_filter=11)
    return parser.parse_args()
