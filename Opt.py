import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default=None, choices=['train', 'test'])
    parser.add_argument('--data_type', default='utk', choices=['utk', 'cacd'], type=str, help='dataset option')
    parser.add_argument('--data_path', default='./datalists/UTKFace/images/', type=str, help='path to dataset images')
    parser.add_argument('--model_path', type=str, default='./trained_models/' + 'utk' + '.pt', help='path to trained_models model')
    parser.add_argument('--checkpoint_path', type=str, default='./trained_models/' + 'utk' + '.pt', help='path to save newly trained_models model')
    parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
    parser.add_argument('--batch', default=64, type=int, help="batch size")
    parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float)
    parser.add_argument('--gpu', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    return args
