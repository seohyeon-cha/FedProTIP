
import argparse

def args_parser():
    
    # Optimization options

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--epochs', default=35, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=128, type=int, metavar='N',
                        help='train/test batchsize')

    parser.add_argument('-d', '--dataset', default='CIFAR10', type=str)

    parser.add_argument('-n', '--n_clients', default=10, type=int)
    
    parser.add_argument('--sampling_rate', default=1.0, type=float, help='sampling rate for clients')
    
    parser.add_argument('--n_classes', default=10, type=int)

    parser.add_argument('--alpha', default=0.1, type=float)
    
    parser.add_argument('--beta', default=0.05, type=float)

    parser.add_argument('--local_epochs', default=50, type=int, metavar='N',
                    help='Interval between pruning is performed')
    
    parser.add_argument('--thre', default=2.0, type=float, metavar='N',
                    help='Pruning threshold')
    
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
    
    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
    
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')

    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
    
    parser.add_argument('-c', '--checkpoint', default='./checkpoints', type=str, metavar='PATH',
                    help='path to save checkpoint')


    parser.add_argument('--decay', type=float, default=0.01, metavar='D',
                    help='decay for bit-sparse regularizer (default: 0.01)')
    
    
    args = parser.parse_args('')
    return args

