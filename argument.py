import argparse

def parser():
    parser = argparse.ArgumentParser(description='Defensive Perturbation')
    parser.add_argument('--dataset', default='MNIST', help='use what dataset')
    parser.add_argument('--log_root', default='log', 
        help='the directory to save the logs or other imformations (e.g. images)')
    parser.add_argument('--model_root', default='./ckpt/', help='the directory to save the models')
    parser.add_argument('--load_checkpoint', default='./best_model_CNN.pth')
    parser.add_argument('--affix', default='', help='the affix for the save folder')
    parser.add_argument('--data_root', default='./data/dp/',
        help='the directory to save the dataset')
    parser.add_argument('--max_epoch', '-m_e', type=int, default=1, 
        help='the maximum numbers of the model see a sample')
    # parameters for generating adversarial examples
    parser.add_argument('--epsilon', '-e', type=float, default=2.5,
        help='maximum perturbation of adversaries')
    parser.add_argument('--alpha', '-a', type=float, default=0.075,
        help='movement multiplier per iteration when generating adversarial examples')
    parser.add_argument('--k', '-k', type=int, default=40,
        help='maximum iteration when generating adversarial examples')
    parser.add_argument('--max_iter', type=int, default=80, 
        help='maximum iteration when generating defensive examples')



    parser.add_argument('--batch_size', '-b', type=int, default=512, help='batch size')
    #parser.add_argument('--learning_rate', '-lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--learning_rate_for_def', '-lrd', type=float, default=0.1, help='learning rate')

    parser.add_argument('--n_store_image_step', type=int, default=100, 
        help='number of iteration to save adversaries')
    parser.add_argument('--perturbation_type', '-p', choices=['linf', 'l2'], default='l2',
        help='the type of the perturbation (linf or l2)')
    
    return parser.parse_args()

def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.info('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))
