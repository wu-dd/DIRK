from data.fmnist import load_fmnist
from data.kmnist import load_kmnist
from data.cifar10 import load_cifar10
from data.cifar100 import load_cifar100
from data.cub200 import load_cub200
from data.flower102 import load_flower102
from data.pet37 import load_pet37

def get_loader(args,k=None):
    if args.dataset == 'fmnist':
        num_classes = 10
        train_loader, train_partialY_matrix, test_loader = load_fmnist(args)

    elif args.dataset == 'kmnist':
        num_classes = 10
        train_loader, train_partialY_matrix, test_loader = load_kmnist(args)

    elif args.dataset == "cifar10":
        num_classes = 10
        train_loader, train_partialY_matrix, test_loader = load_cifar10(args)

    elif args.dataset == 'cifar100':
        num_classes = 100
        train_loader, train_partialY_matrix, test_loader = load_cifar100(args)

    elif args.dataset == 'cub200':
        num_classes = 200
        train_loader, train_partialY_matrix, test_loader = load_cub200(args)

    elif args.dataset == 'flower102':
        num_classes = 102
        train_loader, train_partialY_matrix, test_loader = load_flower102(args)

    elif args.dataset == 'pet37':
        num_classes = 37
        train_loader, train_partialY_matrix, test_loader = load_pet37(args)

    return train_loader,test_loader, num_classes











