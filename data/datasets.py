import torch
import torchvision
import os
from torchvision import transforms

def get_datasets(args):
    device = torch.device('cuda:'+f'{args.cuda}') if torch.cuda.is_available() else 'cpu'
    if "Gradnorm" in args.cal_method:
        args.batch_size = 1
    # switch to in-distribution dataset 
    if args.dataset == 'cifar10':
        args.num_classes = num_class = 10
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        transform = transforms.Compose([
            transforms.CenterCrop(size=(32, 32)),
            transforms.ToTensor(),
            normalize
        ])
        ood_name = ['svhn',  'lsun_c',  'tinyimagenet_c',  'places', 'textures']
        in_test = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10(args.data_dir, train=False, transform=transform), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        svhn = torch.utils.data.DataLoader(torchvision.datasets.SVHN(args.data_dir, split='test', transform=transform),
                                                                                                batch_size=args.batch_size, num_workers=args.num_workers)

        lsun_c = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'LSUN'),
                                                                                        transform=transform),
                                                                                        batch_size=args.batch_size, num_workers=args.num_workers)

        tinyimagenet_c = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'Imagenet'),
                                                                                                transform=transform),
                                                                                                batch_size=args.batch_size, num_workers=args.num_workers)

        places = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'Places'),
                                                                                                transform=transform),
                                                               batch_size=args.batch_size, num_workers=args.num_workers)

        textures = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'textures/images'),
                                                                                                transform=transform),
                                                                batch_size=args.batch_size, num_workers=args.num_workers)
        ood_datasets = [svhn,  lsun_c,  tinyimagenet_c,  places, textures]
    elif args.dataset == 'cifar100':
        args.num_classes = num_class = 100
        normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        transform = transforms.Compose([
            transforms.CenterCrop(size=(32, 32)),
            transforms.ToTensor(),
            normalize
        ])

        ood_name = ['svhn',  'lsun_c',  'tinyimagenet_c',   'places', 'textures']

        in_test = torch.utils.data.DataLoader(torchvision.datasets.CIFAR100(args.data_dir, train=False, transform=transform), batch_size=args.batch_size, num_workers=args.num_workers)
        svhn = torch.utils.data.DataLoader(torchvision.datasets.SVHN(args.data_dir, split='test', transform=transform),
                                                batch_size=args.batch_size, num_workers=args.num_workers)

        lsun_c = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'LSUN'),
                                                                                        transform=transform),
                                                        batch_size=args.batch_size, num_workers=args.num_workers)


        tinyimagenet_c = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'Imagenet'),
                                                                                                transform=transform),
                                                                batch_size=args.batch_size, num_workers=args.num_workers)

        places = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'Places'),
                                                                                                transform=transform),
                                                               batch_size=args.batch_size, num_workers=args.num_workers)

        textures = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'textures/images'),
                                                                                                transform=transform),
                                                                batch_size=args.batch_size, num_workers=args.num_workers)
        ood_datasets = [svhn,  lsun_c,  tinyimagenet_c,  places, textures]

    elif args.dataset == 'imagenet':
        args.num_classes = num_class = 1000
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        crop = 480
        transform = transforms.Compose([
            transforms.Resize(size=(crop, crop)),
            transforms.ToTensor(),
            normalize
        ])
        
        ood_name = ['places', 'textures', 'sun', 'iNaturalist']
        in_test = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'imagenet', 'val'),
                                                                                            transform=transform),
                                                            batch_size=args.batch_size, num_workers=args.num_workers)

        places = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'Places'),
                                                                                                transform=transform),
                                                                batch_size=args.batch_size, num_workers=args.num_workers)

        textures = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'textures/images'),
                                                                                                transform=transform),
                                                                batch_size=args.batch_size, num_workers=args.num_workers)

        sun = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'SUN'),
                                                                                                transform=transform),
                                                                batch_size=args.batch_size, num_workers=args.num_workers)

        iNaturalist = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'iNaturalist'),
                                                                                                transform=transform),
                                                                batch_size=args.batch_size, num_workers=args.num_workers)
        ood_datasets = [places, textures, sun, iNaturalist]
    print("data load over")
    return in_test, ood_datasets, ood_name