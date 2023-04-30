#!/usr/bin/python3
import argparse
import os.path

import torch
import torchvision
import yaml

from pytorch_scripts.utils import build_model

DATA_DIR = "/home/carol/git_research/diehardnet/data"
INTERMEDIATE_LAYERS = dict()
INTERMEDIATE_LAYERS_OUTPUT_PATH = "/tmp/intermediate_layers_output"
INJECTION_COUNTER_FILE = "/tmp/injection_counter.txt"


def load_cifar100(data_dir: str, transform: torchvision.transforms.Compose) -> torch.utils.data.DataLoader:
    """Load CIFAR 100 from <data dir>"""
    # Get a dataset
    test_set = torchvision.datasets.cifar.CIFAR100(root=data_dir, download=False, train=False,
                                                   transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
    return test_loader


def load_cifar10(data_dir: str, transform: torchvision.transforms.Compose) -> torch.utils.data.DataLoader:
    """Load CIFAR 10 from <data dir>"""
    # Get a dataset
    test_set = torchvision.datasets.cifar.CIFAR10(root=data_dir, download=False, train=False,
                                                  transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
    return test_loader


def load_ptl_model(args):
    # Build model (Resnet only up to now)
    optim_params = {'optimizer': args.optimizer, 'epochs': args.epochs, 'lr': args.lr, 'wd': args.wd}
    n_classes = 10 if args.dataset == 'cifar10' else 100

    ptl_model = build_model(model=args.model, n_classes=n_classes, optim_params=optim_params,
                            loss=args.loss, inject_p=args.inject_p,
                            order=args.order, activation=args.activation, affine=args.affine)
    checkpoint = torch.load(args.ckpt)
    ptl_model.load_state_dict(checkpoint['state_dict'])
    return ptl_model.model


def hook_fn(model, hook_input, hook_output):
    global INTERMEDIATE_LAYERS
    INTERMEDIATE_LAYERS[model] = hook_output


def get_all_layers(net):
    for name, layer in net._modules.items():
        # If it is a sequential, don't register a hook on it
        # but recursively register hook on all it's module children
        if isinstance(layer, torch.nn.Sequential):
            get_all_layers(layer)
        else:
            # it's a non-sequential. Register a hook
            layer.register_forward_hook(hook_fn)


def perform_fault_injection_for_a_model(args, config_file_name):
    global INTERMEDIATE_LAYERS
    img_indexes = [0, 1, 30, 150, 200, 250, 330, 1000, 1002, 1010]
    gold_path = str(config_file_name).replace(".yaml", "_golden.pt")
    generate = args.generate
    save_layers = args.savelayers
    model = load_ptl_model(args=args)
    model.eval()
    model = model.to("cuda")
    load_data = load_cifar100
    if args.dataset == "cifar10":
        load_data = load_cifar10

    test_loader = load_data(data_dir=DATA_DIR,
                            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                      torchvision.transforms.Normalize(
                                                                          mean=[0.485, 0.456, 0.406],
                                                                          std=[0.229, 0.224, 0.225])]))
    # Get only the first 10 images
    images = list()
    for i, (image, label) in enumerate(test_loader):
        images.append((image, label))
        if i == img_indexes[-1]:
            break

    gold_probabilities_list = list()
    if generate is False:
        gold_probabilities_list = torch.load(gold_path)

    if save_layers:
        get_all_layers(net=model)

    injection_counter = 0
    if save_layers:
        if os.path.isfile(INJECTION_COUNTER_FILE):
            with open(INJECTION_COUNTER_FILE) as fp:
                injection_counter = int(fp.readline())
        with open(INJECTION_COUNTER_FILE, "w") as fp:
            injection_counter += 1
            fp.write(str(injection_counter))

    # total_time = time.time()
    with torch.no_grad():
        for i in range(len(img_indexes)):
            img_index = img_indexes[i]
            image, label = images[img_index]
            image_gpu = image.to("cuda")
            # Golden execution
            # model_time = time.time()
            dnn_output = model(image_gpu, inject=False)
            # model_time = time.time() - model_time

            probabilities = dnn_output.to("cpu")
            top1_label = int(torch.topk(probabilities, k=1).indices.squeeze(0))
            top1_prob = torch.softmax(probabilities, dim=1)[0, top1_label].item()
            # cmp_gold_prob[7] = 333333
            if generate is False:
                gold_probabilities = gold_probabilities_list[i]
                gold_top1_label = int(torch.topk(gold_probabilities, k=1).indices.squeeze(0))
                gold_top1_prob = torch.softmax(gold_probabilities, dim=1)[0, gold_top1_label].item()
                cmp_gold_prob = torch.flatten(gold_probabilities)
                cmp_out_prob = torch.flatten(probabilities)
                if torch.any(torch.not_equal(cmp_gold_prob, cmp_out_prob)):
                    print(f"SDC detected. IMG INDEX {img_index}")
                    for it, (g, f) in enumerate(zip(cmp_gold_prob, cmp_out_prob)):
                        if g != f:
                            print(f"{it} e:{g} r:{f}")
                    if gold_top1_label != top1_label:
                        print(f"Critical SDC detected. "
                              f"e_label:{gold_top1_label} r_label:{top1_label} "
                              f"e_prob:{gold_top1_prob} r_prob:{top1_prob}")
                    if save_layers:
                        torch.save(INTERMEDIATE_LAYERS, f"{INTERMEDIATE_LAYERS_OUTPUT_PATH}_{injection_counter}.pt")
            else:
                gold_probabilities_list.append(probabilities)
        # total_time = time.time() - total_time
        # print(f"TOTAL TIME:{total_time:.2f} MODEL TIME:{model_time:.2f}")

    if generate is True:
        torch.save(gold_probabilities_list, gold_path)
    print("Finish computation.")


def main() -> None:
    config_parser = argparse.ArgumentParser(description='Criticality eval', add_help=False)
    config_parser.add_argument('--config', default='', type=str, metavar='FILE',
                               help='YAML config file specifying default arguments.')
    args, remaining_argv = config_parser.parse_known_args()

    defaults = {"option": "default"}

    if args.config:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
        defaults.update(**cfg)
    config_file_name = args.config
    # Parse rest of arguments
    # Don't suppress add_help here so it will handle -h
    parser = argparse.ArgumentParser(
        # Inherit options from config_parser
        parents=[config_parser]
    )
    parser.set_defaults(**defaults)
    parser.add_argument('--generate', default=False, action="store_true",
                        help="Set this flag to generate the golds and reprogram the board")
    parser.add_argument('--savelayers', default=False, action="store_true",
                        help="Set this flag to save the intermediate layers of the DNN")
    args = parser.parse_args(remaining_argv)
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print()
    perform_fault_injection_for_a_model(args, config_file_name)


if __name__ == '__main__':
    main()
