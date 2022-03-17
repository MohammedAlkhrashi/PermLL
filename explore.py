# %%
from dataset import cifar_10_dataloaders
from group_utils import create_group_optimizer
from model import GroupModel, create_group_model
from torch.optim import SGD


def get_config():
    config = dict()
    config["networks_lr"] = 0.001
    config["permutation_lr"] = 0.001
    config["epochs"] = 100
    config["batch_size"] = 16
    config["pretrained"] = False

    config["noise"] = 0
    config["upperbound_exp"] = False

    config["networks_per_group"] = 3
    config["num_groups"] = 1

    config["gpu_num"] = "0"
    config["num_workers"] = 8
    return config


config = get_config()
loaders = cifar_10_dataloaders(
    batch_size=config["batch_size"],
    noise=config["noise"],
    num_workers=config["num_workers"],
    upperbound=config["upperbound_exp"],
)
model: GroupModel = create_group_model(
    config["num_networks"],
    num_classes=10,
    pretrained=config["pretrained"],
    dataset_targets=loaders["train"].dataset.dataset.targets,
)
optimizer = create_group_optimizer(
    model, networks_lr=config["networks_lr"], permutation_lr=config["permutation_lr"]
)
for name, p in model.named_parameters():
    print(name)
    print(p)
# %%
networks_params = []
permutations_params = []
for name, param in model.named_parameters():
    if name.startswith("perm_model"):
        permutations_params.append(param)
    else:
        print(name)
        networks_params.append(param)
# %%
n_optimizer = SGD(networks_params, lr=config["learning_rate"])
p_optimizer = SGD(permutations_params, lr=config["learning_rate"])
# %%
p_optimizer.param_groups
# %%
len(list(model.parameters()))
# %%
len(networks_params) + len(permutations_params)

# %%
optimizer = create_group_optimizer(
    model, networks_lr=config["networks_lr"], permutation_lr=config["permutation_lr"]
)
# %%

