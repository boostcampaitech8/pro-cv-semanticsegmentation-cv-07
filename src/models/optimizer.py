import torch.optim as optim


def get_optimizer(cfg, model):
    optim_name = cfg.optimizer.lower()

    optim_dict = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "sgd": optim.SGD,
    }

    if optim_name not in optim_dict:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")

    if optim_name == "sgd":
        return optim_dict[optim_name](
            model.parameters(),
            lr=0.05,
            momentum=0.9,
            weight_decay=1e-4
        )
    else:
        return optim_dict[optim_name](
            model.parameters(),
            lr=cfg.lr,
            weight_decay=1e-6,
        )