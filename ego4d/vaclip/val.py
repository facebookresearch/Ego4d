import hydra
import torch
import copy
import torch.nn.functional as F
from torch.nn import Identity

from tqdm.auto import tqdm
from omegaconf import OmegaConf
from ego4d.vaclip.config import EgoPreprocessConfig, TrainConfig

from ego4d.features.config import (
    FeatureExtractConfig,
    load_model,
)
from ego4d.vaclip.dataset import KineticsDset, create_data_loader


# taken from: https://github.com/mlfoundations/open_clip/blob/main/src/training/zero_shot.py#L29
def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def eval_classification(model, classifier, loader, device=None):
    with torch.no_grad():
        acc1, acc5, n = 0.0, 0.0, 0
        for x, target in tqdm(loader):
            if device is not None:
                x = x.to(device)
                target = target.to(device)

            v = model(x)

            # https://github.com/mlfoundations/open_clip/blob/main/src/training/zero_shot.py#L49
            if isinstance(classifier, torch.Tensor):
                v = F.normalize(v, dim=-1)
                logits = v @ classifier
            else:
                logits = classifier(v)

            a1, a5 = accuracy(logits, target, topk=(1, 5))

            # This is to test correctness
            # hot1 = torch.zeros(x.shape[0], 400)
            # for i in range(x.shape[0]):
            #     hot1[i, target[i]] = 1.0
            # a1, a5 = accuracy(hot1.to(self.device), target, topk=(1, 5))

            acc1 += a1
            acc5 += a5
            n += x.shape[0]

    acc1 /= n
    acc5 /= n
    return {
        "acc1": 100 * acc1,
        "acc5": 100 * acc5,
    }


def eval_k400_on_features(config: TrainConfig, feature_extract_config: FeatureExtractConfig):
    # NOTE: only works for omnivore right now
    dset = KineticsDset(config)
    config = copy.deepcopy(config)
    val_loader = create_data_loader(dset, config)

    omni_model = load_model(feature_extract_config, patch_final_layer=False)
    omni_model = omni_model.cuda()
    classifier = omni_model.model.heads.video[1]  # linear layer
    assert isinstance(classifier, torch.nn.Linear)

    model = Identity()
    classifier.eval()

    return eval_classification(model, classifier, val_loader, "cuda")


@hydra.main(config_path="configs", config_name=None)
def main(config: TrainConfig):
    feature_extract_config = OmegaConf.load(config.k400_pre_config.feature_extract_config_path)
    res = eval_k400_on_features(config, feature_extract_config)
    print(res)


if __name__ == "__main__":
    main()  # pyre-ignore
