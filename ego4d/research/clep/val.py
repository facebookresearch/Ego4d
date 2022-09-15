import copy

import hydra
import torch
import torch.nn.functional as F

from ego4d.features.config import FeatureExtractConfig, load_model
from ego4d.research.clep.config import TrainConfig
from ego4d.research.clep.dataset import create_data_loader, create_kinetics_dset
from ego4d.research.clep.utils import charades_map
from omegaconf import OmegaConf
from torch.nn import Identity

from tqdm.auto import tqdm


# taken from: https://github.com/mlfoundations/open_clip/blob/main/src/training/zero_shot.py#L29
def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


def eval_multi_class_classification(model, classifier, loader, device=None):
    pred_arr = []
    gt_arr = []
    with torch.no_grad():
        for x, target in tqdm(loader):
            assert x.shape[0] == 1

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

            pred = torch.nn.Softmax(dim=1)(logits.mean(1))
            pred_arr.append(pred.squeeze().cpu())
            gt_arr.append(target.squeeze().cpu())

    # res = mAP(torch.stack(pred_arr).numpy(), torch.stack(gt_arr).numpy())
    res = charades_map(torch.stack(pred_arr).numpy(), torch.stack(gt_arr).numpy())
    return {
        "mAP": res[0] * 100,
    }


def eval_classification(model, classifier, loader, device=None, avg_logits=False):
    with torch.no_grad():
        acc1, acc5, n = 0.0, 0.0, 0
        for x, target in tqdm(loader):
            if avg_logits:
                assert x.shape[0] == 1

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

            if avg_logits:
                a1, a5 = accuracy(logits.mean(1), target, topk=(1, 5))
            else:
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


def eval_k400_on_features(
    config: TrainConfig, feature_extract_config: FeatureExtractConfig
):
    # NOTE: only works for omnivore right now
    dset, _ = create_kinetics_dset(config)
    config = copy.deepcopy(config)
    config.batch_size = 1
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
    feature_extract_config = OmegaConf.load(
        config.input_config.feature_extract_config_path
    )
    res = eval_k400_on_features(config, feature_extract_config)
    print(res)


if __name__ == "__main__":
    main()  # pyre-ignore
