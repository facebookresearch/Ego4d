import hydra
from ego4d.features.config import FeatureExtractConfig
from ego4d.research.clep.config import TrainConfig
from ego4d.research.clep.preprocess.cc import preprocess_cc
from ego4d.research.clep.preprocess.charades import preprocess_ego_charade
from ego4d.research.clep.preprocess.ego4d_data import (
    preprocess_ego_features,
    preprocess_ego_narrations,
)
from ego4d.research.clep.preprocess.kinetics import preprocess_k400_data


@hydra.main(config_path="configs", config_name=None)
def preprocess(config: TrainConfig):
    if config.pre_config.mode == "ego4d_narr":
        preprocess_ego_narrations(config, config.pre_config.ego4d_narr)
    elif config.pre_config.mode == "ego4d_features":
        preprocess_ego_features(
            config.input_config.feature_path,
            config,
            config.pre_config.ego4d_features,
        )
    elif config.pre_config.mode == "k400":
        preprocess_k400_data(config, config.pre_config.k400)
    elif config.pre_config.mode == "ego_charade":
        preprocess_ego_charade(config, config.pre_config.ego_charade)
    elif config.pre_config.mode == "cc":
        preprocess_cc(config, config.pre_config.cc)
    else:
        raise AssertionError(f"{config.pre_config.mode} not supported")


if __name__ == "__main__":
    preprocess()  # pyre-ignore
