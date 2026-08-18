import os
import random
import faulthandler
import signal
from typing import List, Tuple
from pathlib import Path
import logging
import pickle
from datetime import datetime

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"

faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)


class LocalMetadataDDPStrategy(DDPStrategy):
    """DDP strategy that avoids NCCL object broadcasts for identical local metadata."""

    def broadcast(self, obj, src: int = 0):
        return obj

def dist_ready():
    return dist.is_available() and dist.is_initialized()


def _load_token_filter(token_file: str) -> List[str]:
    """Load one scene token per line for restricting training to precomputed BEV features."""
    token_path = Path(token_file)
    if not token_path.is_file():
        raise FileNotFoundError(f"scene_filter_token_file does not exist: {token_path}")

    with open(token_path, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    
    print("Train without caching....")
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    token_file = cfg.get("scene_filter_token_file", None)
    if token_file:
        train_scene_filter.tokens = _load_token_filter(token_file)
        logger.info("Loaded %d scene filter tokens from %s", len(train_scene_filter.tokens), token_file)

    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs
    

    print("len(train_scene_filter.log_names) ", len(train_scene_filter.log_names))

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if token_file:
        val_scene_filter.tokens = train_scene_filter.tokens

    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """

    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")
    os.environ["DRIVOR_TRAIN_OUTPUT_DIR"] = str(cfg.output_dir)

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    # Load agent.checkpoint_path before Lightning wraps the module so fine-tuning
    # actually starts from the requested baseline weights.
    agent.initialize()

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        token_file = cfg.get("scene_filter_token_file", None)
        selected_tokens = _load_token_filter(token_file) if token_file else None
        if selected_tokens is not None:
            logger.info("Restricting cache-only datasets to %d scene tokens from %s", len(selected_tokens), token_file)
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
            tokens=selected_tokens,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
            tokens=selected_tokens,
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True,drop_last=True)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False,drop_last=True)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")

    # automatically resume training
    # find latest ckpt
    import glob
    def find_latest_checkpoint(search_pattern):
        # List all files matching the pattern
        list_of_files = glob.glob(search_pattern, recursive=True)
        # Find the file with the latest modification time
        if not list_of_files:
            return None
        latest_file = max(list_of_files, key=os.path.getmtime)
        return latest_file


    auto_resume_training = bool(cfg.get("auto_resume_training", True))
    if cfg.train_ckpt_path is None and auto_resume_training:
        # Pattern to match all .ckpt files in the base_path recursively
        search_pattern = "/".join(str(cfg.output_dir).split("/")[:-1]) + "/*/lightning_logs/version_*/checkpoints/" + '*.ckpt'
        print("/".join(str(cfg.output_dir).split("/")[:-1]))
        print("search_pattern ", search_pattern)
        cfg.train_ckpt_path = find_latest_checkpoint(search_pattern)
        print("cfg.train_ckpt_path ", cfg.train_ckpt_path)
    elif not auto_resume_training:
        logger.info("Automatic training checkpoint resume is disabled")

    # Hydra configs with _target_ (e.g. WandbLogger) must be instantiated; passing a DictConfig
    # as logger makes PL iterate it like a sequence of loggers but yields key strings → crash.
    trainer_params = OmegaConf.to_container(cfg.trainer.params, resolve=True)
    log_conf = trainer_params.get("logger", None)
    if isinstance(log_conf, dict) and "_target_" in log_conf:
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
        trainer_params["logger"] = instantiate(log_conf) if local_rank == 0 else False
    elif log_conf in [None, "none", "None", "false", "False"]:
        trainer_params["logger"] = False
    strategy = trainer_params.get("strategy")
    if isinstance(strategy, str) and strategy.startswith("ddp"):
        trainer_params["strategy"] = LocalMetadataDDPStrategy(
            find_unused_parameters=("find_unused_parameters_true" in strategy),
            start_method="popen",
        )
    callbacks = agent.get_training_callbacks()
    if trainer_params.get("logger") is False:
        callbacks = [callback for callback in callbacks if not isinstance(callback, LearningRateMonitor)]
    trainer = pl.Trainer(**trainer_params, callbacks=callbacks)

    if cfg.validation_run:
        logger.info("Starting Validation")
        timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        dump_root = os.path.join(os.getenv('SUBSCORE_PATH'), "navsim1_pdm_scores", cfg.experiment_name)
        os.makedirs(dump_root, exist_ok=True)
        dump_path = os.path.join(dump_root, f"{timestamp}.pkl")
        trainer.validate(
            model=lightning_module,
            dataloaders=[val_dataloader],
            ckpt_path=cfg.train_ckpt_path,
            verbose=True
        )
        logger.info("Running predictions to collect trajectories")
        predictions = trainer.predict(
            AgentLightningModule(agent=agent, for_viz=True),
            val_dataloader,
            return_predictions=True
        )

        if dist_ready():
            dist.barrier()
        
        world_size = dist.get_world_size() if dist_ready() else 1
        all_predictions = [None for _ in range(world_size)]

        if dist_ready():
            dist.all_gather_object(all_predictions, predictions)
        else:
            all_predictions = [predictions]

        rank = dist.get_rank() if dist_ready() else 0
        if rank != 0:
            return None

        merged_predictions = {}
        for proc_prediction in all_predictions:
            for d in proc_prediction:
                merged_predictions.update(d)

        pickle.dump(predictions, open(dump_path, 'wb'))
    else:
        logger.info("Starting Training")
        trainer.fit(
            model=lightning_module,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=cfg.train_ckpt_path
        )


if __name__ == "__main__":
    main()
