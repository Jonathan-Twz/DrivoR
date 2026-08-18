from typing import Any, List, Dict, Union

import csv
import glob
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from pathlib import Path
import pickle
import subprocess
from .drivor_model import DrivoRModel
from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.training.dataset import load_feature_target_from_pickle
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, ProgressBar, LearningRateMonitor
from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import SensorConfig
from .drivor_features import DrivoRTargetBuilder
from .drivor_features import DrivoRFeatureBuilder
import sys
from omegaconf import OmegaConf
import math
from numbers import Number


def _format_progress_metric(value: Any) -> str:
    if torch.is_tensor(value):
        if value.numel() == 1:
            value = value.detach().item()
        else:
            return str(value)
    if isinstance(value, Number):
        return f"{value:.3f}"
    return str(value)


class LitProgressBar(ProgressBar):

    def __init__(self):
        super().__init__()
        self.enable = True
        self._train_pbar = None
        self._val_pbar = None
        self._epoch_start_time = None

    def disable(self):
        self.enable = False

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        import time
        from tqdm import tqdm
        self._epoch_start_time = time.time()
        total = self.total_train_batches
        self._train_pbar = tqdm(
            total=total,
            desc=f"Epoch {trainer.current_epoch}/{trainer.max_epochs-1} [train]",
            dynamic_ncols=True,
            leave=True,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if self._train_pbar is not None:
            self._train_pbar.update(1)
            if batch_idx % 50 == 0:
                metrics = self.get_metrics(trainer, pl_module)
                short = {k.split("/")[-1]: _format_progress_metric(v) for k, v in metrics.items() if "train/" in k}
                self._train_pbar.set_postfix(short, refresh=False)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        import time
        super().on_train_epoch_end(trainer, pl_module)
        if self._train_pbar is not None:
            self._train_pbar.close()
            self._train_pbar = None
        elapsed = time.time() - self._epoch_start_time if self._epoch_start_time else 0
        metrics = self.get_metrics(trainer, pl_module)
        train_metrics = {k: v for k, v in metrics.items() if "train/" in k}
        val_metrics = {k: v for k, v in metrics.items() if "val/" in k}
        other_metrics = {k: v for k, v in metrics.items() if "train/" not in k and "val/" not in k}
        print(f"\n###########  Epoch {trainer.current_epoch} ({elapsed:.0f}s) ##########")
        for k, v in train_metrics.items():
            print(f"{k},{_format_progress_metric(v)}")
        for k, v in val_metrics.items():
            print(f"{k},{_format_progress_metric(v)}")
        for k, v in other_metrics.items():
            print(f"{k},{_format_progress_metric(v)}")
        print(f"###########\n")

    def on_validation_epoch_start(self, trainer, pl_module):
        super().on_validation_epoch_start(trainer, pl_module)
        from tqdm import tqdm
        total = self.total_val_batches
        self._val_pbar = tqdm(
            total=total,
            desc=f"Epoch {trainer.current_epoch}/{trainer.max_epochs-1} [val]",
            dynamic_ncols=True,
            leave=True,
        )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self._val_pbar is not None:
            self._val_pbar.update(1)

    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        if self._val_pbar is not None:
            self._val_pbar.close()
            self._val_pbar = None


class NavsimV1PDMSEvalCallback(Callback):
    """Runs the official NAVSIM-v1 PDMS evaluator after validation epochs."""

    def __init__(self):
        self.enabled = os.environ.get("DRIVOR_EPOCH_PDMS_EVAL", "0") == "1"
        self.every_n_epochs = max(1, int(os.environ.get("DRIVOR_EPOCH_PDMS_EVERY_N_EPOCHS", "1")))
        self.log_prefix = os.environ.get("DRIVOR_EPOCH_PDMS_LOG_PREFIX", "test").strip("/")
        self.timeout_sec = int(os.environ.get("DRIVOR_EPOCH_PDMS_TIMEOUT_SEC", "7200"))
        self.drivor_root = Path(os.environ.get("DRIVOR_ROOT", Path(__file__).resolve().parents[3]))
        self.script_path = Path(
            os.environ.get(
                "DRIVOR_EPOCH_PDMS_SCRIPT",
                self.drivor_root / "scripts/evaluation/run_drivor_bev_decoder_evaluation.sh",
            )
        )
        self.cuda_visible_devices = os.environ.get(
            "DRIVOR_EPOCH_PDMS_CUDA_VISIBLE_DEVICES",
            os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"),
        )
        self.decoder_bev_lora_rank = os.environ.get(
            "DRIVOR_EPOCH_PDMS_DECODER_BEV_LORA_RANK",
            os.environ.get("DECODER_BEV_LORA_RANK", ""),
        )

    @staticmethod
    def _dist_ready() -> bool:
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    @staticmethod
    def _is_number(value: str) -> bool:
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False

    def _find_latest_csv(self, experiment_name: str) -> Path:
        navsim_exp_root = Path(os.environ.get("NAVSIM_EXP_ROOT", self.drivor_root / "exp"))
        candidates = glob.glob(str(navsim_exp_root / "ke" / experiment_name / "**" / "*.csv"), recursive=True)
        candidates += glob.glob(str(navsim_exp_root / "navsim1_pdm_scores" / experiment_name / "**" / "*.csv"), recursive=True)
        if not candidates:
            raise FileNotFoundError(f"No NAVSIM-v1 PDMS csv found for experiment {experiment_name}")
        return Path(max(candidates, key=os.path.getmtime))

    def _read_average_metrics(self, csv_path: Path) -> Dict[str, float]:
        with csv_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise RuntimeError(f"PDMS csv is empty: {csv_path}")

        average_row = next((row for row in rows if row.get("token") == "average"), rows[-1])
        metrics = {}
        for key, value in average_row.items():
            if key in {"", "token", "valid"} or not self._is_number(value):
                continue
            metric_key = "pdms" if key == "score" else key
            metrics[f"{self.log_prefix}/{metric_key}"] = float(value)
        return metrics

    def _log_metrics(self, trainer, metrics: Dict[str, float]) -> None:
        if not metrics:
            return
        logger = getattr(trainer, "logger", None)
        if logger:
            logger.log_metrics(metrics, step=trainer.global_step)
        print("NAVSIM-v1 PDMS epoch metrics:")
        for key, value in sorted(metrics.items()):
            print(f"{key},{value:.6f}")

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.enabled or getattr(trainer, "sanity_checking", False):
            return

        epoch = int(trainer.current_epoch)
        should_run = (epoch + 1) % self.every_n_epochs == 0
        if not should_run:
            return

        is_rank_zero = getattr(trainer, "is_global_zero", True)
        try:
            if is_rank_zero:
                try:
                    self._run_eval_and_log(trainer)
                except Exception as exc:
                    self._log_metrics(
                        trainer,
                        {
                            f"{self.log_prefix}/pdms_eval_failed": 1.0,
                            f"{self.log_prefix}/epoch": float(epoch),
                        },
                    )
                    print(f"NAVSIM-v1 PDMS eval failed with exception: {exc}")
        finally:
            if self._dist_ready():
                torch.distributed.barrier()

    def _run_eval_and_log(self, trainer) -> None:
        if not self.script_path.exists():
            raise FileNotFoundError(f"NAVSIM-v1 PDMS script does not exist: {self.script_path}")

        epoch = int(trainer.current_epoch)
        global_step = int(trainer.global_step)
        output_root = Path(os.environ.get("DRIVOR_TRAIN_OUTPUT_DIR", getattr(trainer, "default_root_dir", ".")))
        checkpoint_dir = output_root / "checkpoints" / "epoch_pdms"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"epoch={epoch:02d}-step={global_step}.ckpt"
        trainer.save_checkpoint(str(checkpoint_path))

        base_name = os.environ.get("DRIVOR_EPOCH_PDMS_EXPERIMENT_PREFIX")
        if not base_name:
            base_name = os.environ.get("WANDB_RUN_NAME", os.environ.get("EXPERIMENT", "drivor-epoch-pdms"))
        eval_name = f"{base_name.replace('/', '_')}-epoch{epoch:02d}-step{global_step}-navsim-v1-pdms"

        log_dir = output_root / "pdms_eval"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"epoch={epoch:02d}-step={global_step}.log"

        env = os.environ.copy()
        env.update(
            {
                "CKPT_PATH": str(checkpoint_path),
                "EXPERIMENT_NAME": eval_name,
                "CUDA_VISIBLE_DEVICES": self.cuda_visible_devices,
                "DRIVOR_EPOCH_PDMS_EVAL": "0",
                "PYTHON_BIN": sys.executable,
            }
        )
        if self.decoder_bev_lora_rank:
            env["DECODER_BEV_LORA_RANK"] = self.decoder_bev_lora_rank

        print(
            f"Running NAVSIM-v1 PDMS eval for epoch {epoch} step {global_step}: "
            f"{self.script_path} -> {log_path}"
        )
        with log_path.open("w") as log_file:
            result = subprocess.run(
                ["bash", str(self.script_path)],
                cwd=str(self.drivor_root),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=self.timeout_sec,
                check=False,
            )

        if result.returncode != 0:
            self._log_metrics(
                trainer,
                {
                    f"{self.log_prefix}/pdms_eval_failed": 1.0,
                    f"{self.log_prefix}/epoch": float(epoch),
                },
            )
            print(f"NAVSIM-v1 PDMS eval failed with code {result.returncode}; see {log_path}")
            return

        csv_path = self._find_latest_csv(eval_name)
        metrics = self._read_average_metrics(csv_path)
        metrics[f"{self.log_prefix}/epoch"] = float(epoch)
        metrics[f"{self.log_prefix}/pdms_eval_failed"] = 0.0
        self._log_metrics(trainer, metrics)


class DrivoRAgent(AbstractAgent):
    def __init__(
            self,
            config,
            lr_args: dict,
            checkpoint_path: str = None,
            loss: nn.Module = None,
            progress_bar: bool = True,
            scheduler_args: dict = None,
            batch_size: int = 64,
            num_gpus: int = 1,
    ):
        super().__init__()
        self._config = config
        self._lr_args = lr_args
        self._checkpoint_path = checkpoint_path
        self.progress_bar = progress_bar
        self.scheduler_args = scheduler_args
        self.batch_size = batch_size
        self.num_gpus = num_gpus


        cache_data=False

        if not cache_data:
            self._drivor_model = DrivoRModel(config)

        # Training infra: enable whenever a loss module is provided via Hydra,
        # regardless of whether a pre-trained checkpoint is also being loaded
        # (e.g. Phase-1 fine-tuning with frozen backbone).
        training_mode = (not cache_data) and (loss is not None)
        if training_mode:
            self.bce_logit_loss = nn.BCEWithLogitsLoss()
            self.b2d = config.b2d

            self.ray = bool(config.get("use_ray_score", True))

            if self.ray:
                from navsim.planning.utils.multithreading.worker_ray_no_torch import RayDistributedNoTorch
                from nuplan.planning.utils.multithreading.worker_utils import worker_map
                self.worker = RayDistributedNoTorch(threads_per_node=8)
                self.worker_map=worker_map


            from .score_module.compute_navsim_score import get_scores

            metric_cache = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_cache"))
            try:
                # add synthetic metric_cache
                metric_cache_synthetic_0 = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_synthetic_reaction_pdm_v1.0-0"))
                metric_cache_synthetic_1 = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_synthetic_reaction_pdm_v1.0-1"))
                metric_cache_synthetic_2 = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_synthetic_reaction_pdm_v1.0-2"))
                metric_cache_synthetic_3 = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_synthetic_reaction_pdm_v1.0-3"))
                metric_cache_synthetic_4 = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_synthetic_reaction_pdm_v1.0-4"))

                self.train_metric_cache_paths_synthetic = metric_cache_synthetic_0.metric_cache_paths
                self.train_metric_cache_paths_synthetic.update(metric_cache_synthetic_0.metric_cache_paths)
                self.train_metric_cache_paths_synthetic.update(metric_cache_synthetic_1.metric_cache_paths)
                self.train_metric_cache_paths_synthetic.update(metric_cache_synthetic_2.metric_cache_paths)
                self.train_metric_cache_paths_synthetic.update(metric_cache_synthetic_3.metric_cache_paths)
                self.train_metric_cache_paths_synthetic.update(metric_cache_synthetic_4.metric_cache_paths)

                self.test_metric_cache_paths_synthetic = self.train_metric_cache_paths_synthetic
            except:
                self.test_metric_cache_paths_synthetic = self.train_metric_cache_paths_synthetic = None

            self.test_metric_cache_paths_synthetic = self.train_metric_cache_paths_synthetic
            self.train_metric_cache_paths = metric_cache.metric_cache_paths
            self.test_metric_cache_paths = metric_cache.metric_cache_paths

            self.get_scores = get_scores

            self.loss = loss
            


    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""

        if self._checkpoint_path != "":
            if torch.cuda.is_available():
                state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
            else:
                state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                    "state_dict"]
            mapped = {k.replace("agent._drivor_model", "_drivor_model"): v for k, v in state_dict.items()}

            strict_override = self._config.get("load_checkpoint_strict", None)
            use_bev = bool(self._config.get("use_bev_feature", False))
            use_future_bev = bool(self._config.get("use_privileged_future_bev", False))
            strict = bool(strict_override) if strict_override is not None else (not (use_bev or use_future_bev))

            missing, unexpected = self.load_state_dict(mapped, strict=strict)

            if not strict:
                import logging

                logger = logging.getLogger(__name__)
                expected_missing_prefixes = (
                    "_drivor_model.bev_tokenizer.",
                    "_drivor_model.bev_residual_proposal_refiner.",
                    "_drivor_model.future_bev_time_embed",
                )

                def _is_expected_missing(name: str) -> bool:
                    if name.startswith(expected_missing_prefixes):
                        return True
                    if name.startswith("_drivor_model.scorer_attention.layers."):
                        parts = name.split(".", 4)
                        if len(parts) < 5:
                            return False
                        rest = parts[4]
                        return (
                            rest.startswith("cross_attn_bev")
                            or rest.startswith("self_attn_lora")
                            or rest.startswith("cross_attn_lora")
                            or rest.startswith("mlp_lora")
                        )
                    if name.startswith("_drivor_model.trajectory_decoder.layers."):
                        parts = name.split(".", 4)
                        if len(parts) < 5:
                            return False
                        rest = parts[4]
                        return (
                            rest.startswith("cross_attn_bev")
                            or rest.startswith("self_attn_lora")
                            or rest.startswith("cross_attn_lora")
                            or rest.startswith("mlp_lora")
                        )
                    return False

                expected = [k for k in missing if _is_expected_missing(k)]
                unexpected_missing = [k for k in missing if not _is_expected_missing(k)]
                logger.info(
                    "Checkpoint loaded with strict=False. expected_missing=%d, "
                    "unexpected_missing=%d, unexpected_keys=%d",
                    len(expected),
                    len(unexpected_missing),
                    len(unexpected),
                )
                if unexpected_missing:
                    logger.warning(
                        "Unexpected missing keys (not matching BEV/LoRA prefixes): %s",
                        unexpected_missing[:20],
                    )
                if unexpected:
                    logger.warning("Unexpected keys in checkpoint: %s", unexpected[:20])

    def get_sensor_config(self) :
        """Inherited, see superclass."""
        # return SensorConfig(
        #     cam_f0=[3],
        #     cam_l0=[3],
        #     cam_l1=[],
        #     cam_l2=[],
        #     cam_r0=[3],
        #     cam_r1=[],
        #     cam_r2=[],
        #     cam_b0=[3],
        #     lidar_pc=[],
        # )
        return SensorConfig(
            cam_f0=OmegaConf.to_object(self._config["cam_f0"]),
            cam_l0=OmegaConf.to_object(self._config["cam_l0"]),
            cam_l1=OmegaConf.to_object(self._config["cam_l1"]),
            cam_l2=OmegaConf.to_object(self._config["cam_l2"]),
            cam_r0=OmegaConf.to_object(self._config["cam_r0"]),
            cam_r1=OmegaConf.to_object(self._config["cam_r1"]),
            cam_r2=OmegaConf.to_object(self._config["cam_r2"]),
            cam_b0=OmegaConf.to_object(self._config["cam_b0"]),
            lidar_pc=OmegaConf.to_object(self._config["lidar_pc"]),
        )
    
    def get_target_builders(self) :
        return [DrivoRTargetBuilder(config=self._config)]

    def get_feature_builders(self) :
        return [DrivoRFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._drivor_model(features)

    def forward_train(
            self,
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        privileged_future_bev = None
        if bool(self._config.get("use_privileged_future_bev", False)):
            privileged_future_bev = targets.get("privileged_future_bev", None)
            if privileged_future_bev is None:
                raise KeyError(
                    "agent.config.use_privileged_future_bev=true but targets do not contain "
                    "'privileged_future_bev'. Rebuild the dataset cache with the same config or "
                    "run with use_cache_without_dataset=false."
                )
        return self._drivor_model(features, privileged_future_bev=privileged_future_bev)

    def compute_score(self, targets, proposals, test=True):
        if self.training:
            metric_cache_paths = self.train_metric_cache_paths
            metric_cache_paths_synthetic = self.train_metric_cache_paths_synthetic
        else:
            metric_cache_paths = self.test_metric_cache_paths
            metric_cache_paths_synthetic = self.test_metric_cache_paths_synthetic

        target_trajectory = targets["trajectory"]
        proposals=proposals.detach()

        
        data_points = [
            {
                "token": metric_cache_paths[token] if token in metric_cache_paths else metric_cache_paths_synthetic[token],
                "poses": poses,
                "test": test
            }
            for token, poses in zip(targets["token"], proposals.cpu().numpy())
        ]

        if self.ray:
            all_res = self.worker_map(self.worker, self.get_scores, data_points)
        else:
            all_res = self.get_scores(data_points)

        target_scores = torch.FloatTensor(np.stack([res[0] for res in all_res])).to(proposals.device)

        final_scores = target_scores[:, :, -1]

        best_scores = torch.amax(final_scores, dim=-1)

        if test:
            l2_2s = torch.linalg.norm(proposals[:, 0] - target_trajectory, dim=-1)[:, :4]

            return final_scores[:, 0].mean(), best_scores.mean(), final_scores, l2_2s.mean(), target_scores[:, 0]
        else:
            key_agent_corners = torch.FloatTensor(np.stack([res[1] for res in all_res])).to(proposals.device)

            key_agent_labels = torch.BoolTensor(np.stack([res[2] for res in all_res])).to(proposals.device)

            all_ego_areas = torch.BoolTensor(np.stack([res[3] for res in all_res])).to(proposals.device)

            return final_scores, best_scores, target_scores, key_agent_corners, key_agent_labels, all_ego_areas

    def compute_loss(
            self,
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            pred: Dict[str, torch.Tensor],
    ) -> Dict:
        loss_dict = self.loss(targets, pred, self._config, self.compute_score)
        refiner = getattr(self._drivor_model, "bev_residual_proposal_refiner", None)
        if refiner is not None and isinstance(loss_dict, dict):
            loss_dict["residual_alpha"] = refiner.alpha.detach()
        return loss_dict

    def _collect_trainable_params(self):
        """Select parameters for the optimizer.

        When ``freeze_pretrained_except_bev_scorer`` is True, only the new BEV
        tokenizer, side-LoRA adapters and the new ``cross_attn_bev*`` sublayers
        inside ``scorer_attention`` are trainable; everything else is frozen.
        Otherwise all parameters are trained.
        """

        freeze_bev_only = bool(self._config.get("freeze_pretrained_except_bev_scorer", False))
        if not freeze_bev_only:
            return list(self._drivor_model.parameters())

        def _is_trainable(name: str) -> bool:
            if name.startswith("bev_tokenizer."):
                return True
            if name == "future_bev_time_embed":
                return True
            if name.startswith("bev_residual_proposal_refiner."):
                return True
            if name.startswith("scorer_attention.layers."):
                # Strip the "scorer_attention.layers.<i>." prefix to inspect the
                # module name inside the BevAwareBlock.
                parts = name.split(".", 3)
                if len(parts) < 4:
                    return False
                rest = parts[3]
                return (
                    rest.startswith("cross_attn_bev")
                    or rest.startswith("self_attn_lora")
                    or rest.startswith("cross_attn_lora")
                    or rest.startswith("mlp_lora")
                )
            if name.startswith("trajectory_decoder.layers."):
                # New gated BEV cross-attn (+ optional side-LoRA) inside the
                # trajectory generator.
                parts = name.split(".", 3)
                if len(parts) < 4:
                    return False
                rest = parts[3]
                return (
                    rest.startswith("cross_attn_bev")
                    or rest.startswith("self_attn_lora")
                    or rest.startswith("cross_attn_lora")
                    or rest.startswith("mlp_lora")
                )
            return False

        params = []
        for n, p in self._drivor_model.named_parameters():
            trainable = _is_trainable(n)
            p.requires_grad = bool(trainable)
            if trainable:
                params.append(p)
        if not params:
            raise RuntimeError(
                "freeze_pretrained_except_bev_scorer=True but no bev_tokenizer / "
                "BEV injection parameters were found; make sure use_bev_feature=True "
                "and at least one BEV injection module is enabled."
            )
        import logging

        logging.getLogger(__name__).info(
            "freeze_pretrained_except_bev_scorer=True: training %d parameter tensors.",
            len(params),
        )
        return params

    def get_optimizers(self):

        global_batchsize = self.batch_size * self.num_gpus
        params = self._collect_trainable_params()
        if self._lr_args["name"] == "Adam":
            lr = self._lr_args["base_lr"] * math.sqrt(global_batchsize / self._lr_args["base_batch_size"])
            optimizer = torch.optim.Adam(params, lr=lr)
        elif self._lr_args["name"] == "AdamW":
            lr = self._lr_args["base_lr"] * math.sqrt(global_batchsize / self._lr_args["base_batch_size"])
            optimizer = torch.optim.AdamW(params, lr=lr)
        else:
            raise NotImplementedError

        if self.scheduler_args is not None:

            T_max = int(math.ceil(self.scheduler_args.dataset_size / global_batchsize) *  self.scheduler_args.num_epochs)

            # classic cosine
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            #     optimizer,
            #     T_max=T_max, 
            #     eta_min=0.0, last_epoch=-1
            # )

            # Ramp + cosine
            T_max_ramp = int(T_max * 0.1)
            scheduler_ramp = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, total_iters=T_max_ramp)
            T_max_cosine = T_max - T_max_ramp
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=T_max_cosine, 
                eta_min=0.0, last_epoch=-1
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[scheduler_ramp, scheduler_cosine],
                milestones=[T_max_ramp],
            )           

            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        
        else:
            return [optimizer]

    def get_training_callbacks(self):

        checkpoint_dir = None
        train_output_dir = os.environ.get("DRIVOR_TRAIN_OUTPUT_DIR")
        if train_output_dir:
            checkpoint_dir = str(Path(train_output_dir) / "checkpoints")

        checkpoint_cb_best = ModelCheckpoint(save_top_k=5,
                                        monitor='val/score_epoch',
                                        filename='best-{epoch}-{step}',
                                        mode="max",
                                        dirpath=checkpoint_dir,
                                        )
        
        checkpoint_cb = ModelCheckpoint(save_last=True, dirpath=checkpoint_dir)

        lr_monitor = LearningRateMonitor(logging_interval="step", 
                                            log_momentum=False,
                                            log_weight_decay=False)
        callbacks = [checkpoint_cb_best, checkpoint_cb]
        epoch_pdms_eval = NavsimV1PDMSEvalCallback()
        if epoch_pdms_eval.enabled:
            callbacks.append(epoch_pdms_eval)
        
        if self.progress_bar:
            return callbacks + [lr_monitor]
        else:
            progress_bar = LitProgressBar()
            return callbacks + [progress_bar, lr_monitor]
