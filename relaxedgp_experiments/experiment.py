import logging

import hydra
import wandb
from gpflow.config import set_default_jitter
from omegaconf import DictConfig, OmegaConf

from relaxedgp_experiments.base import Experiment
from relaxedgp_experiments.dgp import DGPExperiment
from relaxedgp_experiments.utils.misc import set_default_float, set_seeds


@hydra.main(version_base=None, config_path="conf", config_name="experiment")
def main(cfg: DictConfig) -> None:
    hydra_cfg = hydra.core.hydra_config.HydraConfig().get()
    hydra_run_dir = hydra_cfg.runtime.output_dir

    logger = logging.getLogger(hydra_run_dir)

    if "rsvgp" in cfg.model.name or "lsvgp" in cfg.model.name:
        set_default_jitter(0.0)
    else:
        set_default_jitter(1e-6)

    if cfg.eval.log_wandb:
        wandb.init(
            project="relaxedgp",
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            save_code=True,
            group=cfg.experiment_name,
            notes=cfg.experiment_question,
            dir=hydra_run_dir,
        )

    set_default_float(cfg.default_float)
    set_seeds(cfg.seed)

    if cfg.model.kind == "shallow":
        experiment = Experiment.setup_from_config(
            cfg,
            hydra_run_dir,
            logger,
        )
    else:
        experiment = DGPExperiment.setup_from_config(
            cfg,
            hydra_run_dir,
            logger,
        )
    experiment.run_from_config(cfg)


if __name__ == "__main__":
    main()
