import wandb
from isaac_rlhf.config.rlhf_cfg import RlhfCfg
from isaac_rlhf.runners.rlhf_runner import RlhfRunner


def main():
    seeds = list(range(1, 11))
    algos = ["vanilla", "ts_last"]

    for seed in seeds:
        for algo in algos:
            # build cfg
            base_cfg = RlhfCfg()
            cfg = base_cfg.replace(base_seed=seed, rlhf_algorithm=algo)

            # start W&B run with a custom name + config
            run = wandb.init(
                project="isaac_rlhf", config=cfg.to_dict(), name=f"{algo}_seed{seed}"
            )

            # launch your experiment
            runner = RlhfRunner(cfg)
            runner.run()

            # finish & flush
            run.finish()


if __name__ == "__main__":
    main()
