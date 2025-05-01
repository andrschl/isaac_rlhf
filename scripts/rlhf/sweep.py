import wandb
from isaac_rlhf.config.rlhf_cfg import RlhfCfg
from isaac_rlhf.runners.rlhf_runner import RlhfRunner

sweep_config = {
    "method": "grid",
    "parameters": {
        "base_seed": {"values": list(range(1, 11))},
        "rlhf_algorithm": {"values": ["vanilla", "ts_last"]},
    },
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="isaac_rlhf")

    def sweep_run():
        # 1) init the W&B run (config is injected by the agent)
        run = wandb.init(project="isaac_rlhf")

        # 2) now it's safe to read run.config
        seed = run.config["base_seed"]
        algo = run.config["rlhf_algorithm"]

        # 3) set a custom run name
        run.name = f"{algo}_seed{seed}"

        # 4) build your RlhfCfg from the sweep config
        cfg = RlhfCfg().replace(**dict(run.config))

        # 5) run the experiment
        runner = RlhfRunner(cfg)
        runner.run()

        run.finish()

    wandb.agent(sweep_id, function=sweep_run)
