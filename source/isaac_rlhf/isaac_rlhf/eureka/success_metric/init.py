import importlib

def load_success_metric(rl_task_type: str="cartpole"):
    module_path = f"isaaclab_rlhf.eureka.success_metric.{rl_task_type}"
    mod = importlib.import_module(module_path)

    if not hasattr(mod, "compute_success_metric"):
        raise AttributeError(f"Module '{module_path}' does not define compute_success_metric()")
    
    return mod.compute_success_metric