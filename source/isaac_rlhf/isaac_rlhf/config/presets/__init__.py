from __future__ import annotations

"""Preset helpers for common RLHF configurations."""

from typing import Callable, Dict, Iterable

from ..rlhf_cfg import RlhfCfg

PresetFactory = Callable[[], RlhfCfg]


def _register_presets() -> tuple[
    Dict[str, PresetFactory], Dict[PresetFactory, str]
]:
    """Create the preset registry."""

    registry: Dict[str, PresetFactory] = {}
    canonical_names: Dict[PresetFactory, str] = {}

    def register(
        canonical: str,
        factory: PresetFactory,
        *aliases: str,
    ) -> None:
        names: Iterable[str] = (canonical, *aliases)
        for name in names:
            key = name.strip().lower()
            if key in registry and registry[key] is not factory:
                raise ValueError(f"Conflicting preset registered for '{name}'.")
            registry[key] = factory
        canonical_names[factory] = canonical

    def gridworld() -> RlhfCfg:
        return RlhfCfg()

    def isaac_cartpole() -> RlhfCfg:
        return RlhfCfg().replace(
            task="Isaac-Cartpole-v0",
            ignored_reward_terms=["terminating"],
            num_rl_iterations=30,
            num_rlhf_iterations=20,
        )

    def isaac_velocity_flat() -> RlhfCfg:
        return RlhfCfg().replace(
            task="Isaac-Velocity-Flat-H1-v0",
            num_rl_iterations=300,
        )
    
    def isaac_humanoid() -> RlhfCfg:
        return RlhfCfg().replace(
            task="Isaac-Humanoid-v0",
            num_rl_iterations=200,
            num_rlhf_iterations=20,
            trajectory_length=1000,
        )

    register("gridworld", gridworld)
    register(
        "cartpole",
        isaac_cartpole,
        "cartpole",
        "isaac_cartpole",
    )
    register(
        "humanoid",
        isaac_velocity_flat,
        "velocity_flat",
        "isaac_velocity_flat",
    )
    register(
        "isaac_humanoid",
        isaac_humanoid    
    )
    
    return registry, canonical_names


_PRESET_FACTORIES, _CANONICAL_NAMES = _register_presets()


def list_presets() -> list[str]:
    """Return the available preset names."""
    return sorted(set(_CANONICAL_NAMES.values()))


def get_preset(name: str) -> RlhfCfg:
    """Return the configuration preset with the given name."""
    key = name.strip().lower()
    if key not in _PRESET_FACTORIES:
        raise ValueError(
            f"Unknown preset '{name}'. Available presets: {', '.join(list_presets())}"
        )
    return _PRESET_FACTORIES[key]()
