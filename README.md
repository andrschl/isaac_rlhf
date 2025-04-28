# isaac_rlhf (Preference-Based RL Library for Robotic Manipulation)
[**Installation**](#installation) | [**Training**](#training)
| [**Contributing**](#contributing)

Library for inverse reinforcement learning and RLHF for robotic manipulation.


## Installation
1. Install IsaacLab v2.0 following the instructions in on the [IsaacLab documentation](https://isaac-sim.github.io/IsaacLab/v2.0.2/source/setup/installation/index.html).
2. Clone this repository
3. Activate the virtual environment with the isaaclab installation and run `python -m pip install -e source/isaac_rlhf`

## Training
### Inverse Reinforcement Learning
To record data and run IRL training, run the following commands:
- `python scripts/rsl_rl/train_rl.py --task Isaac-Lift-Cube-Franka-v0 --headless`
- `python scripts/recording/record_synthetic_demos.py --task Isaac-Lift-Cube-Franka-v0 --num_demos 1000`
- `python scripts/irl/train_irl.py --task Isaac-Lift-Cube-Franka-v0 --headless --expert_data_path 'logs/rsl_rl/franka_lift/demos/hdf_dataset.hdf5'`

### Reinforcement Learning from Human Feedback
TBD

## Contributing
If you would like to contribute to the project please reach out to [Andreas Schlaginhaufen](mailto:andreas.schlaginhaufen@epfl.ch?subject=[isaac_reward_learning]%20Contribution%20to%20isaac_reward_learning). If you found this library useful in your research, please consider citing the following paper:
```
@inproceedings{NEURIPS2024_2628d4d3,
 author = {Schlaginhaufen, Andreas and Kamgarpour, Maryam},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
 pages = {21461--21501},
 publisher = {Curran Associates, Inc.},
 title = {Towards the Transferability of Rewards Recovered via Regularized Inverse Reinforcement Learning},
 url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/2628d4d3b054c2d7ad33ab03435204f4-Paper-Conference.pdf},
 volume = {37},
 year = {2024}
}
```
