"""
Plotting.
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import glob

mpl.use("TkAgg")
font = {'family' : 'normal',
        'weight': 'black',
        'size'   : 20}
mpl.rc('font', **font)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})
plt.rcParams['text.usetex'] = True
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r"\usepackage{amsmath}"
                              r"\usepackage{bm}")
float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

def quantile_plot(ax, x, data, label, show_errorbars=True, show_markers=True):
    """
    x: 1d‐array of shape (T,)
    data: 2d‐array of shape (N_runs, T)
    """
    # compute statistics across N_runs (axis=0)
    medians    = np.median(data, axis=0)
    lower_q    = np.percentile(data, 20, axis=0)
    upper_q    = np.percentile(data, 80, axis=0)
    # asymmetric errors for errorbar
    lower_err  = medians - lower_q
    upper_err  = upper_q - medians

    if show_errorbars or show_markers:
        yerr = [lower_err, upper_err] if show_errorbars else None
        fmt  = '-o' if show_markers  else '-'
        ax.errorbar(x, medians,
                    yerr=yerr,
                    fmt=fmt, capsize=3,
                    label=label, alpha=.75)
    else:
        ax.plot(x, medians, '-', label=label, alpha=.75)


    ax.fill_between(x, lower_q, upper_q, alpha=.25)
    return ax

base_dir = os.path.join('logs', 'Isaac-Cartpole-v0', 'pure_exploration')
base_dir_rl = os.path.join('logs', 'Isaac-Cartpole-v0')
beta_str = '/beta1_0.01/beta2_100.0'
alg_paths = [
    # 'vanilla', 
    'ts_last' + beta_str, 
    'ts_last_lazy' + beta_str, 
    'ts_last_lazy_opt_design' + beta_str,
]
# labels = ["Entropy", r"\texttt{RPO}", r"\texttt{LRPO}", r"\texttt{LRPO-OD}"]
labels = [r"\texttt{RPO}", r"\texttt{LRPO}", r"\texttt{LRPO-OD}"]

fig, ax = plt.subplots(figsize=(10, 6))

# first, plot RL as a constant dashed line at the avg last gt_reward
rl_pattern = os.path.join(base_dir_rl, 'rl', 'seed_[1-3]', 'logs.csv')
rl_files   = sorted(glob.glob(rl_pattern))
rl_dfs     = [pd.read_csv(f) for f in rl_files]
# assume same steps across seeds
steps      = rl_dfs[0]['step'].to_numpy()
avg_last   = np.mean([df['rlhf/gt_reward'].iloc[-1] for df in rl_dfs])
ax.hlines(
    y=avg_last,
    xmin=steps[0],
    xmax=steps[-1],
    colors='C0',
    linestyles='--',
    label='RL benchmark',
)

for alg_path, label in zip(alg_paths[::-1], labels[::-1]):
    pattern = os.path.join(base_dir, alg_path, 'seed_[1-7]', 'logs.csv')
    files = sorted(glob.glob(pattern))
    dfs   = [pd.read_csv(f) for f in files]

    steps = dfs[0]['step'].to_numpy()
    print(steps, [df['rlhf/gt_reward'].to_numpy().shape for df in dfs])
    arr = np.stack([df['rlhf/gt_reward'].to_numpy() for df in dfs], axis=0)
    quantile_plot(ax, steps[:-1], arr[:,:-1], label=label, show_errorbars=False, show_markers=False)

# reorder legend: keep “RL benchmark” first, then algorithms in their original order
handles, lg_labels = ax.get_legend_handles_labels()
# RL is at index 0; the rest were plotted in reverse—reverse them back
new_handles = [handles[0]] + handles[1:][::-1]
new_labels  = [lg_labels[0]] + lg_labels[1:][::-1]
ax.legend(new_handles, new_labels)

ax.set_xlabel('RLHF step')
ax.set_ylabel('Ground truth reward')
ax.grid(True)
plt.tight_layout()
save_path = 'fig/gt_reward_pure_exploration.png'
save_dir  = os.path.dirname(save_path)
if save_dir and not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)
fig.savefig(save_path, dpi=300)
plt.show()

# second figure: num_queries vs RLHF step
fig2, ax2 = plt.subplots(figsize=(10, 6))
for alg_path, label in zip(alg_paths[::-1], labels[::-1]):
    pattern = os.path.join(base_dir, alg_path, 'seed_[1-7]', 'logs.csv')
    files   = sorted(glob.glob(pattern))
    if not files:
        continue
    dfs   = [pd.read_csv(f) for f in files]
    steps = dfs[0]['step'].to_numpy()
    # stack num_queries: shape (n_runs, n_steps)
    arr_q = np.stack([df['rlhf/num_queries'].to_numpy() for df in dfs], axis=0)
    quantile_plot(ax2, steps, arr_q,
                  label=label,
                  show_errorbars=False,
                  show_markers=False)

ax2.set_xlabel('RLHF step')
ax2.set_ylabel('Number of preference queries')
# reorder legend entries back to original order
handles2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(handles2[::-1], labels2[::-1])

ax2.grid(True)
plt.tight_layout()

# ensure save directory exists and save
save_path2 = 'fig/num_queries_pure_exploration.png'
save_dir2  = os.path.dirname(save_path2)
if save_dir2 and not os.path.exists(save_dir2):
    os.makedirs(save_dir2, exist_ok=True)
fig2.savefig(save_path2, dpi=300)
plt.show()

# third figure: only the *final* time‐step
fig3, ax3 = plt.subplots(figsize=(10, 6))
xs     = np.arange(len(alg_paths))

# draw RL benchmark as a constant line at its final gt_reward
rl_pattern = os.path.join(base_dir_rl, 'rl', 'seed_[1-3]', 'logs.csv')
rl_files   = sorted(glob.glob(rl_pattern))
rl_dfs     = [pd.read_csv(f) for f in rl_files]
avg_last_rl = np.mean([df['rlhf/gt_reward'].iloc[-1] for df in rl_dfs])
ax3.hlines(
    y=avg_last_rl,
    xmin=-0.5,
    xmax=xs[-1]+0.5,
    colors='C0',
    linestyles='--',
    label='RL benchmark',
)

medians = []
lowers  = []
uppers  = []
for alg_path, label in zip(alg_paths, labels):
    pattern    = os.path.join(base_dir, alg_path, 'seed_[1-7]', 'logs.csv')
    files      = sorted(glob.glob(pattern))
    dfs        = [pd.read_csv(f) for f in files]
    last_vals  = np.array([df['rlhf/gt_reward'].iloc[-1] for df in dfs])
    m          = np.median(last_vals)
    l          = m - np.percentile(last_vals, 20)
    u          = np.percentile(last_vals, 80) - m
    medians.append(m)
    lowers .append(l)
    uppers .append(u)

# bar colors: RPO green, LRPO orange, LRPO-OD blue
bar_colors = ['tab:green', 'tab:orange', 'tab:blue']
ax3.bar(xs, medians, yerr=[lowers, uppers],
        capsize=3,
        color=bar_colors,
        alpha=.7)

ax3.set_xticks(xs)
ax3.set_xticklabels(labels)
ax3.set_ylabel('Final ground truth reward')
ax3.grid(True)

# reorder legend so RL benchmark appears first
handles3, labels3 = ax3.get_legend_handles_labels()
# RL is first plotted, so it's already handles3[0]
ax3.legend(handles3, labels3, loc='lower right')

plt.tight_layout()

save_path3 = 'fig/gt_reward_last_step.png'
save_dir3  = os.path.dirname(save_path3)
if save_dir3 and not os.path.exists(save_dir3):
    os.makedirs(save_dir3, exist_ok=True)
fig3.savefig(save_path3, dpi=300)
plt.show()