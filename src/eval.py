"""
Evaluation Harness and Comparative Leaderboard Generator for Review 2:
Multi-seed simulation of Model A, Model B, and Model C on Standard and Sparse regimes.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.env import build_env

# ─────────────────────────────────────────────────────────────────────────────
# Baseline Routers (for context)
# ─────────────────────────────────────────────────────────────────────────────
class GreedyRouter:
    """Selects the neighbor offering maximum forward progress toward the sink."""
    def select_action(self, state: np.ndarray, mask: np.ndarray) -> int:
        best_act = 0
        best_progress = -float('inf')
        for i in range(6):
            if mask[i] == 1.0:
                prog = state[2 + 4 * i + 3]
                if prog > best_progress:
                    best_progress = prog
                    best_act = i
        return best_act


class EnergyAwareRouter:
    """Balances distance progress with neighbor residual battery."""
    def select_action(self, state: np.ndarray, mask: np.ndarray) -> int:
        best_act = 0
        best_score = -float('inf')
        for i in range(6):
            if mask[i] == 1.0:
                prog = state[2 + 4 * i + 3]
                e_next = state[2 + 4 * i + 2]
                score = prog * 0.5 + e_next * 0.5
                if score > best_score:
                    best_score = score
                    best_act = i
        return best_act


# ─────────────────────────────────────────────────────────────────────────────
# Core Evaluation Function
# ─────────────────────────────────────────────────────────────────────────────
def eval_policy(router, cfg: dict, uq=None, temporal_model=None, seed: int = 42,
                rounds: int = 100, packets_per_round: int = 30,
                n_nodes: int | None = None, comm_range: float | None = None) -> dict:
    """
    Evaluates a routing policy over multiple operational rounds with mobility.
    """
    env, nodes = build_env(
        cfg, uq=uq, temporal_model=temporal_model, seed=seed,
        n_nodes=n_nodes, comm_range=comm_range
    )

    total_packets = 0
    delivered_packets = 0
    total_hops_delivered = 0
    lifetime_round = rounds  # Round when first node died

    for r in range(rounds):
        # Route packets in this round
        for _ in range(packets_per_round):
            state, mask, acts = env.reset_packet()
            if state is None:
                break  # Network partitioned or completely depleted

            total_packets += 1
            done = False
            while not done:
                action_idx = router.select_action(state, mask)
                step_res, reward, done, info = env.step(action_idx)
                if done:
                    if info.get('delivered', False):
                        delivered_packets += 1
                        total_hops_delivered += info.get('hops', 0)
                else:
                    state, mask = step_res

        # Mobility step once per round
        env.apply_mobility()

        # Check if first node died
        stats = env.network_stats()
        if lifetime_round == rounds and stats['alive_count'] < (len(nodes) - 1):
            lifetime_round = r

    pdr = (delivered_packets / max(1, total_packets))
    avg_delay = (total_hops_delivered / max(1, delivered_packets))
    throughput = (delivered_packets / max(1, rounds))
    end_stats = env.network_stats()

    return {
        'PDR': float(pdr),
        'AvgDelay': float(avg_delay),
        'Throughput': float(throughput),
        'Lifetime': int(lifetime_round),
        'AliveEnd': int(end_stats['alive_count']),
        'AvgEnergyEnd': float(end_stats['avg_energy']),
    }


def run_benchmark_leaderboard(routers: dict, cfg: dict, seeds: list[int] = [42, 43, 44],
                              scenario_name: str = "Standard (40 Nodes)",
                              n_nodes: int | None = None,
                              comm_range: float | None = None) -> pd.DataFrame:
    """
    Evaluates multiple policies across multiple seeds and aggregates Mean ± Std.
    """
    records = []
    for name, r_info in routers.items():
        router = r_info['router']
        uq = r_info.get('uq', None)
        temporal_model = r_info.get('temporal_model', None)

        seed_results = []
        for s in seeds:
            res = eval_policy(
                router, cfg, uq=uq, temporal_model=temporal_model, seed=s,
                rounds=cfg['eval_rounds'], packets_per_round=cfg['packets_per_round'],
                n_nodes=n_nodes, comm_range=comm_range
            )
            res['Model'] = name
            res['Seed'] = s
            seed_results.append(res)
            records.append(res)

    df_all = pd.DataFrame(records)

    # Compute aggregation
    summary = []
    for name in routers.keys():
        sub = df_all[df_all['Model'] == name]
        summary.append({
            'Model': name,
            'Scenario': scenario_name,
            'PDR (%)': f"{sub['PDR'].mean()*100:.1f} ± {sub['PDR'].std()*100:.1f}",
            'Avg Delay (hops)': f"{sub['AvgDelay'].mean():.2f} ± {sub['AvgDelay'].std():.2f}",
            'Lifetime (rounds)': f"{sub['Lifetime'].mean():.1f} ± {sub['Lifetime'].std():.1f}",
            'Lifetime Std': float(sub['Lifetime'].std()),
            'Alive at End': f"{sub['AliveEnd'].mean():.1f}",
            'Throughput (pkt/rnd)': f"{sub['Throughput'].mean():.1f}",
        })

    return pd.DataFrame(summary), df_all


def plot_comparative_charts(summary_standard: pd.DataFrame, summary_sparse: pd.DataFrame, save_path: str):
    """Generates comparative visualizations for Review 2 presentation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Standard vs Sparse PDR Comparison
    models = summary_standard['Model'].tolist()
    pdr_std = [float(x.split()[0]) for x in summary_standard['PDR (%)']]
    pdr_sp = [float(x.split()[0]) for x in summary_sparse['PDR (%)']]

    x = np.arange(len(models))
    width = 0.35

    axes[0].bar(x - width/2, pdr_std, width, label='Standard (40 nodes, 40m)', color='#2b5c8f')
    axes[0].bar(x + width/2, pdr_sp, width, label='Sparse Stress (15 nodes, 30m)', color='#e06666')
    axes[0].set_ylabel('Packet Delivery Ratio (%)', fontsize=12)
    axes[0].set_title('PDR Comparison: Standard vs Sparse Stress', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=15, ha='right', fontsize=10)
    axes[0].set_ylim(0, 105)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    axes[0].legend(fontsize=10)

    # Lifetime Variance Comparison (Differentiator for Model C)
    var_std = summary_standard['Lifetime Std'].tolist()
    var_sp = summary_sparse['Lifetime Std'].tolist()

    axes[1].bar(x - width/2, var_std, width, label='Standard Lifetime Std', color='#6fa8dc')
    axes[1].bar(x + width/2, var_sp, width, label='Sparse Lifetime Std', color='#cc0000')
    axes[1].set_ylabel('Lifetime Standard Deviation (Lower is Better)', fontsize=12)
    axes[1].set_title('Risk-Sensitivity: Network Lifetime Variance', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=15, ha='right', fontsize=10)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
