"""
Phase 5 Master Benchmark Suite:
Four 6G Physical Stress Environments (Sparse, Obstacle LoS Blockage, High Mobility, Energy-Scarce)
Benchmarking Model A (Alanazi) vs Model B (Rahul Anand) vs Model C+ (Our Complete Extension).
"""
import os
import sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
for _p in [
    os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib"),
    "C:/Users/Reet/AppData/Local/Programs/Python/Python310/lib/site-packages/torch/lib",
]:
    if os.path.exists(_p):
        try:
            os.add_dll_directory(_p)
        except Exception:
            pass

import math
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from src.config import CFG, ensure_dirs
from src.dataset import load_or_create_dataset, split_dataset, generate_temporal_sequences
from sklearn.preprocessing import StandardScaler
from src.models_link import (
    LinkMLP, LinkUQNet, UncertaintyModel,
    TemporalLinkTransformer, TemporalUncertaintyModel,
    train_point_model, train_temporal_model
)
from src.env import build_env
from src.models_rl import (
    TabularQRouter, DQN, DoubleDQNRouter,
    QRDQN, QRDQNRouter, ReplayBuffer,
    compute_dqn_loss, compute_qrdqn_loss, masked_argmax
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def gini_coefficient(array):
    """Computes Gini index on node energy consumption (0 = perfectly equal, 1 = unequal)."""
    array = np.array(array, dtype=np.float64).flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)
    array += 1e-9
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return float((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))


def train_model_a(cfg: dict, episodes: int = 300, seed: int = 42):
    """Trains Model A Tabular Q-Router."""
    router = TabularQRouter(max_a=cfg['MAX_A'], lr=cfg['tabular_lr'], tau=cfg['tabular_temp'])
    env, _ = build_env(cfg, seed=seed)
    for ep in range(episodes):
        s, m, acts = env.reset_packet()
        if s is None:
            env, _ = build_env(cfg, seed=seed + ep)
            s, m, acts = env.reset_packet()
        if s is None: continue
        done = False
        while not done:
            a = router.select_action(s, m)
            step_res, r, done, _ = env.step(a)
            sn, mn = step_res if not done else (None, None)
            router.update(s, a, r, sn, mn, done)
            if not done: s, m = sn, mn
        if ep % 50 == 0: env.apply_mobility()
    return router


def train_model_b(cfg: dict, uq_model, state_dim: int, episodes: int = 300, seed: int = 42):
    """Trains Model B Double-DQN Router (Risk-Neutral)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    pol = DQN(state_dim, cfg['MAX_A'], hidden=cfg['dqn_hidden']).to(DEVICE)
    tgt = DQN(state_dim, cfg['MAX_A'], hidden=cfg['dqn_hidden']).to(DEVICE)
    tgt.load_state_dict(pol.state_dict())
    opt = torch.optim.Adam(pol.parameters(), lr=cfg['lr_dqn'])
    rb = ReplayBuffer(cfg['replay_cap'])
    env, _ = build_env(cfg, uq=uq_model, seed=seed)

    eps = cfg['eps_start']
    for ep in range(episodes):
        s, m, acts = env.reset_packet()
        if s is None:
            env, _ = build_env(cfg, uq=uq_model, seed=seed + ep)
            s, m, acts = env.reset_packet()
        if s is None: continue
        done = False
        while not done:
            valid = [i for i in range(cfg['MAX_A']) if m[i] == 1.0]
            if np.random.uniform(0, 1) < eps:
                a = np.random.choice(valid) if valid else 0
            else:
                with torch.no_grad():
                    q = pol(torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(DEVICE))[0]
                    a = masked_argmax(q, torch.tensor(m).to(DEVICE))
            step_res, r, done, _ = env.step(a)
            sn = step_res[0] if not done else np.zeros(state_dim, dtype=np.float32)
            mn = step_res[1] if not done else np.zeros(cfg['MAX_A'], dtype=np.float32)
            rb.push(s, m, a, r, sn, mn, float(done))
            if not done: s, m = sn, mn

            if len(rb) >= cfg['batch_size']:
                batch = rb.sample(cfg['batch_size'])
                loss = compute_dqn_loss(pol, tgt, batch, gamma=cfg['gamma'], use_double=True, device=str(DEVICE))
                opt.zero_grad()
                loss.backward()
                opt.step()
        eps = max(cfg['eps_end'], eps * cfg['eps_decay'])
        if ep % 200 == 0: tgt.load_state_dict(pol.state_dict())
        if ep % 50 == 0: env.apply_mobility()
    return DoubleDQNRouter(pol, device=str(DEVICE))


def train_model_c_plus(cfg: dict, temporal_model, state_dim: int, episodes: int = 300, seed: int = 42):
    """Trains Model C+ Distributional QR-DQN with CVaR-alpha and LayerNorm."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    pol = QRDQN(state_dim, cfg['MAX_A'], num_quantiles=cfg['qr_quantiles'],
                hidden=cfg['dqn_hidden'], use_layernorm=True).to(DEVICE)
    tgt = QRDQN(state_dim, cfg['MAX_A'], num_quantiles=cfg['qr_quantiles'],
                hidden=cfg['dqn_hidden'], use_layernorm=True).to(DEVICE)
    tgt.load_state_dict(pol.state_dict())
    opt = torch.optim.Adam(pol.parameters(), lr=cfg['qr_lr'])
    rb = ReplayBuffer(cfg['replay_cap'])
    env, _ = build_env(cfg, temporal_model=temporal_model, seed=seed)

    eps = cfg['eps_start']
    for ep in range(episodes):
        s, m, acts = env.reset_packet()
        if s is None:
            env, _ = build_env(cfg, temporal_model=temporal_model, seed=seed + ep)
            s, m, acts = env.reset_packet()
        if s is None: continue
        done = False
        while not done:
            valid = [i for i in range(cfg['MAX_A']) if m[i] == 1.0]
            if np.random.uniform(0, 1) < eps:
                a = np.random.choice(valid) if valid else 0
            else:
                with torch.no_grad():
                    q = pol(torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(DEVICE))[0]
                    k = max(1, int(math.ceil(cfg['cvar_alpha'] * cfg['qr_quantiles'])))
                    q_sorted, _ = torch.sort(q, dim=-1)
                    cvar_vals = q_sorted[:, :k].mean(dim=-1)
                    a = masked_argmax(cvar_vals, torch.tensor(m).to(DEVICE))
            step_res, r, done, _ = env.step(a)
            sn = step_res[0] if not done else np.zeros(state_dim, dtype=np.float32)
            mn = step_res[1] if not done else np.zeros(cfg['MAX_A'], dtype=np.float32)
            rb.push(s, m, a, r, sn, mn, float(done))
            if not done: s, m = sn, mn

            if len(rb) >= cfg['batch_size']:
                batch = rb.sample(cfg['batch_size'])
                loss = compute_qrdqn_loss(pol, tgt, batch, gamma=cfg['gamma'],
                                          alpha=cfg['cvar_alpha'], device=str(DEVICE))
                opt.zero_grad()
                loss.backward()
                opt.step()
        eps = max(cfg['eps_end'], eps * cfg['eps_decay'])
        if ep % 200 == 0: tgt.load_state_dict(pol.state_dict())
        if ep % 50 == 0:
            env.service_queues()
            env.apply_mobility()
    return QRDQNRouter(pol, alpha=cfg['cvar_alpha'], device=str(DEVICE))


def evaluate_scenario(router, cfg: dict, scenario: str, uq_model=None, temporal_model=None,
                      rounds: int = 60, pkts_per_round: int = 30, seed: int = 42):
    """Evaluates router within a specific Phase 5 stress scenario."""
    env, nodes = build_env(cfg, uq=uq_model, temporal_model=temporal_model, seed=seed, scenario=scenario)
    total_pkts = 0
    deliv_pkts = 0
    total_hops = 0
    first_death = rounds

    for r in range(rounds):
        for _ in range(pkts_per_round):
            s, m, acts = env.reset_packet()
            if s is None: break
            total_pkts += 1
            done = False
            while not done:
                a = router.select_action(s, m)
                step_res, r_val, done, info = env.step(a)
                if done:
                    if info.get('delivered', False):
                        deliv_pkts += 1
                        total_hops += info.get('hops', 0)
                    break
                s, m = step_res

        env.service_queues()
        env.apply_mobility()

        alive = sum(1 for n in nodes if n.alive and not n.is_sink)
        if alive < (len(nodes) - 1) and first_death == rounds:
            first_death = r + 1

    pdr = (deliv_pkts / total_pkts * 100.0) if total_pkts > 0 else 0.0
    avg_hops = (total_hops / deliv_pkts) if deliv_pkts > 0 else 0.0
    throughput = (deliv_pkts / max(1, rounds))
    init_e = float(env.cfg.get('init_energy', 1.0))
    consumed = [max(0.0, init_e - n.energy) for n in nodes if not n.is_sink]
    gini = gini_coefficient(consumed)
    surviving = sum(1 for n in nodes if n.alive and not n.is_sink)

    return {
        'pdr': pdr,
        'hops': avg_hops,
        'fnd': first_death,
        'throughput': throughput,
        'gini': gini,
        'surviving': surviving,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=250, help='RL training episodes')
    parser.add_argument('--rounds', type=int, default=50, help='Evaluation rounds per scenario')
    parser.add_argument('--pkts_per_round', type=int, default=25, help='Packets per round')
    parser.add_argument('--seeds', type=str, default='42,101,2024', help='Evaluation seeds')
    parser.add_argument('--trans_epochs', type=int, default=25, help='Transformer training epochs')
    parser.add_argument('--mc_samples', type=int, default=15, help='MC dropout sample count')
    args = parser.parse_args()

    ensure_dirs()
    seeds = [int(s) for s in args.seeds.split(',')]

    print("=" * 85)
    print("  PHASE 5 BENCHMARK: Four 6G Physical Stress Environments")
    print(f"  Scenarios: {list(CFG['PHASE5_SCENARIOS'].keys())}")
    print(f"  Device: {DEVICE} | Episodes: {args.episodes} | Rounds: {args.rounds} | Seeds: {seeds}")
    print("=" * 85)
    sys.stdout.flush()

    # 1. Dataset & Predictors
    print("\n[Step 1] Loading Dataset & Training Link Predictors...")
    sys.stdout.flush()
    df = load_or_create_dataset(CFG['data_path'], seed=CFG['global_seed'])
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df, seed=CFG['global_seed'])
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_val_s = scaler.transform(X_train), scaler.transform(X_val)

    X_seq_train, y_seq_train = generate_temporal_sequences(X_train_s, y_train, seq_len=CFG['seq_len'], rho=CFG['ar1_rho'], seed=CFG['global_seed'])
    X_seq_val, y_seq_val = generate_temporal_sequences(X_val_s, y_val, seq_len=CFG['seq_len'], rho=CFG['ar1_rho'], seed=CFG['global_seed'])

    print("  -> Training Model B LinkUQNet...")
    net_b = train_point_model(LinkUQNet(hidden=CFG['link_hidden'], dropout=CFG['link_dropout']),
                              X_train_s, y_train, X_val_s, y_val, epochs=40, lr=CFG['link_lr'], device=str(DEVICE))
    uq_b = UncertaintyModel(net_b, scaler, device=str(DEVICE), mc_samples=args.mc_samples)

    print(f"  -> Training Model C+ Temporal Link Transformer ({args.trans_epochs} epochs)...")
    net_c = train_temporal_model(TemporalLinkTransformer(d_model=CFG['trans_d_model'], nhead=CFG['trans_nhead'], num_layers=CFG['trans_layers']),
                                 X_seq_train, y_seq_train, X_seq_val, y_seq_val, epochs=args.trans_epochs, lr=CFG['trans_lr'], device=str(DEVICE))
    temporal_model = TemporalUncertaintyModel(net_c, scaler, device=str(DEVICE), mc_samples=args.mc_samples, seq_len=CFG['seq_len'])

    # 2. Train Routers
    print("\n[Step 2] Training Routing Policies...")
    sys.stdout.flush()
    cfg_base = dict(CFG)
    cfg_base['enable_phase4'] = False
    state_dim_base = CFG['STATE_DIM_PHASE3']

    cfg_p4 = dict(CFG)
    cfg_p4['enable_phase4'] = True
    state_dim_p4 = CFG['STATE_DIM_PHASE4']

    print("  -> Training Model A (Alanazi Tabular Q)...")
    router_a = train_model_a(cfg_base, episodes=args.episodes, seed=seeds[0])

    print("  -> Training Model B (Rahul Anand Double-DQN)...")
    router_b = train_model_b(cfg_base, uq_b, state_dim_base, episodes=args.episodes, seed=seeds[0])

    print("  -> Training Model C+ (Our Complete Extension QR-DQN/CVaR + Phase 4)...")
    router_c_plus = train_model_c_plus(cfg_p4, temporal_model, state_dim_p4, episodes=args.episodes, seed=seeds[0])

    # 3. Evaluate Across All 4 Scenarios
    scenarios = ['sparse', 'obstacle', 'high_mobility', 'energy_scarce']
    records = []

    print("\n[Step 3] Evaluating All Models Across the 4 Physical Stress Scenarios...")
    sys.stdout.flush()

    for sc in scenarios:
        sc_name = CFG['PHASE5_SCENARIOS'][sc]['name']
        print(f"\n================================================================================")
        print(f"  ENVIRONMENT: {sc_name}")
        print(f"================================================================================")
        sys.stdout.flush()

        for model_name, router, uq, temp, cfg_use in [
            ('Model A (Alanazi et al.)', router_a, None, None, cfg_base),
            ('Model B (Rahul Anand)', router_b, uq_b, None, cfg_base),
            ('Model C+ (Our Extension)', router_c_plus, None, temporal_model, cfg_p4),
        ]:
            res_list = []
            for s in seeds:
                res = evaluate_scenario(router, cfg_use, sc, uq_model=uq, temporal_model=temp,
                                        rounds=args.rounds, pkts_per_round=args.pkts_per_round, seed=s)
                res_list.append(res)

            pdr_m, pdr_s = float(np.mean([r['pdr'] for r in res_list])), float(np.std([r['pdr'] for r in res_list]))
            hop_m, hop_s = float(np.mean([r['hops'] for r in res_list])), float(np.std([r['hops'] for r in res_list]))
            fnd_m, fnd_s = float(np.mean([r['fnd'] for r in res_list])), float(np.std([r['fnd'] for r in res_list]))
            th_m, th_s = float(np.mean([r['throughput'] for r in res_list])), float(np.std([r['throughput'] for r in res_list]))
            gini_m, gini_s = float(np.mean([r['gini'] for r in res_list])), float(np.std([r['gini'] for r in res_list]))

            print(f"  {model_name:<26} | PDR: {pdr_m:5.1f}% ± {pdr_s:4.1f}% | Latency: {hop_m:4.2f} hops | FND: {fnd_m:4.1f} rnd | Gini: {gini_m:.3f}")
            sys.stdout.flush()

            records.append({
                'Scenario': sc,
                'Scenario_Name': sc_name,
                'Model': model_name,
                'PDR_mean': pdr_m,
                'PDR_std': pdr_s,
                'Latency_hops_mean': hop_m,
                'Latency_hops_std': hop_s,
                'FND_rounds_mean': fnd_m,
                'FND_rounds_std': fnd_s,
                'Throughput_pkts_rnd_mean': th_m,
                'Throughput_pkts_rnd_std': th_s,
                'Gini_mean': gini_m,
                'Gini_std': gini_s,
            })

    # 4. Save Leaderboard CSV
    df_results = pd.DataFrame(records)
    csv_path = os.path.join(CFG['artifacts_dir'], 'phase5_stress_leaderboard.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"\n[Saved Leaderboard] {csv_path}")

    # 5. Generate Multi-Scenario Comparative Chart
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes_flat = axes.flatten()
    models = ['Model A (Alanazi)', 'Model B (Rahul)', 'Model C+ (Ours)']
    palette = ['#94a3b8', '#3b82f6', '#10b981']

    for idx, sc in enumerate(scenarios):
        sc_df = df_results[df_results['Scenario'] == sc]
        ax = axes_flat[idx]
        pdr_means = sc_df['PDR_mean'].values
        pdr_stds = sc_df['PDR_std'].values

        bars = ax.bar(models, pdr_means, yerr=pdr_stds, capsize=6, color=palette, alpha=0.88, edgecolor='black')
        ax.set_title(CFG['PHASE5_SCENARIOS'][sc]['name'], fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel('Packet Delivery Ratio (PDR %)', fontsize=11)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        for bar, val in zip(bars, pdr_means):
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 3.0, f'{val:.1f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.suptitle('6G Physical Stress Environments: Model A vs Model B vs Model C+', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_path = os.path.join(CFG['figures_dir'], 'phase5_stress_matrix.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[Saved Matrix Plot] {fig_path}")
    print("\n[Phase 5 Benchmark Complete] All 4 stress environments evaluated successfully.")


if __name__ == '__main__':
    main()
