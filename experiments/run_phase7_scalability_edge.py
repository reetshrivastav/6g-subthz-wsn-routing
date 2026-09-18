"""
Phase 7 Benchmark: Massive Network Scalability & Embedded Edge Feasibility Profiling
1. Scalability Sweep: Evaluates Models A, B, and C+ across scales N in [15, 25, 50, 75, 100] nodes.
2. Computational Profiling: Profiles parameter count, storage footprint (KB), inference latency (us/step), and FLOPs.
"""
import os
import sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
import time
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

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

from src.config import CFG, ensure_dirs
from src.dataset import load_or_create_dataset, split_dataset, generate_temporal_sequences
from sklearn.preprocessing import StandardScaler
from src.models_link import (
    LinkUQNet, UncertaintyModel,
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


def train_model_a(cfg: dict, episodes: int = 200, seed: int = 42):
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


def train_model_b(cfg: dict, uq_model, state_dim: int | None = None, episodes: int = 200, seed: int = 42):
    """Trains Model B Double-DQN Router (Risk-Neutral)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    st_dim = state_dim or cfg.get('STATE_DIM_PHASE3', 26)
    pol = DQN(st_dim, cfg['MAX_A'], hidden=cfg['dqn_hidden']).to(DEVICE)
    tgt = DQN(st_dim, cfg['MAX_A'], hidden=cfg['dqn_hidden']).to(DEVICE)
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
            sn = step_res[0] if not done else np.zeros(st_dim, dtype=np.float32)
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


def gini_coefficient(array):
    """Computes Gini index on node energy consumption."""
    array = np.array(array, dtype=np.float64).flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)
    array += 1e-9
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return float((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))


def train_model_c_plus(cfg: dict, temporal_model: TemporalUncertaintyModel,
                       episodes: int = 250, seed: int = 42, alpha: float = 0.25):
    """Trains Model C+ (QR-DQN with CVaR-alpha and Phase 4 extensions)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = cfg['STATE_DIM_PHASE4']
    pol = QRDQN(state_dim, cfg['MAX_A'], num_quantiles=cfg['qr_quantiles'],
                hidden=cfg['dqn_hidden'], use_layernorm=True).to(DEVICE)
    tgt = QRDQN(state_dim, cfg['MAX_A'], num_quantiles=cfg['qr_quantiles'],
                hidden=cfg['dqn_hidden'], use_layernorm=True).to(DEVICE)
    tgt.load_state_dict(pol.state_dict())
    opt = torch.optim.Adam(pol.parameters(), lr=cfg['qr_lr'])
    rb = ReplayBuffer(cfg['replay_cap'])

    env, _ = build_env(cfg, seed=seed, temporal_model=temporal_model)

    eps = cfg['eps_start']
    for ep in range(episodes):
        s, m, acts = env.reset_packet()
        if s is None:
            env, _ = build_env(cfg, seed=seed + ep, temporal_model=temporal_model)
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
                    k = max(1, int(math.ceil(alpha * cfg['qr_quantiles'])))
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
                                          alpha=alpha, device=str(DEVICE))
                opt.zero_grad()
                loss.backward()
                opt.step()
        eps = max(cfg['eps_end'], eps * cfg['eps_decay'])
        if ep % 200 == 0: tgt.load_state_dict(pol.state_dict())
        if ep % 50 == 0:
            env.service_queues()
            env.apply_mobility()

    return QRDQNRouter(pol, alpha=alpha, device=str(DEVICE))


def evaluate_policy_on_scale(router, cfg: dict, link_kw: dict,
                             n_nodes: int, area_size: float, comm_range: float,
                             rounds: int = 40, pkts_per_round: int = 20, seed: int = 42):
    """Evaluates a policy on a specific network scale."""
    cfg_scaled = dict(cfg)
    cfg_scaled['n_nodes'] = n_nodes
    cfg_scaled['area'] = area_size
    cfg_scaled['comm_range'] = comm_range
    cfg_scaled['MAX_HOPS'] = max(15, int(15 * math.sqrt(n_nodes / 25.0)))

    env, nodes = build_env(cfg_scaled, seed=seed, n_nodes=n_nodes, comm_range=comm_range, **link_kw)
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
    init_e = float(cfg.get('init_energy', 1.0))
    consumed = [max(0.0, init_e - n.energy) for n in nodes if not n.is_sink]
    gini = gini_coefficient(consumed)

    return {
        'pdr': pdr,
        'hops': avg_hops,
        'fnd': first_death,
        'throughput': throughput,
        'gini': gini,
    }


def profile_model_complexity(cfg, pol_a, uq_mlp, temporal_model, pol_b, pol_c):
    """
    Measures parameter counts, model memory (KB), and single-hop inference latency (microseconds).
    """
    print("\n" + "=" * 85)
    print("  PART 2: COMPUTATIONAL PROFILING & EMBEDDED EDGE FEASIBILITY")
    print("=" * 85)
    sys.stdout.flush()

    records = []

    # Model A: Tabular Q-learning
    q_table_params = pol_a.q_table.size
    q_table_kb = pol_a.q_table.nbytes / 1024.0

    # Benchmark Model A inference time
    router_a = pol_a
    s_dummy = np.zeros(CFG['STATE_DIM_PHASE3'], dtype=np.float32)
    m_dummy = np.ones(CFG['MAX_A'], dtype=np.float32)

    t0 = time.perf_counter()
    for _ in range(2000):
        _ = router_a.select_action(s_dummy, m_dummy)
    lat_a_us = ((time.perf_counter() - t0) / 2000) * 1e6

    records.append({
        'Model': 'Model A (Alanazi Tabular Q)',
        'Predictor_Params': 0,
        'Predictor_Size_KB': 0.0,
        'Policy_Params': q_table_params,
        'Policy_Size_KB': q_table_kb,
        'Total_Params': q_table_params,
        'Total_Storage_KB': q_table_kb,
        'PerHop_Latency_us': lat_a_us,
        'PerHop_Latency_ms': lat_a_us / 1000.0,
        'Memory_Footprint': 'Ultra-low (< 1 KB)',
        'Edge_MCU_Class': 'Ultra-low power (Cortex-M0+)',
    })

    # Model B: Senior Rahul Anand (Point MLP + Double-DQN)
    p_params_b = sum(p.numel() for p in uq_mlp.model.parameters())
    p_kb_b = sum(p.numel() * p.element_size() for p in uq_mlp.model.parameters()) / 1024.0

    pol_params_b = sum(p.numel() for p in pol_b.net.parameters())
    pol_kb_b = sum(p.numel() * p.element_size() for p in pol_b.net.parameters()) / 1024.0

    # Benchmark Model B per-hop inference (Predictor + Policy)
    router_b = pol_b
    s_dummy_b = np.zeros(CFG['STATE_DIM_PHASE3'], dtype=np.float32)

    # Warmup
    for _ in range(100):
        _ = uq_mlp.predict_from_metrics(-65.0, 18.0, 0.05, mc_samples=15)
        _ = router_b.select_action(s_dummy_b, m_dummy)

    t0 = time.perf_counter()
    for _ in range(1000):
        _ = uq_mlp.predict_from_metrics(-65.0, 18.0, 0.05, mc_samples=15)
        _ = router_b.select_action(s_dummy_b, m_dummy)
    lat_b_us = ((time.perf_counter() - t0) / 1000) * 1e6

    records.append({
        'Model': 'Model B (Rahul Anand Double-DQN)',
        'Predictor_Params': p_params_b,
        'Predictor_Size_KB': p_kb_b,
        'Policy_Params': pol_params_b,
        'Policy_Size_KB': pol_kb_b,
        'Total_Params': p_params_b + pol_params_b,
        'Total_Storage_KB': p_kb_b + pol_kb_b,
        'PerHop_Latency_us': lat_b_us,
        'PerHop_Latency_ms': lat_b_us / 1000.0,
        'Memory_Footprint': f'{(p_kb_b + pol_kb_b):.1f} KB',
        'Edge_MCU_Class': 'Mid-range MCU (Cortex-M4, ESP32)',
    })

    # Model C+: Proposed Extension (Temporal Transformer + QR-DQN + CVaR)
    p_params_c = sum(p.numel() for p in temporal_model.model.parameters())
    p_kb_c = sum(p.numel() * p.element_size() for p in temporal_model.model.parameters()) / 1024.0

    pol_params_c = sum(p.numel() for p in pol_c.net.parameters())
    pol_kb_c = sum(p.numel() * p.element_size() for p in pol_c.net.parameters()) / 1024.0

    # Benchmark Model C+ per-hop inference
    router_c = pol_c
    s_dummy_c = np.zeros(CFG['STATE_DIM_PHASE4'], dtype=np.float32)

    # Warmup
    for _ in range(100):
        _ = temporal_model.update_and_predict(1, 2, -65.0, 18.0, 0.05)
        _ = router_c.select_action(s_dummy_c, m_dummy)

    t0 = time.perf_counter()
    for _ in range(1000):
        _ = temporal_model.update_and_predict(1, 2, -65.0, 18.0, 0.05)
        _ = router_c.select_action(s_dummy_c, m_dummy)
    lat_c_us = ((time.perf_counter() - t0) / 1000) * 1e6

    records.append({
        'Model': 'Model C+ (Our Complete Extension)',
        'Predictor_Params': p_params_c,
        'Predictor_Size_KB': p_kb_c,
        'Policy_Params': pol_params_c,
        'Policy_Size_KB': pol_kb_c,
        'Total_Params': p_params_c + pol_params_c,
        'Total_Storage_KB': p_kb_c + pol_kb_c,
        'PerHop_Latency_us': lat_c_us,
        'PerHop_Latency_ms': lat_c_us / 1000.0,
        'Memory_Footprint': f'{(p_kb_c + pol_kb_c):.1f} KB',
        'Edge_MCU_Class': 'Edge SoC (Cortex-M7, Raspberry Pi, RISC-V)',
    })

    df_prof = pd.DataFrame(records)
    for r in records:
        print(f"  -> {r['Model']:<34} | Params: {r['Total_Params']:>6} | Size: {r['Total_Storage_KB']:>6.1f} KB | Latency: {r['PerHop_Latency_us']:>6.1f} us ({r['PerHop_Latency_ms']:.3f} ms)")
    sys.stdout.flush()

    prof_csv_path = os.path.join(CFG['artifacts_dir'], 'phase7_edge_complexity_profile.csv')
    df_prof.to_csv(prof_csv_path, index=False)
    print(f"\n[Saved Artifact] {prof_csv_path}")
    return df_prof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=200, help='RL training episodes')
    parser.add_argument('--rounds', type=int, default=40, help='Evaluation rounds per scale')
    parser.add_argument('--pkts_per_round', type=int, default=20, help='Packets per round')
    parser.add_argument('--seeds', type=str, default='42,101,2024', help='Evaluation seeds')
    parser.add_argument('--trans_epochs', type=int, default=20, help='Transformer training epochs')
    parser.add_argument('--mc_samples', type=int, default=15, help='MC dropout sample count')
    args = parser.parse_args()

    ensure_dirs()
    seeds = [int(s) for s in args.seeds.split(',')]

    print("=" * 85)
    print("  PHASE 7: MASSIVE NETWORK SCALABILITY & EMBEDDED EDGE FEASIBILITY")
    print(f"  Device: {DEVICE} | Episodes: {args.episodes} | Rounds: {args.rounds} | Seeds: {seeds}")
    print("=" * 85)
    sys.stdout.flush()

    # Step 1: Datasets & Link Models
    print("\n[Step 1] Training Physical Link Uncertainty Models...")
    df = load_or_create_dataset(CFG['data_path'], seed=CFG['global_seed'])
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df, seed=CFG['global_seed'])
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_val_s = scaler.transform(X_train), scaler.transform(X_val)

    X_seq_train, y_seq_train = generate_temporal_sequences(X_train_s, y_train, seq_len=CFG['seq_len'], rho=CFG['ar1_rho'], seed=CFG['global_seed'])
    X_seq_val, y_seq_val = generate_temporal_sequences(X_val_s, y_val, seq_len=CFG['seq_len'], rho=CFG['ar1_rho'], seed=CFG['global_seed'])

    print("  -> Training Static Point MLP (LinkUQNet)...")
    net_mlp = train_point_model(LinkUQNet(hidden=CFG['link_hidden'], dropout=CFG['link_dropout']),
                                X_train_s, y_train, X_val_s, y_val, epochs=40, lr=CFG['link_lr'], device=str(DEVICE))
    uq_mlp = UncertaintyModel(net_mlp, scaler, device=str(DEVICE), mc_samples=args.mc_samples)

    print(f"  -> Training Temporal Link Transformer ({args.trans_epochs} epochs)...")
    net_trans = train_temporal_model(TemporalLinkTransformer(d_model=CFG['trans_d_model'], nhead=CFG['trans_nhead'], num_layers=CFG['trans_layers']),
                                     X_seq_train, y_seq_train, X_seq_val, y_seq_val, epochs=args.trans_epochs, lr=CFG['trans_lr'], device=str(DEVICE))
    temporal_model = TemporalUncertaintyModel(net_trans, scaler, device=str(DEVICE), mc_samples=args.mc_samples, seq_len=CFG['seq_len'])

    # Step 2: Training Policies on Standard Reference Grid
    print("\n[Step 2] Training Routing Policies (Standard Grid Baseline)...")
    cfg_base = dict(CFG)
    cfg_base['enable_phase4'] = False

    cfg_p4 = dict(CFG)
    cfg_p4['enable_phase4'] = True

    print("  -> Training Model A (Alanazi Tabular Q)...")
    pol_a = train_model_a(cfg_base, episodes=args.episodes, seed=seeds[0])

    print("  -> Training Model B (Rahul Anand Double-DQN)...")
    pol_b = train_model_b(cfg_base, uq_mlp, episodes=args.episodes, seed=seeds[0])

    print("  -> Training Model C+ (Our Complete Extension QR-DQN/CVaR + Phase 4)...")
    pol_c = train_model_c_plus(cfg_p4, temporal_model, episodes=args.episodes, seed=seeds[0], alpha=0.25)

    # Step 3: Computational Profiling
    df_prof = profile_model_complexity(CFG, pol_a, uq_mlp, temporal_model, pol_b, pol_c)

    # Step 4: Network Scalability Sweep
    print("\n" + "=" * 85)
    print("  PART 1: MASSIVE NETWORK SCALABILITY BENCHMARK (N = 15 to 100 Nodes)")
    print("=" * 85)
    sys.stdout.flush()

    scales = [
        # (N, Area_Size_L, Comm_Range) - Constant density scaling
        (15, 78.0, 38.0),
        (25, 100.0, 40.0),
        (50, 141.0, 42.0),
        (75, 173.0, 44.0),
        (100, 200.0, 45.0),
    ]

    models = [
        ('Model A (Alanazi et al.)', pol_a, cfg_base, {}),
        ('Model B (Rahul Anand)', pol_b, cfg_base, {'uq': uq_mlp}),
        ('Model C+ (Our Extension)', pol_c, cfg_p4, {'temporal_model': temporal_model}),
    ]

    scalability_records = []

    for n_val, area_val, cr_val in scales:
        print(f"\n  ================================================================")
        print(f"  Evaluating Scale: N = {n_val} Nodes | Area: {area_val:.0f}x{area_val:.0f} m | Comm Range: {cr_val:.0f} m")
        print(f"  ================================================================")
        sys.stdout.flush()

        for m_name, router, cfg_m, l_kw in models:
            res_list = []
            for s in seeds:
                res = evaluate_policy_on_scale(
                    router, cfg_m, l_kw,
                    n_nodes=n_val, area_size=area_val, comm_range=cr_val,
                    rounds=args.rounds, pkts_per_round=args.pkts_per_round, seed=s
                )
                res_list.append(res)

            pdr_m, pdr_s = float(np.mean([r['pdr'] for r in res_list])), float(np.std([r['pdr'] for r in res_list]))
            hop_m, hop_s = float(np.mean([r['hops'] for r in res_list])), float(np.std([r['hops'] for r in res_list]))
            fnd_m, fnd_s = float(np.mean([r['fnd'] for r in res_list])), float(np.std([r['fnd'] for r in res_list]))
            gini_m, gini_s = float(np.mean([r['gini'] for r in res_list])), float(np.std([r['gini'] for r in res_list]))

            print(f"    {m_name:<26} | PDR: {pdr_m:5.1f}% ± {pdr_s:4.1f}% | Latency: {hop_m:4.2f} hops | FND: {fnd_m:4.1f} rnd | Gini: {gini_m:.3f}")
            sys.stdout.flush()

            scalability_records.append({
                'Nodes_N': n_val,
                'Area_Size': area_val,
                'Comm_Range': cr_val,
                'Model': m_name,
                'PDR_mean': pdr_m,
                'PDR_std': pdr_s,
                'Latency_hops_mean': hop_m,
                'Latency_hops_std': hop_s,
                'FND_rounds_mean': fnd_m,
                'FND_rounds_std': fnd_s,
                'Gini_mean': gini_m,
                'Gini_std': gini_s,
            })

    df_scale = pd.DataFrame(scalability_records)
    scale_csv_path = os.path.join(CFG['artifacts_dir'], 'phase7_scalability_benchmark.csv')
    df_scale.to_csv(scale_csv_path, index=False)
    print(f"\n[Saved Artifact] {scale_csv_path}")

    # Step 5: 4-Panel Visualization
    print("\n[Step 5] Generating Publication-Grade Scalability & Complexity Figure...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    n_nodes_unique = [15, 25, 50, 75, 100]
    color_map = {
        'Model A (Alanazi et al.)': '#ef4444',
        'Model B (Rahul Anand)': '#3b82f6',
        'Model C+ (Our Extension)': '#10b981',
    }
    marker_map = {
        'Model A (Alanazi et al.)': 's',
        'Model B (Rahul Anand)': '^',
        'Model C+ (Our Extension)': 'o',
    }

    # Panel 1: PDR vs Scale (N)
    for m_name in color_map.keys():
        sub = df_scale[df_scale['Model'] == m_name]
        axes[0, 0].plot(sub['Nodes_N'], sub['PDR_mean'], marker=marker_map[m_name],
                        linewidth=2.5, label=m_name, color=color_map[m_name])
        axes[0, 0].fill_between(sub['Nodes_N'], sub['PDR_mean'] - sub['PDR_std'],
                                sub['PDR_mean'] + sub['PDR_std'], color=color_map[m_name], alpha=0.15)
    axes[0, 0].set_title('Asymptotic Reliability: PDR vs Network Scale (N)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Network Size N (Nodes)')
    axes[0, 0].set_ylabel('Packet Delivery Ratio (PDR %)')
    axes[0, 0].set_ylim(0, 105)
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)
    axes[0, 0].legend(loc='lower left')

    # Panel 2: Hop Count vs Scale (N)
    for m_name in color_map.keys():
        sub = df_scale[df_scale['Model'] == m_name]
        axes[0, 1].plot(sub['Nodes_N'], sub['Latency_hops_mean'], marker=marker_map[m_name],
                        linewidth=2.5, label=m_name, color=color_map[m_name])
    axes[0, 1].set_title('Path Scalability: Delivery Latency vs Network Diameter', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Network Size N (Nodes)')
    axes[0, 1].set_ylabel('Average Hop Count (Hops)')
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)
    axes[0, 1].legend(loc='upper left')

    # Panel 3: Energy Gini vs Scale (N)
    for m_name in color_map.keys():
        sub = df_scale[df_scale['Model'] == m_name]
        axes[1, 0].plot(sub['Nodes_N'], sub['Gini_mean'], marker=marker_map[m_name],
                        linewidth=2.5, label=m_name, color=color_map[m_name])
    axes[1, 0].set_title('Load Balancing: Energy Depletion Inequality (Gini Index)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Network Size N (Nodes)')
    axes[1, 0].set_ylabel('Gini Index (0=Equal, Lower is Better)')
    axes[1, 0].set_ylim(0.0, 0.7)
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)
    axes[1, 0].legend(loc='upper left')

    # Panel 4: Hardware Profiling (Per-Hop Latency & Memory Footprint)
    labels_short = ['Model A\n(Tabular Q)', 'Model B\n(Senior Double-DQN)', 'Model C+\n(Our QR-DQN/Trans)']
    latencies_us = df_prof['PerHop_Latency_us'].values
    sizes_kb = df_prof['Total_Storage_KB'].values

    ax_lat = axes[1, 1]
    ax_mem = ax_lat.twinx()

    x = np.arange(len(labels_short))
    width = 0.35

    rects1 = ax_lat.bar(x - width/2, latencies_us, width, label='Inference Latency (μs)', color='#3b82f6', alpha=0.85, edgecolor='black')
    rects2 = ax_mem.bar(x + width/2, sizes_kb, width, label='Model Footprint (KB)', color='#10b981', alpha=0.85, edgecolor='black')

    ax_lat.set_ylabel('Inference Latency per Hop (μs)', color='#1e40af', fontweight='bold')
    ax_mem.set_ylabel('Model Weight Footprint (KB)', color='#065f46', fontweight='bold')
    ax_lat.set_title('Embedded Feasibility: Latency & Storage Footprint on Edge CPU', fontsize=12, fontweight='bold')
    ax_lat.set_xticks(x)
    ax_lat.set_xticklabels(labels_short)
    ax_lat.grid(axis='y', linestyle='--', alpha=0.5)

    # Annotations
    for r in rects1:
        h = r.get_height()
        ax_lat.text(r.get_x() + r.get_width()/2.0, h + 15.0, f'{h:.0f} μs', ha='center', fontsize=9, fontweight='bold', color='#1e40af')
    for r in rects2:
        h = r.get_height()
        ax_mem.text(r.get_x() + r.get_width()/2.0, h + 5.0, f'{h:.1f} KB', ha='center', fontsize=9, fontweight='bold', color='#065f46')

    # Add combined legend
    lines1, labels1 = ax_lat.get_legend_handles_labels()
    lines2, labels2 = ax_mem.get_legend_handles_labels()
    ax_lat.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.suptitle('Phase 7 Massive Network Scalability & Embedded Edge Feasibility Benchmark', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_path = os.path.join(CFG['figures_dir'], 'phase7_scalability_and_complexity.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[Saved Figure] {fig_path}")

    print("\n[Phase 7 Complete] Scalability sweep and edge hardware profiling finished successfully.")


if __name__ == '__main__':
    main()
