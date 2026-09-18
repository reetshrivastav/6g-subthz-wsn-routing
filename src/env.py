"""
Canonical WSN Simulation Environment:
Contains Node, EnergyModel, WirelessChannel (with temporal dynamics), MobilityModel,
pad_actions, build_state, RoutingEnv, and build_env.
"""
import numpy as np
import torch

class Node:
    """Represents a single wireless sensor node or the sink."""
    def __init__(self, idx: int, x: float, y: float, energy: float, is_sink: bool = False, max_queue: int = 20):
        self.idx = int(idx)
        self.x = float(x)
        self.y = float(y)
        self.energy = float(energy)
        self.alive = True
        self.is_sink = is_sink
        self.queue_len = 0
        self.max_queue = int(max_queue)
        self.loss_history = []

    def distance_to(self, other: 'Node') -> float:
        return float(np.hypot(self.x - other.x, self.y - other.y))

    def __repr__(self):
        kind = 'SINK' if self.is_sink else 'Node'
        return f'{kind}({self.idx}) pos=({self.x:.1f},{self.y:.1f}) E={self.energy:.4f}J alive={self.alive} Q={self.queue_len}'


class EnergyModel:
    """
    First-order radio energy model:
      E_tx(k, d) = k * E_elec + k * E_amp * d^2
      E_rx(k)    = k * E_elec (0 for sink, which has infinite power)
    """
    def __init__(self, E_elec: float = 50e-9, E_amp: float = 100e-12, packet_bits: int = 4000):
        self.E_elec = float(E_elec)
        self.E_amp = float(E_amp)
        self.packet_bits = int(packet_bits)

    def tx_cost(self, d: float) -> float:
        k = self.packet_bits
        return k * self.E_elec + k * self.E_amp * (d ** 2)

    def rx_cost(self) -> float:
        return self.packet_bits * self.E_elec

    def transmit(self, sender: Node, receiver: Node) -> tuple[float, float]:
        """Deducts energy from sender and receiver; returns (e_tx, e_rx)."""
        d = sender.distance_to(receiver)
        e_tx = self.tx_cost(d)
        e_rx = 0.0 if receiver.is_sink else self.rx_cost()

        sender.energy = max(0.0, sender.energy - e_tx)
        if sender.energy <= 0.0:
            sender.alive = False

        if not receiver.is_sink:
            receiver.energy = max(0.0, receiver.energy - e_rx)
            if receiver.energy <= 0.0:
                receiver.alive = False

        return e_tx, e_rx


class BlockageZone:
    """Represents a physical rectangular structural obstacle causing 6G Sub-THz LoS blockage."""
    def __init__(self, xmin: float, ymin: float, xmax: float, ymax: float):
        self.xmin = min(xmin, xmax)
        self.ymin = min(ymin, ymax)
        self.xmax = max(xmin, xmax)
        self.ymax = max(ymin, ymax)

    def intersects(self, p1: tuple[float, float], p2: tuple[float, float]) -> bool:
        """Liang-Barsky 2D line segment clipping against axis-aligned bounding box."""
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1

        p = [-dx, dx, -dy, dy]
        q = [x1 - self.xmin, self.xmax - x1, y1 - self.ymin, self.ymax - y1]

        u1 = 0.0
        u2 = 1.0

        for i in range(4):
            if p[i] == 0:
                if q[i] < 0:
                    return False
            else:
                t = q[i] / p[i]
                if p[i] < 0:
                    if t > u1:
                        u1 = t
                else:
                    if t < u2:
                        u2 = t
                if u1 > u2:
                    return False
        return True


class WirelessChannel:
    """
    Distance-based physical link simulator (RSSI, SNR, PLR).
    Also supports autoregressive AR(1) temporal memory for Model C and 6G LoS blockage.
    """
    def __init__(self, comm_range: float = 40.0, path_loss_exp: float = 2.5,
                 noise_rssi: float = 3.0, noise_snr: float = 2.0, noise_plr: float = 0.03,
                 rng=None):
        self.comm_range = float(comm_range)
        self.ple = float(path_loss_exp)
        self.ns_r = noise_rssi
        self.ns_s = noise_snr
        self.ns_p = noise_plr
        self.rng = rng or np.random.default_rng(42)

    def get_metrics(self, distance: float, is_blocked: bool = False) -> tuple[float, float, float]:
        """Returns (rssi, snr, plr) for a given Euclidean distance, applying LoS blockage attenuation if obstructed."""
        d = float(np.clip(distance / max(self.comm_range, 1e-6), 1e-3, 1.3))

        rssi_base = -40.0 - 60.0 * (d ** self.ple) + self.rng.normal(0, self.ns_r)
        snr_base = 40.0 * max(0.0, 1.0 - d / 1.3) + self.rng.normal(0, self.ns_s)
        plr_raw = 1.0 / (1.0 + np.exp(-6.0 * (d - 0.75))) + self.rng.normal(0, self.ns_p)

        if is_blocked:
            # Severe Sub-THz LoS blockage penalty: 28 dB RSSI drop, 20 dB SNR drop, +0.65 PLR
            rssi_base -= 28.0
            snr_base -= 20.0
            plr_raw += 0.65

        rssi = float(np.clip(rssi_base, -120.0, -30.0))
        snr = float(np.clip(snr_base, -10.0, 45.0))
        plr = float(np.clip(plr_raw, 0.0, 1.0))
        return rssi, snr, plr


class MobilityModel:
    """Random Waypoint mobility for a configurable mobile fraction."""
    def __init__(self, mobile_ids: set, area: float = 100.0, speed: float = 2.0, rng=None):
        self.mobile_ids = set(mobile_ids)
        self.area = float(area)
        self.speed = float(speed)
        self.rng = rng or np.random.default_rng(42)

    def step(self, nodes: list[Node]):
        for node in nodes:
            if node.is_sink or (node.idx not in self.mobile_ids) or (not node.alive):
                continue
            theta = self.rng.uniform(0.0, 2.0 * np.pi)
            dx = self.speed * np.cos(theta)
            dy = self.speed * np.sin(theta)
            nx = node.x + dx
            ny = node.y + dy
            # Wall reflection
            if nx < 0 or nx > self.area:
                nx = float(np.clip(node.x - dx, 0.0, self.area))
            if ny < 0 or ny > self.area:
                ny = float(np.clip(node.y - dy, 0.0, self.area))
            node.x = nx
            node.y = ny


def pad_actions(neighbour_ids: list[int], max_a: int) -> tuple[list[int], np.ndarray]:
    """Pad a neighbor list to fixed length max_a with -1 and return valid binary mask."""
    trunc = neighbour_ids[:max_a]
    n = len(trunc)
    acts = trunc + [-1] * (max_a - n)
    mask = np.array([1.0] * n + [0.0] * (max_a - n), dtype=np.float32)
    return acts, mask


def build_state(current_node: Node, sink: Node, acts: list[int], mask: np.ndarray,
                nodes_by_id: dict[int, Node], uq_model, channel: WirelessChannel,
                init_energy: float, area_diag: float, max_a: int,
                temporal_model=None, enable_phase4: bool = False,
                max_queue_cap: int = 20, obstacles: list = None) -> np.ndarray:
    """
    Constructs the 1D state vector:
    If not enable_phase4 (Canonical Phase 3, dimension = 2 + 4 * max_a = 26):
      [0]  current node energy (normalised)
      [1]  distance to sink (normalised)
      For each action slot i in [0, max_a):
        [2+4i+0] link success probability mean
        [2+4i+1] link uncertainty std (0 for Model A)
        [2+4i+2] next-hop residual energy
        [2+4i+3] forward progress toward sink
    If enable_phase4 (Context-Aware Phase 4, dimension = 3 + 7 * max_a = 45):
      [0]  current node energy (normalised)
      [1]  distance to sink (normalised)
      [2]  current node queue ratio (normalised)
      For each action slot i in [0, max_a):
        [3+7i+0] link success probability mean
        [3+7i+1] link uncertainty std
        [3+7i+2] next-hop residual energy
        [3+7i+3] forward progress toward sink
        [3+7i+4] next-hop queue ratio (q_j / max_queue)
        [3+7i+5] link stability metric (1 - loss rate)
        [3+7i+6] normalized forward progress ratio
    """
    if enable_phase4:
        state_dim = 3 + 7 * max_a
        state = np.zeros(state_dim, dtype=np.float32)
        state[0] = current_node.energy / max(init_energy, 1e-9)
        state[1] = current_node.distance_to(sink) / max(area_diag, 1e-9)
        state[2] = (current_node.queue_len / max(max_queue_cap, 1)) if not current_node.is_sink else 0.0

        valid_indices = [i for i in range(max_a) if acts[i] != -1 and mask[i] == 1.0]
        for i in valid_indices:
            nid = acts[i]
            nbr = nodes_by_id[nid]
            dist = current_node.distance_to(nbr)
            base_i = 3 + 7 * i

            is_blocked = any(obs.intersects((current_node.x, current_node.y), (nbr.x, nbr.y)) for obs in obstacles) if obstacles else False

            if temporal_model is not None:
                r, s, p = channel.get_metrics(dist, is_blocked=is_blocked)
                mean_prob, std_prob = temporal_model.update_and_predict(current_node.idx, nid, r, s, p)
            elif uq_model is not None:
                r, s, p = channel.get_metrics(dist, is_blocked=is_blocked)
                mean_prob, std_prob = uq_model.predict_from_metrics(r, s, p, sender_id=current_node.idx, receiver_id=nid)
            else:
                mean_prob = float(np.clip(1.0 - dist / max(channel.comm_range, 1e-6), 0.0, 1.0))
                if is_blocked:
                    mean_prob = max(0.0, mean_prob - 0.5)
                std_prob = 0.0

            state[base_i + 0] = mean_prob
            state[base_i + 1] = std_prob
            state[base_i + 2] = 1.0 if nbr.is_sink else (nbr.energy / max(init_energy, 1e-9))
            progress = (current_node.distance_to(sink) - nbr.distance_to(sink)) / max(area_diag, 1e-9)
            state[base_i + 3] = progress
            state[base_i + 4] = 0.0 if nbr.is_sink else (nbr.queue_len / max(max_queue_cap, 1))
            stability = 1.0 - (float(np.mean(nbr.loss_history[-5:])) if nbr.loss_history else 0.0)
            state[base_i + 5] = float(np.clip(stability, 0.0, 1.0))
            adv_ratio = (current_node.distance_to(sink) - nbr.distance_to(sink)) / max(dist, 1e-3)
            state[base_i + 6] = float(np.clip(adv_ratio, -1.0, 1.0))
        return state
    else:
        state_dim = 2 + 4 * max_a
        state = np.zeros(state_dim, dtype=np.float32)
        state[0] = current_node.energy / max(init_energy, 1e-9)
        state[1] = current_node.distance_to(sink) / max(area_diag, 1e-9)

        valid_indices = [i for i in range(max_a) if acts[i] != -1 and mask[i] == 1.0]
        for i in valid_indices:
            nid = acts[i]
            nbr = nodes_by_id[nid]
            dist = current_node.distance_to(nbr)
            base_i = 2 + 4 * i

            is_blocked = any(obs.intersects((current_node.x, current_node.y), (nbr.x, nbr.y)) for obs in obstacles) if obstacles else False

            if temporal_model is not None:
                r, s, p = channel.get_metrics(dist, is_blocked=is_blocked)
                mean_prob, std_prob = temporal_model.update_and_predict(current_node.idx, nid, r, s, p)
            elif uq_model is not None:
                r, s, p = channel.get_metrics(dist, is_blocked=is_blocked)
                mean_prob, std_prob = uq_model.predict_from_metrics(r, s, p, sender_id=current_node.idx, receiver_id=nid)
            else:
                mean_prob = float(np.clip(1.0 - dist / max(channel.comm_range, 1e-6), 0.0, 1.0))
                if is_blocked:
                    mean_prob = max(0.0, mean_prob - 0.5)
                std_prob = 0.0

            state[base_i + 0] = mean_prob
            state[base_i + 1] = std_prob
            state[base_i + 2] = 1.0 if nbr.is_sink else (nbr.energy / max(init_energy, 1e-9))
            progress = (current_node.distance_to(sink) - nbr.distance_to(sink)) / max(area_diag, 1e-9)
            state[base_i + 3] = progress
        return state


class RoutingEnv:
    """
    Packet-level episodic WSN routing simulator environment.
    Supports Model A, Model B, and Model C policies seamlessly.
    """
    def __init__(self, nodes: list[Node], mobile_ids: set, cfg: dict,
                 uq=None, temporal_model=None, seed: int = 42):
        self.nodes = nodes
        self.sink = next(n for n in nodes if n.is_sink)
        self.mobile_ids = set(mobile_ids)
        self.cfg = cfg
        self.uq = uq
        self.temporal_model = temporal_model
        self.rng = np.random.default_rng(seed)
        self.enable_phase4 = bool(cfg.get('enable_phase4', False))
        self.max_queue_cap = int(cfg.get('max_queue_cap', 20))
        self.obstacles = [BlockageZone(*b) for b in cfg.get('obstacles', [])]
        self.adversary_ids = set(cfg.get('adversary_ids', []))
        self.adversary_drop_prob = float(cfg.get('adversary_drop_prob', 1.0))

        self.energy_model = EnergyModel(cfg['E_elec'], cfg['E_amp'], cfg['packet_bits'])
        self.channel = WirelessChannel(cfg['comm_range'], rng=self.rng)
        self.mobility = MobilityModel(self.mobile_ids, cfg['area'], cfg['mobility_speed'], rng=self.rng)
        self.area_diag = float(np.hypot(cfg['area'], cfg['area']))
        self.nodes_by_id = {n.idx: n for n in self.nodes}

        # Episode tracking
        self.current_node = None
        self.hop_count = 0
        self.last_acts = None
        self.last_mask = None

    def _get_alive_neighbours(self, node: Node) -> list[int]:
        """Returns sorted list of reachable alive neighbor node indices."""
        nbrs = []
        for other in self.nodes:
            if other.idx == node.idx:
                continue
            if not other.alive and not other.is_sink:
                continue
            d = node.distance_to(other)
            if d <= self.cfg['comm_range']:
                nbrs.append((d, other.idx))
        nbrs.sort(key=lambda x: x[0])
        return [idx for _, idx in nbrs]

    def reset_packet(self, source_idx: int | None = None):
        """Starts routing a new packet from a random alive non-sink node."""
        alive_sources = [n for n in self.nodes if n.alive and not n.is_sink]
        if not alive_sources:
            return None, None, None  # All sensor nodes dead

        if source_idx is not None:
            self.current_node = self.nodes_by_id[source_idx]
        else:
            self.current_node = self.rng.choice(alive_sources)

        self.hop_count = 0
        nbrs = self._get_alive_neighbours(self.current_node)
        acts, mask = pad_actions(nbrs, self.cfg['MAX_A'])
        self.last_acts = acts
        self.last_mask = mask

        state = build_state(
            self.current_node, self.sink, acts, mask, self.nodes_by_id,
            self.uq, self.channel, self.cfg['init_energy'], self.area_diag,
            self.cfg['MAX_A'], temporal_model=self.temporal_model,
            enable_phase4=self.enable_phase4, max_queue_cap=self.max_queue_cap,
            obstacles=self.obstacles
        )
        return state, mask, acts

    def step(self, action_idx: int):
        """Advances packet routing by one hop according to chosen action index."""
        if self.last_acts is None or action_idx >= len(self.last_acts):
            return None, self.cfg['R_fail'], True, {'delivered': False, 'hops': self.hop_count}
        target_nid = self.last_acts[action_idx]
        if target_nid == -1 or target_nid not in self.nodes_by_id:
            return None, self.cfg['R_fail'], True, {'delivered': False, 'hops': self.hop_count}
        target_node = self.nodes_by_id[target_nid]

        # Phase 8: Byzantine Adversary Interception Check (Black-hole attack)
        if self.adversary_ids and target_nid in self.adversary_ids:
            if self.rng.uniform(0.0, 1.0) < self.adversary_drop_prob:
                self.hop_count += 1
                e_tx, e_rx = self.energy_model.transmit(self.current_node, target_node)
                reward = self.cfg['R_fail'] - self.cfg['w_delay'] * 1.0 - self.cfg['w_energy'] * (e_tx + e_rx)
                done = True
                info = {'delivered': False, 'hops': self.hop_count, 'reason': 'adversary_blackhole', 'intercepted': True}
                return None, reward, done, info

        # Phase 4: Buffer overflow check
        if self.enable_phase4 and not target_node.is_sink:
            if target_node.queue_len >= self.max_queue_cap:
                target_node.loss_history.append(1.0)
                reward = self.cfg['R_fail'] - self.cfg['w_delay'] * 1.0
                done = True
                info = {'delivered': False, 'hops': self.hop_count, 'reason': 'buffer_overflow'}
                return None, reward, done, info
            target_node.queue_len += 1

        # Energy deduction
        is_blocked = any(obs.intersects((self.current_node.x, self.current_node.y), (target_node.x, target_node.y)) for obs in self.obstacles) if self.obstacles else False
        e_tx, e_rx = self.energy_model.transmit(self.current_node, target_node)
        if is_blocked:
            e_tx *= 1.4  # Additional power spent during directional recovery
        self.hop_count += 1

        if self.enable_phase4:
            e_norm = target_node.energy / max(self.cfg['init_energy'], 1e-9)
            e_crit = float(self.cfg.get('E_crit', 0.30))
            if e_norm < e_crit and not target_node.is_sink:
                barrier_penalty = float(self.cfg.get('lambda_energy_exp', 3.0)) * np.exp(5.0 * (e_crit - e_norm))
            else:
                barrier_penalty = 0.0
            energy_penalty = self.cfg['w_energy'] * (e_tx + e_rx) + barrier_penalty
            q_ratio = (target_node.queue_len / max(self.max_queue_cap, 1)) if not target_node.is_sink else 0.0
            queue_penalty = float(self.cfg.get('gamma_queue', 2.0)) * (q_ratio ** 2)
            beta_t = self.cfg['w_uncert'] * (1.0 + max(0.0, 1.0 - e_norm))
        else:
            energy_penalty = self.cfg['w_energy'] * (e_tx + e_rx)
            queue_penalty = 0.0
            beta_t = self.cfg['w_uncert']

        delay_penalty = self.cfg['w_delay'] * 1.0

        if not self.current_node.alive or not target_node.alive:
            reward = self.cfg['R_fail'] - energy_penalty - delay_penalty - queue_penalty
            done = True
            info = {'delivered': False, 'hops': self.hop_count, 'reason': 'energy_death'}
            return None, reward, done, info

        if target_node.is_sink:
            reward = self.cfg['R_deliver'] - energy_penalty - delay_penalty - queue_penalty
            done = True
            info = {'delivered': True, 'hops': self.hop_count, 'reason': 'delivered'}
            return None, reward, done, info

        if self.hop_count >= self.cfg['MAX_HOPS']:
            reward = self.cfg['R_fail'] - energy_penalty - delay_penalty - queue_penalty
            done = True
            info = {'delivered': False, 'hops': self.hop_count, 'reason': 'max_hops'}
            return None, reward, done, info

        # Move to next hop
        self.current_node = target_node
        nbrs = self._get_alive_neighbours(self.current_node)
        if not nbrs:
            reward = self.cfg['R_fail'] - energy_penalty - delay_penalty - queue_penalty
            done = True
            info = {'delivered': False, 'hops': self.hop_count, 'reason': 'routing_hole'}
            return None, reward, done, info

        acts, mask = pad_actions(nbrs, self.cfg['MAX_A'])
        self.last_acts = acts
        self.last_mask = mask

        next_state = build_state(
            self.current_node, self.sink, acts, mask, self.nodes_by_id,
            self.uq, self.channel, self.cfg['init_energy'], self.area_diag,
            self.cfg['MAX_A'], temporal_model=self.temporal_model,
            enable_phase4=self.enable_phase4, max_queue_cap=self.max_queue_cap,
            obstacles=self.obstacles
        )
        if self.enable_phase4:
            q_std = next_state[3 + 7 * action_idx + 1] if action_idx < self.cfg['MAX_A'] else 0.0
        else:
            q_std = next_state[2 + 4 * action_idx + 1] if action_idx < self.cfg['MAX_A'] else 0.0

        reward = 0.0 - energy_penalty - delay_penalty - queue_penalty - beta_t * q_std
        done = False
        info = {'delivered': False, 'hops': self.hop_count}
        return (next_state, mask), reward, done, info

    def service_queues(self):
        """Processes queued packets at each sensor node."""
        svc = int(self.cfg.get('queue_service_rate', 4))
        for n in self.nodes:
            if not n.is_sink:
                n.queue_len = max(0, n.queue_len - svc)

    def apply_mobility(self):
        """Moves mobile nodes according to random waypoint physics."""
        self.mobility.step(self.nodes)
        if self.temporal_model is not None and hasattr(self.temporal_model, 'clear_cache'):
            self.temporal_model.clear_cache()
        if self.uq is not None and hasattr(self.uq, 'clear_cache'):
            self.uq.clear_cache()

    def network_stats(self) -> dict:
        alive_nodes = [n for n in self.nodes if n.alive and not n.is_sink]
        energies = [n.energy for n in alive_nodes]
        return {
            'alive_count': len(alive_nodes),
            'avg_energy': float(np.mean(energies)) if energies else 0.0,
            'min_energy': float(np.min(energies)) if energies else 0.0,
        }


def build_env(cfg: dict, uq=None, temporal_model=None, seed: int = 42,
              scenario: str | None = None, n_nodes: int | None = None, comm_range: float | None = None,
              adversary_ids: list[int] | None = None, adversary_drop_prob: float = 1.0):
    """Factory helper to build a full WSN topology and RoutingEnv with optional Phase 5 scenario preset and Phase 8 adversary support."""
    rng = np.random.default_rng(seed)
    
    # Load scenario override if specified
    sc_dict = {}
    if scenario and 'PHASE5_SCENARIOS' in cfg and scenario in cfg['PHASE5_SCENARIOS']:
        sc_dict = cfg['PHASE5_SCENARIOS'][scenario]

    N = int(n_nodes or sc_dict.get('n_nodes', cfg['n_nodes']))
    area = float(cfg['area'])
    cr = float(comm_range or sc_dict.get('comm_range', cfg['comm_range']))
    init_e = float(sc_dict.get('init_energy', cfg['init_energy']))
    mob_frac = float(sc_dict.get('mobile_fraction', cfg['mobile_fraction']))
    mob_speed = float(sc_dict.get('mobility_speed', cfg['mobility_speed']))
    obstacles = list(sc_dict.get('obstacles', cfg.get('obstacles', [])))

    # Sink placed in center (50, 50)
    sink = Node(0, area / 2.0, area / 2.0, energy=100.0, is_sink=True)
    nodes = [sink]

    max_q = int(cfg.get('max_queue_cap', 20))
    for i in range(1, N):
        x = rng.uniform(0.0, area)
        y = rng.uniform(0.0, area)
        nodes.append(Node(i, x, y, energy=init_e, max_queue=max_q))

    n_mobile = int(mob_frac * (N - 1))
    mobile_ids = set(rng.choice(range(1, N), size=n_mobile, replace=False))

    env_cfg = dict(cfg)
    env_cfg['n_nodes'] = N
    env_cfg['comm_range'] = cr
    env_cfg['init_energy'] = init_e
    env_cfg['mobile_fraction'] = mob_frac
    env_cfg['mobility_speed'] = mob_speed
    env_cfg['obstacles'] = obstacles
    env_cfg['adversary_ids'] = list(adversary_ids if adversary_ids is not None else cfg.get('adversary_ids', []))
    env_cfg['adversary_drop_prob'] = float(adversary_drop_prob if adversary_drop_prob is not None else cfg.get('adversary_drop_prob', 1.0))

    env = RoutingEnv(nodes, mobile_ids, env_cfg, uq=uq, temporal_model=temporal_model, seed=seed)
    return env, nodes
