"""
Shared building blocks for the PPO topology proposer used by ``MyAgent`` (hybrid
PPO + simulate-expert submission).

This module is deliberately self-contained (it does NOT depend on l2rpn_baselines):
it re-implements the small, high-value ideas from ``rl4pnc-survey-main`` on top of
stable-baselines3 + grid2op's gym_compat layer:

  * a *rule-based, reduced topology action space* (substation ``set_bus`` actions only,
    restricted to "controllable" substations)   -> ``build_reduced_actions``
  * a compact, normalized observation space                       -> ``make_obs_space``
  * a *heuristic gym environment* that mirrors rl4pnc's rho-gating
    ("do nothing when the grid is safe") and automatic line reconnection
    (equivalent to the competition's ``GymEnvWithRecoWithDN``)      -> ``HeuristicGymEnv``

These building blocks only ever produce topology (``set_bus``) actions. Redispatch,
storage and curtailment are *not* modeled here: in ``MyAgent`` they are handled
separately by a rule-based last resort (see ``my_agent.py``), so this module can stay
purely about the learned topology policy.
"""

from typing import Dict, List, Tuple

import numpy as np
from grid2op.Action import BaseAction
from grid2op.Observation import BaseObservation
from grid2op.Reward import BaseReward
from grid2op.gym_compat import BoxGymObsSpace, DiscreteActSpace, GymEnv


# ---------------------------------------------------------------------------
# Defaults (also stored in config.json so training and inference stay in sync)
# ---------------------------------------------------------------------------
DEFAULT_OBS_ATTR = [
    "rho",
    "line_status",
    "p_or",
    "topo_vect",
    "timestep_overflow",
    "gen_p",
    "load_p",
    "hour_of_day",
    "day_of_week",
]

# attributes we try to normalize (discrete-ish attrs are left as-is if it fails)
_NORMALIZE_ATTR = {"rho", "p_or", "gen_p", "load_p", "timestep_overflow"}


# ---------------------------------------------------------------------------
# Rule-based reduced topology action space
# ---------------------------------------------------------------------------
def build_reduced_actions(
    env,
    min_elems: int = 4,
    max_elems: int = 8,
    max_per_sub: int = 30,
    max_actions: int = 400,
) -> List[BaseAction]:
    """Generate a reduced list of unary substation ``set_bus`` (topology) actions.

    Mirrors rl4pnc's "changeable substations" idea: only substations that carry enough
    elements are worth splitting, and the combinatorics are bounded so the action space
    stays light for the 118-substation grid.

    * index 0 is always the *do-nothing* action
    * only substations whose number of elements is in ``[min_elems, max_elems]`` are used
    * at most ``max_per_sub`` topologies per substation and ``max_actions`` in total
    """
    actions: List[BaseAction] = [env.action_space({})]  # index 0 = do-nothing

    # controllable substations, larger ones first (they relieve congestion the most)
    sub_sizes = [(sub, int(env.sub_info[sub])) for sub in range(env.n_sub)]
    controllable = sorted(
        (sub for sub, n in sub_sizes if min_elems <= n <= max_elems),
        key=lambda s: int(env.sub_info[s]),
        reverse=True,
    )

    for sub in controllable:
        try:
            sub_acts = env.action_space.get_all_unitary_topologies_set(
                env.action_space, sub_id=sub
            )
        except Exception:
            continue
        # drop the "all on bus 1" reference topology (it is a no-op) and cap per sub
        kept = 0
        for act in sub_acts:
            if kept >= max_per_sub:
                break
            actions.append(act)
            kept += 1
        if len(actions) >= max_actions:
            break

    return actions[:max_actions]


def serialize_actions(actions: List[BaseAction]) -> List[dict]:
    """Serialize grid2op actions (skip the do-nothing at index 0, rebuilt implicitly)."""
    return [act.as_serializable_dict() for act in actions[1:]]


def deserialize_actions(env, action_dicts: List[dict]) -> List[BaseAction]:
    """Rebuild the action list (index 0 do-nothing + serialized topology actions)."""
    actions: List[BaseAction] = [env.action_space({})]
    for d in action_dicts:
        actions.append(env.action_space(d))
    return actions


def screen_actions_by_simulation(
    env,
    actions: List[BaseAction],
    n_states: int = 40,
    rho_low: float = 0.85,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[List[BaseAction], List[int]]:
    """Keep only topology actions that *ever help* on genuinely stressed grid states.

    Mirrors rl4pnc's rho-filtering idea. We collect a sample of near-overloaded states by
    running do-nothing, then for each candidate action we use grid2op's one-step lookahead
    (`obs.simulate`) and count how often it lowers the maximum rho below what do-nothing
    would give. Actions that never help are dropped; do-nothing (index 0) is always kept.
    The result is a much smaller, higher-quality menu for the policy to choose from.
    """
    do_nothing = actions[0]

    # 1. collect stressed states by stepping do-nothing across scenarios
    states: List[BaseObservation] = []
    env.seed(seed)
    obs = env.reset()
    guard = 0
    max_guard = max(n_states * 300, 3000)
    while len(states) < n_states and guard < max_guard:
        guard += 1
        obs, _, done, _ = env.step(env.action_space({}))
        if done:
            obs = env.reset()
            continue
        if obs.rho.max() >= rho_low:
            states.append(obs.copy())
    if verbose:
        print(f"  collected {len(states)} stressed states (rho >= {rho_low}).")

    # 2. count how often each action beats do-nothing (lower predicted max rho)
    help_count = [0] * len(actions)
    for st in states:
        try:
            sim_dn, _, done_dn, _ = st.simulate(do_nothing, time_step=1)
            base = np.inf if done_dn else float(sim_dn.rho.max())
        except Exception:
            base = np.inf
        for i in range(1, len(actions)):
            try:
                sim, _, done_i, _ = st.simulate(actions[i], time_step=1)
                val = np.inf if done_i else float(sim.rho.max())
            except Exception:
                val = np.inf
            if val < base - 1e-4:
                help_count[i] += 1

    # 3. keep do-nothing + helpful actions, most useful first
    ranked = sorted(range(1, len(actions)), key=lambda i: help_count[i], reverse=True)
    kept_idx = [0] + [i for i in ranked if help_count[i] > 0]
    kept_actions = [actions[i] for i in kept_idx]
    kept_counts = [help_count[i] for i in kept_idx]
    if verbose:
        print(f"  kept {len(kept_actions)} / {len(actions)} actions (incl. do-nothing).")
    return kept_actions, kept_counts


class SurvivalReward(BaseReward):
    """Survival- and safety-shaped reward used for *training* the topology policy.

    Per surviving step it returns ``clip(1 - max_rho, reward_min + eps, 1)`` — positive
    while the grid is comfortably loaded, shrinking toward zero as a line approaches its
    limit, and increasingly negative beyond the thermal limit — and a hard ``reward_min``
    on a failure or an illegal/ambiguous action. This points the policy directly at what
    op_score rewards (stay alive, keep flows away from the limit), unlike the environment's
    default cost-based reward. It does NOT affect the evaluation score, which is computed
    separately.

    v2: ``reward_min`` was widened from -1 to -2 and the floor is only ever hit by a
    terminal failure/illegal/ambiguous action. A non-terminal overload's reward is left
    unclipped down to ``reward_min + eps`` instead of saturating at -1 the moment
    ``max_rho`` passes ~2 (as the v1 ``clip(..., -1, 1)`` did). The v1 version gave every
    overload past that point an identical gradient-free -1, which is exactly the regime
    the agent is queried in most often (it only acts once gated past ``safe_max_rho``), so
    the policy had little signal to distinguish a mild overload from a severe one. This
    change only affects future PPO training runs -- ``saved_model/ppo_model.zip`` was
    trained under the v1 reward and should be retrained to benefit from it.
    """

    def __init__(self, logger=None):
        BaseReward.__init__(self, logger=logger)
        self.reward_min = -2.0
        self.reward_max = 1.0

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            return self.reward_min
        obs = env.get_obs()
        if obs is None:
            return 0.0
        return float(
            np.clip(1.0 - float(obs.rho.max()), self.reward_min + 0.01, self.reward_max)
        )


# ---------------------------------------------------------------------------
# Observation / action gym spaces
# ---------------------------------------------------------------------------
def make_obs_space(env, obs_attr: List[str]) -> BoxGymObsSpace:
    """Compact, normalized Box observation space (rl4pnc-style attribute selection)."""
    obs_space = BoxGymObsSpace(env.observation_space, attr_to_keep=list(obs_attr))
    for attr in obs_attr:
        if attr in _NORMALIZE_ATTR:
            try:
                obs_space.normalize_attr(attr)
            except Exception:
                pass
    return obs_space


def make_act_space(env, actions: List[BaseAction]) -> DiscreteActSpace:
    """Discrete action space over the reduced topology action list."""
    return DiscreteActSpace(env.action_space, action_list=actions)


# ---------------------------------------------------------------------------
# Heuristic gym environment (rho-gating + line reconnection)
# ---------------------------------------------------------------------------
class HeuristicGymEnv(GymEnv):
    """Gym environment that applies rl4pnc-style expert rules around the RL agent.

    Between two decisions of the neural network we automatically:
      1. reconnect every powerline that can be reconnected;
      2. do nothing while the grid is "safe" (``max(rho) <= safe_max_rho``).

    The PPO policy is therefore only queried on genuinely risky states, which both speeds
    up training and matches the behaviour of :class:`MyAgent` at inference time.
    Equivalent in spirit to the competition's ``GymEnvWithRecoWithDN``.
    """

    def __init__(
        self,
        env_init,
        obs_attr: List[str],
        actions: List[BaseAction],
        safe_max_rho: float = 0.9,
        reward_cumul: str = "init",
        **kwargs,
    ):
        super().__init__(env_init, **kwargs)
        self._safe_max_rho = safe_max_rho
        self._reward_cumul = reward_cumul
        self._previous_obs = None
        self.observation_space = make_obs_space(env_init, obs_attr)
        self.action_space = make_act_space(env_init, actions)

    # -- expert rules ------------------------------------------------------
    def heuristic_actions(self, g2op_obs: BaseObservation) -> List[BaseAction]:
        to_reco = (g2op_obs.time_before_cooldown_line == 0) & (~g2op_obs.line_status)
        if np.any(to_reco):
            return [
                self.init_env.action_space({"set_line_status": [(int(line_id), +1)]})
                for line_id in np.where(to_reco)[0]
            ]
        if g2op_obs.rho.max() <= self._safe_max_rho:
            return [self.init_env.action_space()]  # do nothing while safe
        return []  # hand control to the RL agent

    def _apply_heuristics(
        self, g2op_obs: BaseObservation, reward: float, done: bool, info: Dict
    ) -> Tuple[BaseObservation, float, bool, Dict]:
        res_reward = reward
        need_action = True
        while need_action and not done:
            need_action = False
            for g2op_act in self.heuristic_actions(g2op_obs):
                need_action = True
                g2op_obs, tmp_reward, done, info = self.init_env.step(g2op_act)
                if self._reward_cumul == "max":
                    res_reward = max(tmp_reward, res_reward)
                elif self._reward_cumul == "sum":
                    res_reward += tmp_reward
                elif self._reward_cumul == "last":
                    res_reward = tmp_reward
                if done:
                    break
        return g2op_obs, res_reward, done, info

    # -- gym API -----------------------------------------------------------
    def step(self, gym_action):
        g2op_act = self.action_space.from_gym(gym_action)
        g2op_obs, reward, done, info = self.init_env.step(g2op_act)
        if not done:
            g2op_obs, reward, done, info = self._apply_heuristics(
                g2op_obs, reward, done, info
            )
        self._previous_obs = g2op_obs
        gym_obs = self.observation_space.to_gym(g2op_obs)
        return gym_obs, float(reward), done, False, info

    def reset(self, *, seed=None, options=None):
        done = True
        info: Dict = {}
        gym_obs = None
        while done:
            super()._aux_reset_new(seed, options)
            g2op_obs = self.init_env.get_obs()
            reward = self.init_env.reward_range[0]
            g2op_obs, reward, done, info = self._apply_heuristics(
                g2op_obs, reward, False, info
            )
            if not done:
                self._previous_obs = g2op_obs
                gym_obs = self.observation_space.to_gym(g2op_obs)
                break
        return gym_obs, info
