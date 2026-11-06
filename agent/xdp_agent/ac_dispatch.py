"""AC-anchored per-generator dispatch program.

The OptimCVXPY baseline optimizes a lossless DC model: both the sensitivity of flows to injections and
the base flows come from that model. Grid2Op computes the observation with an AC power flow, so the
observed flows are exact. This program takes the base flows from the observation and uses distribution
factors only for the change that the controls cause.

OptimCVXPY also decides per bus and then splits each decision across the generators of that bus, so
individual ramp limits are respected only approximately. Here:

  * base flows come from the observation, and their changes from distribution factors;
  * each generator has its own decision variable, bounded by its own ramp margins, which also keeps the
    fleet's total downward movement within its downward room;
  * curtailment and storage are decided in the same program as redispatch;
  * the solution is converted to one action, which the agent plays only if a forecast confirms a lower
    peak loading.
"""
from typing import Optional, List
import os
import numpy as np


class ACAnchoredDispatch:
    """Redispatch / curtailment / storage chosen on an AC-anchored linear model of line flows."""

    def __init__(self, env, max_iter: int = 2, overload_weight: float = 1000.0,
                 redisp_cost: float = 0.03, curtail_cost: float = 50.0, storage_cost: float = 0.1,
                 margin: float = 0.95, down_room_margin: float = 0.0, solver: str = "OSQP"):
        self.n_line = int(env.n_line)
        self.n_gen = int(env.n_gen)
        self.gen_to_subid = np.asarray(env.gen_to_subid, dtype=int)
        self.load_to_subid = np.asarray(env.load_to_subid, dtype=int)
        self.gen_redispatchable = np.asarray(env.gen_redispatchable, dtype=bool)
        self.gen_renewable = np.asarray(env.gen_renewable, dtype=bool)
        self.gen_pmax = np.asarray(env.gen_pmax, dtype=float)
        self.n_sub = int(env.n_sub)
        self.n_storage = int(getattr(env, "n_storage", 0) or 0)
        self.storage_to_subid = (np.asarray(env.storage_to_subid, dtype=int)
                                 if self.n_storage else np.zeros(0, dtype=int))
        self.max_iter = int(max_iter)
        self.w_ovl = float(overload_weight)
        self.c_redisp = float(redisp_cost)
        self.c_curtail = float(curtail_cost)
        self.c_storage = float(storage_cost)
        # Actuators of the program (EXP_ACT3): 0 = redispatch only; 1 = redispatch, curtailment and storage
        # (default); 2 = redispatch and storage; 3 = redispatch and curtailment. Deciding them in one program
        # prices their different effects on every line at once, which separate candidates cannot do.
        _a3 = int(os.environ.get("EXP_ACT3", "1"))
        self._act3 = _a3 > 0
        self._act3_curt = _a3 in (1, 3)
        self._act3_stor = _a3 in (1, 2)

        # --- Security against scheduled outages (EXP_SECN1) ---
        # Withholding already keeps the opponent's attempts from landing, so the outages the agent cannot prevent
        # are the scheduled ones, published through `time_next_maintenance`. For a line whose outage starts within
        # the lead window, the post-outage flows of the lines it would overload are kept within their ratings
        # through a slack (at most EXP_SECN1_MAX pairs). The slack is linear and priced below the curtailment cost,
        # so relaxing a security limit is always cheaper than curtailing to meet it, while redispatch and storage
        # remain cheaper still.
        self._secn1 = int(os.environ.get("EXP_SECN1", "1"))
        self._secn1_lead = int(os.environ.get("EXP_SECN1_LEAD", "48"))     # steps of warning to act on
        self._secn1_margin = float(os.environ.get("EXP_SECN1_MARGIN", "1.0"))
        self._secn1_w = float(os.environ.get("EXP_SECN1_W", "5.0"))  # must stay below curtail_cost
        self._secn1_max = int(os.environ.get("EXP_SECN1_MAX", "40"))
        self._secn1_armed = 0      # solves with a pending outage in the window
        self._secn1_cons = 0       # monitored lines constrained across those solves
        self.margin = float(margin)
        self.down_room_margin = float(down_room_margin)
        self.solver = solver
        self._prev = None                      # warm start: the previous solution
        self.stats = {"solved": 0, "infeasible": 0, "iters": 0, "improved": 0}

    # ------------------------------------------------------------------
    def _ptdf(self, env):
        """Distribution factors for the current topology of ``env``, rebuilt at every call (about 0.3 ms) so
        that they follow topology changes of that environment.
        """
        try:
            return np.asarray(env.backend._grid.get_ptdf(), dtype=float)
        except Exception:
            return None

    def solve(self, obs, env, targets: Optional[List[int]] = None):
        """Return a grid2op action, or None when nothing feasible and useful was found."""
        try:
            import cvxpy as cp
        except Exception:
            return None
        P = self._ptdf(env)
        if P is None or P.shape[0] != self.n_line:
            return None

        rho = np.asarray(obs.rho, dtype=float)
        lim = np.asarray(obs.thermal_limit, dtype=float)
        base = np.asarray(obs.p_or, dtype=float)  # AC flows from the observation
        status = np.asarray(obs.line_status, dtype=bool)
        # a flow limit expressed in MW at the observed operating point; rho is |S|/limit, so scaling the
        # observed active flow by rho recovers the MW value the limit corresponds to
        with np.errstate(divide="ignore", invalid="ignore"):
            mw_lim = np.where(rho > 1e-6, np.abs(base) / np.maximum(rho, 1e-6), np.abs(base) + 1.0)
        mw_lim = np.where(np.isfinite(mw_lim) & (mw_lim > 1e-3), mw_lim, np.abs(base) + 1.0)

        if targets is None:
            targets = [int(i) for i in np.where(status & (rho > self.margin))[0]]
        if not targets:
            return None

        # ---- decision variables, one per device ----
        redisp_ids = [g for g in range(self.n_gen) if self.gen_redispatchable[g]]
        if not redisp_ids:
            return None
        up = np.asarray(obs.gen_margin_up, dtype=float)[redisp_ids]
        dn = np.asarray(obs.gen_margin_down, dtype=float)[redisp_ids]
        dp = cp.Variable(len(redisp_ids))

        # Curtailment gives up renewable output at a renewable generator; storage power is positive
        # when charging, which is a load. Both enter the injection map with a negative sign below.
        curt_ids = [g for g in range(self.n_gen) if self.gen_renewable[g]] if self._act3_curt else []
        p_ren = np.asarray(obs.gen_p, dtype=float)[curt_ids] if curt_ids else np.zeros(0)
        curt = cp.Variable(len(curt_ids), nonneg=True) if curt_ids else None
        ns = self.n_storage if self._act3_stor else 0
        stor = cp.Variable(ns) if ns else None

        cons = [dp <= up, dp >= -dn]                      # per-generator ramp limits, exactly
        if curt is not None:
            cons.append(curt <= np.maximum(p_ren, 0.0))   # cannot give up more than is produced
        if stor is not None:
            try:
                smax = np.asarray(obs.storage_max_p_prod, dtype=float)    # discharge limit (MW)
                smin = np.asarray(obs.storage_max_p_absorb, dtype=float)  # charge limit (MW)
            except Exception:
                smax = smin = np.full(ns, 10.0)
            cons += [stor <= smin, stor >= -smax]
        # The environment rejects a dispatch whose required downward movement exceeds what the fleet can
        # deliver; asking for more is not a small error but an immediate game over. Constrain it directly.
        room = float(np.sum(dn))
        cons.append(cp.sum(cp.neg(dp)) <= max(0.0, room - self.down_room_margin))
        if not self._act3:
            cons.append(cp.sum(dp) == 0)                  # redispatch is balanced by construction

        # ---- map device changes to bus injections, then to flows ----
        nb = P.shape[1]
        A = np.zeros((nb, len(redisp_ids)))
        for k, g in enumerate(redisp_ids):
            b = int(self.gen_to_subid[g])
            if 0 <= b < nb:
                A[b, k] = 1.0
        inj = A @ dp
        if curt is not None:
            Ac = np.zeros((nb, len(curt_ids)))
            for k, g in enumerate(curt_ids):
                b = int(self.gen_to_subid[g])
                if 0 <= b < nb:
                    Ac[b, k] = -1.0  # curtailment removes injection at that bus
            inj = inj + Ac @ curt
        if stor is not None:
            As = np.zeros((nb, ns))
            for k in range(ns):
                b = int(self.storage_to_subid[k])
                if 0 <= b < nb:
                    As[b, k] = -1.0                       # charging is a load: negative injection
            inj = inj + As @ stor
        if self._act3:
            # Curtailment removes generation and charging consumes it, so both must be matched by
            # the dispatchable fleet. Writing the balance across the three together is exactly what
            # lets the optimiser trade one against another.
            cons.append(cp.sum(inj) == 0)
        dflow = P @ inj

        # ---- objective: relieve the targeted overloads, spend as little as possible doing it ----
        ovl = cp.Variable(len(targets), nonneg=True)
        obj = self.w_ovl * cp.sum_squares(ovl) + self.c_redisp * cp.sum_squares(dp)
        if curt is not None:
            obj = obj + self.c_curtail * cp.sum(curt)     # curtailed MWh is scored against us directly
        if stor is not None:
            obj = obj + self.c_storage * cp.sum_squares(stor)
        for j, l in enumerate(targets):
            cap = self.margin * mw_lim[l]
            cons += [base[l] + dflow[l] <= cap + ovl[j],
                     -(base[l] + dflow[l]) <= cap + ovl[j]]

        # ---- security against the pending maintenance outage ----
        if self._secn1:
            try:
                tnm = np.asarray(getattr(obs, "time_next_maintenance",
                                         np.full(self.n_line, -1)), dtype=float)
                L = np.asarray(env.backend._grid.get_lodf(), dtype=float)
                # Only lines with a published outage inside the warning window, still in service, and
                # whose removal the linear model can describe at all.
                cset = [k for k in range(self.n_line)
                        if status[k] and 0.0 < tnm[k] <= self._secn1_lead
                        and np.isfinite(L[:, k]).all()]
                pairs = []
                for k in cset:
                    post = np.abs(base + L[:, k] * base[k]) / np.maximum(mw_lim, 1e-6)
                    post[k] = 0.0
                    post[~status] = 0.0
                    for m in np.where(post > self._secn1_margin)[0]:
                        pairs.append((float(post[m]), int(m), k))
                pairs.sort(reverse=True)
                pairs = pairs[: max(0, self._secn1_max)]
                if pairs:
                    sl = cp.Variable(len(pairs), nonneg=True)
                    # linear and priced below curtail_cost: relaxing the limit is always cheaper than curtailing to meet
                    # it, so security is never bought with renewable output
                    obj = obj + self._secn1_w * cp.sum(sl)
                    for j, (_r, m, k) in enumerate(pairs):
                        fpost = (base[m] + dflow[m]) + float(L[m, k]) * (base[k] + dflow[k])
                        cap = self._secn1_margin * mw_lim[m]
                        cons += [fpost <= cap + sl[j], -fpost <= cap + sl[j]]
                    self._secn1_armed += 1
                    self._secn1_cons += len(pairs)
            except Exception:
                pass  # if the security block fails, solve without it

        prob = cp.Problem(cp.Minimize(obj), cons)
        try:
            if self._prev is not None and len(self._prev) == len(redisp_ids):
                dp.value = self._prev                     # warm start; upstream disables this outright
            prob.solve(solver=self.solver, warm_start=True)
        except Exception:
            try:
                prob.solve(warm_start=True)
            except Exception:
                self.stats["infeasible"] += 1
                return None
        if dp.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
            self.stats["infeasible"] += 1
            return None

        self._prev = np.asarray(dp.value, dtype=float)
        self.stats["solved"] += 1
        vals = np.asarray(dp.value, dtype=float)
        entries = [(int(g), float(vals[k])) for k, g in enumerate(redisp_ids) if abs(vals[k]) > 1e-3]
        if self._act3:
            act_dict = {}
            if entries:
                act_dict["redispatch"] = entries
            if curt is not None and curt.value is not None:
                cv = np.asarray(curt.value, dtype=float)
                # grid2op takes curtailment as a ratio of the generator's maximum output, so the solved megawatts
                # are converted here
                ce = [(int(g), float(max(0.0, min(1.0, (p_ren[k] - cv[k]) / float(self.gen_pmax[g])))))
                      for k, g in enumerate(curt_ids) if cv[k] > 1e-3 and self.gen_pmax[g] > 1e-6]
                if ce:
                    act_dict["curtail"] = ce
            if stor is not None and stor.value is not None:
                sv = np.asarray(stor.value, dtype=float)
                se = [(int(k), float(sv[k])) for k in range(ns) if abs(sv[k]) > 1e-3]
                if se:
                    act_dict["set_storage"] = se
            if not act_dict:
                return None
            try:
                a = env.action_space(act_dict)
                # The feasibility clamp the other continuous levers already use: an over-large
                # curtailment or storage setpoint is an immediate game over, not a small error.
                try:
                    a.limit_curtail_storage(obs)
                except Exception:
                    pass
                return a
            except Exception:
                # Fall back to redispatch alone rather than losing the solve entirely.
                if entries:
                    try:
                        return env.action_space({"redispatch": entries})
                    except Exception:
                        return None
                return None
        if not entries:
            return None
        try:
            return env.action_space({"redispatch": entries})
        except Exception:
            return None
