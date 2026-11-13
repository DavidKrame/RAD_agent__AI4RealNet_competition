"""Evaluation harness: plays an agent on a range or list of scenarios and writes one record per scenario.

Each record holds the steps played, how the episode ended, the operating cost, the low-carbon term, the
attacks suffered, and the counters of each mechanism, so that failures can be characterized across many
scenarios.

Modes:
  fast  - steps survived and diagnostics; no reference statistics are needed.
  score - additionally computes the official operations and low-carbon scores, which requires the
          scorer's do-nothing and reference statistics for the scenarios used.

Sharding: pass a scenario range; each shard writes its own file, and files merge by concatenation.
Seeding is per scenario index, so a run is reproducible record by record.

Usage:
  python eval_full.py --submission <folder> --from 0 --to 764 --mode fast --out results/shard000.json
"""
import argparse, json, math, os, sys, time, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
AGENTS = os.path.join(os.path.dirname(HERE), "agent")   # the agent packages live in <repo>/agent/
sys.path.insert(0, AGENTS)
import numpy as np

# Mechanism counters, read from the agent when it defines them and recorded per scenario as their change
# over that scenario, so that a record shows which mechanisms acted. Agents that lack a counter leave it
# unrecorded.
COUNTERS = [
            "_ovfcycle_used", "_selfopen_used",
            "_isl_armed", "_isl_deep_used", "_isl_deep_fired", "_isl_ts0_fired",
            "_isl_escape_lines", "_isl_escape_splits", "_isl_escape_won", "_isl_fail",
            # Superimposed redispatching (post_val): armed / solved / offered / won / fail, separately.
            "_supr_armed", "_supr_solved", "_supr_offered", "_supr_won", "_supr_fail",
            "_dpfix_clash",
            "_isl_gla_yield", "_isl_cutall_offered", "_isl_vet_checked", "_isl_vet_blocked",
            # Registered with the mechanism, not after it: a counter absent from this list
            # is invisible in every run, and a pre-registered "read the firing count first" check is
            # then impossible.
            "_secfix_armed", "_secfix_kept", "_runway_armed",
            # These two live on the dispatch module, reached by the fallback in the reader below.
            "_tprice_applied", "_secn1_armed", "_secprep_solves",
            "_dispatch_capped", "_gla_clear", "_lodfcut_offered", "_lodfcut_screened", "_mg_cleared",
            "_cvx_computed", "_cvx_won", "_gla_fired", "_perarea_fired", "_perarea_multi",
            "_perarea_coexist_used", "_trip_fired", "_rb_fired", "_revert_done",
            "_n1_filter_blocked", "_sustain_fired",
            # Opponent starvation: how often a reconnection was withheld to keep an area's attacker
            # disarmed, and how often loading forced the hold to be given up. A mechanism that never
            # fires and one that fires and is immediately overridden look identical in the score.
            "_starve_held", "_starve_released", "_starve_maint_released",
            "_starve_foreseen", "_starve_fc_calls", "_starve_fc_fail", "_bigpool_restored",
            # relief added to the opening of the withheld lines
            "_fs_composed", "_fs_compose_declined", "_fs_compose_illegal", "_fs_compose_fail",
            "_compc_kept", "_compc_declined", "_compc_storage", "_compc_flex", "_compc_fail", "_cancel_fired",
            "_topoc_kept", "_topoc_declined", "_topoc_empty", "_topoc_fail",
            "_dpc_armed", "_dpc_kept", "_dpc_declined", "_dpc_topo", "_dpc_fail",
            "_svc_armed", "_svc_kept", "_svc_fail",
            "_corr_armed", "_corr_applied", "_corr_declined", "_corr_nosens",
            "_corr_mwsum", "_corr_fail",
            "_glac_armed", "_glac_composed", "_glac_declined", "_glac_illegal", "_glac_collide",
            "_glac_fail", "_glac_nocont",
            # Feasible-redispatch clamp. `_feas_clamped` is the only one that changes an action;
            # `_feas_zeroed` separates "shrunk" from "declined entirely", and `_feas_fail` keeps a raised
            # attempt distinguishable from a mechanism with nothing to do.
            "_feas_clamped", "_feas_zeroed", "_feas_fail",
            "_feas_shrunk", "_feas_blocked", "_unwind_band_fired",
            "_outage_fired", "_outage_skipped", "_outage_nofix", "_outage_pending",
            # A mechanism arm is only diagnosable if its counters are recorded. Two arms have
            # already been read without them: the superimposed-redispatch build left no record of
            # how often it fired or won, and the starvation posture none of how often it armed.
            "_fs_forced", "_joint_built", "_joint_won", "_joint_skipped",
            "_maint_released", "_maint_forced",
            # scheduled-outage look-ahead
            "_mg_probed", "_mg_skipped", "_mg_fired", "_mg_nofix", "_mg_table_used",
            "_solve_hits",
            "_pool_normal_played", "_pool_big_played",
            # Flexibility top-up. `_flex_fired` is the only one that changes an action; `_flex_declined`
            # separates "the guard refused it" from `_flex_nogain`'s "there was nothing to recover",
            # and those two have opposite implications for a null result -- one says the mechanism is
            # being suppressed, the other that the state it targets never arose.
            "_flex_fired", "_flex_declined", "_flex_nogain",
            "_hp_scored", "_hp_switched", "_hp_vetoed",
            "_latch_cleared",
            # Comfortable-band escalation on the published maintenance countdown. `_fe_armed` counts the
            # steps the trigger held and `_fe_escalated` the subset where the forward model predicted the
            # episode ending -- so `armed` high with `escalated` zero says the comfortable band genuinely
            # is safe on this set, and both zero says the trigger never fired at all.
            "_fe_armed", "_fe_escalated",
            "_og_screened", "_og_fatal", "_og_fired", "_og_nofix",
            "_po_truncated", "_po_endpoints",
            # Evidence-ranked candidate selection and the live-path injection. `_pool_rank_trunc`
            # counts the decisions where the per-area budget actually bound, which is the only place
            # ranking can differ from the positional prefix; `_pool_live_used` counts decisions where
            # deep pool entries were offered on the generated-candidate branch as well.
            "_pool_rank_trunc", "_pool_live_used",
            "_lodf_fired", "_lodf_maint_fired", "_topo_budget_hits", "_topo_scored",
            "_lodf_armed",
            "_busopt_solved", "_busopt_added", "_busopt_won",
            # N-1 security gate on the starvation hold: armings refused, allowed, and those the
            # linear model could not judge. `blocked` is the mechanism; `abstain` says how often the
            # gate was silent, which distinguishes "the gate did nothing" from "the gate was never
            # able to run".
            "_fs_n1_blocked", "_fs_n1_passed", "_fs_n1_abstain",
            "_merge_raced", "_merge_dc_won", "_merge_ac_won",
            # Cascade-aware selection: candidates whose afterstate was priced for propagating a
            # trip, and the total lines the screen expected to fall. Without these a null result
            # cannot be told apart from a term that never fired.
            "_casc_priced", "_casc_lines",
            # Separation guard: candidates priced for coming apart, and the two components of that
            # price. Without them a null cannot be told apart from a term that never fired.
            "_sepg_fired", "_sepg_split", "_sepg_bridge",
            "_ip_solved", "_ip_added", "_ip_declined", "_ip_won", "_ip_moved",
            "_anticip_opf_fired", "_dispatch_blocked", "_attack_counter",
            "_maint_armed", "_maint_fired",
            # v2: how often the hold was given up because the fleet's downward room was thin. A mechanism
            # that never fires and one that fires and is immediately overridden look identical in the score.
            "_starve_lowroom",
            # v3: holds given up because they had run long enough for the area's next scheduled attack to
            # have fallen due, after which further holding buys no immunity and only carries outage risk.
            "_starve_expired",
            # v4: swaps of the starving outage onto a cheaper attackable line, and those the forecast vetoed.
            "_starve_swap_fired", "_starve_swap_vetoed",
            # v5: proactive disarming of areas the adversary has not yet attacked.
            "_starve_pro_fired", "_starve_pro_vetoed",
            # Headroom invariant: how often the fleet's aggregate downward room fell below the floor,
            # and how often an unwind could actually be composed there.
            "_headroom_seen", "_headroom_fired",
            # How the AC-anchored proposal actually reaches the grid. It has two routes -- a direct
            # admission gate that compares peak loading against doing nothing, and the ranked relief pool --
            # and they answer different questions: `kept` counts the gate, `won` counts the pool. Both were
            # present in the agent and absent from this list, so no run had ever recorded which route
            # carries the behaviour, on the one path whose advantage is measurable.
            "_acopf_fired", "_acopf_kept", "_acopf_won",
            # Set by forks that rank candidates under a survival-indifference band; counts decisions the
            # band actually changed, so a null result separates "did not fire" from "fired and did nothing".
            "_eps_fired",
            # Announced-outage pre-positioning. `armed` and `abstained` are recorded separately because
            # they are opposite diagnoses of the same null result: the screen never fired, versus it fired
            # and the optimiser declined to commit. `blocked` counts positions the forecast refused.
            "_maint_armed", "_maint_abstained", "_maint_applied", "_maint_blocked",
            "_maint_released", "_maint_fired",
            "_gnn_tried", "_gnn_used", "_gnn_rejected", "_gnn_feas_blocked"]


def snapshot(agent):
    # Counters may live on the agent OR on the dispatch module it owns: anything inside the convex
    # program (the security block's arming count, the curtailment-price scale) increments on that object,
    # and reading only the agent reports those as a confident zero. The fallback closes that gap without
    # needing every mechanism to mirror its own counter up.
    sub = getattr(agent, "_acopf_obj", None)
    def _get(c):
        v = getattr(agent, c, None)
        if v is None and sub is not None:
            v = getattr(sub, c, None)
        return int(v or 0)
    return {c: _get(c) for c in COUNTERS}


def islanding_report(obs, env):
    """Whether the final state has a substation group carrying demand it cannot serve, which is the
    signature of the dominant failure mode: a trip splits off a pocket with no local generation."""
    try:
        status = np.asarray(obs.line_status).astype(bool)
        adj = {}
        for l in range(len(status)):
            if not status[l]:
                continue
            a, b = int(env.line_or_to_subid[l]), int(env.line_ex_to_subid[l])
            adj.setdefault(a, []).append(b); adj.setdefault(b, []).append(a)
        n = int(env.n_sub); seen = np.zeros(n, dtype=bool); comps = []
        for s in range(n):
            if seen[s]: continue
            stack, comp = [s], []
            seen[s] = True
            while stack:
                u = stack.pop(); comp.append(u)
                for v in adj.get(u, ()):
                    if not seen[v]: seen[v] = True; stack.append(v)
            comps.append(comp)
        if len(comps) <= 1:
            return {"split": False, "islands": 1, "stranded_load_mw": 0.0}
        worst = 0.0
        for comp in comps:
            cs = set(comp)
            lm = np.isin(env.load_to_subid, list(cs)); gm = np.isin(env.gen_to_subid, list(cs))
            if not lm.any(): continue
            demand = float(np.asarray(obs.load_p)[lm].sum())
            supply = float(np.asarray(obs.gen_p)[gm].sum()) if gm.any() else 0.0
            worst = max(worst, demand - supply)
        return {"split": True, "islands": len(comps), "stranded_load_mw": round(worst, 2)}
    except Exception:
        return {"split": None, "islands": None, "stranded_load_mw": None}


# --------------------------------------------------------------------------------------------------
# Low-carbon term.
#
# grid2op's renewable-usage score is a function of one ratio, the renewable energy delivered over the
# renewable energy available before curtailment, and both are attributes of the observation, so
# accumulating them per step reproduces the official term exactly. The operations term needs the
# do-nothing and reference runs and is not reconstructed here; a completion-anchored surrogate replaces
# it, used only to compare agents on the same scenarios.
OP_ANCHOR = 79.8  # mean operations score of a completed episode in the competition's returned results


def _surlinear(x, center=80.0):
    """grid2op's sur-linear curtailment transform, on f(v) = v*log(v) centred at `center`.

    The transform is asymmetric on purpose: above the centre it is normalised by the distance to a
    perfect 100, below it by the distance to 50, so that losing renewable output is penalised far more
    steeply than gaining it is rewarded."""
    x = max(float(x), 1e-6)
    f = lambda v: v * math.log(v)
    fc = lambda v: f(v) - f(center)
    return fc(x) / (fc(100.0) if x >= center else -fc(50.0))


def nres_from(num, den):
    """The official low-carbon score from accumulated (delivered, available) renewable energy.

    A scenario carrying no renewable generation at all yields a perfect score rather than a division by
    zero, matching the scorer, and the result is floored at -100 as the scorer's scaling does."""
    ratio = 100.0 if abs(den) <= 1e-6 else 100.0 * num / den
    return 100.0 * max(_surlinear(ratio), -1.0)


def episode_score(played, maxsteps, num, den):
    """The competition's per-episode score, with the operations term replaced by its surrogate.

    Survival dominates by construction: a full episode is worth 0.8*OP_ANCHOR = 63.8 points against a
    low-carbon term spanning 20, so the low-carbon channel breaks ties between candidates that survive
    equally rather than competing with survival."""
    op = OP_ANCHOR * min(played, maxsteps) / max(1, maxsteps)
    return 0.8 * op + 0.2 * nres_from(num, den)



def run(args):
    import grid2op, importlib
    try:
        from lightsim2grid import LightSimBackend as BK; backend = BK()
    except Exception:
        from grid2op.Backend import PandaPowerBackend; backend = PandaPowerBackend()
    # other_rewards={"cost": ...} makes the environment report the competition's operating cost at every
    # step, in `info`, without changing what the agent sees, so completed episodes can be compared on cost.
    from grid2op.Reward import L2RPNSandBoxScore
    env = grid2op.make(args.env, backend=backend, other_rewards={"cost": L2RPNSandBoxScore})

    _env_seeds = json.load(open(args.seeds))["env_seeds"] if args.seeds else None

    def _seed_for(ep):
        """The environment seed for episode `ep` -- the index itself unless a seed file was given."""
        return int(_env_seeds[ep]) if _env_seeds is not None else ep
    # The agent is constructed on a freshly made environment without a reset, as the competition's scorer
    # does: a reset before construction changes the state from which the later seeded reset starts.
    mod = importlib.import_module(args.submission)
    path = os.path.join(AGENTS, args.submission)
    agent = mod.make_agent(env, path)
    names = [os.path.basename(p) for p in env.chronics_handler.subpaths]
    # Which generators the low-carbon score counts. Fixed for the environment, so it is read once.
    ren_mask = np.asarray(env.gen_renewable, dtype=bool)
    hi = min(args.to, len(names))
    # An explicit list overrides the contiguous range; ids outside the environment are dropped rather than
    # raising, so a list assembled from another run cannot abort the job partway through.
    eps_list = None
    if getattr(args, "eps", None):
        raw = args.eps
        if raw.endswith(".json") and os.path.exists(raw):
            with open(raw) as fh:
                raw_ids = json.load(fh)
        else:
            raw_ids = [x for x in raw.replace(" ", "").split(",") if x]
        eps_list = sorted({int(x) for x in raw_ids if 0 <= int(x) < len(names)})
    meta = {"submission": args.submission, "env": args.env, "mode": args.mode,
            "range": ([args.from_, hi] if eps_list is None else {"explicit_eps": eps_list}),
            "n_chronics_available": len(names),
            "flags": {k: v for k, v in os.environ.items() if k.startswith("EXP_")},
            # A knob baked as a default leaves no trace in `flags`, which reads the environment only.
            # Two arms have already been scored inert for exactly that reason, so the pool the agent
            # actually resolved is recorded from the constructed agent rather than inferred.
            "pool": {"source": getattr(agent, "_pool_source", "unknown"),
                     "substations": int(getattr(agent, "_pool_subs", 0) or 0),
                     "entries": int(getattr(agent, "_pool_entries", 0) or 0)},
            "grid2op": __import__("grid2op").__version__,
            "optimiser_active": bool(getattr(agent, "_cvx", None) is not None)}
    def reference_run(ep):
        """How the scenario goes with no intervention at all.

        This is the single most informative extra measurement per chronic and it is nearly free, because
        an unmanaged grid usually fails early. It separates the two questions that matter and that an
        agent-only run confuses: is this scenario intrinsically hard, or is it our handling of it? A
        scenario the passive reference completes and ours does not is an own-goal, and one such case was
        worth more than a point of score when it was found by hand on a 70-scenario sample."""
        o = env.reset(seed=_seed_for(ep), options={"time serie id": ep})
        dn = env.action_space({})
        for t in range(int(env.max_episode_duration()) + 5):
            o, _, d, _ = env.step(dn)
            if d:
                return t + 1
        return t + 1

    rows, traces = [], {}
    for ep in (eps_list if eps_list is not None else range(args.from_, hi)):
        seed_of = _seed_for(ep)
        t0 = time.time()
        obs = env.reset(seed=seed_of, options={"time serie id": ep})
        try: agent.seed(0)
        except Exception: pass
        agent.reset(obs)
        before = snapshot(agent)
        maxsteps = int(env.max_episode_duration())
        rho_tail, played, done = [], 0, False
        nres_num, nres_den = 0.0, 0.0        # delivered / available renewable energy, this episode
        ep_cost = 0.0                        # the competition's operating cost, summed over played steps
        end_cause, end_detail, end_illegal, end_ambiguous = "completed", "", False, False
        attacks, first_overload = [], None
        # Attack onsets, counted per line: a line is counted when it is attacked at this step and was not
        # attacked at the previous step. `attacks` (above) records changes of the whole attacked set, which
        # counts an overlap of two attacks more than once ([A], [A,B], [B]) and misses a second attack on
        # the same line after a quiet gap; the onset list counts each attack once.
        attack_onsets, prev_attacked = [], set()
        tr_rho, tr_down, tr_act = [], [], []
        dn_act = env.action_space({})
        prev = obs  # the last state before termination
        for t in range(maxsteps + 5):
            act = agent.act(obs, 0.0, False)
            nxt, _, done, info = env.step(act)
            played = t + 1
            # Accumulated before the done branch, because the step that ends an episode still has an operating
            # cost, which the competition charges.
            try:
                c = (info.get("rewards") or {}).get("cost")
                if c is not None and np.isfinite(c):
                    ep_cost += float(c)
            except Exception:
                pass
            if done:
                # The termination cause, which the state alone does not reveal: an infeasible dispatch and a power flow
                # that fails to converge both end the episode, often on a grid with no overloaded line.
                try:
                    exc = info.get("exception") or []
                    end_cause = type(exc[0]).__name__ if exc else "none"
                    end_detail = str(exc[0])[:400] if exc else ""
                    if "Redispatch" in str(exc):
                        end_cause = "ImpossibleRedispatching"
                except Exception:
                    end_cause, end_detail = "unknown", ""
                end_illegal = bool(info.get("is_illegal", False))
                end_ambiguous = bool(info.get("is_ambiguous", False))
                break                 # on game over the returned observation is blanked; keep `prev`
            prev = obs = nxt
            # Accumulated only for steps that were actually played and did not end the episode,
            # which is the same set the official scorer sums over.
            try:
                nres_num += float(np.asarray(obs.gen_p)[ren_mask].sum())
                nres_den += float(np.asarray(obs.gen_p_before_curtail)[ren_mask].sum())
            except Exception:
                pass
            # the adversary's schedule, as it actually unfolded: what was attacked and when
            try:
                atk = info.get("opponent_attack_line", None)
                if atk is not None:
                    hit = [int(i) for i in np.where(np.asarray(atk).astype(bool))[0]]
                    if hit and (not attacks or attacks[-1]["lines"] != hit):
                        attacks.append({"step": played, "lines": hit})
                cur_attacked = set(int(i) for i in np.where(np.asarray(atk).astype(bool))[0]) \
                    if atk is not None else set()
                for line in sorted(cur_attacked - prev_attacked):
                    attack_onsets.append({"step": played, "line": line})
                prev_attacked = cur_attacked
            except Exception:
                pass
            rmax = float(np.nanmax(obs.rho))
            if first_overload is None and rmax > 1.0:
                first_overload = played
            if args.trace:
                # Half precision is ample for a loading value and keeps a full 764-scenario trace small
                # enough to move around and analyse anywhere, which is the point: the expensive machine
                # produces the breadth, and any single scenario can then be examined without re-running it.
                tr_rho.append(np.float16(rmax))
                tr_down.append(np.uint8(min(int((~np.asarray(obs.line_status)).sum()), 255)))
                tr_act.append(np.uint8(0 if act == dn_act else 1))
            rho_tail.append(round(rmax, 3))
            if len(rho_tail) > 12: rho_tail.pop(0)
        after = snapshot(agent)
        # Diagnose from the last live observation: the terminal one reports every line open and zero loading.
        diag = prev
        rho = np.asarray(diag.rho, dtype=float)
        st = np.asarray(diag.line_status).astype(bool)
        row = {
            "ep": ep, "chronic": names[ep], "steps": played, "max_steps": maxsteps,
            "completed": bool(played >= maxsteps - 1),
            "seconds": round(time.time() - t0, 1),
            # The competition's operating cost over the played steps, from which the operations score is
            # interpolated.
            "ep_cost": round(ep_cost, 4),
            # why it ended: the state at the final step, enough to classify the failure without a replay
            "final_rho_max": round(float(np.nanmax(rho)) if rho.size else -1.0, 3),
            "overloaded_lines": [int(l) for l in np.where((rho > 1.0) & st)[0]][:12],
            "lines_down": [int(l) for l in np.where(~st)[0]][:12],
            "n_lines_down": int((~st).sum()),
            "rho_max_last_12_steps": rho_tail,
            "islanding": islanding_report(diag, env),
            "end_cause": end_cause,
            "end_detail": end_detail,
            "end_illegal": end_illegal,
            "end_ambiguous": end_ambiguous,
            # fleet flexibility at the end: how many MW the generators could still shed that step.
            # A dispatch concentrated on slow units runs out of this and the episode ends even though
            # nothing is overloaded, so it is recorded alongside the loading picture.
            "down_ramp_room_mw": round(float(np.minimum(
                np.maximum(np.asarray(diag.gen_p)[env.gen_redispatchable] - env.gen_pmin[env.gen_redispatchable], 0.0),
                env.gen_max_ramp_down[env.gen_redispatchable]).sum()), 1),
            # was this scenario hard, or did we make it hard?
            "reference_steps": None,        # filled below when --reference is on
            "first_overload_step": first_overload,
            "grace_steps": (played - first_overload) if first_overload else None,
            "n_attacks": len(attacks),
            # The full schedule, not a prefix. Truncating this while still reporting n_attacks biased every
            # attack-conditioned statistic: the truncation fell on 101 of 764 episodes and 98 of them were
            # completions, so exposure was systematically undercounted among survivors and every
            # attack-conditioned odds ratio was inflated.
            "attacks": attacks,
            "n_attack_onsets": len(attack_onsets),
            "attack_onsets": attack_onsets,
            # The low-carbon term, reconstructed exactly, recorded per episode.
            "nres_score": round(nres_from(nres_num, nres_den), 6),
            "nres_ratio": round(100.0 * nres_num / nres_den, 4) if nres_den > 1e-6 else None,
            "episode_score": round(episode_score(played, maxsteps, nres_num, nres_den), 6),
            # which parts of the controller acted during this scenario
            "fired": {k: after[k] - before[k] for k in COUNTERS if after[k] - before[k] != 0},
        }
        if args.reference:
            row["reference_steps"] = reference_run(ep)
            row["verdict"] = ("both completed" if row["completed"] and row["reference_steps"] >= maxsteps - 1
                              else "we completed, passive failed" if row["completed"]
                              else "OWN GOAL: passive completed, we failed" if row["reference_steps"] >= maxsteps - 1
                              else "both failed")
        if args.trace:
            traces[str(ep)] = np.stack([np.asarray(tr_rho, dtype=np.float16).astype(np.float32),
                                        np.asarray(tr_down, dtype=np.float32),
                                        np.asarray(tr_act, dtype=np.float32)]).astype(np.float16)
        rows.append(row)
        print(f"  ep{ep} {names[ep]}: {played}/{maxsteps} "
              f"{'COMPLETED' if row['completed'] else 'ENDED'} rho={row['final_rho_max']} "
              f"down={row['n_lines_down']} island={row['islanding']['split']} "
              f"atk={row['n_attacks']} grace={row['grace_steps']} cause={row['end_cause']} "
              f"{row.get('verdict','')} "
              f"fired={row['fired']} {row['seconds']}s", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"meta": meta, "rows": rows}, open(args.out, "w"), indent=1)
    if args.trace and traces:
        tp = os.path.splitext(args.out)[0] + "_trace.npz"
        np.savez_compressed(tp, **traces)
        print(f"[eval_full] per-step traces -> {tp} ({os.path.getsize(tp)/1e6:.1f} MB)", flush=True)
    comp = sum(1 for r in rows if r["completed"])
    mean_score = sum(r["episode_score"] for r in rows) / max(1, len(rows))
    mean_nres = sum(r["nres_score"] for r in rows) / max(1, len(rows))
    print(f"[eval_full] {comp}/{len(rows)} completed  "
          f"mean_episode_score={mean_score:.4f}  mean_nres={mean_nres:.4f} -> {args.out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--env", default="l2rpn_idf_2023_train")
    ap.add_argument("--from", dest="from_", type=int, default=0)
    ap.add_argument("--to", type=int, default=10**9)
    # An explicit episode list, for targeting a diagnosed subset instead of a contiguous block. Seeding is
    # per-episode (seed=ep, agent.seed(0)), so a subset is scored exactly as it would be inside a full run
    # and the numbers are directly comparable with one. Accepts "3,17,204" or a path to a JSON list.
    ap.add_argument("--eps", default=None,
                    help="comma-separated episode ids, or a .json file holding a list of ids")
    ap.add_argument("--mode", choices=["fast", "score"], default="fast")
    ap.add_argument("--trace", action="store_true",
                    help="also save a per-step trace (peak loading, lines down, whether the agent acted)")
    ap.add_argument("--reference", action="store_true",
                    help="also run the scenario with no intervention, to separate hard scenarios from own goals")
    ap.add_argument("--out", required=True)
    # An environment that ships its own reference statistics also ships the seeds they were computed
    # under, and the opponent's attack draw and the maintenance schedule are both functions of the
    # environment seed. Passing them makes this harness describe the same run the anchored scorer
    # scores. Omitted, the seed remains the episode index, which is what every recorded result used.
    ap.add_argument("--seeds", default=None,
                    help="JSON file with an env_seeds list; seed episode i with env_seeds[i]")
    run(ap.parse_args())


