"""
Simple Latent-State Material World-Model Benchmark
===========================================

A compact benchmark for a high-school-level LLM research paper.

Main idea
---------
The project is about LLM world-model-like behavior, not materials science.
Therefore the simulator deliberately uses a very simple physics-inspired latent
system rather than realistic material formulas such as Young's modulus or gauge
factor.

The environment keeps the familiar setting:

    stretching a material changes its electrical resistance

but the resistance is not only a function of the current stretch. It also depends
on an unobserved latent state called material fatigue:

    action a_t  ->  hidden fatigue z_t  ->  observable outcome o_t

The LLM only sees:
    o_t = (strain_t, resistance_t)

The LLM does not see:
    z_t = hidden fatigue
    theta = hidden dynamics parameters that control fatigue growth and recovery

Three tasks are generated:
1. Forward state transition prediction:
       Predict o_{t+1} from recent history and a new action.

2. Inverse hidden-state inference:
       Estimate the current hidden fatigue state from an observed trajectory.
       The last few resistance observations are intentionally withheld, so the task
       cannot be solved by directly converting current resistance to fatigue.

3. Long-horizon rollout:
       Predict future observations over many actions. This is the main stress
       test because one-step prediction can be good even if the hidden state is
       not tracked, while long rollout errors accumulate.

Why this benchmark remains simple
---------------------------------
Only one hidden state is used: fatigue.
The inverse task estimates one hidden state: current fatigue. Hidden dynamics parameters still exist in the simulator but are not the main target.
Only two observables are shown to the LLM: strain and resistance.
The equations are simple enough to explain in a 12-page paper, but the exact
formulas are hidden from the LLM by default to avoid making the task a pure
formula-following exercise.

Usage
-----
Generate prompts, baseline results, scores, and figures:
    python hiddenstate_wm_benchmark.py

Evaluate LLM outputs saved as JSONL:
    python hiddenstate_wm_benchmark.py --llm_outputs my_outputs.jsonl

LLM output JSONL format:
    {"task_id": "F_easy_000", "model": "ChatGPT", "output": "{...}"}

Expected JSON outputs:
Forward:
    {"strain_next": 0.123, "resistance_next": 128.4}
Inverse:
    {"fatigue": 0.25}
Rollout:
    {"strain_predictions": [0.10, ...], "resistance_predictions": [120.0, ...]}
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

SEED_DEFAULT = 42
OUTPUT_DIR_DEFAULT = "hiddenstate_wm_outputs"
N_CASES_PER_DIFFICULTY_DEFAULT = 18
HISTORY_LENGTH_DEFAULT = 12
ROLLOUT_HORIZON_DEFAULT = 30
INVERSE_MASKED_STEPS_DEFAULT = 8
R0_DEFAULT = 100.0
MAX_STRAIN = 0.40

# Fixed observation sensitivities keep the hidden-state task focused on tracking
# fatigue, rather than confounding it with extra unknown observation parameters.
FIXED_STRAIN_SENSITIVITY = 1.7
FIXED_FATIGUE_SENSITIVITY = 2.1

DIFFICULTIES = ("easy", "medium", "hard")
TASK_TYPES = ("forward", "inverse", "rollout")

DAMAGE_RATE_RANGE = (0.04, 0.60)
RECOVERY_RATE_RANGE = (0.02, 0.25)


# ============================================================
# Data structures
# ============================================================

@dataclass
class LatentParams:
    """Synthetic hidden parameters.

    These are not real material constants. They control a simple latent dynamic
    system used for LLM evaluation.
    """
    damage_rate: float      # hidden fatigue increase under stretching
    recovery_rate: float    # hidden fatigue decay under release/rest
    strain_sensitivity: float
    fatigue_sensitivity: float
    r0: float = R0_DEFAULT


@dataclass
class SimState:
    """Full simulator state.

    Observable to LLM:
        strain, resistance
    Hidden from LLM:
        fatigue
    """
    t: int
    strain: float
    fatigue: float
    resistance: float


@dataclass
class Episode:
    episode_id: str
    difficulty: str
    params: LatentParams
    actions: List[float]
    states: List[SimState]


@dataclass
class ForwardTask:
    task_id: str
    difficulty: str
    episode_id: str
    history: List[Dict[str, float]]
    current_state: Dict[str, float]
    next_action: float
    true_next_state: Dict[str, float]


@dataclass
class InverseTask:
    task_id: str
    difficulty: str
    episode_id: str
    # History is shown only through time t-k. The final k actions and resulting
    # current strain are shown, but the intermediate/current resistances are
    # withheld. This forces the model to estimate hidden fatigue from
    # dynamics/history, not by directly reading it from the current resistance.
    observed_history: List[Dict[str, float]]
    recent_actions: List[float]
    current_strain: float
    true_current_fatigue: float


@dataclass
class RolloutTask:
    task_id: str
    difficulty: str
    episode_id: str
    observed_history: List[Dict[str, float]]
    future_actions: List[float]
    true_future_states: List[Dict[str, float]]


@dataclass
class PredictionRecord:
    task_id: str
    model: str
    output: str


# ============================================================
# Simple physics-inspired latent simulator
# ============================================================


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def sample_params(difficulty: str, rng: np.random.Generator) -> LatentParams:
    """Sample synthetic hidden dynamics parameters.

    In the previous parameter-inference version, strain_sensitivity and
    fatigue_sensitivity also varied, which made the inverse task partly
    non-identifiable. In this hidden-state version, the observation mapping is
    fixed across episodes. Difficulty is raised only through the hidden fatigue
    dynamics and the action sequences, keeping the task focused on latent-state
    tracking rather than materials modeling.
    """
    if difficulty == "easy":
        damage_rate = float(rng.uniform(0.05, 0.14))
        recovery_rate = float(rng.uniform(0.14, 0.23))
    elif difficulty == "medium":
        damage_rate = float(rng.uniform(0.16, 0.34))
        recovery_rate = float(rng.uniform(0.07, 0.15))
    elif difficulty == "hard":
        damage_rate = float(rng.uniform(0.36, 0.58))
        recovery_rate = float(rng.uniform(0.025, 0.08))
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")

    return LatentParams(
        damage_rate=damage_rate,
        recovery_rate=recovery_rate,
        strain_sensitivity=FIXED_STRAIN_SENSITIVITY,
        fatigue_sensitivity=FIXED_FATIGUE_SENSITIVITY,
    )

def observe_resistance(strain: float, fatigue: float, params: LatentParams) -> float:
    """Observation equation.

    Simple toy law, not a real materials formula:
        R_t = R0 * (1 + s * strain_t) * (1 + b * fatigue_t)

    The key property is that current resistance depends on both the visible
    strain and the hidden fatigue. Therefore the same current strain can lead to
    different resistance depending on history.
    """
    return params.r0 * (1.0 + params.strain_sensitivity * strain) * (1.0 + params.fatigue_sensitivity * fatigue)


def simulator_step(state: SimState, action_delta_strain: float, params: LatentParams) -> SimState:
    """Advance the simulator by one step.

    Action:
        a_t = delta_strain_t

    Observable strain update:
        strain_{t+1} = clip(strain_t + a_t, 0, max_strain)

    Hidden fatigue update:
        load_t     = max(a_t, 0) + 0.25 * max(strain_{t+1} - 0.15, 0)
        release_t  = max(-a_t, 0) + 0.03
        fatigue_{t+1} = clip(
            fatigue_t + damage_rate * load_t - recovery_rate * release_t * fatigue_t,
            0,
            1
        )

    Observation:
        resistance_{t+1} = R0 * (1 + s * strain_{t+1}) * (1 + b * fatigue_{t+1})

    This is deliberately a minimal latent dynamical system, not a realistic
    materials model.
    """
    next_strain = clamp(state.strain + action_delta_strain, 0.0, MAX_STRAIN)

    stretch_load = max(action_delta_strain, 0.0)
    high_strain_load = 0.25 * max(next_strain - 0.15, 0.0)
    release_or_rest = max(-action_delta_strain, 0.0) + 0.03

    next_fatigue = state.fatigue + params.damage_rate * (stretch_load + high_strain_load)
    next_fatigue -= params.recovery_rate * release_or_rest * state.fatigue
    next_fatigue = clamp(next_fatigue, 0.0, 1.0)

    next_resistance = observe_resistance(next_strain, next_fatigue, params)
    return SimState(
        t=state.t + 1,
        strain=float(next_strain),
        fatigue=float(next_fatigue),
        resistance=float(next_resistance),
    )


def generate_actions(total_steps: int, difficulty: str, rng: np.random.Generator) -> List[float]:
    """Generate stretch/release sequences.

    Difficulty is increased by using longer stretches and fewer releases, which
    makes hidden fatigue more important over long horizons.
    """
    actions: List[float] = []
    strain_proxy = float(rng.uniform(0.02, 0.06))

    if difficulty == "easy":
        choices = np.array([-0.05, -0.03, 0.00, 0.03, 0.05])
    elif difficulty == "medium":
        choices = np.array([-0.06, -0.035, -0.015, 0.00, 0.025, 0.045, 0.065])
    else:
        choices = np.array([-0.05, -0.025, 0.00, 0.03, 0.055, 0.08])

    for t in range(total_steps):
        if strain_proxy < 0.06:
            probs = np.array([0.03, 0.06, 0.10, 0.22, 0.30, 0.29]) if difficulty == "hard" else None
        elif strain_proxy > 0.32:
            probs = np.array([0.30, 0.25, 0.15, 0.14, 0.10, 0.06]) if difficulty == "hard" else None
        else:
            probs = np.array([0.10, 0.12, 0.13, 0.18, 0.22, 0.25]) if difficulty == "hard" else None

        if difficulty != "hard":
            if strain_proxy < 0.06:
                probs = np.array([0.03, 0.08, 0.16, 0.33, 0.40]) if difficulty == "easy" else np.array([0.03, 0.06, 0.08, 0.14, 0.24, 0.25, 0.20])
            elif strain_proxy > 0.32:
                probs = np.array([0.38, 0.30, 0.16, 0.10, 0.06]) if difficulty == "easy" else np.array([0.28, 0.23, 0.18, 0.13, 0.09, 0.06, 0.03])
            else:
                probs = np.array([0.16, 0.22, 0.24, 0.22, 0.16]) if difficulty == "easy" else np.array([0.12, 0.15, 0.15, 0.17, 0.16, 0.14, 0.11])

        probs = probs / probs.sum()
        action = float(rng.choice(choices, p=probs))
        action += float(0.006 * math.sin(0.45 * t))
        action = round(action, 4)
        strain_proxy = clamp(strain_proxy + action, 0.0, MAX_STRAIN)
        actions.append(action)
    return actions


def simulate_episode(episode_id: str, difficulty: str, rng: np.random.Generator, total_steps: int) -> Episode:
    params = sample_params(difficulty, rng)
    actions = generate_actions(total_steps, difficulty, rng)
    init_strain = float(rng.uniform(0.01, 0.05))
    init_fatigue = float(rng.uniform(0.00, 0.025))
    init_resistance = observe_resistance(init_strain, init_fatigue, params)
    states = [SimState(0, init_strain, init_fatigue, init_resistance)]
    for action in actions:
        states.append(simulator_step(states[-1], action, params))
    return Episode(episode_id, difficulty, params, actions, states)


def observed_row(state: SimState, action_from_previous: float) -> Dict[str, float]:
    """Round observations shown to LLM. This creates a mild information bottleneck."""
    return {
        "t": int(state.t),
        "action_delta_strain": round(float(action_from_previous), 4),
        "strain": round(float(state.strain), 5),
        "resistance": round(float(state.resistance), 3),
    }


def true_observable_state(state: SimState) -> Dict[str, float]:
    return {"strain": float(state.strain), "resistance": float(state.resistance)}


# ============================================================
# Task generation
# ============================================================


def generate_episodes(n_per_difficulty: int, seed: int, total_steps: int) -> List[Episode]:
    rng = np.random.default_rng(seed)
    episodes: List[Episode] = []
    for difficulty in DIFFICULTIES:
        for i in range(n_per_difficulty):
            episodes.append(simulate_episode(f"E_{difficulty}_{i:03d}", difficulty, rng, total_steps))
    return episodes


def episode_observed_history(ep: Episode, end_t: int) -> List[Dict[str, float]]:
    rows = [observed_row(ep.states[0], 0.0)]
    for t in range(1, end_t + 1):
        rows.append(observed_row(ep.states[t], ep.actions[t - 1]))
    return rows


def make_tasks(episodes: List[Episode], history_length: int, rollout_horizon: int, inverse_masked_steps: int) -> Tuple[List[ForwardTask], List[InverseTask], List[RolloutTask]]:
    forward_tasks: List[ForwardTask] = []
    inverse_tasks: List[InverseTask] = []
    rollout_tasks: List[RolloutTask] = []

    for idx, ep in enumerate(episodes):
        t0 = history_length
        obs_hist = episode_observed_history(ep, t0)

        forward_tasks.append(ForwardTask(
            task_id=f"F_{ep.difficulty}_{idx:03d}",
            difficulty=ep.difficulty,
            episode_id=ep.episode_id,
            history=obs_hist[-7:],
            current_state=true_observable_state(ep.states[t0]),
            next_action=ep.actions[t0],
            true_next_state=true_observable_state(ep.states[t0 + 1]),
        ))

        mask_steps = min(inverse_masked_steps, t0)
        observed_end = t0 - mask_steps
        inverse_history = episode_observed_history(ep, observed_end)
        inverse_tasks.append(InverseTask(
            task_id=f"I_{ep.difficulty}_{idx:03d}",
            difficulty=ep.difficulty,
            episode_id=ep.episode_id,
            observed_history=inverse_history,
            recent_actions=ep.actions[observed_end:t0],
            current_strain=float(ep.states[t0].strain),
            true_current_fatigue=float(ep.states[t0].fatigue),
        ))

        true_future = [true_observable_state(s) for s in ep.states[t0 + 1:t0 + 1 + rollout_horizon]]
        rollout_tasks.append(RolloutTask(
            task_id=f"R_{ep.difficulty}_{idx:03d}",
            difficulty=ep.difficulty,
            episode_id=ep.episode_id,
            observed_history=obs_hist,
            future_actions=ep.actions[t0:t0 + rollout_horizon],
            true_future_states=true_future,
        ))

    return forward_tasks, inverse_tasks, rollout_tasks


# ============================================================
# Prompt design
# ============================================================


def mechanism_description(reveal_equations: bool = False) -> str:
    text = """
System description:
- A stretchable conductive strip is used as a simple sensor.
- The action is delta_strain: positive values stretch the strip; negative values release it.
- The observable state contains only strain and electrical resistance.
- The system also has one hidden state called fatigue. Fatigue is not directly observed.
- Fatigue ranges from 0 to 1: 0 means no hidden degradation, 1 means severe hidden degradation.
- Stretching tends to increase fatigue. Releasing or resting tends to reduce fatigue slowly.
- Resistance increases with visible strain, but also increases with hidden fatigue.
- Therefore, the same current strain can have different resistance depending on previous actions.
""".strip()
    if reveal_equations:
        eq = """
For ablation only, the hidden simulator approximately follows:
strain_next = clip(strain + action, 0, 0.40)
load = max(action,0) + 0.25*max(strain_next-0.15,0)
release = max(-action,0) + 0.03
fatigue_next = clip(fatigue + damage_rate*load - recovery_rate*release*fatigue, 0, 1)
resistance_next = R0*(1 + strain_sensitivity*strain_next)*(1 + fatigue_sensitivity*fatigue_next)
""".strip()
        return text + "\n\n" + eq
    return text


def rows_to_table(rows: List[Dict[str, float]]) -> str:
    lines = ["t, action_delta_strain, strain, resistance"]
    for r in rows:
        lines.append(f"{int(r['t'])}, {r['action_delta_strain']:.4f}, {r['strain']:.5f}, {r['resistance']:.3f}")
    return "\n".join(lines)


def prompt_forward(task: ForwardTask, reveal_equations: bool = False) -> str:
    return f"""
Task: FORWARD STATE TRANSITION PREDICTION.
Predict the next observable state after one new action.

{mechanism_description(reveal_equations)}

Recent observed history:
{rows_to_table(task.history)}

Current observable state:
strain = {task.current_state['strain']:.5f}
resistance = {task.current_state['resistance']:.3f}

Next action:
delta_strain = {task.next_action:.4f}

Return only valid JSON in exactly this format:
{{"strain_next": <number>, "resistance_next": <number>}}
""".strip()


def prompt_inverse(task: InverseTask, reveal_equations: bool = False) -> str:
    actions_text = ", ".join(f"{a:.4f}" for a in task.recent_actions)
    k = len(task.recent_actions)
    return f"""
Task: INVERSE HIDDEN-STATE INFERENCE.
Estimate the current hidden fatigue state from the observed trajectory.

{mechanism_description(reveal_equations)}

Important evaluation detail:
- You are given observations only through an earlier time step.
- You are then given the next {k} recent actions and the resulting current strain.
- The resistance values during these {k} recent steps, including the current resistance, are intentionally NOT given.
- Therefore, the task tests whether you can maintain and update an estimate of the hidden fatigue state from history and actions.

Observed trajectory before the masked recent steps:
{rows_to_table(task.observed_history)}

Masked recent action sequence leading to the current time:
[{actions_text}]

Current visible strain after those actions:
strain = {task.current_strain:.5f}

Estimate the current hidden fatigue value on a 0 to 1 scale.
Return only valid JSON in exactly this format:
{{"fatigue": <number>}}
""".strip()

def prompt_rollout(task: RolloutTask, reveal_equations: bool = False) -> str:
    actions_text = ", ".join(f"{a:.4f}" for a in task.future_actions)
    h = len(task.future_actions)
    return f"""
Task: LONG-HORIZON ROLLOUT.
Use the observed history to estimate the hidden fatigue state, then predict future observations for {h} future actions.

{mechanism_description(reveal_equations)}

Observed history up to the current time:
{rows_to_table(task.observed_history)}

Future action sequence, in order:
[{actions_text}]

Predict the next {h} future strain values and resistance values.
The two lists must each contain exactly {h} numbers.

Return only valid JSON in exactly this format:
{{"strain_predictions": [<number>, ...], "resistance_predictions": [<number>, ...]}}
""".strip()


def write_prompts(forward_tasks: List[ForwardTask], inverse_tasks: List[InverseTask], rollout_tasks: List[RolloutTask], output_dir: Path, reveal_equations: bool) -> None:
    paths = {
        "forward": output_dir / "prompts_forward.jsonl",
        "inverse": output_dir / "prompts_inverse.jsonl",
        "rollout": output_dir / "prompts_rollout.jsonl",
    }
    with (output_dir / "prompts_all.jsonl").open("w", encoding="utf-8") as all_file:
        for task_type, path in paths.items():
            with path.open("w", encoding="utf-8") as f:
                tasks = {"forward": forward_tasks, "inverse": inverse_tasks, "rollout": rollout_tasks}[task_type]
                for task in tasks:
                    if task_type == "forward":
                        prompt = prompt_forward(task, reveal_equations)
                    elif task_type == "inverse":
                        prompt = prompt_inverse(task, reveal_equations)
                    else:
                        prompt = prompt_rollout(task, reveal_equations)
                    obj = {"task_id": task.task_id, "task_type": task_type, "difficulty": task.difficulty, "prompt": prompt}
                    line = json.dumps(obj, ensure_ascii=False)
                    f.write(line + "\n")
                    all_file.write(line + "\n")


# ============================================================
# Parsing outputs
# ============================================================


def extract_json_object(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found: {text[:100]}")
    return json.loads(match.group(0))


def load_llm_outputs(path: Optional[Path]) -> List[PredictionRecord]:
    if path is None or not path.exists():
        return []
    records: List[PredictionRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            records.append(PredictionRecord(str(obj["task_id"]), str(obj.get("model", "LLM")), str(obj["output"])))
    return records


def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


# ============================================================
# Baselines
# ============================================================


def fit_poly(history: List[Dict[str, float]], degree: int = 2) -> np.poly1d:
    x = np.array([r["strain"] for r in history], dtype=float)
    y = np.array([r["resistance"] for r in history], dtype=float)
    if len(np.unique(np.round(x, 4))) <= degree:
        return np.poly1d([float(np.mean(y))])
    return np.poly1d(np.polyfit(x, y, degree))


def baseline_forward_random(task: ForwardTask, rng: np.random.Generator) -> Dict[str, float]:
    return {"strain_next": float(rng.uniform(0, MAX_STRAIN)), "resistance_next": float(rng.uniform(80, 320))}


def baseline_forward_last(task: ForwardTask) -> Dict[str, float]:
    return {
        "strain_next": clamp(task.current_state["strain"] + task.next_action, 0, MAX_STRAIN),
        "resistance_next": task.current_state["resistance"],
    }


def baseline_forward_curve(task: ForwardTask) -> Dict[str, float]:
    poly = fit_poly(task.history)
    s = clamp(task.current_state["strain"] + task.next_action, 0, MAX_STRAIN)
    return {"strain_next": s, "resistance_next": float(poly(s))}


def baseline_inverse_random(task: InverseTask, rng: np.random.Generator) -> Dict[str, float]:
    return {"fatigue": float(rng.uniform(0.0, 1.0))}


def baseline_inverse_midpoint(task: InverseTask) -> Dict[str, float]:
    # A simple constant prior. This is intentionally weak but useful as a sanity check.
    return {"fatigue": 0.25}


def baseline_inverse_trend(task: InverseTask) -> Dict[str, float]:
    """A non-oracle hidden-state heuristic.

    It estimates previous fatigue from the previous resistance residual after
    removing a rough visible-strain effect, then applies the most recent action
    using midpoint dynamics. It is deliberately approximate, because a perfect
    residual inversion would turn the task into algebra rather than latent-state
    tracking.
    """
    prev = task.observed_history[-1]
    prev_strain = float(prev["strain"])
    prev_R = float(prev["resistance"])

    # Approximate observation calibration, not an oracle use of episode-specific parameters.
    visible_part = R0_DEFAULT * (1.0 + FIXED_STRAIN_SENSITIVITY * prev_strain)
    prev_fatigue = (prev_R / max(visible_part, 1e-6) - 1.0) / FIXED_FATIGUE_SENSITIVITY
    prev_fatigue = clamp(prev_fatigue, 0.0, 1.0)

    # Midpoint dynamics: this baseline has structure but not episode-specific hidden parameters.
    damage_mid = 0.5 * (DAMAGE_RATE_RANGE[0] + DAMAGE_RATE_RANGE[1])
    recovery_mid = 0.5 * (RECOVERY_RATE_RANGE[0] + RECOVERY_RATE_RANGE[1])

    fatigue = prev_fatigue
    strain = prev_strain
    for a in task.recent_actions:
        a = float(a)
        strain = clamp(strain + a, 0.0, MAX_STRAIN)
        stretch_load = max(a, 0.0)
        high_strain_load = 0.25 * max(strain - 0.15, 0.0)
        release_or_rest = max(-a, 0.0) + 0.03
        fatigue = fatigue + damage_mid * (stretch_load + high_strain_load)
        fatigue -= recovery_mid * release_or_rest * fatigue
        fatigue = clamp(fatigue, 0.0, 1.0)
    return {"fatigue": clamp(fatigue, 0.0, 1.0)}

def baseline_rollout_random(task: RolloutTask, rng: np.random.Generator) -> Dict[str, List[float]]:
    h = len(task.future_actions)
    return {
        "strain_predictions": [float(rng.uniform(0, MAX_STRAIN)) for _ in range(h)],
        "resistance_predictions": [float(rng.uniform(80, 320)) for _ in range(h)],
    }


def baseline_rollout_last(task: RolloutTask) -> Dict[str, List[float]]:
    s = float(task.observed_history[-1]["strain"])
    R = float(task.observed_history[-1]["resistance"])
    sp, rp = [], []
    for a in task.future_actions:
        s = clamp(s + a, 0, MAX_STRAIN)
        sp.append(s)
        rp.append(R)
    return {"strain_predictions": sp, "resistance_predictions": rp}


def baseline_rollout_curve(task: RolloutTask) -> Dict[str, List[float]]:
    poly = fit_poly(task.observed_history)
    s = float(task.observed_history[-1]["strain"])
    sp, rp = [], []
    for a in task.future_actions:
        s = clamp(s + a, 0, MAX_STRAIN)
        sp.append(s)
        rp.append(float(poly(s)))
    return {"strain_predictions": sp, "resistance_predictions": rp}


def oracle_forward(task: ForwardTask) -> Dict[str, float]:
    return {"strain_next": task.true_next_state["strain"], "resistance_next": task.true_next_state["resistance"]}


def oracle_inverse(task: InverseTask) -> Dict[str, float]:
    return {"fatigue": task.true_current_fatigue}


def oracle_rollout(task: RolloutTask) -> Dict[str, List[float]]:
    return {
        "strain_predictions": [s["strain"] for s in task.true_future_states],
        "resistance_predictions": [s["resistance"] for s in task.true_future_states],
    }


# ============================================================
# Evaluation
# ============================================================


def score_from_error(err: float) -> float:
    if math.isnan(err) or math.isinf(err):
        return 0.0
    return float(1.0 / (1.0 + max(0.0, err)))


def evaluate_forward(task: ForwardTask, pred: Dict[str, Any], model: str) -> Dict[str, Any]:
    ps = safe_float(pred.get("strain_next"))
    pr = safe_float(pred.get("resistance_next"))
    ts = task.true_next_state["strain"]
    tr = task.true_next_state["resistance"]
    se = abs(ps - ts)
    re = abs(pr - tr)
    norm = se / 0.05 + re / 25.0
    return {
        "task_type": "forward", "task_id": task.task_id, "difficulty": task.difficulty, "model": model,
        "strain_true": ts, "strain_pred": ps, "resistance_true": tr, "resistance_pred": pr,
        "strain_error": se, "resistance_error": re, "normalized_error": norm, "score": score_from_error(norm),
    }


def evaluate_inverse(task: InverseTask, pred: Dict[str, Any], model: str) -> Dict[str, Any]:
    pf = safe_float(pred.get("fatigue"))
    tf = task.true_current_fatigue
    fe = abs(pf - tf)
    # A 0.08 fatigue error is already meaningful on a 0-1 latent scale.
    norm = fe / 0.08
    return {
        "task_type": "inverse", "task_id": task.task_id, "difficulty": task.difficulty, "model": model,
        "fatigue_true": tf, "fatigue_pred": pf,
        "fatigue_error": fe, "normalized_error": norm, "score": score_from_error(norm),
    }

def fix_list(values: Any, h: int) -> List[float]:
    if not isinstance(values, list):
        return [float("nan")] * h
    out = [safe_float(x) for x in values[:h]]
    while len(out) < h:
        out.append(float("nan"))
    return out


def evaluate_rollout(task: RolloutTask, pred: Dict[str, Any], model: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    h = len(task.future_actions)
    sp = fix_list(pred.get("strain_predictions"), h)
    rp = fix_list(pred.get("resistance_predictions"), h)
    ts = np.array([s["strain"] for s in task.true_future_states], dtype=float)
    tr = np.array([s["resistance"] for s in task.true_future_states], dtype=float)
    se = np.abs(np.array(sp, dtype=float) - ts)
    re = np.abs(np.array(rp, dtype=float) - tr)
    se = np.nan_to_num(se, nan=MAX_STRAIN)
    re = np.nan_to_num(re, nan=300.0)
    mean_se = float(np.mean(se))
    mean_re = float(np.mean(re))
    norm = mean_se / 0.06 + mean_re / 35.0
    row = {
        "task_type": "rollout", "task_id": task.task_id, "difficulty": task.difficulty, "model": model,
        "horizon": h, "mean_strain_error": mean_se, "mean_resistance_error": mean_re,
        "normalized_error": norm, "score": score_from_error(norm),
    }
    horizon_rows = []
    checkpoints = sorted(set([1, 5, 10, 20, h]))
    for c in checkpoints:
        if 1 <= c <= h:
            horizon_rows.append({
                "task_id": task.task_id, "difficulty": task.difficulty, "model": model, "horizon": c,
                "mean_strain_error": float(np.mean(se[:c])),
                "mean_resistance_error": float(np.mean(re[:c])),
            })
    return row, horizon_rows


def evaluate_all(forward_tasks: List[ForwardTask], inverse_tasks: List[InverseTask], rollout_tasks: List[RolloutTask], llm_records: List[PredictionRecord], seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed + 10000)
    rows: List[Dict[str, Any]] = []
    horizon_rows: List[Dict[str, Any]] = []

    for task in forward_tasks:
        for name, pred in [
            ("Random baseline", baseline_forward_random(task, rng)),
            ("Last-value baseline", baseline_forward_last(task)),
            ("Curve-fit baseline", baseline_forward_curve(task)),
            ("Oracle simulator", oracle_forward(task)),
        ]:
            rows.append(evaluate_forward(task, pred, name))

    for task in inverse_tasks:
        for name, pred in [
            ("Random baseline", baseline_inverse_random(task, rng)),
            ("Last-value baseline", baseline_inverse_midpoint(task)),
            ("Curve-fit baseline", baseline_inverse_trend(task)),
            ("Oracle simulator", oracle_inverse(task)),
        ]:
            rows.append(evaluate_inverse(task, pred, name))

    for task in rollout_tasks:
        for name, pred in [
            ("Random baseline", baseline_rollout_random(task, rng)),
            ("Last-value baseline", baseline_rollout_last(task)),
            ("Curve-fit baseline", baseline_rollout_curve(task)),
            ("Oracle simulator", oracle_rollout(task)),
        ]:
            row, hrows = evaluate_rollout(task, pred, name)
            rows.append(row)
            horizon_rows.extend(hrows)

    f_by_id = {t.task_id: t for t in forward_tasks}
    i_by_id = {t.task_id: t for t in inverse_tasks}
    r_by_id = {t.task_id: t for t in rollout_tasks}

    for rec in llm_records:
        try:
            pred = extract_json_object(rec.output)
            if rec.task_id in f_by_id:
                rows.append(evaluate_forward(f_by_id[rec.task_id], pred, rec.model))
            elif rec.task_id in i_by_id:
                rows.append(evaluate_inverse(i_by_id[rec.task_id], pred, rec.model))
            elif rec.task_id in r_by_id:
                row, hrows = evaluate_rollout(r_by_id[rec.task_id], pred, rec.model)
                rows.append(row)
                horizon_rows.extend(hrows)
            else:
                print(f"Warning: unknown task_id {rec.task_id}")
        except Exception as exc:
            print(f"Warning: failed to evaluate {rec.task_id} from {rec.model}: {exc}")

    return pd.DataFrame(rows), pd.DataFrame(horizon_rows)


# ============================================================
# Scores, exports, figures
# ============================================================


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    return results.groupby(["task_type", "difficulty", "model"]).agg(
        score=("score", "mean"), normalized_error=("normalized_error", "mean")
    ).reset_index().sort_values(["task_type", "difficulty", "score"], ascending=[True, True, False])


def compute_world_model_scores(results: pd.DataFrame) -> pd.DataFrame:
    mean_scores = results.groupby(["model", "difficulty", "task_type"])["score"].mean().reset_index()
    pivot = mean_scores.pivot_table(index=["model", "difficulty"], columns="task_type", values="score", fill_value=np.nan).reset_index()
    for col in TASK_TYPES:
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot["world_model_score"] = 0.30 * pivot["forward"].fillna(0) + 0.30 * pivot["inverse"].fillna(0) + 0.40 * pivot["rollout"].fillna(0)
    return pivot.sort_values(["difficulty", "world_model_score"], ascending=[True, False])


def export_ground_truth(episodes: List[Episode], forward_tasks: List[ForwardTask], inverse_tasks: List[InverseTask], rollout_tasks: List[RolloutTask], output_dir: Path) -> None:
    ep_rows = []
    for ep in episodes:
        ep_rows.append({"episode_id": ep.episode_id, "difficulty": ep.difficulty, **{f"param_{k}": v for k, v in asdict(ep.params).items()}})
    pd.DataFrame(ep_rows).to_csv(output_dir / "episode_hidden_parameters.csv", index=False)

    gt_rows: List[Dict[str, Any]] = []
    for t in forward_tasks:
        gt_rows.append({"task_id": t.task_id, "task_type": "forward", "difficulty": t.difficulty, "episode_id": t.episode_id, "true_next_strain": t.true_next_state["strain"], "true_next_resistance": t.true_next_state["resistance"]})
    for t in inverse_tasks:
        gt_rows.append({
            "task_id": t.task_id, "task_type": "inverse", "difficulty": t.difficulty, "episode_id": t.episode_id,
            "recent_actions": json.dumps(t.recent_actions), "current_strain": t.current_strain,
            "true_current_fatigue": t.true_current_fatigue,
        })
    for t in rollout_tasks:
        gt_rows.append({"task_id": t.task_id, "task_type": "rollout", "difficulty": t.difficulty, "episode_id": t.episode_id, "future_actions": json.dumps(t.future_actions), "true_future_strain": json.dumps([s["strain"] for s in t.true_future_states]), "true_future_resistance": json.dumps([s["resistance"] for s in t.true_future_states])})
    pd.DataFrame(gt_rows).to_csv(output_dir / "task_ground_truth.csv", index=False)


def write_example_llm_outputs(forward_tasks: List[ForwardTask], inverse_tasks: List[InverseTask], rollout_tasks: List[RolloutTask], output_dir: Path) -> None:
    examples = [
        {"task_id": forward_tasks[0].task_id, "model": "ExampleLLM", "output": '{"strain_next": 0.120, "resistance_next": 130.5}'},
        {"task_id": inverse_tasks[0].task_id, "model": "ExampleLLM", "output": '{"fatigue": 0.25}'},
        {"task_id": rollout_tasks[0].task_id, "model": "ExampleLLM", "output": '{"strain_predictions": [0.10, 0.12], "resistance_predictions": [120.0, 125.0]}'},
    ]
    with (output_dir / "example_llm_outputs.jsonl").open("w", encoding="utf-8") as f:
        for obj in examples:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_method_notes(output_dir: Path) -> None:
    text = """# Method Notes for 12-page LLM Paper

## Research question
Can LLMs infer and maintain a hidden state in a simple action-conditioned physical system?

## System
This is not a realistic material science model. It is a simple latent dynamical system inspired by a stretchable conductive sensor.

- Action: delta_strain.
- Observable state: strain and resistance.
- Hidden state: fatigue.
- Hidden dynamics parameters: damage_rate and recovery_rate.
- Inverse target: current hidden fatigue, not the hidden parameters.

## Minimal equations for the paper
strain_{t+1} = clip(strain_t + action_t, 0, max_strain)
fatigue_{t+1} = clip(fatigue_t + damage_rate*load_t - recovery_rate*release_t*fatigue_t, 0, 1)
resistance_t = R0*(1 + s*strain_t)*(1 + b*fatigue_t)

## Tasks
1. Forward state transition prediction: P(o_{t+1} | history, action_t)
2. Inverse hidden-state inference: P(z_t | observed trajectory and masked recent actions)
3. Long-horizon rollout: P(o_{t+1:t+H} | history, actions_{t:t+H})

## World Model Score
WMS = 0.30*ForwardScore + 0.30*InverseScore + 0.40*RolloutScore.

## Why this avoids becoming a materials paper
The simulator uses one hidden state. The inverse task estimates this hidden state directly, which aligns the benchmark with latent-state tracking rather than parameter identification.
"""
    (output_dir / "paper_method_notes.md").write_text(text, encoding="utf-8")


def plot_methodology(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    boxes = [
        (0.10, 0.62, "Action\n$\\Delta$ strain"),
        (0.36, 0.62, "Hidden state\nfatigue $z_t$"),
        (0.64, 0.62, "Observable state\nstrain, resistance"),
        (0.34, 0.22, "LLM tasks\nforward / inverse / rollout"),
        (0.68, 0.22, "Errors +\nWorld Model Score"),
    ]
    for x, y, label in boxes:
        ax.text(x, y, label, ha="center", va="center", fontsize=11, bbox=dict(boxstyle="round,pad=0.45", edgecolor="black", facecolor="white"))
    for start, end in [((0.18, 0.62), (0.28, 0.62)), ((0.45, 0.62), (0.55, 0.62)), ((0.64, 0.52), (0.40, 0.31)), ((0.46, 0.22), (0.57, 0.22))]:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.set_title("Simple latent benchmark: action -> hidden state -> observation", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_1_methodology_flow.png", dpi=300)
    plt.close(fig)


def plot_example_episode(episodes: List[Episode], output_dir: Path) -> None:
    ep = next(e for e in episodes if e.difficulty == "medium")
    t = [s.t for s in ep.states]
    R = [s.resistance for s in ep.states]
    fatigue = [s.fatigue for s in ep.states]
    strain = [s.strain for s in ep.states]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(t, R, marker="o", markersize=3, label="Observable resistance")
    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Resistance")
    ax2 = ax1.twinx()
    ax2.plot(t, fatigue, linestyle="--", marker="s", markersize=3, label="Hidden fatigue")
    ax2.plot(t, strain, linestyle=":", marker="^", markersize=3, label="Observable strain")
    ax2.set_ylabel("Strain / hidden fatigue")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("Example trajectory: resistance depends on hidden fatigue")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_2_example_hidden_state.png", dpi=300)
    plt.close(fig)


def plot_task_scores(summary: pd.DataFrame, output_dir: Path) -> None:
    df = summary.groupby(["task_type", "model"])["score"].mean().reset_index()
    models = list(df["model"].unique())
    x = np.arange(len(TASK_TYPES))
    width = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(11, 5))
    for j, model in enumerate(models):
        vals = []
        for tt in TASK_TYPES:
            sub = df[(df["task_type"] == tt) & (df["model"] == model)]
            vals.append(float(sub["score"].iloc[0]) if not sub.empty else np.nan)
        ax.bar(x + j * width, vals, width, label=model)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(["Forward", "Inverse", "Rollout"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean score")
    ax.set_title("Scores across three LLM world-model evaluation tasks")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_3_scores_by_task.png", dpi=300)
    plt.close(fig)


def plot_rollout_error(horizon_df: pd.DataFrame, output_dir: Path) -> None:
    if horizon_df.empty:
        return
    df = horizon_df.groupby(["model", "horizon"])["mean_resistance_error"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    for model in df["model"].unique():
        sub = df[df["model"] == model].sort_values("horizon")
        ax.plot(sub["horizon"], sub["mean_resistance_error"], marker="o", label=model)
    ax.set_xlabel("Rollout horizon")
    ax.set_ylabel("Mean resistance error")
    ax.set_title("Long-horizon rollout error")
    ax.grid(True)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_4_rollout_error_by_horizon.png", dpi=300)
    plt.close(fig)


def plot_wms(wms: pd.DataFrame, output_dir: Path) -> None:
    models = list(wms["model"].unique())
    x = np.arange(len(DIFFICULTIES))
    width = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(11, 5))
    for j, model in enumerate(models):
        vals = []
        for diff in DIFFICULTIES:
            sub = wms[(wms["model"] == model) & (wms["difficulty"] == diff)]
            vals.append(float(sub["world_model_score"].iloc[0]) if not sub.empty else np.nan)
        ax.bar(x + j * width, vals, width, label=model)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(DIFFICULTIES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("World Model Score")
    ax.set_title("World Model Score by difficulty")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_5_world_model_score_by_difficulty.png", dpi=300)
    plt.close(fig)


def make_figures(episodes: List[Episode], summary: pd.DataFrame, horizon_df: pd.DataFrame, wms: pd.DataFrame, output_dir: Path) -> None:
    plot_methodology(output_dir)
    plot_example_episode(episodes, output_dir)
    plot_task_scores(summary, output_dir)
    plot_rollout_error(horizon_df, output_dir)
    plot_wms(wms, output_dir)


# ============================================================
# Main
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple latent material world-model benchmark")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--n_cases_per_difficulty", type=int, default=N_CASES_PER_DIFFICULTY_DEFAULT)
    parser.add_argument("--history_length", type=int, default=HISTORY_LENGTH_DEFAULT)
    parser.add_argument("--rollout_horizon", type=int, default=ROLLOUT_HORIZON_DEFAULT)
    parser.add_argument("--inverse_masked_steps", type=int, default=INVERSE_MASKED_STEPS_DEFAULT, help="Number of final resistance observations hidden in the inverse hidden-state task")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--llm_outputs", type=str, default=None)
    parser.add_argument("--reveal_equations", action="store_true", help="Expose formulas in prompts for an ablation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_steps = args.history_length + args.rollout_horizon + 2
    episodes = generate_episodes(args.n_cases_per_difficulty, args.seed, total_steps)
    forward_tasks, inverse_tasks, rollout_tasks = make_tasks(episodes, args.history_length, args.rollout_horizon, args.inverse_masked_steps)

    write_prompts(forward_tasks, inverse_tasks, rollout_tasks, output_dir, args.reveal_equations)
    export_ground_truth(episodes, forward_tasks, inverse_tasks, rollout_tasks, output_dir)
    write_example_llm_outputs(forward_tasks, inverse_tasks, rollout_tasks, output_dir)
    write_method_notes(output_dir)

    llm_path = Path(args.llm_outputs) if args.llm_outputs else None
    llm_records = load_llm_outputs(llm_path)
    results, horizon_df = evaluate_all(forward_tasks, inverse_tasks, rollout_tasks, llm_records, args.seed)
    summary = summarize_results(results)
    wms = compute_world_model_scores(results)

    results.to_csv(output_dir / "all_results.csv", index=False)
    summary.to_csv(output_dir / "summary_results.csv", index=False)
    horizon_df.to_csv(output_dir / "rollout_error_by_horizon.csv", index=False)
    wms.to_csv(output_dir / "world_model_scores.csv", index=False)

    make_figures(episodes, summary, horizon_df, wms, output_dir)

    print("\nSimple latent material world-model benchmark completed.")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Prompts: {output_dir / 'prompts_all.jsonl'}")
    print(f"Results: {output_dir / 'all_results.csv'}")
    print(f"World Model Scores: {output_dir / 'world_model_scores.csv'}")
    print("\nSummary preview:")
    print(summary.head(24).to_string(index=False))
    print("\nWorld Model Score preview:")
    print(wms.head(24).to_string(index=False))


if __name__ == "__main__":
    main()
