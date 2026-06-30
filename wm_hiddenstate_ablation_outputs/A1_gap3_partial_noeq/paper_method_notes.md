# Method Notes for 12-page LLM Paper

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
