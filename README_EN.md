# Myelination-style Developmental Learning System

[English](./README_EN.md) | [中文](./README.md)

An AGI-oriented **developmental learning architecture**: structure is not a prerequisite but a *product* of learning. Seed neurons unfold dimensions under environment-driven signals, wire by co-firing, myelinate, differentiate and split. All learning signals come from immediate environment feedback — no backpropagation, no labels, no pretrained embeddings.

> **In one sentence**: treat developmental neurobiology mechanisms (myelination, use-it-or-lose-it selection, sleep replay, observational learning) as *engineering constraints* for building a "light-interior, heavy-coupling" continually-learning system.

---

## Design Doctrines

1. **Reaction over prediction** — the system learns *actions that control feedback loops*, not sequence statistics. Autoregressive prediction is merely a distillation vehicle; adjudication always belongs to the world's response to action.
2. **Logic before statistics** — symbolic action readout uses *confirmation voting*, not magnitude-weighted sums: each pathway votes for a candidate token, weighted by the world's per-transition confirmation record. Magnitude never decides.
3. **Bare encoding** — all modalities enter as one-hot / raw physical quantities; no distance structure is imposed. Cross-modal binding must be grown by the system itself. "Nested pre-encoding" (injecting pretrained similarity structure) is forbidden.
4. **Anti-divergence** — development contains multiple positive feedback loops (myelin thickening, attention modulation, three-factor learning-rate modulation); their biological failure mode is epilepsy. An energy brake and tension governance are *existential* guarantees, not performance components.
5. **The ratchet** — developmental state persists across runs (`persistence.py`): accumulate on prior results and never slide back. Any "learning succeeded" claim must first pass trivial baselines (persistence / majority class).

## Architecture Overview

```
Environment(deposits/sandbox) ⇄ sensory channels vis/aud/... ⇄ Ports (bare encoding + normalizer)
        ⇅
┌──────────────────────────────────────────────────────────┐
│ Reptilian brain (frozen reflexes = limb action surfaces)  │  ← pluggable: babble/echo/LLM/sandbox
│   ReflexArc pre-orchestrated routes (baseline at λ=0)     │
├──────────────────────────────────────────────────────────┤
│ Developmental substrate: seed neuron → unfold → co-firing │
│   wiring → myelination                                    │
│   Myelin = second operator (delay × gain)                 │
│   Selection = use/disuse (contribution > ρ·(1-prot))      │
│   T2 cross-step event queue / T3 eligibility / T8 tension │
├──────────────────────────────────────────────────────────┤
│ Higher brain: autoregressive residual (delta rule) +      │
│   three-factor RPE (drive satisfaction)                   │
│   Shadow mode → alternating trials → reversible handover  │
│   λ (context-conditioned)                                 │
│   Confirmation-vote readout / attention / spikes (decoupled extensions)
├──────────────────────────────────────────────────────────┤
│ Sleep: NREM homeostatic scaling → generative replay →     │
│   REM counterfactual → forward preplay                    │
│   Attribution ledger: per-phase, per-sheath gain deltas   │
└──────────────────────────────────────────────────────────┘
```

Core files: [`mcp/developmental/myelin.py`](mcp/developmental/myelin.py) (sheaths/dispatch/co-firing), [`system.py`](mcp/developmental/system.py) (main loop), [`neuron.py`](mcp/developmental/neuron.py), [`selection.py`](mcp/developmental/selection.py), [`attention.py`](mcp/developmental/attention.py), [`spikes.py`](mcp/developmental/spikes.py), [`geometry.py`](mcp/developmental/geometry.py) (cross-modal geometry analytics), [`persistence.py`](mcp/developmental/persistence.py) (brain serialization).

## Quick Start

```bash
git clone <repo>
cd devo_project
pip install numpy            # only hard dependency
pip install torch            # optional: real PyTorch (CPU/GPU); numpy_torch_shim otherwise

# Grammar acquisition (autoregressive completion over an FSM token stream, 16 tokens, 3 seeds)
python baby_grammar.py

# Unified-vocabulary continual curriculum: grammar → image → movie → grammar revisited
python unified_curriculum.py

# Deposit reading: any file (byte-level typewriter, 256 bare keys)
python deposit_reader.py baby_grammar.py

# Multi-limb infant: two action surfaces + coordination-demand world
python baby_multilimb.py
```

## Tests

Every mechanism has a TDD red→green verification script (`docs/devo-project/verify/`):

```bash
python docs/devo-project/verify/test_batched_dispatch.py   # batched scatter bitwise equivalence
python docs/devo-project/verify/test_event_queue.py        # cross-step event arrival timing
python docs/devo-project/verify/test_eprop.py              # eligibility traces
python docs/devo-project/verify/test_context_competence.py # context-conditioned competence
python docs/devo-project/verify/test_tension.py            # threshold-tension drift governance
python docs/devo-project/verify/test_sandbox_bridge.py     # world bridge HTTP endpoints
python docs/devo-project/verify/test_persistence.py        # brain serialization / ratchet
python docs/devo-project/verify/quant_audit.py             # quantification audit (trivial baselines)
```

## Main Experimental Results

### Grammar acquisition (FSM token stream, 16 tokens, 10000 steps, no backprop)

| Configuration | top1 (next-token accuracy) |
|---|---|
| Uniform random | 0.062 |
| Echo baseline | 0.071 |
| Bigram statistical learner (reference) | 0.556–0.611 |
| **This system (w_norm=1.1)** | **0.243 ± 0.055** |
| **This system (w_norm=3.0)** | **0.383 ± 0.017** |

Ground-truth probe: 9/16 FSM transitions recovered (all deterministic ones hit). Ratchet retention 1.02 (grammar skill survives studying image and movie deposits).

### Ablations (6000 steps, 3 seeds)

| Configuration | top1 | Notes |
|---|---|---|
| baseline | 0.291±0.156 | Closed loop works after judge alignment (λ≈0.93, handover 93%) |
| +attention | 0.294±0.134 | Neutral: keys learned, readout dominated by confirmation votes |
| +spike | 0.108–0.118 | Negative: refractory cuts vote count; a safety mechanism, not a performance one |

### Honest Limitations

- **Quantification audit lesson**: self-made media tasks were beaten by trivial baselines (movie persistence=0.924 vs system 0.44) — any "learning succeeded" claim must pass persistence/majority checks first. A predictive-coding image deposit has been introduced (task signal exists: first-order conditional optimum 0.423); system extraction is still climbing.
- **Credit-assignment depth**: single-layer W + local rules; hierarchical composition unsolved (eligibility traces pave the way).
- **Skill decay**: in the multi-limb experiment, learned skills were lost; four interventions ruled out arbitration/reward coupling; suspects narrowed to drift without counter-selection / normalizer nonstationarity / vote-W interaction (threshold tension is the first response).
- This repository is a research prototype, not a product.

## Repository Structure

```
mcp/developmental/     architecture core (myelin/system/neuron/selection/higher_brain/
                       attention/spikes/geometry/persistence/reflex/...)
baby_grammar.py        grammar acquisition experiment (includes the off-by-one judge lesson)
baby_loop.py           infant vocal-control loop (reaction vs prediction contrast)
baby_multilimb.py      multi-limb infant (dual action surfaces + coordination demands)
deposit_reader.py      deposit reader (byte-level typewriter)
deposit_media.py       image/movie deposit adapters (predictive-coding encoding)
unified_curriculum.py  unified-vocabulary continual curriculum
devo_control.py        embodied control task (unstable system + use/disuse)
mcp/developmental/builtin/sandbox_bridge.py  world bridge (HTTP JSON, Three.js-compatible)
docs/devo-project/     macdev artifacts: specs/plans/logs (dual-track)
docs/devo-project/verify/  all verification & diagnostic scripts (8 TDD contract tests +
                       quant_audit + archived experiment scripts)
paper/preprint.Rmd     arXiv preprint draft (Chinese)
```

## Roadmap

Done: batched vectorization, cross-step event queue, e-prop eligibility, multi-limb, world bridge skeleton, quantification audit, threshold tension, brain serialization, RWKV-7-style adaptive decay.

In progress: Moving MNIST benchmark, Crafter integration, context-conditioned competence validation on media, three-suspect investigation of skill decay (drift rollback landed), Transformer teacher organ (make-or-buy fusion).

## References

**Direct precedents and theoretical support**

- Åström & Wittenmark (1973). On Self-Tuning Regulators. *Automatica* 9(2) — nearest precedent for signal-driven online learning
- Brooks (1986). Subsumption Architecture — direct precedent for baseline-layer separation; this system adds the learning and memory it lacks
- Conant & Ashby (1970). Every good regulator of a system must be a model of that system — basis for lazy dimension unfolding
- Oja (1982) — mathematical form of weight-layer use/disuse
- Tononi & Cirelli (2019). Sleep and synaptic down-selection — source of sleep homeostatic scaling
- Asokan, Chhabria & Chakravarthy (2015). Adaptive myelination of temporal delays (CNS*2015 poster) — only direct precedent for myelin as a trainable quantity
- Lefebvre et al. (2025). Myelin-induced gain control in nonlinear neural networks. *Communications Physics* — theory for the myelin gain axis

**Learning-mechanism comparisons**

- Fritzke (1995). Growing Neural Gas — correspondence for the edge-age mechanism
- Rao & Ballard (1999). Predictive coding in the visual cortex — the academic formalization of our autoregressive-residual channel
- Barlow (1961). Efficient coding hypothesis — common root of redundancy reduction and use/disuse dedup
- Hosoya et al. (2005). Dynamic predictive coding by the retina — the canonical adaptive-statistics encoder
- Hinton (2022). Forward-Forward — modern reference for local learning without backprop
- Ding et al. Information-saturation structural developmental NN (CAAI TIT) — comparison for split timing and parent-child relations
- Baxter & Levy (2019). Adaptive synaptogenesis network — comparison for autophagy/pruning

**Continual-learning and sequence-modeling paradigms**

- Peng et al. (2025). RWKV-7 "Goose" (arXiv:2503.14456) — per-channel in-context learning rates, adopted as adaptive decay
- Sun et al. Learning to (Learn at Test Time) — state as test-time-trainable memory
- Mamba-CL (arXiv:2411.15469) — selective gating as the forgetting controller

**Benchmarks**

- Crafter, MineDojo/MineStudio (embodied agents)
- Moving MNIST (video prediction), CIFAR-10 (images, label-verdict protocol)

The full 66-source literature survey with per-source verification notes is in [`docs/research/发育智能系统_相似研究调研.md`](docs/research/发育智能系统_相似研究调研.md) (Chinese).

## License

MIT
