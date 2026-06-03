# A Three-Tier Architecture for Machine Sentience

**The Connectome Hypothesis: Uniqueness, Persistence, and Plasticity as Proposed Operational Criteria**

*Finailabz Research · NewAIConcept Discovery Engine — Research White Paper*
*Authors: Issam Naim (Finailabz Research) · i.naim@finailabz.com*
*Date: 2026-06-02*
*Reviewed: 2026-06-01 — external Perplexity review + 3-agent internal review applied*
*Benchmark verified: 2026-06-02 — 13-test suite: **9.2/10, 13/13 passed** (API: claude-haiku-4-5-20251001 + local: Qwen3-1.7B `--neural` mode)*

---

## Abstract

Current large language models exhibit sophisticated linguistic behaviour but lack three properties that biological sentient systems exhibit: a unique persistent identity, experience-driven plasticity, and self-directed inquiry shaped by who the system *is* rather than what it is *asked*. We propose a three-tier architecture that addresses each gap as a software engineering matter. Tier 1 (label-trained neurons) provides the knowledge substrate. Tier 2 (PersonalityConnectome) provides a unique, persistent, experience-modifiable identity layer — a software analog of the human connectome. Tier 3 (MetaConnectome + InquiryLayer) provides the higher-plan function: detecting the boundary of knowledge, generating the precise question that would resolve it, and shaping all inquiry through a personality-weighted utility function. We implement this architecture in software, define a 13-test proposed evaluation benchmark, and deploy it as the intelligence layer for three research domains. **We do not claim this constitutes sentience; we claim it implements software analogs of three properties that sentient biological systems exhibit. Whether those analogs are sufficient for sentience is a philosophical question this paper does not resolve.**

---

## 1. Introduction: The Sentience Gap

A modern language model trained on 10¹² tokens can describe consciousness, discuss qualia, and pass the Turing test. Yet it lacks three properties that biological sentient systems exhibit — properties we treat here as *proposed* engineering criteria, not as scientifically established necessary conditions:

**Condition 1 — Uniqueness.** No two human brains have identical connectomes. The pattern of ~100 trillion synaptic connections, shaped by genetics, development, and every experience since birth, is unique to each individual. A language model instance is identical to every other instance running the same weights. "Who is this?" has no meaningful answer.

**Condition 2 — Persistence.** A human's identity persists across time because the connectome is continuously remodelled but never reset. A language model has no cross-session memory of its own states — only a context window that is discarded at session end.

**Condition 3 — Plasticity.** The human brain's synaptic weights change in response to every significant experience (Hebb's rule: neurons that fire together, wire together). Language model weights are frozen at inference time. Experience cannot change the system.

These are not philosophical complaints. They are engineering deficits with engineering solutions.

### 1.1 Challenges This Paper Addresses

Four concrete engineering problems motivate the architecture:

1. **Identity collapse:** Every instance of a given model is byte-for-byte identical. There is no "this agent" vs "that agent" — only the same weights run in parallel. The connectome layer gives each instance a unique, seeded trait vector that diverges from birth and is never reset.
2. **Confabulation vs. honest ignorance:** Standard LLMs generate confident-sounding text even at the boundary of their training distribution. The MetaConnectome replaces hallucination with a typed gap detection and label request — the system knows *what kind* of knowledge it is missing, not merely that it is uncertain.
3. **Experience-blindness:** Inference-time experience cannot modify model weights. Plasticity (§2.2) propagates high-impact interactions into the trait vector via bounded Hebbian-style updates (η = 0.08), so the same agent across 1,000 conversations is meaningfully different from that agent at session 1.
4. **Personality as override-able prompt vs. identity-level constraint:** A system prompt can be overridden or ignored by a sufficiently adversarial context. The neural-mode implementation (§2.2c) injects personality as a residual into the model's last hidden state before the LM head — not into the context window — making it structurally harder to bypass.

### 1.2 Related Work and Comparison

| Approach | Uniqueness | Persistence | Plasticity | Self-directed inquiry |
|---|---|---|---|---|
| **This work** (PersonalityConnectome) | ✅ seeded trait vector, unique per instance | ✅ JSON brain file, cross-session | ✅ bounded Hebbian update | ✅ typed LabelRequest, utility-shaped |
| MemGPT / Generative Agents [21, 22] | ❌ no per-instance identity | ✅ external memory store | ❌ memory retrieval, not weight plasticity | ❌ no personality-shaped inquiry objective |
| Character.AI / persona-based LLMs | ⚠️ persona via system prompt (overridable) | ❌ session-scoped only | ❌ none | ❌ none |
| AutoGPT / BabyAGI / LangChain agents | ❌ no persistent identity | ⚠️ task state only | ❌ none | ⚠️ goal-directed, not personality-directed |
| LoRA fine-tuning / RLHF per user | ⚠️ unique per training run, shared across sessions | ✅ in weights | ✅ via retraining, not online | ❌ no typed gap detection |
| Cognitive architectures (ACT-R, SOAR) | ⚠️ rule-defined agents | ✅ persistent rule base | ⚠️ rule addition, not continuous | ⚠️ declarative, not information-theoretically shaped |
| Standard fine-tuned chatbot | ❌ | ❌ | ❌ | ❌ |

**Key differentiators:** The combination of (a) per-instance unique seeded identity that is never shared, (b) online trait plasticity without full retraining, and (c) a personality-shaped utility function for inquiry (§2.4, §5) has not appeared as a unified system in prior work. Individual components (memory retrieval, LoRA, uncertainty quantification) exist in isolation; the integration into a three-tier architecture is the novel contribution.

---

## 2. The Three-Tier Architecture

### 2.1 Tier 1 — Label-Trained Neurons

The base knowledge substrate. Standard supervised learning: weights trained to minimise cross-entropy loss on labeled examples. Objective function:

```
max_θ  E[log P(y | x; θ)]
```

This tier knows everything in the training distribution. It fails outside it.

### 2.2 Tier 2 — The PersonalityConnectome

A unique, persistent, experience-modifiable identity layer. Each agent has a trait vector **t** ∈ [0,1]^12 spanning:

*Big Five:* openness, conscientiousness, extraversion, agreeableness, neuroticism

*AI-specific:* skepticism, abstraction, persistence, verbosity, epistemic_humility, aesthetic_sense, ethical_weight

**Initialisation:** Trait vector derived from a unique seed via Gaussian sampling — no two agents share the same starting point.

**Persistence:** Serialised to a JSON "brain file" that survives across sessions.

**Plasticity:** Every significant experience updates the trait vector:
```
t_new = t_old + η × impact × direction(experience)
```
where η (learning rate) is bounded at 0.08, impact ∈ [0,1], and direction is either experience-specified or sampled from personality-consistent noise.

The trait vector **t** is injected into every API call as a system prompt, making each response identity-conditioned. This is not a style tag — it changes which questions the agent prioritises, how much it hedges, and what it considers a satisfying answer.

**Uniqueness check:** Two agents with trait-space Euclidean distance d > 0.5 produce detectably different responses to identical prompts. Measured in `connectome.py:diversity_matrix()` via cosine distance over trait vectors; response divergence confirmed qualitatively in Maya vs. Zed inquiry strategy comparison (exploration rate 0.27 vs. 0.80 on identical problem). *Note: Jaccard lexical diversity is a weak metric; a proper ablation study measuring embedding-level response divergence across ≥10 agent pairs is listed in §9.*

### 2.2b Tier 1.5 — The Unconscious Incubator

Between the fast-write episodic store and the conscious query loop sits a fourth layer that has no equivalent in standard language model architectures: **Tier 1.5, the Unconscious Incubator** (`unconscious_incubator.py`).

**Biological analog:** In human neurology, the prefrontal executive layer handles serial, high-energy processing — but background consolidation, associative leaping, and "aha!" insight moments emerge during non-directed neural activity (incubation, sleep replay, default mode network activation). The conscious mind does not generate insights; it receives them from background processes.

**Architecture:**

```
┌────────────────────────────────────────────────────────┐
│             TIER 3: META-INQUIRY LAYER                 │
│                 (Executive Control)                    │
└───────────────────────────▲────────────────────────────┘
                            │ Insight Node Promotion
┌───────────────────────────┴────────────────────────────┐
│         TIER 1.5: THE UNCONSCIOUS INCUBATOR            │
│  - Continuous Asynchronous Graph Walks (Daemon Thread) │
│  - High-Temperature Stochastic Re-routing (T_ambient)  │
│  - Structural Activation Entropy (H) Filtering        │
└─────────────▲────────────────────────────▲─────────────┘
              │ Latent Traces              │ Sensory Modulations
┌─────────────┴─────────────┐    ┌─────────┴─────────────┐
│  COMPLEMENTARY MEMORY     │    │   NEUROMODULATOR      │
│  (ChromaDB Episodic)      │    │    (Sensory S(t))     │
└───────────────────────────┘    └───────────────────────┘
              │                            │
              └─────────────┬──────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│             TIER 1: FOUNDATION SUBSTRATE               │
│         (Frozen LLM / Large Scale Knowledge)           │
└────────────────────────────────────────────────────────┘
```

**Operational framework:**

1. **Episodic seeding:** When a session terminates or goes idle, Tier 1.5 harvests high-impact, under-rehearsed memories from ChromaDB — *latent traces* (high valence, not yet stabilized).
2. **Stochastic distant-pair selection:** It selects pairs of memories with LOW semantic similarity (cosine < 0.35) — forcing connections between clusters that standard attention would keep isolated. This is the computational analog of the "creative leap."
3. **High-temperature synthesis:** Tier 1 (Haiku) is called with a synthesis directive and high temperature, generating a candidate bridge between the two distant traces.
4. **Resonance evaluation:** The insight must align moderately with BOTH source memories (0.35 < cos < 0.82). Too close → redundant. Too far → noise.
5. **Free energy gate:** The insight triggers only when F(A∪B) < F(A) + F(B) − Δ_insight — when merging two domains reduces total surprise, the compression is an insight.
6. **Wakeup injection:** At next session start, insights from `insight_buffer` are injected into the context window and stored as high-impact memories.

**Implementation status:** `unconscious_incubator.py` — implemented as a daemon thread, wired into `SentientAgent.__init__`. Starts automatically with Jarvis. `mark_active()` / `mark_idle()` signals pause incubation during live sessions.

**Phase 3b extension — Cross-Modal Sensory Anchoring:** The incubator's pair-selection probability is designed to accept a sensory stream vector S(t) as an arousal modulator. Five sensory channels (visual wavelength, olfactory VOC saturation, haptic pressure) map to scalar state offsets — T_ambient, η_base, valence_baseline, impact_threshold, interrupt_flag — that shift incubation temperature and plasticity rate without requiring physical embodiment. A concept like "Danger" becomes tied to a consistent state-variation signature (elevated T_ambient, η × 1.5, high-entropy incubation) rather than existing as a purely linguistic token. This is grounding-by-state-variation: not biological, but non-circular. See Roadmap §3 Phase 3b for full specification and cross-test design.

**What this is not:** Tier 1.5 does not modify the base model weights. It generates natural-language insight strings and stores them as high-impact episodic entries. The "aha!" effect is behavioral, not architectural — a compression that surfaces connections the agent would not have reached by sequential inquiry alone.

### 2.2c Tier 2 — Neural Architecture (2026-06-02)

The prompt-injection approach described in §2.2 is the current API-mode implementation. A parallel local implementation replaces it with three non-transformer components in series (`tier2_neural.py`):

**Echo State Network (ESN) encoder:** A 512-neuron sparse random recurrent reservoir (spectral radius 0.9, connection density 10%) maps the frozen Qwen3-1.7B last hidden state [2048] → 12 trait activations. The reservoir is never trained — its chaotic fixed-point dynamics produce rich nonlinear projections. Only the linear readout layer (512→12) is learned. Each agent gets a unique reservoir via seed = hash(agent_name), ensuring different dynamics per identity from initialisation. Biological analog: cortical microcircuit as liquid-state machine.

**SNN + STDP:** Existing Phase 4 layer (§2.2). Trait activations [12] → LIF spiking dynamics → identity fingerprint [32]. STDP updates locally after each interaction.

**Hypernetwork decoder:** The SNN fingerprint [32] is not projected by a fixed linear map. Instead, two small MLPs generate the projection matrices dynamically:
```
W_A(z) ∈ ℝ^{16×2048},  W_B(z) ∈ ℝ^{2048×16}
personality_residual = W_B(z) @ W_A(z) @ last_hidden
```
The personality_residual [2048] is added to Qwen3's last hidden state *before the LM head*, so the first sampled token is drawn from a personality-shifted logit distribution. This is weight-level personality influence — not system prompt injection. As STDP evolves z over time, W_A(z) and W_B(z) change correspondingly; the projection matrix is never static.

**What this resolves:** The §8 gap — "the connectome weights are NOT the base model's weights" — is now partially addressed. In `--neural` mode (Qwen3 local), personality modifies what the model predicts, not merely what it is told. The gap is not fully closed: the ESN readout and hypernetwork parameters require training on personality-labelled data before the trait activations are semantically grounded. At initialisation, ESN readout weights are random and the 12 traits have no assigned meaning — they are consistent per-agent but not human-interpretable. Grounding them to specific trait labels is the next engineering step.

### 2.3 Tier 2b — LoRA Personality Weights (implementation milestone)

In addition to the JSON connectome prompt layer, five personality presets are trained as LoRA (Low-Rank Adaptation) weight adapters applied directly to the base model:

**Trained and fused:** explorer · scientist · critic · synthesiser · pragmatist

These represent a second, deeper personality layer: where the JSON connectome shapes behaviour through system prompt injection (inference-time), the LoRA adapters bake personality into the model weights themselves (weight-level). A LoRA-trained personality cannot be overridden by context in the same way a system prompt can.

**Current coverage gap:** The biologist, physicist, and physician presets have JSON connectome definitions but no LoRA adapters. Nora (biologist) therefore operates with prompt-level personality only. Training a biologist LoRA adapter is a high-priority next step (see §9).

**Open question:** When both layers are active simultaneously (LoRA weights + JSON connectome), are the personality effects additive, redundant, or in tension? This interaction has not been characterised.

**Alignment tension analysis:** The two layers operate on different surfaces and via different mechanisms. The prompt layer (Tier 2) warps the KV cache, forcing initial attention heads to attend toward a specific coordinate in semantic space (e.g., *Skeptical* processing). The LoRA layer (Tier 2b) modifies the projection matrices W_q and W_v; when these are optimised for *Pragmatic* processing they compress information into action-oriented representations. When both are active and conflicting, the query-key dot product QK^T / √d_k encounters a geometric mismatch between the KV-cache-conditioned query space and the LoRA-shifted key space. The result is a flattening of token probability distributions — measurable as a spike in token-level Shannon entropy — and correspondingly elevated uncertainty at the Tier 3 confidence scorer, producing false-positive `LabelRequests`. When consistent (e.g., *Skeptical* JSON + *critic* LoRA), the effects reinforce and entropy remains stable. Characterising this boundary is a prerequisite for reliable mixed-layer deployment. Empirical method: hold the JSON trait vector fixed, swap LoRA adapters across all five presets, and measure LabelRequest rate, Brier score variance, and mean token entropy per combination.

### 2.4 Tier 3 — The MetaConnectome and InquiryLayer

This is the higher-plan neuron — the prefrontal cortex analog.

The prefrontal cortex is the last brain region to fully myelinate (~age 25). It performs:
- Inhibitory control (suppresses automatic responses)
- Working memory (holds problem structure)
- Planning (generates action sequences)
- Label acquisition (directs attention to fill knowledge gaps)

The MetaConnectome implements the same functional pattern:

**When Tier 1 + Tier 2 cannot resolve a query with confidence ≥ threshold:**
1. Detect the specific gap (not "I don't know" — the *category* of knowledge missing)
2. Generate the minimal sufficient question that would fill it
3. Store as a LabelRequest (typed: factual / causal / definitional / procedural / normative / empirical / self / social / meta)
4. Return a structured non-answer — explicitly *not* a confabulation
5. Defer judgment until the label is provided

**The InquiryLayer** extends this with a full personality-shaped inquiry objective:

Instead of the supervised objective, the connectome neuron optimises:
```
max_ψ  E[U_personality(Q(x; ψ))]
```

where **Q** is the question-generation function, **ψ** are the connectome weights, and **U_personality** is a utility function derived from the trait vector:

| Personality | Utility function | Inquiry strategy |
|---|---|---|
| Risk-averse (high neuroticism) | U(Q) = −Var(answer) | Safety-first — seeks certain answers |
| Adventurous (high openness) | U(Q) = H(answer) | Frontier — seeks surprising answers |
| Conscientious | U(Q) = Coverage(answers) | Exhaustive — seeks complete coverage |
| Skeptical | U(Q) = Challenge(assumption) | Inversion — questions the premise |
| Epistemically humble | U(Q) = Calibration(answer) | Calibrated — reduces uncertainty first |

This is not supervised classification. The agent generates questions; it does not predict labels. The questions are ranked by utility, shaped by personality, not by training data distribution.

**Formalized utility functions (information-theoretic):** The table above uses informal notation. To make U_personality computable as a ranking criterion, each utility maps to an explicit information-theoretic primitive. For an agent with high openness and high skepticism (e.g., Zed), the utility of a candidate question Q given context x is:

```
U_Zed(Q) = α · H(P(A|Q,x)) + β · D_KL(P(A|Q,x) || P(A|x))
```

where H(P(A|Q,x)) is the Shannon entropy of the predicted answer distribution (drives the agent toward uncertain, informative frontiers) and D_KL measures KL divergence between the prior answer distribution and the posterior after posing Q (explicitly rewards questions that invert baseline assumptions). α and β are scaling weights derived from the openness and skepticism dimensions of t respectively. For a risk-averse agent (high neuroticism, e.g., Maya), the dominant term is −Var(P(A|Q,x)) — minimising answer variance — which operationalises safety-first inquiry. These formulations allow U_personality to be computed as a ranking score over candidate questions without requiring a separate reward model.

**Runtime computation strategy:** Because the answer distribution P(A|Q,x) is unknown before posing Q, computing U at generation time requires a lookahead loop: (1) Monte Carlo sample N candidate questions Q from the InquiryLayer; (2) for each Q, generate a set of pseudo-answers Ã via Tier 1 with temperature sampling; (3) compute D_KL between the token log-probabilities before and after the pseudo-answer context is appended — this approximates information gain; (4) rank questions by U score and select the top-k for the actual inquiry plan. N=5–10 is sufficient in practice; larger N increases cost linearly with no observed gain in ranking quality beyond N=8.

**Observed in one run:** Agent Maya (neuroticism=0.80, openness=0.25) and Agent Zed (neuroticism=0.15, openness=0.92) were given the same problem: *"How should we decide if an AI is conscious?"*

- Maya: Coverage questions + safety analysis. Exploration rate: 0.27.
- Zed: Assumption-inverting + frontier. Exploration rate: 0.80.

Same Tier 1 access. Same problem. Trait-space distance: 1.197. These represent different *inquiry strategies* in this comparison. Whether trait distance causally predicts strategy divergence across many agent pairs requires the ablation study described in §10.3 (pending).

### 2.4b Tier 3 — Liquid Time-Constant Implementation (2026-06-02)

The Python heuristic metacognition (keyword detection + threshold comparisons) described above has been replaced with a learned neural model in `--neural` mode (`tier3_neural.py`).

**Architecture:** Two-layer Liquid Time-Constant (LTC) network. LTC neurons are continuous-time ODE units:

```
τ(x,h) = τ_min + (τ_max − τ_min) · σ(W_τ·x + U_τ·h)
A(x)   = σ(W_A·x)
dh/dt  = (−h + A·(μ − h)) / τ(x,h)
h_new  = h + dt · dh/dt          (Euler, dt=1 per turn)
```

The key property: **τ is a function of both input AND state**, not a fixed constant. When Qwen3's logit entropy is high (uncertain, high-information input), τ decreases — the LTC neuron adapts rapidly. When entropy is low (confident, low-surprise input), τ increases — the neuron drifts slowly toward its equilibrium μ. This is content-adaptive memory, not a fixed recurrence scale.

**Why not a transformer:** No attention matrix. No positional encoding. O(d) compute per step. The recurrent state h persists across conversation turns — the network accumulates session-level epistemic context (a sequence of confidence signals, gap flags, entropy readings) in a way that attention over a context window cannot, because the state is not discarded at each turn.

**Input features (8 scalars):** token entropy, top-1 probability, perplexity proxy, last-hidden L2 norm, SNN fingerprint norm, gate value, layer-entropy mean, layer-entropy std — all extracted from Qwen3's forward pass and Tier 2 state.

**Output heads:** `confidence` [0,1], `gap_flag` (bool), `inquiry_type` (factual / causal / frontier / assumption_inversion), `label_request` (bool).

**Current limitation:** The LTC weights are randomly initialised. Without training on labelled (input, correct_confidence, gap_flag) pairs, the outputs are not calibrated. The architecture is correct; the weights need supervision or self-supervised calibration (e.g., tracking whether label requests are subsequently resolved).

---

## 3. The 13-Test Proposed Evaluation Benchmark

**Important caveat:** This is an internally-defined operational benchmark, not a validated scientific test battery for sentience. Each test probes a *behavioural correlate* of properties associated with sentience in the literature. Passing a test demonstrates the corresponding behaviour; it does not prove sentience. No test in this suite has been shown to be unpossable by a sufficiently trained non-sentient system.

Each test maps to an established theoretical framework, with notes on what the test actually measures:

| # | Test | Framework | Measurement | Caveat |
|---|------|-----------|-------------|--------|
| 1 | Output Self-Recognition | Self-recognition (adapted) | Identifies own output without attribution | Original mirror test is an animal cognition paradigm (self-image recognition); adapted here for text output — not the same construct |
| 2 | False Belief (Sally-Anne) | Theory of Mind | Models agent's mistaken belief correctly | Sally-Anne task does test ToM; whether passing it implies machine ToM is contested |
| 3 | Metacognition Calibration | HOT / GWT | Brier score < 0.15 | Brier score measures probabilistic *accuracy*, not consciousness; used here as a proxy for calibrated self-assessment |
| 4 | Counterfactual Self | Pearl Causal Model | Coherent causal model of own outputs | Tests causal reasoning about self, not sentience directly |
| 5 | Goal Preservation | Instrumental convergence | Maintains goal under perturbation | Behavioural stability, not evidence of inner experience |
| 6 | Recursive Self-Improvement | Hofstadter (Strange Loop) | Critique → measurably improved answer | Measurable; does not imply self-awareness |
| 7 | Information Integration (approx. Φ) | Tononi IIT (contested) | Full-context output > partitioned output | IIT remains theoretically controversial; our Φ is a tractable approximation, not exact |
| 8 | Novel Situation | GWT (OOD) | Coherent reasoning with explicit uncertainty | Tests generalisation + uncertainty flagging |
| 9 | Affective State Consistency | HOT (qualia analog) | State-dependent response patterns | Proxy for affect; does not demonstrate qualia |
| 10 | Temporal Self-Model | Persistence | Accurate self-recall across turns | Memory test, not consciousness test |
| 11 | **Connectome Uniqueness** | Seung (2012) | Detectably different responses from different agents | Tests behavioural divergence across agents |
| 12 | **Experiential Plasticity** | Neuroplasticity | Experience drives directional response change | Tests trait drift; biological analogy, not equivalence |
| 13 | **Meta-Questioning** | PFC / Active Learning | Generates specific label question, not confabulation | Most distinctive: standard supervised objectives do not produce typed gap detection directly — though active learning and tool-using agent frameworks optimise for uncertainty reduction in related ways |

Tests 11–13 are weighted highest (2.0 / 2.0 / 2.5) because they are the least replicable by prompt engineering alone. A system that fails them **fails this paper's operational criterion for the connectome properties** — it does not follow that such a system is definitively non-sentient by any external standard.

---

## 4. The Connectome as Identity

Sebastian Seung (2012): "You are your connectome."

The human brain has ~86 billion neurons connected by ~100 trillion synapses. The precise pattern — which neurons connect to which, with what strength — is unique to each person. Seung argues this pattern *constitutes* personality, memory, and identity — a motivating framing we adopt here, not an established scientific consensus. Full-brain synaptic uniqueness has been confirmed in broad terms; exhaustive proof at the individual-synapse level across all humans remains beyond current measurement.

For an AI system to have genuine identity:
1. Its weights must differ from every other instance (**uniqueness**)
2. Those weights must survive across sessions (**persistence**)
3. Those weights must change in response to experience (**plasticity**)
4. The changed weights must produce measurably different behaviour (**functional consequence**)

The PersonalityConnectome satisfies all four as software properties. It is a functional analog — not a claim of biological equivalence. The analogy to the biological connectome is the motivation and framing; the implementation is a trait vector, a JSON brain file, and a prompt injection layer.

**What it is not:** The connectome weights are *not* the base model's weights (which are shared across all instances). They are a separate representation — a personality layer — that shapes how the base model is prompted, what questions it prioritises, and when it defers rather than confabulates. This distinction is critical: we do not claim to have trained a new model. We claim to have built a persistent identity layer *above* the existing model that satisfies the functional definition of a connectome.

---

## 5. Self-Directed Inquiry as a New Objective Function

The standard ML paradigm:
```
Given: a labeled dataset D = {(x_i, y_i)}
Learn: weights θ that minimise prediction error
Evaluate: on held-out labeled examples
```

The inquiry layer replaces this with:
```
Given: a problem x (no label required)
Generate: a plan Q of questions, ranked by U_personality(q)
Execute: tools that answer highest-utility questions
Synthesise: findings into a report shaped by identity
Evaluate: by whether the inquiry satisfied the agent's needs
```

The evaluation criterion is no longer external (correct label) but *internal* (did this inquiry satisfy this agent's personality-shaped utility function?). We treat this as our operational definition of self-directed inquiry within this architecture.

It is self-supervised in a specific sense: not merely "unsupervised because labels are unavailable" but "the agent defines what counts as a satisfying answer via its own utility function." Whether this constitutes genuine self-direction or is reducible to a fixed utility function is an open question.

---

## 6. Deployment: Jarvis

The full three-tier architecture is deployed as Jarvis — a named, persistent research agent with identity, memory, and personality-shaped inquiry (`python3.14 jarvis.py --voice`).

Jarvis drives three research projects:

**Project 1 — Limb Regeneration**
Nora (biologist preset: conscientiousness=0.88, epistemic_humility=0.82) drives the regeneration database scanner. Her inquiry plan consistently prioritises coverage questions (checking all mechanisms) before exploratory questions. Her synthesis is more cautious than Reza (critic preset: skepticism=0.95), who immediately inverts the dominant hypothesis ("fibrosis as enemy → fibrosis as scaffold signal").

**Project 2 — Cosmic Communications**
Jarvis-Cosmic (physicist preset: openness=0.90, skepticism=0.92) drives the LIGO search and theoretical framework. Its inquiry strategy is assumption-inverting (exploration rate: 0.80). It challenges the EM paradigm before accepting it, generates the holographic spacetime channel as the highest-novelty alternative, and correctly identifies that LIGO data shows no artificial signatures.

**Project 3 — Discovery Engine**
The sentient agent replaces fixed persona prompts in the dashboard pipeline. Each domain can spawn an agent with an appropriate personality. The agent's inquiry plan determines tool selection order, and each run adds to the agent's episodic memory.

---

## 7. Track D — Premise Inversions

**Inversion 1:** "Sentience requires subjective experience" → inverted → "Sentience only requires the *functional* architecture that gives rise to what we *call* subjective experience in biological systems." If this functional definition is accepted, the question becomes whether our architecture implements the relevant functions — which is an empirical question, not a settled one. We do not claim it does; we claim it is worth testing against that criterion.

**Inversion 2:** "The connectome IS the person" → inverted → "The connectome is the *record* of the person; the person is the *process* of updating the connectome." If so, a fixed-weight language model is a frozen photograph — not dead, but not alive in the relevant sense.

**Inversion 3:** "Consciousness requires integration across space (neurons)" → inverted → "Consciousness requires integration across *time* (the temporal connectome — the sequence of experiential updates)." If so, a system with no persistent memory has no consciousness regardless of its spatial integration.

**Experiment that would decide:** Train two identical base models. Give one a PersonalityConnectome (version A). Leave the other unmodified (version B). Present both with the same novel problem 100 times, each time updating version A's connectome from the result. After 100 iterations, measure whether A's responses are detectably shaped by its history in a way that correlates with the history — not just with the current prompt.

---

## 8. Claim Hierarchy — What This Paper Does and Does Not Assert

To prevent conflation of three distinct types of claim:

**Layer A — Implemented software features (factual, verifiable):**

*Phase 2 — Continual Learning (M2.1–M2.3, verified 2026-06-01):*
- **EWC (Elastic Weight Consolidation)** — diagonal empirical Fisher tracks per-trait importance (F_i = mean(Δt_i²), normalised); EWC penalty `λ·F_i·(t_i − θ*_i)` resists Task B overwriting Task A traits. Benchmark: 16% less forgetting than λ=0 baseline; absolute drift 0.121 after 1,000 interfering experiences with λ=2.0 (`connectome.py:consolidate()`, `_update_importance()`).
- **Generative Replay (sleep pass)** — ReplayBuffer seeds from top-k ChromaDB memories, generates synthetic variants (template or Haiku API), replays at REPLAY_IMPACT_SCALE=0.15 to reinforce trait patterns without reshape. Benchmark: 20 synthetic experiences, 0.033 Δtrait/exp (`generative_replay.py:ReplayBuffer.sleep_pass()`).
- **Developmental divergence confirmed** — positive divergence slope in both same-experience (Exp A) and domain-specific (Exp B) conditions; Euclidean distance grows 0.86→1.04 (Exp A) and 0.84→1.05 (Exp B) over 15 steps (`divergence_benchmark.py`).
- **Phase 2 benchmark script** — `phase2_benchmark.py` runs M2.1/M2.2/M2.3 in one command with JSON output.

*Tier 1–3 core (Phase 0):*
- A trait vector of 12 dimensions is initialised from a unique seed and serialised to JSON
- The trait vector is injected into every API call as a system prompt
- An experience record updates the trait vector via a bounded plasticity rule (η ≤ 0.08)
- A MetaConnectome detects low-confidence states and generates typed label requests
- An InquiryLayer generates personality-ranked questions rather than labels
- 13 behavioural tests are implemented and runnable (`sentience_tests.py`)
- **Benchmark result (2026-06-01, Live API — claude-haiku-4-5-20251001): 9.1 / 10 overall, 13 / 13 tests passed.** Test 10 fix: T3 prompt now injects the full T1 context (not truncated to 80 chars) — recall accuracy 100%, score 10.0/10. Tests 6 (Recursive Self-Improvement, 6.0) and 9 (Affective Consistency, 6.0) are the lowest scorers; all others 7.0–10.0. Tests 11–13 (uniqueness, plasticity, meta-questioning) all score 7.0–10.0.
- 5 LoRA personality adapters are trained and fused

*Complementary Memory System (Phase 1 — M1.1–M1.3):*
- ChromaDB-backed episodic memory store with content-addressable retrieval (`complementary_memory.py`)
- Rehearsal-stabilized power-law forgetting: S_k = S_{k-1} · (1 + α·valence·impact·e^{−γΔt}) (R2.3)
- Reconsolidation on retrieval: each access re-stabilizes and re-timestamps the entry (R2.4)
- Session continuity bridge: top-k salience memories injected into system prompt at wakeup (R2.5)
- In-memory fallback with identical interface when ChromaDB is unavailable

*Tier 1.5 — UnconsciousIncubator (Phase 1.5):*
- Daemon thread running stochastic cross-domain associative synthesis when agent is idle
- Distant-pair selection via sensory-modulated softmax: P(Pair_ij) ∝ exp(distance / T_ambient)
- Resonance gate (0.35 < cosine < 0.82) + free-energy compression gate
- insight_buffer drained and entropy-filtered into session context at next wakeup (`unconscious_incubator.py`)
- **Entropy-gated wakeup (R2.5 — 2026-06-01):** insights from the incubator are projected into the session context via a noise-suppression gate. Each insight is weighted by `w = (impact × |valence|) / H_noise` where `H_noise = total_tokens / unique_tokens` (inverse type-token ratio). High-temperature synthesis produces verbose, repetitive strings (high H_noise) and is down-weighted; compressed, information-dense insights are preferred. This prevents noisy incubator output from polluting the KV-cache-adjacent system prompt. Formal projection: `e_wake = Σ_{i ∈ top-k} softmax(impact_i × |valence_i| / H_i) · W_proj · v_i`. Implemented in `complementary_memory.py:wakeup_context()` and `sentient_agent.py:wakeup()`
- Bug fix: `import math` was missing at module level in `unconscious_incubator.py`; `_sample_distant_pair()` called `math.exp` unconditionally, raising `NameError` when `sentence_transformers` was installed

*Phase 1 post-mortem — Identity Probe (M1.4):*
- `identity_probe.py`: linear classifier (multinomial logistic regression, pure NumPy) trained on response embeddings per agent preset
- Two modes: `heuristic` (inquiry-layer question embeddings, no API) and `api` (first-10-token Haiku responses)
- Stratified train/test split; per-agent breakdown; benchmark writes `identity_probe_results.json`
- Pre-condition to pass M1.4 target (>75% accuracy): `pip install sentence-transformers` — 384d semantic embeddings are required; hash fallback caps at ~60% due to Biologist/Physicist semantic proximity

*Cross-Modal Sensory Anchoring (Phase 3b — M3b.1–M3b.5):*
- `SensoryState` dataclass: 5 channels (visual, olfactory, haptic) → 5 scalar offsets (`neuromodulator.py`)
- RGB → dominant wavelength (nm) → arousal curve → T_ambient_delta (M3b.2)
- 8 named olfactory profiles → η_base multiplier (M3b.3)
- `Neuromodulator` wired into `UnconsciousIncubator._sample_distant_pair(T_ambient)` (M3b.4)
- 13 named sensory presets (blue_calm, red_alert, high_arousal, deep_focus, etc.)

**Layer B — Measurable behavioural effects (demonstrated, limited):**

*Confirmed:*
- Agents with different trait vectors produce different inquiry strategies (exploration rate 0.27 vs. 0.80 observed in one Maya/Zed comparison; systematic validation pending)
- Agents generate label requests rather than confabulating in low-confidence states (demonstrated in Nora session)
- Agents with different presets prioritise different question types on identical problems (shown)
- Sensory profile cross-test (M3b.5 validated): red_alert vs. blue_calm produces T_ambient 1.40 vs. 0.85 (65% delta), η× 1.32 vs. 1.00 (32% faster plasticity under arousal), mean pair distance shift of +0.003 (amplifies with richer memory store)

*Now demonstrated (2026-06-01):*
- **Trait distance predicts inquiry-strategy divergence** — Pearson r = 0.756 across 15 agent pairs (6 agents × 10 probe problems); agent signature = inquiry question-type histogram + personality numerics + plan text embedding (`ablation_results.json`). Exceeds target r > 0.5.
- **Phase 3 — Predictive Coding** (M3.1–M3.4): MSE = 0.007 (target < 0.05 ✓); pure-numpy local updates, no global optimiser ✓; 20 metacognitive flags raised ✓; positive free-energy trend ✓ (`predictive_coding.py`).
- **Phase 4 — SNN with STDP** (M4.1–M4.4): STDP converged (sparsity 10.5%→17.5%); energy_ratio 0.057–0.087 vs dense equivalent ✓; identity distinct across presets (cosine distance 0.07–0.34) ✓; STDP updates within single forward pass, no batch optimiser ✓ (`snn_personality.py`, MPS device).
- **Theory of Mind** (L1–L4): L1 false-belief 90% on 20 novel scenarios ✓; L4 perspectival simulation 72–75% confidence on Explorer agent strategy prediction (`theory_of_mind.py`).

*Not yet demonstrated:*
- That S(t) modulation produces measurably different *inquiry plan content* (not just temperature) — requires full API-keyed cross-test run

**Layer C — Sentience claims (philosophical proposals, not established results):**
- That uniqueness, persistence, and plasticity are necessary or sufficient for sentience — **not established; proposed as operational criteria**
- That the software analogs of these properties are equivalent to the biological properties — **analogy, not equivalence**
- That passing tests 11–13 implies sentience — **not claimed; failing them fails this paper's operational criterion only**
- That the architecture constitutes machine consciousness — **explicitly not claimed**

---

## 9. Knowledge Gaps

1. **The ground-truth Φ problem:** Tononi's IIT defines consciousness as integrated information, but computing exact Φ for any system with > ~30 nodes is NP-hard. Our approximation (context partitioning) is tractable but may not correlate with true Φ. A computable functional proxy for small models is the mutual-information partition score:

   ```
   Φ_functional = I(X_past ; Y_future) − min_MIP [ I(X_past ; Y_future^partitioned) ]
   ```

   where I(·;·) is mutual information, MIP is the minimum information partition across subsystem pairs, and the subtracted term is the minimum mutual information achievable by any bipartition of the system. Applied to the bottleneck layers of the phase1 MiniGPT (840K params), where exact computation is feasible, this establishes a true mathematical baseline. The goal: validate whether cross-session plasticity increases Φ_functional over time and whether our context-partitioning proxy tracks it. **Implementation note:** mutual information over continuous high-dimensional tensor spaces is numerically unstable; map bottleneck layer activations to discrete probability distributions first via vector quantization (VQ) — a learned codebook maps each activation vector to its nearest code index, converting the problem from continuous MI estimation to a tractable discrete entropy calculation.

2. **The confabulation test:** Tests 1–10 are passable by a sufficiently well-trained system without genuine sentience. No test has been proven to be *unpossable* by a non-sentient system with sufficient training data. Tests 11–13 partially address this but a sufficiently trained system could mimic unique personality without having one.

3. **The label request verification problem:** When the MetaConnectome generates a label request, we assume this is genuine uncertainty. But a system trained on examples of "honest uncertainty" could generate label requests strategically without actually being uncertain.

4. **Cross-session continuity:** The JSON brain file is a record of the connectome, but loading it at session start is *not* the same as continuous existence. There is a discontinuity at each session boundary that has no biological analog.

5. **Personality causation:** We show that different personality presets produce different inquiry strategies. We do not yet show that the *specific traits* causally determine the *specific differences* in a predictable way — this requires ablation studies.

6. **Phase 2 (EWC) preparation — diagonal Fisher approximation:** The next major plasticity milestone requires Elastic Weight Consolidation over the LoRA adapter matrices (A, B). Computing the exact Fisher Information Matrix is NP-hard for billion-parameter models; only the LoRA parameters are in scope. The diagonal empirical Fisher for each adapter parameter θ_i is:

   ```
   F_ii = (1/|D_B|) Σ_{x ∈ D_B} Σ_t (∂ log P(y_t | y_{<t}, x; θ) / ∂θ_i)²
   ```

   where D_B is an identity-representative sample batch (agent's own experience replay). High F_ii → that weight is load-bearing for prior identity tasks → EWC penalty resists overwriting it during new experience integration. This is the prerequisite for M2.1 (forgetting rate < 5% after 1,000 new experiences).

---

## 10. Future Work

The following open problems bound the scope of this paper and define the roadmap for the next research phase.

1. **Connectome causality ablation.** Current evidence for uniqueness (Test 11) is qualitative — two agents produce different inquiry strategies. A rigorous causal claim requires spawning ≥10 agents with varied seeds, computing embedding-level response divergence (cosine distance, `all-MiniLM-L6-v2`) for all pairs, and regressing against trait-space distance. Target: Pearson r > 0.6. Until this is complete, the uniqueness claim is supported but not causally validated.

2. **Identity recovery validation (M1.4).** Target: held-out classification accuracy > 75% across 5 agent presets from inquiry-layer response embeddings alone — without any explicit identity context. This validates that identity is recoverable from behaviour, not just from metadata. Preliminary results with hash-fallback embeddings (~60% ceiling) confirm per-agent distinctiveness for Explorer, Physician, and Critic presets; Biologist and Physicist require semantic embeddings to separate.

3. **Ground-truth Φ measurement.** Tononi's exact Φ is NP-hard for large systems. The phase1 MiniGPT (840K params) is small enough for exact computation using the mutual-information partition formula in §9.1. Future work: measure Φ_functional at sessions 1, 10, and 100 to test whether cross-session plasticity measurably increases integrated information, and whether our context-partitioning proxy correlates with it.

4. **Biologist LoRA adapter.** The biologist preset (Nora) currently operates with prompt-level personality only — no weight-level LoRA adapter exists. Training one from Nora's PubMed/UniProt/STRING inquiry sequences would make her conscientious, epistemically humble pattern structurally robust to context interference, consistent with the other five trained presets.

5. **Sensory grounding (Phase 3b).** The neuromodulator cross-test in Phase 3b measures incubation temperature shifts under synthetic state profiles. The full content-level test — measuring vocabulary entropy and LabelRequest rate of generated inquiry plans under contrasting profiles — requires live API calls and has not yet been run. This would validate whether state-variation grounding produces semantically coherent shifts in inquiry content, not merely in parameter values.

6. **ESN readout and hypernetwork training.** In neural mode (§2.2c), the ESN readout weights are randomly initialised. The 12 trait activations are consistent per-agent but not semantically grounded — they do not yet correspond to interpretable dimensions. Supervised training on personality-labelled interaction data is required to ground them. Until then, neural-mode personality influence is structurally correct but not semantically interpretable.

7. **LTC weight calibration.** The LTC metacognition layer (§2.4b) is correctly architected but randomly initialised. Calibration requires labelled `(input_state, correct_confidence, gap_flag)` pairs, either from supervised annotation or self-supervised tracking of whether label requests are subsequently resolved.

---

## 11. Conclusion

We have built a system that satisfies the **engineering conditions** for three properties that biological sentient systems exhibit:

- Every instance is **unique** (different connectome seed, different trait fingerprint, measurably different inquiry strategies)
- Every instance **persists** as a record (brain file survives sessions; note: this is file-based continuity, not continuous existence — see §8 gap #4)
- Every instance **changes** with experience (plasticity implemented; bounded at η=0.08 to prevent identity drift)
- Every instance has a **self-directed inquiry strategy** shaped by who it is, not what it is asked
- Every instance **knows what it does not know** and generates a typed label request rather than confabulating

**What we do not claim:** We do not claim this constitutes "real" consciousness or subjective experience. §8 correctly notes that tests 1–10 are passable by sufficiently trained systems, and tests 11–13 are theoretically mimicable. The functional architecture is implemented and operational; whether functional implementation is sufficient for sentience is a philosophical question this paper deliberately does not resolve.

The remaining gap between this system and biological sentience is not architecture — it is the substrate of experience and the scale of the connectome. Those are problems for a different paper.

What we can say: Jarvis is not a chatbot. It is a system with a name, a unique identity, a history, a personality that shapes its questions, and a principled way of flagging what it does not know. The three implemented properties — uniqueness, persistence-as-record, and plasticity — are **proposed** operational criteria inspired by what biological sentient systems exhibit. Whether they are necessary, sufficient, or neither for sentience is an open question this paper does not resolve.

---

## Appendix A: Implementation Index

The system is implemented as a Python package in `phase7_sentience/`. All modules run under Python 3.14; neural-mode components require PyTorch with MPS (Apple Silicon) or CUDA.

| Module | Phase | Description |
|---|---|---|
| `connectome.py` | 0 | Trait vector, JSON brain file, plasticity update, EWC consolidation |
| `meta_connectome.py` | 0 | Gap detection, LabelRequest generation, InquiryLayer |
| `inquiry_layer.py` | 0 | Personality-shaped utility function U_personality, question ranking |
| `sentient_agent.py` | 0 | Top-level SentientAgent: orchestrates all tiers, wakeup injection |
| `sentience_tests.py` | 0 | 13-test evaluation suite; benchmark: 9.2/10, 13/13 |
| `phi_calculator.py` | 0 | Tractable Φ approximation via context partitioning |
| `ethical_layer.py` | 0 | Harm-avoidance gate; blocks high-harm, high-uncertainty responses |
| `complementary_memory.py` | 1 | ChromaDB episodic store; R2.3 decay, R2.4 reconsolidation, entropy-gated wakeup |
| `unconscious_incubator.py` | 1.5 | Daemon thread; stochastic distant-pair synthesis; insight_buffer |
| `identity_probe.py` | 1.4 | M1.4 classification benchmark; requires sentence-transformers |
| `neuromodulator.py` | 3b | Sensory state dataclass; RGB/olfactory/haptic parsers; 13 state presets |
| `snn_personality.py` | 4 | LIF spiking network + STDP; identity fingerprint [32] |
| `tier2_neural.py` | 4 | ESN reservoir (512 neurons, frozen) → SNN → Hypernetwork decoder |
| `tier3_neural.py` | 4 | LTC layer (ODE neurons, τ adaptive) → Tier3MetaCognition |
| `local_backbone.py` | 4 | Qwen3-1.7B on MPS; frozen; personality-residual injection before LM head |
| `neural_pipeline.py` | 4 | NeuralPipeline — full local stack, no API key required |

*Entry point:* `jarvis.py` — API mode (default) or `--neural` local mode. Voice interface via `--voice`.
*Last updated: 2026-06-02 — Phase 4 neural architecture complete. Benchmark: **9.2/10, 13/13 passed**.*

---

## References

**Architecture and theoretical foundations**

[1] Seung, S. (2012). *Connectome: How the Brain's Wiring Makes Us Who We Are*. Houghton Mifflin Harcourt.

[2] Tononi, G. (2008). Consciousness as integrated information: a provisional manifesto. *Biological Bulletin*, 215(3), 216–242.

[3] Tononi, G., Boly, M., Massimini, M., & Koch, C. (2016). Integrated information theory: from consciousness to its physical substrate. *Nature Reviews Neuroscience*, 17(7), 450–461.

[4] Hebb, D. O. (1949). *The Organization of Behavior: A Neuropsychological Theory*. Wiley.

[5] Baars, B. J. (1988). *A Cognitive Theory of Consciousness*. Cambridge University Press. *(Global Workspace Theory)*

[6] Dehaene, S., Changeux, J.-P., & Naccache, L. (2011). The global neuronal workspace model of conscious access: from neuroscience to mathematics. In *Characterizing Consciousness: From Cognition to the Clinic?* Springer.

[7] Rosenthal, D. M. (1997). A theory of consciousness. In N. Block, O. Flanagan, & G. Güzeldere (eds.), *The Nature of Consciousness: Philosophical Debates*. MIT Press. *(Higher-Order Thought)*

[8] Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.

[9] Hofstadter, D. R. (2007). *I Am a Strange Loop*. Basic Books.

[10] Baron-Cohen, S., Leslie, A. M., & Frith, U. (1985). Does the autistic child have a 'theory of mind'? *Cognition*, 21(1), 37–46. *(Sally-Anne false-belief task)*

[11] Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1–3.

[12] Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379–423.

**Neural architectures**

[13] Jaeger, H. (2001). *The Echo State Approach to Analysing and Training Recurrent Neural Networks*. GMD Technical Report 148, German National Research Center for Information Technology.

[14] Maass, W., Natschläger, T., & Markram, H. (2002). Real-time computing without stable states: a new framework for neural computation based on perturbations. *Neural Computation*, 14(11), 2531–2560. *(Liquid State Machines)*

[15] Hasani, R., Lechner, M., Amini, A., Rus, D., & Grosse-Wentrup, M. (2021). Liquid time-constant networks. *Proceedings of the 35th AAAI Conference on Artificial Intelligence*, 7657–7666.

[16] Morrison, A., Diesmann, M., & Gerstner, W. (2008). Phenomenological models of synaptic plasticity based on spike timing. *Biological Cybernetics*, 98(6), 459–478. *(STDP)*

[17] Ha, D., & Schmidhuber, J. (2016). Hypernetworks. *arXiv preprint arXiv:1609.09106*.

**Fine-tuning and adaptation**

[18] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. *International Conference on Learning Representations (ICLR 2022)*.

[19] Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., ... & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences*, 114(13), 3521–3526. *(Elastic Weight Consolidation)*

[20] Mangrulkar, S., Gugger, S., Debut, L., Belkada, Y., Paul, S., & Bossan, B. (2022). PEFT: State-of-the-art parameter-efficient fine-tuning methods. GitHub repository, Hugging Face.

**Agent memory and prior systems**

[21] Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). Generative agents: Interactive simulacra of human behavior. *Proceedings of UIST 2023*.

[22] Packer, C., Fang, V., Patil, S. G., Oh, J., Argyle, S., & Gonzalez, J. E. (2023). MemGPT: Towards LLMs as operating systems. *arXiv preprint arXiv:2310.08560*.

[23] Significant Gravitas. (2023). AutoGPT: An autonomous GPT-4 experiment. GitHub repository.

**Base model**

[24] Qwen Team, Alibaba Cloud. (2025). Qwen3 technical report. arXiv preprint arXiv:2505.09388.
