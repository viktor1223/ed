---
title: "Simulated Student Research: Literature Survey and Recommendation"
description: >-
  Deep research survey of simulated student models for intelligent tutoring
  systems, evaluating preexisting solutions against project requirements,
  with recommendation for a new research path.
author: Viktor Ciroski
ms.date: 2026-03-30
ms.topic: reference
keywords:
  - simulated student
  - intelligent tutoring systems
  - misconception modeling
  - cognitive student models
  - educational simulation
estimated_reading_time: 25
---

<!-- markdownlint-disable MD013 MD033 -->

## Executive Summary

This document surveys the state of simulated student models for intelligent
tutoring systems as of March 2026, evaluating whether any preexisting solution
can replace our invalid BKT-based simulation. After surveying 12 candidate
systems across frameworks, papers, and packages, the answer is clear:

**No preexisting solution meets our Critical requirements.** The field is split
between statistical models (BKT/DKT) that lack misconception fidelity and
LLM-based models that lack controllability and reproducibility.

**Recommendation: Option B - New Research Path.** Build a misconception-aware
simulated student grounded in BEAGLE's neuro-symbolic architecture and
informed by MalAlgoPy's algebraic misconception taxonomy. This warrants a
separate repository for independent validation before integration.

---

## 1. Literature Survey

### 1.1 Existing Simulated Student Frameworks

#### MalAlgoPy (Sonkar et al., 2024)

**Citation:** Sonkar, S., Chen, X., Liu, N., Baraniuk, R.G., & Sachan, M.
(2024). "LLM-based Cognitive Models of Students with Misconceptions."
arXiv:2410.12294.

**What it is:** A Python library that generates datasets reflecting authentic
student algebra solution patterns through a graph-based representation of
algebraic problem-solving. It is used to instruction-tune LLMs into "Cognitive
Student Models" (CSMs) that replicate specific misconceptions while correctly
solving problems where those misconceptions don't apply.

**Key findings:**
- LLMs trained on misconception examples can learn to replicate errors
- But training *diminishes* the model's ability to solve problems correctly
  on problem types where misconceptions are inapplicable
- Calibrating the ratio of correct-to-misconception examples (as low as 0.25)
  can produce CSMs satisfying both properties

**Repository status:** No public repository found. GitHub searches for
`MalAlgoPy`, `sonkarmanish/MalAlgoPy`, `umass-ml4ed/MalAlgoPy`, and
`SonkarS/MalAlgoPy` all return 404. The library appears to be described in
the paper but not publicly released.

**Assessment for our project:**
- Misconception fidelity: Yes (graph-based misconception representation)
- But: Requires an LLM for each simulated student (expensive, slow)
- Not reproducible: LLM responses vary between runs
- No negative transfer model
- No learning dynamics (static misconception profile, no instruction response)

#### BEAGLE (Wang et al., 2026)

**Citation:** Wang, H.D., Cohn, C., Xu, Z., Guo, S., Biswas, G., & Ma, M.
(2026). "BEAGLE: Behavior-Enforced Agent for Grounded Learner Emulation."
arXiv:2602.13280. Under submission at IJCAI.

**What it is:** A neuro-symbolic framework from Vanderbilt University that
addresses LLM "competency bias" (LLMs optimized for efficiency produce correct
solutions rather than novice-like struggle). The architecture has five major
components:

1. **Semi-Markov model**: Governs timing and transitions of 4 metacognitive
   behaviors (Planning, Enacting, Monitoring, Reflecting) and 3 cognitive
   behaviors (Constructing, Debugging, Assessing). Uses Gamma duration
   distributions instead of geometric - critical for capturing "getting stuck"
   patterns (LOW Enacting has CV=1.35, 42% above geometric prediction).
2. **BKT with Explicit Flaw Injection (EFI)**: Goes beyond standard BKT.
   When a KC is unmastered, injects: "CRITICAL CONSTRAINT: You have NEVER
   heard of and CANNOT use [concept]. This concept does not exist in your
   knowledge." This forces the LLM to improvise wrong solutions rather than
   using suppressed knowledge.
3. **Strategist/Executor architecture**: Decouples planning from code
   generation. The Strategist formulates a Goal/Mindset/Directive; the
   Executor implements it. Ablation shows merging them reduces error
   recurrence from 86.2% to 65.3% (21% drop) - the LLM silently self-corrects
   when planning and execution are unified.
4. **Observation filtering**: During impulsive Enacting states, error traces
   are redacted ("[Error]: [output omitted...]"), preventing the agent from
   diagnosing errors it shouldn't understand.
5. **Stochastic interrupts**: Assistance (peaks mid-task, mu=0.5) and
   Off-Topic (peaks late, mu=0.73) modeled as Gaussian over task progress.
   High performers seek MORE help (15% vs 11.7%); Low performers disengage
   MORE (9.2% vs 3.7%).

**Key quantitative results:**
- Error recurrence: BEAGLE 86.2% vs Vanilla 7.8% (real students: 92.0%)
- Behavioral KL divergence: BEAGLE 0.35 vs Vanilla 3.97
- Steps to solve: BEAGLE 29 vs Vanilla 6 (real students take many steps)
- Human Turing test (N=71, 852 classifications): 52.8% accuracy, TOST
  equivalence confirmed (d'=0.15, p_TOST=0.038)
- Performance gap: BEAGLE +40% between High/Low profiles vs Vanilla +0%
- Ablation: removing semi-Markov causes D_KL to jump from 0.35 to 6.76

**Repository status:** No public code. Under submission at IJCAI 2026. Uses
Gemini 2.0/2.5 Flash as LLM backbone.

**Assessment for our project:**
- **Most architecturally relevant candidate found in the survey**
- BKT + EFI + observation filtering is exactly the approach we need to
  prevent unconditional learning in our simulation
- The Strategist/Executor split directly addresses our "any misconception ID
  triggers 2x bonus" problem - the Executor should only apply remediation
  when the Strategist verifies it matches the student's actual gap
- BUT: designed for Python programming tasks, not algebra misconceptions
- BUT: requires an LLM backbone (Gemini 2.0 Flash), making deterministic
  experiments impossible. Each run costs real money and varies.
- BUT: no public code available
- The domain is fundamentally different: BEAGLE simulates code-writing
  trajectories, we need misconception-specific wrong-answer generation
- **We should adopt the architectural principles (semi-Markov behavioral
  control, EFI-style knowledge gating, observation filtering, decoupled
  agent design) but implement them as a deterministic rule-based system
  without an LLM backbone.**

#### Scarlatos et al. (2026) - Simulated Students in Tutoring Dialogues

**Citation:** Scarlatos, A., Lee, J., Woodhead, S., & Lan, A. (2026).
"Simulated Students in Tutoring Dialogues: Substance or Illusion?"
arXiv:2601.04025.

**What it is:** The first rigorous evaluation framework for LLM-simulated
students. Formally defines the student simulation task, proposes evaluation
metrics spanning linguistic, behavioral, and cognitive aspects, and benchmarks
a wide range of simulation methods.

**Key findings (critical for our project):**
- **Error replication is catastrophically bad across ALL methods.** Scores on
  the "Errors" metric (does the simulated student make the same error as the
  real student when both are wrong): Zero-Shot 0.022, OCEAN 0.031, ICL 0.032,
  Reasoning 0.009 (!), SFT 8B 0.066, DPO 8B 0.053. Even Oracle (with leaked
  ground-truth behavior summary) only hits 0.187. No method comes close to
  reliably replicating specific student errors.
- **Prompting generates mostly correct answers.** LLMs default to correctness.
  Distribution analysis shows prompting methods overestimate correct responses
  and underestimate "n/a" conversational turns. Fine-tuned models match the
  real distribution much better.
- **SFT+DPO outperforms prompting** on acts (0.684 vs 0.500), knowledge
  acquisition (0.879 vs 0.808), cosine similarity (0.739 vs 0.546), and
  tutor response induction (0.204 vs 0.191). But still poor on errors.
- **Human evaluation confirms** automated metrics: Cohen's Kappa 0.73 for
  acts, 0.69 for correctness, 0.61 for errors, 0.74 for linguistic similarity.
- **Key quote from conclusions:** "There is a long way to go before LLMs can
  fully resemble real student behavior in dialogues."
- The paper uses the **Eedi Question-Anchored Tutoring Dialogues 2k** dataset
  (1,529 train / 382 test dialogues). This could be a validation resource.
- Also references **TutorGym** (Weitekamp et al., 2025, AIED): "a testbed for
  evaluating AI agents as tutors and students" - worth investigating.

**Repository status:** No public framework code found. Uses proprietary models
(GPT-4.1, GPT-5 mini) for annotation; local models are Llama 3.1 8B and 3.2 3B.

**Assessment:** Essential reading. The Error metric results (0.02-0.19) are
the strongest evidence that LLM-based simulated students cannot reliably
exhibit specific misconceptions. The 6-dimension evaluation framework (acts,
correctness, errors, knowledge, linguistics, tutor response) is directly
applicable to evaluating any simulated student we build.

#### SMART (Scarlatos et al., 2025)

**Citation:** Scarlatos, A., Fernandez, N., Ormerod, C., Lottridge, S., & Lan,
A. (2025). "SMART: Simulated Students Aligned with Item Response Theory for
Question Difficulty Prediction." EMNLP 2025. arXiv:2507.05129.

**What it is:** Uses IRT-aligned simulated students for question difficulty
prediction. More focused on item calibration than tutoring evaluation.

**Assessment:** Tangential to our needs. IRT alignment is useful but this
doesn't model misconceptions or learning dynamics.

#### SimStudent (Matsuda et al., 2005-2015)

**Citation:** Matsuda, N., Cohen, W.W., & Koedinger, K.R. (2015). "Building
Cognitive Tutors with SimStudent." In R. Sottilare et al. (Eds.), Design
Recommendations for Intelligent Tutoring Systems, Vol. 3.

**What it is:** A machine-learning-based simulated student from Carnegie Mellon
that learns production rules by inductive logic programming. Used to
construct step-based cognitive tutors by having the simulated student learn
from example solutions.

**Repository status:** The original SimStudent code is a Java-based system from
the CTAT/LearnLab ecosystem. The GitHub user "SimStudent" is an unrelated
individual. No current public repository for the original CMU SimStudent
found.

**Assessment:**
- Designed to learn tutoring rules, not to simulate realistic student behavior
- Java-based, tightly coupled to CTAT authoring tools
- No misconception persistence model
- Not maintained (last publications ~2015)
- **Not suitable for our use case** - different purpose entirely

#### pyBKT (CAHLR, UC Berkeley)

**Citation:** Badrinath, A., Wang, F., & Pardos, Z.A. (2021). "pyBKT: An
Accessible Python Library of Bayesian Knowledge Tracing Models." EDM 2021.

**Repository:** https://github.com/CAHLR/pyBKT - MIT license, 249 stars,
actively maintained (last commit: March 2026), v1.4.2.

**What it is:** Production-grade Python BKT implementation with variants:
individual student priors, per-item guess/slip, per-resource learn rates,
forgetting. Includes Roster class for cohort simulation.

**Assessment:**
- Excellent BKT implementation, well-tested, actively maintained
- BUT: models binary mastery (knows/doesn't know), not misconceptions
- No misconception-level state tracking
- No negative transfer
- No instruction-response interface
- **Useful as a dependency** for the BKT component of a new model, but
  cannot serve as the simulated student itself

#### GIFT (U.S. Army Research Lab)

**What it is:** Generalized Intelligent Framework for Tutoring. A large
Java enterprise system for authoring and delivering ITSs.

**Repository status:** The GIFT system is available through the Army Research
Lab but is not a simple open-source library. No GitHub organization found.

**Assessment:**
- Enterprise-scale ITS authoring platform
- Not a simulated student model
- Not relevant to our needs

#### Sonkar et al. (2023) - Novice Learner and Expert Tutor

**Citation:** Liu, N., Sonkar, S., Wang, Z., Woodhead, S., & Baraniuk, R.G.
(2023). "Novice Learner and Expert Tutor: Evaluating Math Reasoning Abilities
of Large Language Models with Misconceptions." arXiv:2310.02439.

**Assessment:** Evaluative paper showing LLMs struggle to produce incorrect
answers from specific misconceptions. Confirms the difficulty of the simulated
student task. No framework released.

### 1.2 ITS Literature: Student Modeling Approaches

#### Knowledge Tracing Variants

| Model | Misconception State? | Learning Dynamics? | Notes |
|---|---|---|---|
| BKT (Corbett & Anderson, 1995) | No - binary knows/doesn't-know | Yes (p_learn) | Our current model; insufficient |
| DKT (Piech et al., 2015) | No - latent embedding only | Yes (implicit) | RNN-based; opaque internal state |
| DKVMN (Zhang et al., 2017) | Partial - concept-level memory | Yes | Dynamic key-value memory; could store misconception state |
| AKT (Ghosh et al., 2020) | No | Yes | Attention-based; no misconception primitives |
| simpleKT (Liu et al., 2023) | No | Yes | Simplified transformer KT |
| SAINT (Choi et al., 2020) | No | Yes | Sequence-to-sequence KT |

**Verdict:** No knowledge tracing model tracks per-misconception state or
models negative transfer from incorrect instruction. They model
"knows/doesn't know" per skill, not "holds misconception X which requires
targeted remediation Y."

#### Misconception-Aware Models

| Approach | Era | Misconception Model | Negative Transfer? |
|---|---|---|---|
| BUGGY (Brown & Burton, 1978) | 1978 | Procedural bugs as production rules | No - static bugs, no learning |
| Repair Theory (VanLehn, 1990) | 1990 | Bug generation from incomplete knowledge | No - generative, not responsive |
| Sleeman's diagnostic models (1982) | 1982 | Mal-rules for algebra | No - diagnostic, not simulative |
| Matz (1982) | 1982 | Extrapolation/overgeneralization bugs | No - theory, no simulation |
| MalAlgoPy/CSMs (Sonkar, 2024) | 2024 | LLM-embedded misconceptions | No |
| BEAGLE (Wang, 2026) | 2026 | BKT + flaw injection | Partial (prevents self-correction) |

**Verdict:** The procedural bug tradition (BUGGY, Repair Theory) models
misconceptions as stable production rules, which is the right cognitive
primitive. But these systems are 30-40 years old, have no open-source
implementations, and don't model learning dynamics (how misconceptions
resolve through instruction). BEAGLE is the only modern system that combines
BKT with misconception injection, but it's unpublished code for a different
domain.

#### Cognitive Architectures

| Architecture | Misconception Support | Simulated Student Use | Status |
|---|---|---|---|
| ACT-R (Anderson et al.) | Production rules can encode bugs | Used in cognitive tutor research | Lisp/Java; heavy; not practical for simulation |
| Soar | Impasses can model misconceptions | Theoretical | Complex; no educational deployment |
| Cognitive load theory models | Indirect (overload causes errors) | No direct simulation | Framework, not implementation |

**Verdict:** ACT-R is the most theoretically grounded but impractical.
Building a full ACT-R model for algebra misconceptions would take months and
produce something slow and opaque. Not recommended.

#### LLM-Based Simulated Students (2024-2026)

This is the most active research area, with three approaches:

1. **Prompting:** Give an LLM a persona ("you are a struggling algebra student
   with misconception X") and have it generate responses. Scarlatos (2026)
   shows error replication scores of 0.02-0.03 - essentially zero. Even
   with Oracle-leaked behavior summaries, only 0.19.

2. **Fine-tuning (CSMs):** Instruction-tune an LLM on misconception examples
   (Sonkar, 2024). Calibration of correct-to-misconception ratio (as low as
   0.25) helps. But degrades correct-solving ability and requires expensive
   per-misconception-set fine-tuning. Scarlatos's SFT results (error score
   0.05-0.07) confirm fine-tuning helps but is still inadequate.

3. **Neuro-symbolic hybrid (BEAGLE):** Use a symbolic model (semi-Markov +
   BKT + EFI) to control high-level behavior and an LLM for low-level
   code/language generation. Error recurrence of 86.2% vs 7.8% for vanilla.
   Most promising architecturally but requires LLM backbone (Gemini Flash),
   no code released, designed for programming not math.

4. **TutorGym** (Weitekamp et al., 2025, AIED): A testbed for evaluating
   AI agents as tutors and students. Referenced by Scarlatos as evaluating
   temporal error rates of simulated students. Worth investigating for
   evaluation protocol, though details limited in citations.

**Key insight from this literature:** Pure LLM approaches fail because LLMs
are fundamentally competent - they want to solve problems correctly. Making
them reliably wrong in specific, stable ways is an unsolved problem.
Scarlatos's error scores (0.02-0.19) and BEAGLE's ablations (merging
Strategist/Executor drops error recurrence by 21%) both confirm this. The
neuro-symbolic approach (symbolic cognitive model controlling an LLM) is the
emerging consensus. But for our use case - deterministic simulation of
algebra misconceptions - we do not need the LLM at all. We need the
symbolic control without the neural action.

### 1.3 PyPI Package Search

| Search Term | Results |
|---|---|
| `simulated-student` | No relevant packages |
| `simulated-learner` | No relevant packages |
| `its-evaluation` | No relevant packages |
| `cognitive-student-model` | No relevant packages |
| `pyBKT` | pyBKT 1.4.2 - BKT only, no misconceptions |
| `knowledge-tracing` | Various DKT implementations, none with misconception models |

**Verdict:** No pip-installable simulated student framework exists.

---

## 2. Candidate Evaluation Matrix

Scoring: **Yes** = fully meets | **Partial** = partially meets | **No** = does not meet

| Candidate | Misconception Fidelity (Critical) | Discrimination (Critical) | Negative Transfer (High) | Open Source (High) | Domain Flexible (High) | Integration (Med) | Validated (Med) | Maintained (Low) |
|---|---|---|---|---|---|---|---|---|
| MalAlgoPy/CSMs | Yes | Partial (LLM variability) | No | **No (no public code)** | Partial (algebra only) | Low (needs LLM) | Partial | N/A |
| BEAGLE | Yes (flaw injection) | Yes (Turing test passed) | Partial | **No (no public code)** | No (programming only) | Low (needs LLM) | Yes | N/A |
| Scarlatos eval | N/A (eval framework) | N/A | N/A | **No** | N/A | N/A | Yes | N/A |
| SimStudent | No | No | No | **No** | No | Low (Java/CTAT) | Partial | No |
| pyBKT | No (binary only) | No | No | **Yes** | Partial | Med | Yes | Yes |
| GIFT | No (not a student model) | No | No | Partial | Partial | Low (Java) | N/A | Partial |
| ACT-R | Partial (production rules) | Partial | No | Yes | Low | Low (Lisp) | Yes | No |
| BKT variants | No | No | No | Yes | Yes | High | Yes | Varies |
| DKT/DKVMN | No | No | No | Yes | Yes | Med | Yes | Varies |
| BUGGY/Repair Theory | Yes (bugs as rules) | No (static) | No | No | No (arithmetic) | N/A | Yes (1980s) | No |

**No candidate scores "Yes" on both Critical requirements while also having
available code.**

- BEAGLE comes closest on the requirements but has no code and wrong domain
- MalAlgoPy has the right misconception model but no code and no negative transfer
- pyBKT has excellent code quality but lacks misconception primitives entirely
- Everything else fails on at least one Critical dimension

---

## 3. Recommendation: Option B - New Research Path

No preexisting solution meets both Critical requirements (misconception
fidelity and discrimination) while having available, integrable code. A new
simulated student model is required.

### Problem statement

Build a simulated student model that:
1. Maintains per-misconception state (not just per-concept)
2. Produces measurably different learning outcomes under good vs. bad tutoring
3. Models negative transfer from incorrect instruction
4. Accepts arbitrary knowledge graphs (15-50 concepts)
5. Runs deterministically without an LLM (reproducible experiments)
6. Integrates with our `respond()` / `receive_instruction()` pipeline

### Cognitive theory: Misconception-aware BKT with interference

The model synthesizes four traditions:

1. **BKT** (Corbett & Anderson, 1995) for per-concept mastery tracking
2. **Procedural bug theory** (Brown & Burton, 1978; VanLehn, 1990) for stable
   misconceptions as production rules
3. **Interference theory** (proactive and retroactive interference from
   cognitive psychology) for negative transfer when incorrect instruction is
   given
4. **BEAGLE's architectural principles** (Wang et al., 2026): Explicit Flaw
   Injection (gating what knowledge is accessible), observation filtering
   (limiting what student can diagnose), and decoupled instruction evaluation
   (separating targeting accuracy from learning application)

The key innovation over our current model: **learning is conditional on
instruction quality and misconception resolution is gated on targeting
accuracy.** This is the deterministic, non-LLM analog of BEAGLE's
Strategist/Executor split, applied to algebra misconceptions instead of code.

### Architecture

```
MisconceptionState:
    misconception_id: str
    concept_id: str
    p_active: float           # probability misconception fires
    strength: float            # resistance to resolution (0-1)
    confusion_susceptible: bool # can wrong instruction strengthen this?

ConceptState:
    concept_id: str
    p_know: float              # BKT mastery probability
    p_know_stable: float       # mastery that has "consolidated" (resistant to interference)
    exposure_count: int        # total instruction events for this concept

StudentState:
    concepts: dict[str, ConceptState]
    misconceptions: list[MisconceptionState]
    learning_rate_modifier: float  # individual learning speed
    confusion_threshold: float     # how many wrong instructions before confusion
    confusion_count: dict[str, int] # per-concept: count of mismatched instructions
```

### Misconception lifecycle

```
                    ┌─────────────────┐
                    │   DORMANT       │  (p_active < threshold)
                    │   (resolved)    │
                    └────────▲────────┘
                             │ targeted remediation
                             │ (correct misconception ID)
    ┌────────────────────────┤
    │                        │
    │   ┌────────────────────┴────────┐
    │   │   ACTIVE                    │  (p_active > threshold)
    │   │   fires on relevant problems│
    │   └────────────▲───────────────-┘
    │                │ wrong instruction
    │                │ (strengthens misconception)
    │                │
    │   ┌────────────┴────────────────┐
    │   │   REINFORCED                │  (p_active increases)
    │   │   wrong remediation made    │
    │   │   misconception harder to   │
    │   │   resolve                   │
    │   └─────────────────────────────┘
    │
    │   (generic instruction has near-zero effect on misconception state)
    └──────────────────────────────────
```

State transitions:

| Event | Misconception Effect | p_know Effect |
|---|---|---|
| Correct targeted remediation | `p_active *= (1 - resolution_rate)` | `p_know += (1 - p_know) * p_learn * remediation_bonus` |
| Wrong targeted remediation | `p_active *= (1 + reinforcement_rate)` | `p_know += 0` (no learning; confusion) |
| Generic instruction (no targeting) | `p_active *= (1 - generic_decay)` (very small) | `p_know += (1 - p_know) * p_learn * 0.3` (reduced learning) |
| No instruction | No change | No change |
| Wrong concept instruction | No change to this misconception | Other concept gets confused |

### Negative transfer model

Wrong instruction causes harm through three mechanisms:

1. **Misconception reinforcement:** If the tutor says "you have misconception
   X" but the student actually has misconception Y, the instruction for X is
   irrelevant at best. If X and Y are in the same concept, the confused
   instruction can *strengthen* Y (the student interprets the mismatch as
   evidence their existing approach is correct).

2. **Confusion accumulation:** Repeated mismatched instruction on the same
   concept increments a confusion counter. When confusion exceeds a threshold,
   the student's learning rate for that concept drops (modeling "learned
   helplessness" or "I'll never get this").

3. **Interference with correct knowledge:** If the student has partially
   mastered a concept (`p_know > 0.5`) and receives wrong instruction,
   `p_know_stable` does not increase even if `p_know` would have. This models
   the distinction between fragile and consolidated knowledge.

### Validation plan

The validation test is the exact test experiments 07-09 failed:

**Discrimination test:** Run two conditions with 300+ students each:
- Condition A: Perfect classifier (always identifies correct misconception)
- Condition B: Random classifier (picks random misconception or none)

**Pass criteria:**
- Cohen's d >= 0.5 between conditions on test score gain
- Resolution rate in Condition A >= 2x Condition B
- Condition B should show *lower* gains than no-instruction baseline
  (negative transfer from random targeting)

**Sensitivity test:** Run the Experiment 07 protocol (error rates 0-50%).
The new model must show monotonic degradation in gain as error rate increases
(not the flat line our current model produces).

**BKT fidelity test:** Run the Experiment 08 protocol. Concept selection
accuracy should be meaningfully above random (target: >50% vs oracle) and
BKT parameter perturbation should produce measurable gain changes.

### Implementation plan

| File | Purpose |
|---|---|
| `src/simulated_student_v3.py` | New student model with `ConceptState`, `MisconceptionState` |
| `src/knowledge_graph_v2.py` | Extended KG with 20+ concepts, branching prerequisites |
| `data/knowledge_graph_v2.json` | Expanded algebra KG (20 concepts, ~60 misconceptions) |
| `data/problem_bank_v2.json` | Expanded problem bank (10+ per concept) |
| `tests/test_discrimination.py` | Automated discrimination test (must pass before merge) |
| `tests/test_negative_transfer.py` | Verify wrong instruction hurts |
| `tests/test_misconception_lifecycle.py` | Verify resolution, reinforcement, reactivation |
| `experiments/10_v3_discrimination/run.py` | Full discrimination experiment |
| `experiments/11_v3_error_propagation/run.py` | Re-run Exp 07 with new model |

### Separate repository recommendation

This model is **not novel enough** to warrant an independent research
publication or separate repository. It is an engineering synthesis of
well-established cognitive primitives (BKT + procedural bugs + interference
theory) applied to a specific problem. It should be built in the main `ed`
repo under a clear versioning scheme (`simulated_student_v3`).

However, if during implementation the interference/confusion model proves
to have broader applicability or produces surprising results that warrant
controlled experimentation, it could be extracted into a standalone package
at that point.

---

## 4. Next Steps

1. **Read these papers first** (in priority order):
   - Scarlatos et al. (2026): "Simulated Students in Tutoring Dialogues" -
     [arXiv:2601.04025](https://arxiv.org/abs/2601.04025) - evaluation
     framework and why prompting fails
   - Sonkar et al. (2024): "LLM-based Cognitive Models of Students with
     Misconceptions" - [arXiv:2410.12294](https://arxiv.org/abs/2410.12294) -
     MalAlgoPy and the CSM approach
   - Wang et al. (2026): "BEAGLE" -
     [arXiv:2602.13280](https://arxiv.org/abs/2602.13280) - neuro-symbolic
     architecture and BKT with flaw injection
   - VanLehn (1990): "Mind Bugs" (book) - Repair Theory for procedural bugs

2. **Build the expanded knowledge graph** - 20 concepts, branching
   prerequisites, 60+ misconceptions. This is needed before the student
   model because the model's discriminating power depends on a non-trivial
   routing problem. Can be done in parallel with the student model.

3. **Implement `simulated_student_v3.py`** - follow the architecture above.
   Start with the discrimination test as a TDD anchor: write the test first,
   then build the model until it passes.

4. **Re-run experiments 07-09** with the new model. These become experiments
   10-12. If the new model shows proper degradation curves (monotonic gain
   decrease with error rate increase), the simulation is validated.

5. **Then proceed to Phase 0** of the Agentic Roadmap. The simulated student
   is a prerequisite for evaluating anything the agent does.

---

## Appendix: Papers and Resources

### Essential reading

| Paper | Year | Relevance |
|---|---|---|
| Scarlatos et al., "Simulated Students in Tutoring Dialogues" | 2026 | Evaluation framework; confirms prompting fails |
| Wang et al., "BEAGLE" | 2026 | Neuro-symbolic architecture template |
| Sonkar et al., "LLM-based Cognitive Models" | 2024 | MalAlgoPy; algebra misconception taxonomy |
| Liu et al., "Novice Learner and Expert Tutor" | 2023 | LLMs struggle with misconception simulation |
| Corbett & Anderson, "Knowledge Tracing" | 1995 | BKT foundation |
| Brown & Burton, "Diagnostic Models for Procedural Bugs" | 1978 | BUGGY; procedural bug paradigm |
| VanLehn, "Mind Bugs" | 1990 | Repair Theory; misconception generation |

### Useful tools

| Tool | URL | Use For |
|---|---|---|
| pyBKT | https://github.com/CAHLR/pyBKT | Reference BKT implementation; potential dependency |
| Eedi Misconception Dataset | Kaggle "Eedi MAP" competition | Real misconception taxonomy for validation |
| Eedi QA Tutoring Dialogues 2k | Used by Scarlatos (2026) | 1,529 real math tutoring dialogues for evaluation |
| TutorGym | Weitekamp et al., AIED 2025 | Testbed for evaluating AI tutors and simulated students |

### Not recommended

| System | Reason |
|---|---|
| SimStudent (CMU) | Wrong purpose (tutor authoring, not student simulation); dead project |
| GIFT (Army Research Lab) | Enterprise ITS platform, not a student model |
| ACT-R | Too heavy; Lisp-based; months of work for marginal benefit |
| Pure LLM prompting | Scarlatos (2026) demonstrated this doesn't work reliably |
