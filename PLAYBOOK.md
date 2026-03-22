---
title: "AI Algebra Tutor: Implementation Playbook"
description: "Phase-by-phase execution guide for building an LLM-based algebra misconception detection and adaptive tutoring system. Designed as both an AI-promptable workflow and a human-executable checklist."
ms.date: 2026-03-22
---

<!-- markdownlint-disable-file -->

## Purpose

This document rewrites the original research proposal (IRB.md) into an operational playbook. Each phase includes:

- Concrete deliverables with acceptance criteria
- AI-promptable task blocks you can hand directly to an LLM to execute
- Manual evaluation checklists so you can verify the output yourself
- Decision gates that determine whether to proceed or iterate

The system under construction has three components:

1. A knowledge graph of algebra concepts with per-student mastery probabilities
2. A misconception classifier (LLM with LoRA fine-tuning) that reads student free-text answers and predicts error categories
3. An adaptive engine that updates mastery scores and selects the next problem or hint

## Phase 1: Problem Scoping (2 weeks)

### What gets done

Narrow the project to 3-5 algebra concepts and define the exact misconception categories the system will detect. This is the foundation: everything downstream (data, model, evaluation) depends on a tight, well-defined scope.

### AI task block

> You are an expert in K-12 mathematics education and algebra pedagogy. Given the MaE dataset (55 misconceptions across middle-school algebra) and the goal of building a misconception-detection AI tutor scoped to a single RTX 3070 GPU:
>
> 1. Review the 55 misconception categories in the MaE dataset (Otero et al., 2024, arXiv:2412.03765). Group them by algebra subdomain.
> 2. Select 3-5 concepts that (a) are high-frequency in classrooms (cited by >60% of teachers in the MaE survey), (b) have enough labeled examples in MaE to support fine-tuning, and (c) form a connected subgraph (so the adaptive engine has meaningful transitions between them).
> 3. For each selected concept, list: the concept name, its MaE misconception IDs, 2-3 example student errors (with the correct solution for contrast), and the prerequisite relationships between concepts.
> 4. Output a JSON schema for the knowledge graph: nodes (concept ID, name, prerequisite IDs, associated misconception IDs) and edges (prerequisite relationships).
> 5. Define exit criteria: what does "mastery" of each concept look like? Propose a threshold (e.g., 3 consecutive correct answers, or mastery probability > 0.8).

### Manual evaluation checklist

- [ ] The selected concepts are genuinely high-frequency (cross-reference with the MaE teacher survey data: 80% of teachers report encountering these)
- [ ] Each concept has at least 10-15 labeled examples in MaE (or a clear augmentation plan if fewer)
- [ ] The concepts form a connected graph with clear prerequisite ordering (e.g., distribution before combining like terms before solving linear equations)
- [ ] The misconception categories are mutually exclusive enough that a classifier can distinguish them (no two categories describe the same error in different words)
- [ ] The JSON schema is complete and parseable

### Decision gate

Proceed to Phase 2 when you have a finalized concept list, knowledge graph schema, and at least one domain expert (or strong literature backing) confirming the selections make pedagogical sense.

## Phase 2: Literature Review (3 weeks, parallel with Phase 1)

### What gets done

Build a comprehensive, annotated bibliography covering intelligent tutoring systems, NLP-based misconception detection, knowledge tracing, and LLM fine-tuning for education. Identify the specific gap this project fills.

### AI task block

> You are a research assistant specializing in AI for education. Produce an annotated bibliography for a project building an LLM-based algebra misconception detector. For each source:
>
> 1. Provide the full citation (authors, year, title, venue).
> 2. Write a 2-3 sentence summary of the method and findings.
> 3. State what this project can reuse or build on from that work.
> 4. State the limitation or gap this work leaves open.
>
> Cover these categories (minimum 3 sources each):
>
> - Intelligent Tutoring Systems: Cognitive Tutor (Koedinger et al.), AutoTutor, ASSISTments. Focus on how they model student knowledge and adapt.
> - Misconception Detection with NLP: Michalenko et al. (2017), any work from LAK/EDM conferences on open-response analysis. Focus on classification approaches and accuracy.
> - Math Misconception Benchmarks: MaE dataset (Otero et al., 2024), Eedi/Kaggle competition, MATH benchmark. Focus on data availability and evaluation methodology.
> - Knowledge Tracing: Bayesian Knowledge Tracing (Corbett & Anderson, 1995), Deep Knowledge Tracing (Piech et al., 2015). Focus on mastery probability update mechanisms.
> - LLM Fine-Tuning on Small Data: LoRA (Hu et al., 2021), QLoRA, adapter methods. Focus on what's achievable with 8GB VRAM and <1000 training examples.
>
> Conclude with a "Gap Statement" paragraph: what does no existing system do that this project will attempt?

### Manual evaluation checklist

- [ ] Each category has at least 3 substantive sources (not blog posts or press releases)
- [ ] The summaries are accurate (spot-check 3-4 papers by reading their abstracts)
- [ ] The gap statement is defensible: it describes something genuinely unaddressed, not a strawman
- [ ] The bibliography includes at least one source from the last 2 years (2024-2026)
- [ ] LLM fine-tuning sources specifically address low-data regimes (<1000 examples) and consumer GPU constraints

### Decision gate

Proceed when the bibliography is complete, the gap statement is clear, and you can articulate in one sentence what this project does that prior work does not.

## Phase 3: Dataset Strategy (4 weeks)

### What gets done

Assemble a labeled dataset of student algebra answers tagged by misconception category. Combine existing data (MaE) with synthetic augmentation. Produce train/validation/test splits.

### AI task block: data acquisition

> You are a data engineer for an education AI project. The target dataset maps student free-text algebra answers to misconception labels. Perform these steps:
>
> 1. Download and parse the MaE dataset (Hugging Face: the "Math Misconceptions and Errors" dataset). Report: total examples, distribution across misconception categories, format (fields per example).
> 2. Filter to the 3-5 concepts selected in Phase 1. Report how many labeled examples remain per concept.
> 3. Identify any class imbalance. If any concept has fewer than 50 examples, flag it for synthetic augmentation.

### AI task block: synthetic data generation

> You are an algebra teacher creating realistic wrong answers to algebra problems. For each problem-misconception pair below, generate 10 distinct student responses that exhibit the specified misconception. Requirements:
>
> - Vary the language: some students write equations only, some explain in words, some mix both
> - Vary the sophistication: some responses are terse ("x = 4"), some show partial work
> - Include plausible but incorrect reasoning, not random garbage
> - Label each generated response with: problem ID, misconception ID, the incorrect answer, and a 1-sentence explanation of what the student likely thought
>
> [Insert problem-misconception pairs from Phase 1 output here]

### AI task block: data validation

> You are a quality assurance reviewer for an education dataset. Given these synthetic student responses [insert batch], verify:
>
> 1. Does each response genuinely exhibit the labeled misconception? (not a different error or a correct answer)
> 2. Is the language realistic for a high-school student? (not overly formal or nonsensical)
> 3. Are there any duplicates or near-duplicates?
> 4. Rate each response: "valid", "borderline", or "reject". Provide a 1-sentence reason for borderline/reject.

### Manual evaluation checklist

- [ ] The final dataset has at least 100 examples per target concept (combining MaE + synthetic)
- [ ] Train/validation/test splits are stratified by concept and misconception label (e.g., 70/15/15)
- [ ] No data leakage: identical or near-identical examples do not appear in both train and test
- [ ] At least 20% of synthetic examples have been manually reviewed by a human and confirmed as realistic
- [ ] The dataset is saved in a structured format (JSON or CSV) with clear column definitions
- [ ] A dataset card documenting source, size, label distribution, and generation method exists

### Decision gate

Proceed when you have a clean, labeled dataset with sufficient examples per class, validated splits, and a documented generation methodology.

## Phase 4: Prototype Building (6 weeks)

This is the largest phase. Break it into three parallel workstreams.

### Workstream A: Misconception Classifier

#### AI task block: model selection and setup

> You are an ML engineer setting up a text classification system on a single RTX 3070 (8GB VRAM). The task: classify student algebra responses into misconception categories (3-5 classes plus "correct" and "other/unknown").
>
> 1. Evaluate these model options for feasibility on 8GB VRAM:
>    - LLaMA-2-7B with 4-bit quantization + LoRA (via bitsandbytes + peft)
>    - Mistral-7B with 4-bit quantization + LoRA
>    - FLAN-T5-XL (3B parameters, may fit in 8-bit)
>    - FLAN-T5-Large (780M, fits comfortably)
>    - A fine-tuned BERT/RoBERTa baseline (for comparison)
> 2. For each: estimate VRAM usage during training (batch size 1-4 with gradient checkpointing) and inference. Flag any that won't fit.
> 3. Recommend the primary model and a lighter fallback.
> 4. Write the training script skeleton: data loading, tokenization, LoRA config, training loop with evaluation, checkpoint saving. Use HuggingFace Transformers + peft + bitsandbytes.

#### AI task block: training and tuning

> You are training a misconception classifier. Given:
> - Model: [selected model from above]
> - Dataset: [path to train/val splits from Phase 3]
> - Task: multi-class classification (input: student response text, output: misconception label)
>
> 1. Implement two classification approaches:
>    a. Prompt-based: format each example as "Student answer: {text}\nQuestion topic: {topic}\nMisconception:" and train the model to generate the label
>    b. Head-based: add a classification head on top of the LLM's last hidden state
> 2. Train both for 3-5 epochs on the training set. Log training loss and validation accuracy/F1 per epoch.
> 3. Report: which approach performs better on validation? What is the per-class F1?
> 4. Run the "topic ablation": train once with topic metadata included in the prompt, once without. Report the accuracy difference.

#### Manual evaluation checklist

- [ ] The training script runs end-to-end without OOM errors on the RTX 3070
- [ ] Validation F1 is at least 70% (minimum viable) or 80% (target)
- [ ] The topic-constrained model outperforms the unconstrained model (confirming Otero et al.'s finding)
- [ ] The model checkpoints are saved and reproducible (fixed random seeds, logged hyperparameters)
- [ ] At least one baseline (BERT or logistic regression on TF-IDF) has been trained and compared

### Workstream B: Knowledge Graph and Mastery Model

#### AI task block

> You are building a student knowledge model for an adaptive tutor. Implement in Python:
>
> 1. A `KnowledgeGraph` class that loads from the JSON schema (Phase 1 output). Nodes are concepts, edges are prerequisites.
> 2. A `StudentState` class that tracks per-concept mastery probability (initialized at 0.5). Implement Bayesian Knowledge Tracing updates:
>    - On correct answer: P(mastery) increases (use standard BKT parameters: P(learn)=0.1, P(guess)=0.2, P(slip)=0.1, or make them configurable)
>    - On misconception detected: P(mastery) for the specific concept decreases proportionally to the classifier's confidence
> 3. A `next_action(state, graph)` function that returns:
>    - "remediate: {concept}" if any concept's mastery is below threshold and has been attempted
>    - "progress: {next_concept}" if current concepts are mastered and a prerequisite-unlocked concept exists
>    - "review: {concept}" if no new concepts are available but some are borderline
> 4. Unit tests covering: mastery increases on correct, decreases on error, prerequisite gating works, remediation triggers at threshold.

#### Manual evaluation checklist

- [ ] BKT update math is correct (verify with a hand-calculated example: 3 correct answers in a row should push mastery above 0.8)
- [ ] Prerequisite gating works (a student cannot be assigned a concept whose prerequisites are unmastered)
- [ ] The `next_action` function handles edge cases: all concepts mastered, no concepts attempted yet, only one concept in the graph
- [ ] Unit tests pass

### Workstream C: Integration and Interface

#### AI task block

> You are building the integration layer and a minimal interface. Implement:
>
> 1. A `TutorSession` class that ties together the classifier, knowledge graph, and student state:
>    - `present_problem()`: selects a problem for the current target concept
>    - `evaluate_response(text)`: runs the classifier, updates mastery, returns feedback
>    - `get_hint(concept, misconception)`: generates a targeted hint using the LLM (prompt: "The student made this error: {misconception}. Give a short, encouraging hint that addresses this specific misunderstanding without giving the answer.")
>    - `session_summary()`: returns current mastery state across all concepts
> 2. A CLI interface (or Jupyter notebook) where a user can:
>    - See a problem
>    - Type an answer
>    - Receive feedback (correct/incorrect + specific misconception diagnosis + hint)
>    - See their mastery state update
>    - Get the next recommended problem
> 3. A problem bank: at minimum, 5 problems per target concept, stored in JSON with fields: problem_id, concept, difficulty, problem_text, correct_answer.

#### Manual evaluation checklist

- [ ] A full session loop works end-to-end: problem shown, answer entered, misconception detected, mastery updated, next problem selected
- [ ] Hints are specific to the detected misconception (not generic "try again")
- [ ] The problem bank covers all target concepts with at least 5 problems each
- [ ] Session state persists across problems within one session

### Phase 4 decision gate

Proceed when: the classifier meets minimum accuracy on validation data, the knowledge graph and mastery model pass unit tests, and a full tutoring loop runs end-to-end in the CLI/notebook without errors.

## Phase 5: Offline Evaluation (4 weeks)

### What gets done

Rigorous quantitative evaluation of the classifier and the full adaptive system on held-out data and simulated student runs.

### AI task block: classification evaluation

> You are evaluating a misconception classifier. Given the test set [path] and the trained model [path]:
>
> 1. Run inference on the full test set. Compute:
>    - Overall accuracy
>    - Per-class precision, recall, F1
>    - Confusion matrix
>    - Macro and weighted F1
> 2. Run ablation experiments:
>    - With topic metadata vs. without
>    - LoRA fine-tuned vs. zero-shot (same base model, no adapter)
>    - Single-turn vs. multi-turn (allow one follow-up "Can you explain your reasoning?" prompt)
> 3. Compare against baselines:
>    - Logistic regression on TF-IDF features
>    - BERT-base fine-tuned
>    - Keyword/regex heuristic (define 5-10 patterns per misconception)
>    - Random/majority-class baseline
> 4. Error analysis: for each misclassified example, report the predicted vs. true label and the student response text. Identify patterns: are errors concentrated in specific misconceptions? Short vs. long responses? Ambiguous cases?

### AI task block: simulated student evaluation

> You are evaluating the adaptive tutoring loop. Create a simulation:
>
> 1. Define 5 "student profiles" with different initial mastery states:
>    - Profile A: strong overall, one weak concept (distribution)
>    - Profile B: weak overall, all concepts below 0.5
>    - Profile C: mixed, strong on arithmetic but weak on algebraic concepts
>    - Profile D: strong everywhere (ceiling test)
>    - Profile E: random noise (some responses correct, some with random misconceptions)
> 2. For each profile, simulate 20 interaction rounds. At each round:
>    - The system selects a problem
>    - The simulated student answers based on their profile (correct for mastered concepts, with the appropriate misconception for unmastered ones)
>    - The system classifies, updates mastery, and selects the next action
> 3. Measure:
>    - Concept identification accuracy: did the system correctly identify which concepts were weak?
>    - Remediation targeting: what percentage of remediation problems targeted the genuinely weak concept?
>    - Mastery trajectory: graph the mastery probability over time for each concept per profile
>    - Convergence: how many rounds until the system's mastery estimates stabilize?
> 4. Compare: adaptive system vs. random problem selection vs. fixed sequence.

### Manual evaluation checklist

- [ ] Classification accuracy meets or exceeds 70% overall (minimum), 80% (target)
- [ ] The topic-ablation confirms a significant accuracy boost with topic metadata (>5 percentage points)
- [ ] The fine-tuned model outperforms all baselines
- [ ] Error analysis reveals interpretable patterns (not random failures)
- [ ] Simulated students with known weak concepts receive remediation targeting those concepts >80% of the time
- [ ] Mastery trajectories look reasonable (monotonically improving for consistent students, noisy for Profile E)
- [ ] All results are logged in a reproducible evaluation script with fixed random seeds

### Decision gate

Proceed to Phase 6 if classification accuracy is above 70% and the adaptive loop correctly identifies weak concepts in simulation. If accuracy is below 70%, return to Phase 4 to iterate on the model (more data, different architecture, better prompts). If the adaptive loop misbehaves despite good classification, debug the mastery update logic.

## Phase 6: Human Evaluation and IRB (6 weeks)

### What gets done

Design and execute a small pilot study with real human participants. This requires IRB approval first.

### AI task block: IRB protocol draft

> You are writing an IRB protocol for a minimal-risk educational study. Draft the following sections:
>
> **Study Title**: "AI-Supported Algebra Tutoring: A Pilot Study of Misconception Detection and Adaptive Practice"
>
> **Study Design**:
> - Between-subjects, two conditions: AI-feedback group (misconception-specific hints) vs. control group (generic correct/incorrect feedback)
> - Participants: 20-40 adults (18+) recruited from a university participant pool or online
> - Procedure: consent form, 5-minute pre-test (10 algebra problems), 20-minute learning session (solve problems with system feedback), 5-minute post-test (10 parallel problems), 5-minute survey
> - Primary outcome: pre-to-post score improvement
> - Secondary outcomes: survey ratings of feedback helpfulness (5-point Likert), time per problem
>
> **Draft these documents**:
> 1. Informed consent form (plain language, covers: purpose, procedure, risks, benefits, confidentiality, voluntary participation, contact info)
> 2. Pre-test and post-test problem sets (10 problems each, matched difficulty, covering the target concepts)
> 3. Post-session survey (10 questions: feedback clarity, perceived learning, system usability, willingness to use again)
> 4. Data management plan (how data is collected, stored, de-identified, retained, and destroyed)
>
> **Risk Assessment**: minimal risk (solving algebra problems, no deception, no sensitive data). Possible frustration from difficult problems, mitigated by encouraging framing and the ability to skip.

### AI task block: study execution plan

> Given IRB approval, outline the execution logistics:
>
> 1. Recruitment plan: where to recruit (university listservs, online platforms), inclusion criteria (18+, English-speaking, not a math major), target N=30, compensation ($10 gift card or course credit)
> 2. Randomization: how to assign participants to conditions (block randomization, ensuring balanced groups)
> 3. Session protocol: step-by-step script for each session (what the participant sees, timing, data logged at each step)
> 4. Data collection: what gets logged automatically (responses, timestamps, classifier outputs, mastery states) vs. manually (survey responses)
> 5. Analysis plan:
>    - Primary: paired t-test or Wilcoxon signed-rank on pre-post improvement, between-group comparison with independent t-test or Mann-Whitney
>    - Secondary: descriptive statistics on survey ratings, qualitative coding of open-ended feedback
>    - Effect size reporting (Cohen's d)
>    - Power analysis: given N=30, what effect size can we detect at alpha=0.05, power=0.8?

### Manual evaluation checklist

- [ ] The IRB protocol is complete and addresses all required sections for your institution
- [ ] Consent form is written in plain language (8th-grade reading level)
- [ ] Pre-test and post-test are matched in difficulty and cover the same concepts
- [ ] The study design can detect a meaningful effect (the power analysis confirms this)
- [ ] Data management plan specifies de-identification before analysis, secure storage, and a destruction timeline
- [ ] A pilot run with 2-3 friendly volunteers has been conducted to catch usability issues before the real study

### Decision gate

Proceed to Phase 7 when you have IRB approval and either (a) completed data collection, or (b) decided to publish with simulated-only results and note the human study as future work.

## Phase 7: Writeup and Publication (4 weeks)

### What gets done

Produce a conference-quality paper or technical report documenting the system, experiments, and findings.

### AI task block: paper structure

> You are drafting a research paper for submission to AIED (AI in Education) or EDM (Educational Data Mining). Generate the full structure with section-by-section guidance:
>
> 1. Abstract (150 words): problem, approach, key result, implication
> 2. Introduction: motivate the problem (algebra misconceptions are common, current tutors don't analyze free-text), state the contribution (an LLM-based system that detects misconceptions and adapts), preview results
> 3. Related Work: subsections for ITS history, NLP misconception detection, knowledge tracing, LLM fine-tuning. Position this work in the gap.
> 4. System Design: knowledge graph, classifier architecture, adaptive engine. Include a system diagram.
> 5. Experimental Setup: dataset (MaE + synthetic), models compared, evaluation metrics, simulated student profiles, human study design (if applicable)
> 6. Results: classification performance table, ablation results, simulation outcomes, human study outcomes (if applicable)
> 7. Discussion: what worked, what didn't, limitations (small data, simulated vs. real, single GPU constraints), implications for practice
> 8. Conclusion and Future Work
>
> For each section, write a 2-3 sentence description of what goes there and flag which Phase's outputs provide the content.

### AI task block: figures and tables

> Generate specifications for the key figures and tables:
>
> 1. System architecture diagram (box-and-arrow: student input -> classifier -> misconception label -> mastery update -> next action)
> 2. Classification results table (model x metric matrix)
> 3. Confusion matrix heatmap
> 4. Ablation results table (with/without topic, with/without LoRA, single/multi-turn)
> 5. Mastery trajectory plots (one per simulated student profile, mastery probability over interaction rounds)
> 6. Human study results (if applicable): bar chart of pre-post improvement by condition, survey rating distributions

### Manual evaluation checklist

- [ ] The paper tells a coherent story: problem, approach, evidence, conclusion
- [ ] All claims are supported by data from the experiments
- [ ] Limitations are honestly stated (not buried or minimized)
- [ ] The paper meets the target venue's formatting requirements (page limit, citation style)
- [ ] All figures are readable and all tables have clear headers
- [ ] A colleague or advisor has reviewed a draft and provided feedback

### Decision gate

Submit when the paper has been reviewed internally and revised at least once.

## Quick Reference: Reduced Scope Fallback

If time, compute, or data constraints force scope reduction at any point, here is the minimum viable version:

- One concept only: distributive property in linear equations
- Data: MaE examples for distribution misconceptions + 50 synthetic examples
- Model: FLAN-T5-Large (780M, fits easily on 3070) or even zero-shot prompting of a quantized 7B model
- Evaluation: offline classification accuracy on held-out examples only (no simulation, no human study)
- Deliverable: a technical report showing the classifier's performance vs. baselines on this single concept

This fallback still produces a publishable contribution if the accuracy is strong and the error analysis is insightful.

## Prompt Chaining Guide

For using these task blocks with an AI assistant, follow this sequence. Each step feeds its output into the next.

1. Run Phase 1 AI task block. Save the output (concept list + JSON schema) to `data/knowledge_graph.json`.
2. Run Phase 2 AI task block. Save the annotated bibliography to `docs/literature_review.md`.
3. Run Phase 3 acquisition block, then generation block (passing Phase 1 concepts), then validation block (passing generated data). Save the final dataset to `data/dataset/`.
4. Run Phase 4 workstream blocks in parallel (A, B, C all take Phase 1 and Phase 3 outputs). Save code to `src/`.
5. Run Phase 5 evaluation blocks (they take Phase 4 model and Phase 3 test set). Save results to `results/`.
6. Run Phase 6 blocks if pursuing human evaluation. Save IRB documents to `docs/irb/`.
7. Run Phase 7 blocks (they reference all prior outputs). Save the paper draft to `docs/paper/`.

Each AI task block is self-contained: copy it, fill in the bracketed placeholders with outputs from prior phases, and execute.
