# Annotated Bibliography: AI Algebra Tutoring Project

---

## Category 1: Intelligent Tutoring Systems

### 1.1 Koedinger & Corbett (2006) - Cognitive Tutors

**Citation:** Koedinger, K.R. & Corbett, A.T. (2006). Cognitive Tutors: Technology bringing learning science to the classroom. In R.K. Sawyer (Ed.), *The Cambridge Handbook of the Learning Sciences* (pp. 61-78). Cambridge University Press.

**Summary:** This chapter introduces Cognitive Tutors, grounded in the ACT-R cognitive architecture. The system uses two core mechanisms - *model tracing* (comparing each student step against a cognitive model of correct and buggy production rules) and *knowledge tracing* (Bayesian estimation of whether each knowledge component has been mastered to a 95% threshold). Across multiple school deployments, the Algebra Cognitive Tutor led students to outperform traditionally-taught peers on standardized tests and real-world transfer problems, with benefits particularly pronounced for special-education, ESL, and low-income students.

**Reusable for our project:** The knowledge-tracing (BKT) math and the "skillometer" concept map directly inform our mastery model. The production-rule decomposition of algebra skills into individual knowledge components provides a validated blueprint for our skill graph.

**Limitation/gap:** Cognitive Tutors are rigid rule-based systems: authoring new production rules requires an estimated 200:1 development-to-instruction-time ratio. They cannot handle free-form natural language input, only stepwise equation manipulation, leaving open-response misconception detection unaddressed.

---

### 1.2 VanLehn (2011) - Relative Effectiveness of Tutoring Systems

**Citation:** VanLehn, K. (2011). The relative effectiveness of human tutoring, intelligent tutoring systems, and other tutoring systems. *Educational Psychologist*, 46(4), 197-221. DOI: 10.1080/00461520.2011.611369.

**Summary:** This landmark meta-review categorizes tutoring systems by *interaction granularity*: answer-based (CAI), step-based (e.g., Cognitive Tutor), and substep-based (e.g., Andes). Analyzing effect sizes across dozens of controlled experiments, VanLehn finds that step-based ITS achieve an effect size of d = 0.76, statistically indistinguishable from expert human one-on-one tutoring (d = 0.79). Answer-based systems produce significantly lower gains (d = 0.31). The key finding is that interaction granularity - not the mere presence of AI - determines effectiveness.

**Reusable for our project:** Provides the theoretical justification for building a step-level algebra tutor rather than answer-level only. The granularity taxonomy (answer vs. step vs. substep) directly informs our design decision to capture and evaluate intermediate algebraic steps.

**Limitation/gap:** The review predates LLM-based approaches and does not consider natural language dialogue as a form of step-level interaction. It does not address open-response or text-based misconception detection, leaving open whether NLP-based feedback can match traditional step-tracing effectiveness.

---

### 1.3 Graesser et al. (2005) - AutoTutor

**Citation:** Graesser, A.C., Chipman, P., Haynes, B.C. & Olney, A. (2005). AutoTutor: An intelligent tutoring system with mixed-initiative dialogue. *IEEE Transactions on Education*, 48(4), 612-618. DOI: 10.1109/TE.2005.856149.

**Summary:** AutoTutor simulates a human tutor through natural language conversation, augmented by an animated pedagogical agent and 3D interactive simulations. Grounded in constructivist learning theory and tutoring discourse research, the system uses latent semantic analysis (LSA) and dialogue management to maintain mixed-initiative conversations. AutoTutor achieves learning gains of approximately 0.8 sigma (~one letter grade) compared to re-reading a textbook, primarily in conceptual physics and computer literacy domains.

**Reusable for our project:** The mixed-initiative dialogue framework - where the tutor prompts, students explain, and the system provides pumps, hints, and corrections - directly informs our conversational scaffold design. The use of LSA for comparing student explanations against ideal answers is a precursor to our planned embedding-based response matching.

**Limitation/gap:** AutoTutor's NLP pipeline (LSA-based, circa 2005) is far less capable than modern LLMs for understanding algebraic expressions or detecting specific computational errors in free-form math responses. The system was designed for conceptual explanations, not step-by-step algebraic problem solving.

---

### 1.4 Heffernan & Heffernan (2014) - ASSISTments

**Citation:** Heffernan, N.T. & Heffernan, C.L. (2014). The ASSISTments ecosystem: Building a platform that brings scientists and teachers together for minimally invasive research on human learning and teaching. *International Journal of Artificial Intelligence in Education*, 24(4), 470-497. DOI: 10.1007/s40593-014-0024-x.

**Summary:** ASSISTments is a web-based platform that simultaneously tutors and assesses students ("assist + assessment"). Originally a model-tracing tutor, the system evolved into an ecosystem supporting randomized A/B experiments within normal homework assignments. It provides immediate feedback, scaffolded hints, and teacher-facing reporting dashboards. Large-scale evaluations with thousands of students demonstrated that ASSISTments use predicted state standardized test scores better than prior test scores alone, and the platform produces meaningful learning gains in middle-school math.

**Reusable for our project:** The "dual-purpose" design philosophy (assessment *and* tutoring simultaneously) maps to our project's need to both diagnose misconceptions and remediate them. The teacher-dashboard and A/B experimentation infrastructure provide a model for how to build research capabilities into a production tutoring system.

**Limitation/gap:** ASSISTments relies primarily on structured, close-ended question formats (fill-in-the-blank, multiple-choice). Open-ended response analysis is limited, and the system does not use NLP for misconception detection in free-form student work - a central goal of our project.

---

### 1.5 Hooshyar et al. (2025) - LLMs Alone Fall Short for Learner Modelling

**Citation:** Hooshyar, D., Yang, Y., Sir, G., Karkkainen, T., Hamalainen, R. & Cukurova, M. (2025). Problems with large language models for learner modelling: Why LLMs alone fall short for responsible tutoring in K-12 education. *arXiv preprint arXiv:2512.23036*.

**Summary:** This study empirically compares deep knowledge tracing (DKT) against an LLM (zero-shot and fine-tuned) for next-step correctness prediction. DKT achieves AUC = 0.83 and consistently outperforms the LLM. Even after fine-tuning (~198 hours of compute), the LLM remains 6% below DKT and produces higher early-sequence errors. Temporal analysis reveals that DKT maintains stable, directionally correct mastery updates, while LLM variants exhibit inconsistent and wrong-direction updates. The authors argue that responsible K-12 tutoring requires hybrid frameworks combining LLMs with established learner modeling.

**Reusable for our project:** Provides strong empirical evidence for our hybrid architecture - using BKT/DKT for mastery tracking while using LLMs for natural language understanding and feedback generation. The specific failure modes documented (wrong-direction mastery updates, early-sequence errors) are pitfalls to explicitly guard against.

**Limitation/gap:** Focuses on correctness prediction as a proxy for learner modeling, not on misconception *classification* or *diagnosis*. Does not explore how LLMs might complement (rather than replace) knowledge tracing through qualitative analysis of student work.

---

## Category 2: Misconception Detection with NLP

### 2.1 Michalenko, Lan & Baraniuk (2017) - Data-Mining Textual Responses

**Citation:** Michalenko, J.J., Lan, A.S. & Baraniuk, R.G. (2017). Data-mining textual responses to uncover misconception patterns. In *Proceedings of the Fourth ACM Conference on Learning @ Scale (L@S '17)*, pp. 245-248. ACM. DOI: 10.1145/3051457.3053996.

**Summary:** This paper proposes an NLP framework for detecting common misconceptions from students' textual responses to open-ended questions. The method uses text preprocessing, TF-IDF vectorization, and hierarchical clustering to group responses and surface misconception patterns without requiring predefined taxonomies. Evaluated on STEM open-response datasets, the approach successfully identifies clusters corresponding to known misconception categories, enabling instructors to deliver more targeted feedback.

**Reusable for our project:** The unsupervised clustering approach for discovering misconception groupings from raw student text provides a baseline architecture for our misconception-detection pipeline. The TF-IDF + clustering workflow could serve as a lightweight fallback when limited labeled data is available.

**Limitation/gap:** The NLP methods (TF-IDF, basic clustering) are pre-transformer and have limited capacity to capture semantic nuance in algebraic expressions. The evaluation is small-scale, and discovered clusters require manual expert labeling, which does not scale.

---

### 2.2 McNichols, Zhang & Lan (2023) - Algebra Error Classification with LLMs

**Citation:** McNichols, H., Zhang, M. & Lan, A. (2023). Algebra error classification with large language models. In *Proceedings of the 24th International Conference on Artificial Intelligence in Education (AIED 2023)*, LNCS 13916, pp. 365-376. Springer. DOI: 10.1007/978-3-031-36272-9_30.

**Summary:** This paper applies LLMs (GPT-based models) to classify algebra errors in student open-ended responses, comparing against rule-based and data-driven methods. The approach uses few-shot prompting with example error patterns to classify mistakes into predefined categories (sign errors, distribution errors, combining unlike terms, etc.). Results show that LLMs outperform rule-based systems in generalization and achieve competitive accuracy with significantly less engineering effort, crucially without requiring hand-crafted mathematical expression parsers.

**Reusable for our project:** Directly applicable - provides a validated methodology for LLM few-shot prompting to classify algebraic errors from open responses. The error taxonomy (sign errors, distribution errors, variable confusion, etc.) can be adopted as our initial misconception ontology. The prompting strategy templates are immediately reusable.

**Limitation/gap:** Evaluation uses a relatively small labeled dataset, and the method treats each response independently without considering student history or knowledge state. Error classification is performed in isolation - the paper does not integrate it into a tutoring loop or address how to connect classification to targeted remediation.

---

### 2.3 Gorgun & Botelho (2023) - Math Misconceptions with NLP

**Citation:** Gorgun, G. & Botelho, A.F. (2023). Enhancing the automatic identification of common math misconceptions using natural language processing. In *Proceedings of the 24th International Conference on Artificial Intelligence in Education (AIED 2023)*, LNCS 13916, pp. 302-308. Springer. DOI: 10.1007/978-3-031-36336-8_47.

**Summary:** This study extends prior work on wrong-answer frequency analysis by applying NLP to open-ended math question responses rather than only close-ended formats. The authors cluster unique student answers using text similarity and frequency analysis to surface common misconceptions. The method demonstrates as a proof of concept that NLP can identify misconception patterns from open-response math answers, moving beyond the traditional limitation of multiple-choice-only misconception detection.

**Reusable for our project:** Validates applying NLP to open-response math questions for misconception identification - a core requirement of our system. The integration with a computer-based learning platform demonstrates practical feasibility. Frequency-based misconception identification could inform our prioritization of which misconceptions to target with feedback.

**Limitation/gap:** As a short paper / proof of concept, it uses relatively simple NLP methods (text similarity, frequency counting) rather than deep learning. The evaluation is limited in scale, and algebraic expression semantics are not deeply handled (e.g., `2x + 3` and `3 + 2x` would not be recognized as equivalent without symbolic processing).

---

### 2.4 Sonkar et al. (2024) - LLM-Based Cognitive Models of Students

**Citation:** Sonkar, S., Chen, X., Liu, N., Baraniuk, R.G. & Sachan, M. (2024). LLM-based cognitive models of students with misconceptions. *arXiv preprint arXiv:2410.12294*.

**Summary:** This paper introduces *Cognitive Student Models (CSMs)* - LLMs instruction-tuned to faithfully emulate realistic student behavior in algebra, including reproducing specific misconceptions while correctly solving unrelated problems. The authors present MalAlgoPy, a Python library that generates datasets of authentic student solution patterns via graph-based representations of algebraic problem-solving. A key finding is that naive misconception training causes catastrophic degradation of correct-solving ability, but carefully calibrating the correct-to-misconception example ratio (as low as 0.25) enables CSMs that satisfy both properties.

**Reusable for our project:** The MalAlgoPy library for generating synthetic student misconception data is directly reusable for training and testing our misconception detector. The graph-based representation of algebraic problem-solving patterns could enhance our knowledge graph. The calibration insight (correct-to-misconception ratio) is critical for any fine-tuning experiments.

**Limitation/gap:** CSMs are *models of students*, not tutoring systems - they simulate misconceptions but do not provide remediation. The approach requires substantial compute for instruction tuning and does not address real-time deployment. The misconception taxonomy covers algorithmic algebra errors but not conceptual misconceptions (e.g., "variables are labels, not quantities").

---

### 2.5 Otero, Druga & Lan (2025) - Benchmark for Middle School Algebra Misconceptions

**Citation:** Otero, N., Druga, S. & Lan, A. (2025). A benchmark for math misconceptions: Bridging gaps in middle school algebra with AI-supported instruction. *Discover Education*, 4, Article 42. DOI: 10.1007/s44217-025-00742-w.

**Summary:** This study introduces an evaluation benchmark for middle school algebra misconceptions: 55 misconceptions, associated common errors, and 220 diagnostic examples drawn from peer-reviewed literature. The authors evaluate GPT-4's ability to diagnose these misconceptions, finding that while the LLM can identify many, performance varies significantly across misconception types. The benchmark is designed to support the development of AI systems that enhance learners' conceptual understanding by accounting for their current comprehension level.

**Reusable for our project:** The curated dataset of 55 algebra misconceptions with 220 diagnostic examples is directly usable as a ground-truth evaluation set for our misconception detector. The misconception taxonomy (built from peer-reviewed literature) can structure our knowledge graph. The GPT-4 evaluation baseline gives a concrete accuracy target to beat or match.

**Limitation/gap:** The benchmark focuses on *diagnosis* (identifying which misconception a student holds) but does not address *remediation* strategies or measure whether detection leads to improved learning outcomes. The 55-misconception scope may not cover all error types in a full algebra curriculum. The dataset is static and does not capture how misconceptions evolve over student learning trajectories.

---

## Category 3: Knowledge Tracing

### 3.1 Corbett & Anderson (1995) - Bayesian Knowledge Tracing

**Citation:** Corbett, A.T. & Anderson, J.R. (1995). Knowledge tracing: Modeling the acquisition of procedural knowledge. *User Modeling and User-Adapted Interaction*, 4(4), 253-278. Springer. DOI: 10.1007/BF01099821.

**Summary:** This foundational paper introduces the Bayesian Knowledge Tracing (BKT) model for estimating student mastery of individual knowledge components during skill acquisition. BKT uses a Hidden Markov Model with four parameters per skill - P(L0) initial knowledge, P(T) learning transition, P(G) guess, and P(S) slip - to update mastery probability after each student interaction. Evaluated in the context of the ACT Programming Tutor, the model demonstrated that mastery estimates reliably predicted post-test performance, and setting a 95% mastery threshold before advancing enabled individualized pacing that improved learning outcomes.

**Reusable for our project:** BKT is the core mastery-tracking mechanism for our adaptive engine. The four-parameter model provides interpretable, per-skill probability updates that we implement directly via pyBKT. The 95% mastery threshold concept informs our decision gate for when students advance to the next concept in the knowledge graph.

**Limitation/gap:** Standard BKT assumes all students share the same learning parameters per skill, ignoring individual differences in learning rate and prior knowledge. It also treats knowledge components as independent, missing correlations between related algebra skills. The binary correct/incorrect observation model cannot incorporate information from free-text student explanations or partial credit.

---

### 3.2 Piech et al. (2015) - Deep Knowledge Tracing

**Citation:** Piech, C., Bassen, J., Huang, J., Ganguli, S., Sahami, M., Guibas, L.J. & Sohl-Dickstein, J. (2015). Deep knowledge tracing. In *Advances in Neural Information Processing Systems 28 (NeurIPS 2015)*, pp. 505-513.

**Summary:** This paper introduces Deep Knowledge Tracing (DKT), the first application of recurrent neural networks (specifically LSTMs) to the knowledge tracing problem. DKT takes sequences of student interactions (exercise ID, correctness) as input and predicts future performance without requiring explicit encoding of domain knowledge or skill labels. On the ASSISTments and Khan Academy datasets, DKT significantly outperformed BKT in next-step prediction AUC, and the learned representations captured meaningful latent structure in the skill space.

**Reusable for our project:** DKT provides a higher-accuracy alternative or complement to BKT for mastery estimation when sufficient training data is available. The learned hidden representations could serve as student embeddings for personalization. The demonstrated ability to discover skill relationships without explicit labeling could augment our manually defined knowledge graph.

**Limitation/gap:** DKT is a black box - it does not produce interpretable per-skill mastery probabilities the way BKT does, making it harder to explain to students or teachers *why* a particular problem was selected. It can exhibit non-monotonic mastery predictions (mastery decreasing after a correct answer), which Hooshyar et al. (2025) empirically confirmed. It also requires substantially more training data than BKT and offers no mechanism for incorporating qualitative response features beyond binary correctness.

---

### 3.3 Yudelson, Koedinger & Gordon (2013) - Individualized Bayesian Knowledge Tracing

**Citation:** Yudelson, M.V., Koedinger, K.R. & Gordon, G.J. (2013). Individualized Bayesian knowledge tracing models. In *Proceedings of the 16th International Conference on Artificial Intelligence in Education (AIED 2013)*, LNCS 7926, pp. 171-180. Springer. DOI: 10.1007/978-3-642-39112-5_18.

**Summary:** This paper revisits the problem of introducing student-specific parameters into BKT at larger scale. The authors show that adding per-student variability parameters (individual priors and learning rates) leads to tangible improvements in prediction accuracy compared to standard skill-only BKT. They demonstrate that accounting for student-specific differences - particularly in initial knowledge P(L0) and learning rate P(T) - meaningfully enhances model fit on datasets from the Cognitive Tutor, validating the importance of individualization even within a simple probabilistic framework.

**Reusable for our project:** Directly applicable to our BKT implementation. By adding student-specific priors to pyBKT's fitting procedure, we can account for the wide variation in incoming algebra knowledge among middle school students. The individualized P(L0) parameter is especially valuable for our cold-start problem - new students can be initialized based on a brief diagnostic assessment rather than skill-level defaults.

**Limitation/gap:** Individualized BKT adds parameters proportional to the number of students, which complicates model training and risks overfitting with limited per-student data. The paper does not address how to set individualized parameters for entirely new students (cold-start), and the evaluation focuses on prediction accuracy rather than downstream learning outcomes.

---

### 3.4 Badrinath, Wang & Pardos (2021) - pyBKT

**Citation:** Badrinath, A., Wang, F. & Pardos, Z.A. (2021). pyBKT: An accessible Python library of Bayesian knowledge tracing models. In *Proceedings of the 14th International Conference on Educational Data Mining (EDM 2021)*. arXiv:2105.00385.

**Summary:** This paper introduces pyBKT, an open-source Python library implementing BKT and its major extensions from the literature, including individualized parameters, item difficulty, and multiple-subskill models. The library provides data generation, fitting (via EM algorithm), prediction, and cross-validation routines with a simple data helper interface for ingesting standard tutor log formats. Runtime evaluations demonstrate computational efficiency across dataset sizes, and sanity checks using simulated and real-world data validate that pyBKT's implementations reproduce results from the original papers in which model variants were introduced.

**Reusable for our project:** pyBKT is our primary implementation library for the mastery-tracking component. It provides off-the-shelf support for standard BKT, individualized BKT (Yudelson et al.), and item-difficulty extensions, eliminating the need to implement these algorithms from scratch. The cross-validation routines enable rigorous evaluation of our mastery model, and the data helper interface simplifies integration with our student interaction logs.

**Limitation/gap:** pyBKT implements classical BKT variants but does not incorporate deep learning extensions (DKT, AKT, etc.) or transformer-based models. It assumes binary correctness observations and cannot directly ingest the richer signal our LLM misconception classifier produces (e.g., error type, partial credit). Integrating misconception labels into the BKT update equations will require custom extensions beyond what pyBKT provides out of the box.

---

### 3.5 Liu et al. (2023) - simpleKT

**Citation:** Liu, Z., Liu, Q., Chen, J., Huang, S. & Luo, W. (2023). simpleKT: A simple but tough-to-beat baseline for knowledge tracing. In *The 11th International Conference on Learning Representations (ICLR 2023)*. arXiv:2302.06881.

**Summary:** This paper proposes simpleKT, a straightforward knowledge tracing model that uses ordinary dot-product attention to capture temporal learning patterns, combined with explicit question-specific variation modeling inspired by the Rasch model from psychometrics. Despite its simplicity, simpleKT consistently ranks in the top 3 by AUC score across 7 public datasets of different domains, achieving 57 wins, 3 ties, and 16 losses against 12 deep learning KT baseline methods. The paper also highlights the lack of standardized evaluation protocols in KT research, noting that the same model's reported AUC scores vary dramatically across publications (e.g., DKT on ASSISTments2009 ranges from 0.721 to 0.821).

**Reusable for our project:** Provides evidence that a simple attention-based architecture can match or exceed complex KT models, which is valuable for our resource-constrained deployment (single RTX 3070 GPU). The question-level variation modeling is directly relevant since our algebra problems within a single concept may vary in difficulty. If we eventually upgrade from BKT to a neural KT model, simpleKT offers the best accuracy-to-complexity ratio. The pyKT benchmark integration allows straightforward comparison.

**Limitation/gap:** Like DKT, simpleKT operates on binary correctness sequences and does not incorporate information from student response content or misconception type. The evaluation focuses on next-step prediction accuracy rather than adaptive tutoring outcomes. The model requires training data volume that may exceed what we collect in early pilot studies.

---

### 3.6 Xia & Li (2025) - FlatFormer

**Citation:** Xia, X. & Li, H. (2025). FlatFormer: A flat Transformer knowledge tracing model based on cognitive bias injection. *arXiv preprint arXiv:2512.06629*.

**Summary:** FlatFormer proposes a lightweight transformer architecture for knowledge tracing that resolves the "Performance-Complexity Trap" via "Information Injection over Structural Stacking." Rather than using deep hierarchical architectures, it augments a standard flat transformer with two injection mechanisms: (i) hybrid input encoding combining learnable session identifiers with sinusoidal step embeddings, and (ii) a pre-computed power-law bias integrated into attention logits to explicitly model the forgetting curve. On the EdNet dataset, FlatFormer achieves an absolute AUC gain of 8.3% over the strongest hierarchical baseline (HiTSKT) while using less than 15% of the parameters and achieving approximately 3x faster inference.

**Reusable for our project:** The forgetting-curve bias injection directly addresses a known weakness of standard BKT (which has no explicit forgetting mechanism). The lightweight architecture (15% parameters of hierarchical models) is well-suited for our single-GPU constraint. The session-aware encoding could capture important patterns in our tutoring data, where students may have variable session lengths and gaps between sessions.

**Limitation/gap:** As a 2025 preprint, the approach has not yet been peer-reviewed or replicated. The power-law forgetting bias is pre-computed and fixed, which may not capture individual differences in forgetting rates. Like other neural KT models, FlatFormer does not incorporate qualitative response features or misconception information, limiting its use as a standalone model for our misconception-aware adaptive engine.

---

## Category 4: LLM Fine-Tuning on Small Data

### 4.1 Hu et al. (2021) - LoRA

**Citation:** Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L. & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. In *The 10th International Conference on Learning Representations (ICLR 2022)*. arXiv:2106.09685.

**Summary:** LoRA proposes freezing the pre-trained model weights and injecting trainable low-rank decomposition matrices (A and B, where the update is W + BA) into each transformer layer. Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000x and the GPU memory requirement by 3x. Experiments on RoBERTa, DeBERTa, GPT-2, and GPT-3 show that LoRA performs on par with or better than full fine-tuning, with no additional inference latency - unlike adapter modules that add serial computation. The paper also provides an empirical investigation into rank-deficiency in language model adaptation, finding that very low ranks (r = 1 to 4) are often sufficient.

**Reusable for our project:** LoRA is the core adaptation method for our misconception classifier. Its extreme parameter efficiency (as few as 0.01% of total parameters) enables fine-tuning a 7B model on our RTX 3070 (8GB VRAM) when combined with quantization (QLoRA). The finding that low rank (r = 4) suffices is critical for our small-data regime (<1000 labeled examples), where more trainable parameters would risk overfitting. LoRA's weight-merging capability means our fine-tuned model has zero additional inference cost.

**Limitation/gap:** LoRA's original evaluation focuses on large-scale benchmark tasks with thousands to millions of training examples. The paper does not explore the extreme low-data regime (<1000 examples) relevant to our setting. Optimal rank selection requires experimentation, and there is no principled method for choosing rank as a function of dataset size. The paper also does not evaluate on classification tasks (as opposed to generation), which is our primary use case.

---

### 4.2 Dettmers et al. (2023) - QLoRA

**Citation:** Dettmers, T., Pagnoni, A., Holtzman, A. & Zettlemoyer, L. (2023). QLoRA: Efficient finetuning of quantized LLMs. In *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*. arXiv:2305.14314.

**Summary:** QLoRA extends LoRA by backpropagating gradients through a frozen 4-bit quantized model into Low-Rank Adapters, reducing memory requirements enough to fine-tune a 65B parameter model on a single 48GB GPU while preserving full 16-bit fine-tuning performance. The method introduces three innovations: (a) 4-bit NormalFloat (NF4), an information-theoretically optimal data type for normally distributed weights; (b) double quantization to reduce quantization constant overhead; and (c) paged optimizers to manage memory spikes. The resulting Guanaco model family achieves 99.3% of ChatGPT's performance on the Vicuna benchmark with only 24 hours of fine-tuning on a single GPU.

**Reusable for our project:** QLoRA is the enabling technology for our hardware constraint. With 4-bit quantization, a 7B parameter model requires approximately 4-5 GB of VRAM for weights, leaving sufficient headroom on the RTX 3070 (8GB VRAM) for activations and LoRA adapter gradients during fine-tuning. The small high-quality dataset finding ("QLoRA finetuning on a small high-quality dataset leads to state-of-the-art results") directly validates our approach of fine-tuning on a curated misconception dataset of hundreds of examples.

**Limitation/gap:** The primary evaluation uses chatbot benchmarks (Vicuna, Open Assistant) rather than classification or educational tasks. The 4-bit quantization introduces a small accuracy floor that could matter for fine-grained misconception discrimination. The paper does not explore datasets below 1,000 examples, and the interaction between extreme quantization and low-data fine-tuning is not well characterized. Batch size restrictions on consumer GPUs (often batch size 1-2) may affect training stability.

---

### 4.3 Liu et al. (2022) - Few-Shot PEFT vs. In-Context Learning

**Citation:** Liu, H., Tam, D., Muqeeth, M., Mohta, J., Huang, T., Bansal, M. & Raffel, C. (2022). Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning. In *Advances in Neural Information Processing Systems 35 (NeurIPS 2022)*. arXiv:2205.05638.

**Summary:** This paper provides a rigorous empirical comparison of few-shot in-context learning (ICL) and parameter-efficient fine-tuning (PEFT), demonstrating that PEFT offers strictly better accuracy at dramatically lower computational cost. The authors introduce (IA)^3, a method that scales activations by learned vectors with minimal new parameters, and propose T-Few, a simple recipe applied to the T0 model that achieves super-human performance on the RAFT benchmark - the first method to do so - outperforming the previous state-of-the-art by 6% absolute. A key finding is that PEFT with as few as 32 training examples consistently outperforms ICL using the same examples, while requiring only a single forward pass at inference instead of processing all examples every time.

**Reusable for our project:** Provides the empirical justification for choosing PEFT (LoRA/QLoRA) over prompt-based ICL for our misconception classifier. With our expected dataset of 200-500 labeled examples per misconception category, PEFT will outperform few-shot prompting while being cheaper at inference time - critical for real-time tutoring feedback. The T-Few recipe demonstrates that PEFT methods can be applied "out of the box" to new tasks, reducing the need for extensive task-specific tuning.

**Limitation/gap:** The evaluation uses T0 (an encoder-decoder model) as the base; the findings may not transfer directly to decoder-only models (LLaMA, Mistral) preferred for our project. The RAFT benchmark tasks differ substantially from math misconception classification. The paper does not address combining PEFT with quantization (QLoRA), which adds complexity to the training dynamics.

---

### 4.4 Lialin et al. (2023) - Scaling Down to Scale Up: A Guide to PEFT

**Citation:** Lialin, V., Deshpande, V., Yao, X. & Rumshisky, A. (2023). Scaling down to scale up: A guide to parameter-efficient fine-tuning. *arXiv preprint arXiv:2303.15647* (revised November 2024).

**Summary:** This comprehensive survey covers over 50 parameter-efficient fine-tuning methods published between 2019 and 2024, providing a taxonomy that organizes approaches into additive methods (adapters, soft prompts), selective methods (sparse updates, masking), and reparameterization methods (LoRA, Kronecker products). The authors conduct an extensive head-to-head experimental comparison of 15 diverse PEFT methods on models up to 11B parameters. A key finding is that methods previously shown to surpass LoRA face difficulties in resource-constrained settings where hyperparameter optimization is limited and fine-tuning runs for only a few epochs - conditions that match our use case precisely.

**Reusable for our project:** Serves as the definitive reference for choosing our fine-tuning strategy. The practical finding that LoRA is hard to beat in resource-constrained settings validates our design choice. The taxonomy helps us understand the full landscape of alternatives if LoRA underperforms on misconception classification. The practical recommendations section provides guidance on learning rate, rank, and target module selection that we can apply directly.

**Limitation/gap:** The survey evaluates PEFT methods primarily on standard NLP benchmarks (GLUE, SuperGLUE, generation quality) rather than educational or domain-specific classification tasks. The resource-constrained evaluation uses limited epochs but not extremely small datasets (<1000 examples), leaving open the question of how PEFT methods compare in our specific data regime. The field evolves rapidly; methods published after mid-2024 (e.g., DoRA, PiSSA) are not covered.

---

## Category 5: Math Misconception Benchmarks and Datasets

### 5.1 Otero, Druga & Lan (2025) - MaE: A Benchmark for Math Misconceptions

**Citation:** Otero, N., Druga, S. & Lan, A. (2025). A benchmark for math misconceptions: Bridging gaps in middle school algebra with AI-supported instruction. *Discover Education*, 4, Article 42. DOI: 10.1007/s44217-025-00742-w. arXiv:2412.03765.

**Summary:** This study introduces the Math Misconceptions in Education (MaE) benchmark, a curated evaluation dataset comprising 55 middle-school algebra misconceptions, associated common errors, and 220 diagnostic examples drawn from peer-reviewed mathematics education literature. The authors evaluate GPT-4's ability to diagnose these misconceptions, finding performance varies significantly by topic - reaching 83.9% precision when constrained to specific algebra topics and incorporating educator feedback. A survey of five practicing middle-school math educators confirmed that 80% or more encounter these misconceptions among their students, validating the dataset's ecological relevance. Notably, topics such as ratios and proportions prove as difficult for LLMs as they are for students.

**Reusable for our project:** The MaE benchmark is our primary evaluation dataset for the misconception classifier. The 55-misconception taxonomy, organized by algebra subdomain, directly structures our knowledge graph. The GPT-4 baseline scores establish concrete accuracy targets, and the educator survey provides evidence of pedagogical relevance needed for IRB justification. The topic-constrained evaluation methodology informs our planned per-concept accuracy analysis.

**Limitation/gap:** At 220 diagnostic examples, MaE is an evaluation benchmark rather than a training corpus - insufficient for fine-tuning without augmentation. The dataset is static and does not capture how misconceptions evolve over learning trajectories. It covers diagnosis only, not remediation effectiveness. The multiple-choice diagnostic format does not include free-form student explanations, limiting its applicability for testing open-response misconception detection.

---

### 5.2 King et al. (2024) - Eedi: Mining Misconceptions in Mathematics (NeurIPS 2024 Kaggle Competition)

**Citation:** King, J., Burleigh, L., Woodhead, S., Kon, P., Baffour, P., Crossley, S., Reade, W. & Demkin, M. (2024). Eedi - Mining Misconceptions in Mathematics. NeurIPS 2024 Competition Track. Kaggle. https://kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics.

**Summary:** This NeurIPS 2024 competition, organized by Eedi, Vanderbilt University, and The Learning Agency Lab (with support from the Gates Foundation, Schmidt Futures, and CZI), challenges participants to build NLP models that predict the affinity between mathematical misconceptions and incorrect answer distractors in multiple-choice diagnostic questions. The dataset provides thousands of diagnostic questions where each of three distractors is tagged with a specific misconception, and models must rank candidate misconceptions for each distractor using MAP@25. The competition attracted 7,941 entrants across 1,446 teams producing 41,411 submissions, with a separate efficiency track rewarding CPU-only solutions that balance accuracy with computational cost. This builds on the earlier NeurIPS 2020 Eedi Education Challenge (Wang et al., 2020, arXiv:2007.12061) which released over 20 million student answer records from the Eedi platform for student response prediction and personalized question sequencing tasks.

**Reusable for our project:** The competition's dataset of misconception-tagged diagnostic questions provides a large-scale complement to MaE's curated benchmark. Top-performing competition solutions (publicly shared as Kaggle notebooks) offer validated architectures for misconception-distractor matching that we can adapt. The efficiency track results identify models that achieve strong misconception prediction on CPU-only hardware, directly relevant to our single-GPU deployment constraint. The misconception taxonomy embedded in the data augments our knowledge graph.

**Limitation/gap:** The task formulates misconception detection as a ranking/retrieval problem (matching distractors to pre-defined misconception labels) rather than detecting misconceptions from free-form student work. The multiple-choice format is fundamentally different from the open-response setting our tutor targets. Competition data may have licensing restrictions that limit direct reuse for training. The misconception taxonomy is Eedi-specific and may not align with the MaE taxonomy our project adopts.

---

### 5.3 Hendrycks et al. (2021) - MATH: Measuring Mathematical Problem Solving

**Citation:** Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D. & Steinhardt, J. (2021). Measuring mathematical problem solving with the MATH dataset. In *Advances in Neural Information Processing Systems 34 (NeurIPS 2021)*. arXiv:2103.03874.

**Summary:** This paper introduces MATH, a benchmark of 12,500 challenging competition-level mathematics problems spanning seven subjects (Prealgebra, Algebra, Number Theory, Counting and Probability, Geometry, Intermediate Algebra, Precalculus) with five difficulty levels. Each problem includes a full step-by-step solution that can teach models to generate answer derivations and explanations. The authors also contribute a large auxiliary pretraining dataset (AMPS) to teach mathematical fundamentals. Even the largest Transformer models at the time achieved low accuracy, and the authors demonstrated that simply scaling model size and compute would be impractical for achieving strong mathematical reasoning, suggesting the need for new algorithmic advances.

**Reusable for our project:** The Algebra and Prealgebra subsets of MATH (approximately 2,500 problems) provide a validated difficulty-graded problem bank for generating tutoring content. The step-by-step solutions serve as ground-truth reasoning chains for evaluating whether our LLM produces correct worked examples. The five difficulty levels map naturally to our adaptive engine's problem selection: early mastery stages use Level 1-2 problems while advanced stages use Level 3-5. MATH's widespread adoption as an LLM evaluation benchmark means results on it are directly comparable to published model capabilities.

**Limitation/gap:** MATH targets competition-level math and skews harder than typical middle-school algebra curriculum - even the "Prealgebra" subset contains problems beyond standard classroom difficulty. The dataset contains final answers and solutions but no student response data, no misconception annotations, and no information about common errors. It evaluates problem *solving* ability, not *teaching* or *diagnostic* ability. The free-response format requires exact-match or symbolic equivalence checking, which our system will also need to implement.

---

### 5.4 Macina et al. (2025) - MathTutorBench: Measuring Pedagogical Capabilities of LLM Tutors

**Citation:** Macina, J., Daheim, N., Hakimi, I., Kapur, M., Gurevych, I. & Sachan, M. (2025). MathTutorBench: A benchmark for measuring open-ended pedagogical capabilities of LLM tutors. In *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025)*. arXiv:2502.18940.

**Summary:** MathTutorBench is the first comprehensive open-source benchmark for holistic evaluation of LLM tutoring capabilities, covering three high-level teacher skills across seven concrete tasks: problem solving, Socratic questioning, student solution generation, mistake location, mistake correction, scaffolding generation, and pedagogy following. The authors train a pedagogical reward model that discriminates expert from novice teacher responses with high accuracy. Evaluating a wide range of closed- and open-weight models, they find a key insight: subject expertise (mathematical solving ability) does not translate to good teaching. Rather, pedagogy and subject expertise form a trade-off mediated by the degree of tutoring specialization. LearnLM-1.5-Pro leads the leaderboard overall, while math-specialized models like Qwen2.5-Math-7B-Instruct score well on solving but near-zero on pedagogical tasks. Tutoring becomes more challenging in longer dialogues, where simpler questioning strategies break down.

**Reusable for our project:** The mistake location and mistake correction tasks map directly to our misconception detection pipeline. The trained pedagogical reward model (released on HuggingFace as Qwen2.5-1.5B-pedagogical-rewardmodel) can evaluate the quality of our LLM-generated tutoring feedback without expensive human annotation. The benchmark's finding that solving ability and teaching ability are distinct validates our architecture: using one LLM component for mathematical reasoning and a separate fine-tuned component for pedagogical response generation. The public leaderboard provides continuously updated baselines to compare against.

**Limitation/gap:** MathTutorBench evaluates single-turn pedagogical responses, not multi-turn adaptive tutoring sessions where our system operates. The benchmark does not incorporate student learning outcomes - it measures response quality, not whether students actually learn. The reward model is trained on expert vs. novice discrimination, which may not capture all dimensions of effective algebra misconception remediation. The math content spans multiple grade levels and is not specifically targeted at middle-school algebra misconceptions.

---

### 5.5 Sonkar et al. (2024) - MalAlgoPy: Synthetic Math Error Generation for Cognitive Student Models

**Citation:** Sonkar, S., Chen, X., Liu, N., Baraniuk, R.G. & Sachan, M. (2024). LLM-based cognitive models of students with misconceptions. *arXiv preprint arXiv:2410.12294*.

**Summary:** This paper addresses the scarcity of labeled student misconception data by introducing MalAlgoPy, a Python library that generates datasets of authentic student solution patterns via graph-based representations of algebraic problem-solving. Each algebra problem is modeled as a directed acyclic graph (DAG) of procedural steps, and misconceptions are injected by modifying specific nodes (e.g., replacing correct sign handling with a sign-error rule). The library produces paired datasets of correct and misconception-exhibiting solutions at arbitrary scale. The authors use MalAlgoPy to train Cognitive Student Models (CSMs) - LLMs instruction-tuned to emulate realistic student behavior. A critical finding is that misconception training causes catastrophic degradation of correct-solving ability, but carefully calibrating the correct-to-misconception example ratio (as low as 0.25) preserves both properties.

**Reusable for our project:** MalAlgoPy is directly usable to augment our limited training data: we can generate thousands of synthetic misconception examples for fine-tuning beyond the 220 examples in MaE. The graph-based representation of algebra procedures aligns with our knowledge graph architecture. The correct-to-misconception calibration ratio (0.25) provides actionable guidance for our QLoRA training data composition. The generated data can also serve as additional evaluation cases to stress-test our misconception classifier on systematically varied error patterns.

**Limitation/gap:** MalAlgoPy generates *procedural/algorithmic* errors (sign errors, order-of-operations mistakes, incorrect distribution) but does not model *conceptual* misconceptions (e.g., "the equals sign means 'write the answer'" or "variables are labels for objects"). The synthetic data may not reflect the full distribution of errors real students produce, particularly creative or compound errors. The library requires manual specification of misconception rules for each problem type, and it is unclear how well it covers the full MaE taxonomy of 55 misconceptions. Generated data lacks the natural language explanations that real students provide.

---

### 5.6 Wang et al. (2024) - Bridge: Tutoring Conversations for Remediating Math Mistakes

**Citation:** Wang, R.E., Zhang, Q., Robinson, C., Loeb, S. & Demszky, D. (2024). Bridging the novice-expert gap via models of decision-making: A case study on remediating math mistakes. In *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL 2024)*. arXiv:2310.10648.

**Summary:** Bridge contributes a dataset of 700 real tutoring conversations annotated by experts with structured decision-making labels: (A) the student's specific error, (B) the remediation strategy selected, and (C) the tutor's pedagogical intention - elicited via cognitive task analysis of expert tutors. The authors use this framework to evaluate whether LLMs can close the novice-expert tutoring gap. Results show that GPT-4 responses generated with access to expert decisions (e.g., "simplify the problem") are preferred 76% more often than responses without, while random decisions decrease response quality by 97% compared to expert decisions, demonstrating that context-sensitive pedagogical reasoning is critical.

**Reusable for our project:** Bridge's three-part annotation schema (error identification, strategy selection, intention) provides a model for structuring our tutoring system's response pipeline: first classify the misconception, then select a remediation strategy, then generate the response with explicit pedagogical intent. The 700 annotated conversations can serve as few-shot examples for prompting our LLM to follow expert remediation patterns. The demonstrated importance of structured decision-making over raw LLM capability validates building explicit pedagogical decision logic into our adaptive engine rather than relying on end-to-end generation.

**Limitation/gap:** At 700 conversations, the dataset is too small for fine-tuning and serves primarily as an evaluation and prompting resource. The conversations come from an online math tutoring platform and may not represent the specific error patterns of middle-school algebra (MaE's scope). The expert annotations capture *what* experts decide but not the full reasoning chain, making it difficult to train models that explain *why* a particular remediation strategy was chosen. The dataset does not include student learning outcomes, so it is unknown whether the annotated strategies actually improve understanding.

---

### 5.7 Macina et al. (2023) - MathDial: A Dialogue Tutoring Dataset with Rich Pedagogical Properties

**Citation:** Macina, J., Daheim, N., Pal Chowdhury, S., Sinha, T., Kapur, M., Gurevych, I. & Sachan, M. (2023). MathDial: A dialogue tutoring dataset with rich pedagogical properties grounded in math reasoning problems. In *Findings of the Association for Computational Linguistics: EMNLP 2023*. arXiv:2305.14536.

**Summary:** MathDial presents 3,000 one-to-one teacher-student tutoring dialogues grounded in multi-step math reasoning problems, generated through a novel human-AI framework that pairs real human teachers with an LLM prompted to simulate students making common errors. Each dialogue is annotated with a taxonomy of teacher moves (e.g., probing questions, hints, positive reinforcement) to capture scaffolding strategies. The authors show that while LLMs like GPT-3 are effective problem solvers, they fail at tutoring by generating factually incorrect feedback or revealing solutions prematurely. Fine-tuning on MathDial produces models that are measurably more effective tutors, as confirmed by both automatic and human evaluation in interactive settings measuring the trade-off between student solving success and solution revealing.

**Reusable for our project:** MathDial's 3,000 dialogues (publicly released) are the largest available math tutoring conversation dataset and can serve as fine-tuning data for our tutoring response generator. The teacher-move taxonomy provides a validated vocabulary for our scaffolding system. The human-AI collection methodology - pairing real teachers with LLM-simulated students - is directly replicable for generating additional algebra-specific training dialogues. The finding that raw LLMs reveal solutions prematurely is a specific failure mode our system must guard against.

**Limitation/gap:** MathDial covers general math reasoning (GSM8K-style word problems) rather than targeted algebra misconceptions. The LLM-simulated students may not exhibit the full range of real student behaviors, particularly affective responses (frustration, confusion) or compound/creative errors. The dialogues are relatively short (single-problem interactions) and do not capture multi-session learning progressions. The dataset does not include misconception labels, so mapping dialogues to specific misconception categories requires additional annotation.

---

### 5.8 Wang et al. (2020) - Eedi NeurIPS 2020 Education Challenge: 20M+ Diagnostic Question Responses

**Citation:** Wang, Z., Lamb, A., Saveliev, E., Cameron, P., Zaykov, Y., Hernandez-Lobato, J.M., Turner, R.E., Baraniuk, R.G., Barton, C., Peyton Jones, S., Woodhead, S. & Zhang, C. (2021). Instructions and guide for diagnostic questions: The NeurIPS 2020 Education Challenge. *NeurIPS 2020 Competition Track*. arXiv:2007.12061.

**Summary:** This competition paper accompanies the NeurIPS 2020 Education Challenge, which released over 20 million student answer records to mathematics diagnostic multiple-choice questions from Eedi, a platform used daily by thousands of students worldwide. The challenge posed three tasks: (1) accurately predicting which answer option students select, (2) predicting which questions have high diagnostic quality, and (3) determining a personalized sequence of questions that best predicts individual student performance. Each diagnostic question has one correct answer and three distractors, where each distractor is designed to reveal a specific misconception.

**Reusable for our project:** The 20M+ response records constitute the largest publicly available dataset of student interactions with misconception-tagged diagnostic math questions. The response patterns (which distractors students choose, and how often) provide empirical prior distributions over misconception prevalence that can initialize our Bayesian knowledge tracing parameters. Task 1 solutions from the competition (student response prediction) are directly relevant to the "guess" and "slip" parameter estimation in our BKT model. The dataset's scale enables credible statistical analysis of misconception co-occurrence and prerequisite relationships.

**Limitation/gap:** The dataset uses the Eedi misconception taxonomy, which differs from MaE's and may not map cleanly to our project's chosen scope of middle-school algebra. The data captures only selected answer options (A/B/C/D), not free-form student work or explanations. Student identities are anonymized in ways that may limit longitudinal analysis. The questions span all of K-12 mathematics, requiring filtering to extract the algebra-relevant subset. No tutoring dialogue or remediation data is included - only diagnostic responses.

---

## Cross-Cutting Synthesis

| Dimension | Category 1 (ITS) | Category 2 (Misconception + NLP) | Category 3 (Knowledge Tracing) | Category 4 (LLM Fine-Tuning) | Category 5 (Benchmarks + Datasets) |
|---|---|---|---|---|---|
| **Dominant paradigm** | Rule-based / BKT / production systems | Shifting from TF-IDF/clustering to LLM few-shot | BKT evolving toward transformers (simpleKT, FlatFormer) | LoRA-family methods dominate low-resource adaptation | Curated expert benchmarks + large-scale crowd-sourced response data |
| **Interaction type** | Step-level structured input | Open-ended text responses | Binary correct/incorrect sequences | Task-specific labeled examples | Multiple-choice diagnostics, dialogue transcripts, synthetic error patterns |
| **Key gap our project fills** | ITS lack NLP-based open-response handling | NLP misconception work lacks integration into a tutoring loop | KT models ignore response content and misconception type | PEFT not evaluated on education classification tasks | Benchmarks evaluate diagnosis or solving, not integrated tutoring with adaptive remediation |
| **Strongest reuse** | BKT mastery model, skill decomposition, scaffolding | Error taxonomy, LLM prompting templates, benchmark datasets | pyBKT library, individualized parameters, simpleKT baseline | QLoRA for 8GB GPU, LoRA rank selection guidance | MaE evaluation set, MalAlgoPy data augmentation, MathDial fine-tuning corpus, Bridge decision schema, pedagogical reward model |
| **Convergence point** | All five categories converge in our project: BKT-driven adaptive tutoring (Cat 1 + 3) uses LLM-powered misconception classification (Cat 2 + 4) trained and evaluated on curated benchmarks and augmented datasets (Cat 5) to provide step-level feedback on open-response algebra problems |

---

## Gap Statement

Existing intelligent tutoring systems (Cognitive Tutor, ASSISTments) achieve strong learning outcomes through step-level feedback and Bayesian knowledge tracing, but require expert-authored production rules and cannot interpret free-form student language. NLP research has demonstrated that LLMs can classify algebra misconceptions from open responses (McNichols et al., 2023; Otero et al., 2025), and parameter-efficient fine-tuning enables this on consumer hardware (Hu et al., 2022; Dettmers et al., 2023). Knowledge tracing models (BKT, DKT, simpleKT) reliably estimate student mastery from interaction sequences but treat each response as binary correct/incorrect, discarding the rich diagnostic signal that misconception classification provides. Benchmarks and datasets exist for evaluation (MaE), data augmentation (MalAlgoPy), and tutoring dialogue (MathDial, Bridge), but no published system integrates all three components: LLM-based misconception detection from open responses, misconception-informed mastery tracking via BKT, and adaptive problem selection within a connected knowledge graph.

This project fills that gap. It builds the first end-to-end system that (1) uses a LoRA-fine-tuned open-weight LLM to classify free-text algebra answers into specific misconception categories, (2) feeds those classifications into a Bayesian knowledge tracing model that updates per-concept mastery probabilities, and (3) uses the mastery state to adaptively select the next problem or targeted hint within a prerequisite-ordered knowledge graph. The system operates entirely on a single consumer GPU (RTX 3070, 8GB VRAM), uses open-source models and data, and is designed for empirical evaluation through both offline metrics and a pilot study with real learners.
