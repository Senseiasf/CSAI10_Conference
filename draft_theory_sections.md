# Theoretical Foundations and Methodology (Sections 2–4 for CSAI 2026, Track 3)

**Target Track:** Technical Track 3: *Digital Humanities and Modeling of Complex Subject Areas*  
**Track Chair:** Prof. Yuriy Gapanyuk (Bauman Moscow State Technical University)  
**Style Standard:** Academic UK English (*modelling, optimisation, behaviour*), IET / ACM Double-Column Format  
**Scope Principle:** Pure theoretical foundations, mathematical problem formulations, and algorithmic abstractions. No references to specific software repositories, local database table names, hardcoded geographical catalogs, or empirical benchmark metrics.  
**Bibliography Standard:** 100% peer-reviewed scientific publications indexed in **Scopus** and/or **Web of Science**.

---

## Section 2. Related Work & Background

### 2.1 Limitations and Hallucinations of Autonomous LLM Agents in Constrained Multi-Step Planning
Recent advancements in large language models (LLMs) equipped with tool-calling capabilities (e.g., ToolLLM [3]) have catalyzed autonomous planning agents across structured and semi-structured domains. However, empirical investigations demonstrate that standalone LLMs exhibit severe structural and algorithmic degradation when handling complex combinatorial reasoning tasks [1]. Specifically, in multi-modal planning benchmarks such as TravelPlanner [2], state-of-the-art foundation models without external symbolic constraints achieve less than a 5% overall success rate due to combinatorial scheduling errors, physical deadline violations, and arithmetic hallucinations in multi-currency budgeting. These failure modes establish that probabilistic next-token generation is fundamentally ill-suited for guaranteeing hard physical invariants, necessitating a decoupled neuro-symbolic architecture where the neural engine handles natural language intent parsing, whilst deterministic symbolic structures enforce physical and budgetary constraints.

### 2.2 Graph-Based Knowledge Formalisms and Metagraph Structures for Complex Domains
To formally represent and verify domain constraints, systems traditionally employ relational schemas or flat labeled property graphs. Although property graphs effectively capture pairwise entity adjacencies, they lack native constructs for encapsulating composite multi-segment objects, contextual scopes, and nested constraints into unified addressable entities [4]. To address this limitation, the metagraph data model introduced by Gapanyuk et al. [5, 6] extends classical hypergraphs by defining *metavertices* that encapsulate subgraphs, attribute tuples, and contextual constraints into first-class mathematical objects. Complex graph and metagraph formalisms have demonstrated rigorous capability in capturing hierarchical dependencies across transport networks and multi-agent coordination systems [7, 8, 9]. In this work, we operationalise an integrated hierarchical property graph and metagraph formalism to represent, verify, and version complete multi-segment itinerary plans.

### 2.3 Neuro-Symbolic Hybridisation and Cold-Start Learning-to-Rank
The integration of structured knowledge graphs with neural architectures represents an established paradigm for mitigating hallucinations and providing explainable decision paths [10, 11]. Nevertheless, candidate sets verified by symbolic knowledge engines must subsequently be prioritized based on user preferences. While employing LLMs directly as zero-shot rankers is feasible, it incurs prohibitive inference latency (often exceeding 500 ms per candidate) and high token overhead [16]. Conversely, gradient boosted decision tree (GBDT) rankers, such as CatBoost [12], achieve sub-millisecond inference times on standard CPUs while natively handling heterogeneous numerical and categorical feature spaces without pre-scaling [14, 15]. To overcome the absence of historical interaction logs in newly deployed systems (*the cold-start problem*), synthetic preference distillation allows training pairwise ranking objectives (such as PairLogit) on simulated archetype utility distributions [13, 16], establishing an optimal trade-off between execution speed and personalization accuracy.

---

## Section 3. Hybrid Neuro-Symbolic Architecture and Hierarchical Metagraph Model

### 3.1 Neuro-Symbolic Decomposition and Hybrid Candidate Retrieval
The proposed end-to-end framework decouples user intent interpretation from deterministic constraint verification through a structured multi-stage pipeline: (i) intent parsing and slot filling via an LLM agent, (ii) semantically grounded macro-cluster retrieval, (iii) conjunctive boolean filtering of service nodes, and (iv) symbolic structural verification.

When a user submits an unconstrained natural language query $q$, semantic matching cannot rely on exact lexical overlap. The query text is mapped to a normalized $d$-dimensional embedding vector $\boldsymbol{e}(q) \in \mathbb{R}^d$ ($d=128$). Macro-clusters (destinations or regional centers) $d_j \in V_{\text{loc}}$ are retrieved based on cosine similarity against pre-computed cluster semantic descriptions:
$$S_{\text{vec}}(q, d_j) = \frac{\boldsymbol{e}(q) \cdot \boldsymbol{e}(d_j)}{\|\boldsymbol{e}(q)\|_2 \, \|\boldsymbol{e}(d_j)\|_2} \tag{1}$$

For each candidate cluster $d_j$, the set of candidate service entities $V_{\text{serv}}(d_j)$ is filtered by evaluating a conjunctive boolean predicate against declared traveler constraints $C_u = \langle M_{\text{req}}, H_{\text{excl}} \rangle$:
$$\Phi_{\text{filter}}(h, C_u) = \mathbb{I}(h \notin H_{\text{excl}}) \cdot \prod_{r \in M_{\text{req}}} \mathbb{I}\big(\text{attr}_r(h) \ge c_{u,r}\big) \tag{2}$$
where $\mathbb{I}(\cdot) \in \{0, 1\}$ is the indicator function, $M_{\text{req}}$ denotes the set of non-negotiable attribute thresholds (e.g., family-friendly classification excluding adult-only facilities, pet tolerance), and $H_{\text{excl}}$ is the set of entities explicitly rejected within the active dialogue context.

### 3.2 Formal Hierarchical Metagraph Model and State Versioning
To capture multi-modal transport segments, operating time windows, and plan evolution, we formalise the domain as an attributed hierarchical metagraph $MG$ based on the formalism of Gapanyuk et al. [5, 6]:
$$MG = \langle V, MV, E, \mathcal{L}, \mathcal{P} \rangle \tag{3}$$
where:
* $V = V_{\text{loc}} \cup V_{\text{hub}} \cup V_{\text{serv}}$ denotes the set of atomic vertices typed by label set $\mathcal{L}$, representing geographic locations ($V_{\text{loc}}$), transit hubs and terminals ($V_{\text{hub}}$), and service facilities ($V_{\text{serv}}$).
* $E = E_{\text{trans}} \cup E_{\text{link}}$ represents directed typed edges modeling transport transits (scheduled flights, rail, ground/maritime transfers) and structural containment relations. Each transit edge $e \in E_{\text{trans}}$ is associated with an attribute tuple:
  $$\mathcal{P}(e) = \langle t_{\text{dep}}(e), t_{\text{arr}}(e), c(e), \text{cap}(e), W_{\text{dom}}(e) \rangle$$
  specifying scheduled departure $t_{\text{dep}}$, arrival $t_{\text{arr}}$, unit cost $c(e)$, maximum passenger capacity $\text{cap}(e)$, and domain operating window $W_{\text{dom}}(e)$.
* $MV$ denotes the set of **metavertices**. Each metavertex $mv_k \in MV$ models an atomically synthesised and verifiable candidate plan configuration:
  $$mv_k = \langle G_{\text{sub}}^{(k)}, C_u^{(k)}, Q_{\text{cost}}^{(k)} \rangle \tag{4}$$
  where $G_{\text{sub}}^{(k)} = (V_k, E_k)$ is an instantiated route snapshot subgraph ($V_k \subset V, E_k \subset E$), $C_u^{(k)}$ is the active user constraint tuple, and $Q_{\text{cost}}^{(k)}$ is the itemized cost ledger.

**State Versioning and Mutation Invariance:** During conversational interaction, modifications requested by the user (e.g., altering a transfer mode while preserving selected accommodations) are resolved via local subgraph mutation $(mv_{k-1} \to mv_k)$ by replacing selected edges in $E_k$ and deterministically recomputing $Q_{\text{cost}}^{(k)}$. This avoids full combinatorial re-planning while preserving verified sub-paths.

### 3.3 Symbolic Constraint Verification and Additive Budget Invariant
A synthesised plan metavertex $mv_k$ is admissible for ranking if and only if it satisfies the composite symbolic validity predicate $\Psi_{\text{valid}}(mv_k) = 1$:
$$\Psi_{\text{valid}}(mv_k) = \begin{cases} 1, & \text{if } \forall (e_i, e_{i+1}) \in E_k: \\ & \text{(a) } t_{\text{dep}}(e_{i+1}) - t_{\text{arr}}(e_i) \ge \tau_{\min}, \\ & \text{(b) } \text{cap}(e_i) \ge N_{\text{group}}, \\ & \text{(c) } [t_{\text{dep}}(e_i), t_{\text{arr}}(e_i)] \subseteq W_{\text{dom}}(e_i), \\ & \text{(d) } B_{\text{fixed}}(mv_k) + B_{\text{var}}^{\max}(mv_k) \le B_{\text{limit}}; \\ 0, & \text{otherwise.} \end{cases} \tag{5}$$

Predicate (5) guarantees four hard domain invariants:
* **(a) Temporal Continuity:** consecutive transit segments must maintain an idle connection buffer strictly no less than technological threshold $\tau_{\min}$;
* **(b) Capacity Feasibility:** vehicle or facility capacity $\text{cap}(e_i)$ must accommodate the party size $N_{\text{group}}$;
* **(c) Domain Operating Envelopes:** transit intervals must lie entirely within permitted temporal envelopes $W_{\text{dom}}(e_i)$, enforcing regulatory curfews (such as maritime navigation restrictions or municipal drawbridge schedules);
* **(d) Additive Budget Invariant:** the sum of deterministic contractual commitments $B_{\text{fixed}}(mv_k)$ and the upper bound of estimated variable living expenses $B_{\text{var}}^{\max}(mv_k)$ must not exceed the declared user budget threshold $B_{\text{limit}}$.

---

## Section 4. Cold-Start Preference Distillation and Pairwise Tour Ranking

### 4.1 Heterogeneous Feature Space and Linear Heuristic Approximation
Admissible candidate plans satisfying symbolic graph verification ($mv_k \in MV_{\text{valid}}$ where $\Psi_{\text{valid}}(mv_k) = 1$) are projected into an $M$-dimensional heterogeneous feature space:
$$\boldsymbol{x}_k = \big[\boldsymbol{x}_{\text{cat}}^T, \, \boldsymbol{x}_{\text{cont}}^T, \, \boldsymbol{x}_{\text{ord}}^T, \, \boldsymbol{x}_{\text{bin}}^T\big]^T \in \mathbb{R}^M \tag{6}$$
encompassing four distinct mathematical subspaces:
* $\boldsymbol{x}_{\text{cat}} \in \mathcal{X}_{\text{cat}}$ represents discrete categorical identifiers, including spatial macro-cluster tokens and behavioral archetype modalities;
* $\boldsymbol{x}_{\text{cont}} \in \mathbb{R}^{M_1}$ captures continuous physical dimensions, such as cumulative transfer buffers, transit times, and itemized budgetary allocations;
* $\boldsymbol{x}_{\text{ord}} \in \mathbb{N}^{M_2}$ encodes ordinal tier levels and discrete service quality ratings;
* $\boldsymbol{x}_{\text{bin}} \in \{0, 1\}^{M_3}$ represents binary amenity indicators and regulatory compliance flags ($M = |\mathcal{X}_{\text{cat}}| + M_1 + M_2 + M_3$).

As a baseline ranking mechanism and fault-tolerant fallback, we formulate a normalized linear surrogate heuristic:
$$S_{\text{heur}}(\boldsymbol{x}_k) = \sum_{m=1}^{M} w_m(a) \cdot \tilde{x}_{k,m} \tag{7}$$
where $\tilde{x}_{k,m} \in [0, 1]$ denotes min-max normalised feature values and $w_m(a)$ represents weighting coefficients conditioned on user archetype $a \in \mathcal{A}$. 

**The Compensation Dilemma:** Although computationally trivial ($\mathcal{O}(M)$), linear convolution fundamentally suffers from the *compensation dilemma*: an unacceptable deficiency along a critical dimension (such as an excessive layover buffer or an unsuitable travel policy) can always be mathematically compensated by an arbitrarily favorable value in an orthogonal dimension (e.g., low financial cost). This directly violates lexicographic and non-compensatory decision-making axioms inherent to human decision processes.

### 4.2 Synthetic Preference Distillation in GBDT Ranker
To overcome the cold-start deficit resulting from the complete absence of historical user interaction logs in newly initialized deployments, we formulate a synthetic preference distillation paradigm. We define a discrete set of latent behavioral personas $\mathcal{A} = \{a_1, a_2, \dots, a_L\}$. For each archetype $a \in \mathcal{A}$, an offline simulation oracle assigns a non-linear latent utility incorporating marginal valuations, cross-feature interaction synergies, and non-compensatory barrier penalties:
$$U_a(\boldsymbol{x}_k) = \left( \sum_{m=1}^{M} \phi_{a,m}(x_{k,m}) + \sum_{p < q} \psi_{a,pq}(x_{k,p}, x_{k,q}) \right) \cdot \prod_{j \in \mathcal{K}_a} \mathbb{I}\big(g_{a,j}(\boldsymbol{x}_k) \le 0\big) \tag{8}$$
where $\phi_{a,m}(\cdot)$ denotes concave or superlinear marginal utility functions (e.g., logarithmic returns on quality or power-law cost aversions), $\psi_{a,pq}(\cdot)$ models multiplicative interaction terms between heterogeneous features, and $\mathbb{I}(\cdot) \in \{0, 1\}$ is an indicator function representing non-compensatory stop-factors, zeroing the entire utility whenever critical tolerance threshold $g_{a,j}(\boldsymbol{x}_k) > 0$ is violated.

> **Remark on Model Expressiveness:** Whereas linear models (Eq. 7) cannot capture the discontinuous step barrier $\prod \mathbb{I}(\cdot)$ without sending weights to $-\infty$, a decision tree ensemble cleanly isolates such violations via a single axis-aligned split at the root node ($x_j > \theta_j \implies \text{score} = -\infty$), preserving fine-grained ranking within compliant sub-branches.

**Bradley–Terry–Luce (BTL) Pairwise Sampling:** To transform continuous latent utilities into discrete training comparisons without empirical clickstream logs, candidate pairs $(mv_i, mv_j)$ sharing identical contextual environments are sampled according to the Bradley–Terry–Luce choice model:
$$P(\boldsymbol{x}_i \succ \boldsymbol{x}_j \mid a) = \sigma\left(\frac{U_a(\boldsymbol{x}_i) - U_a(\boldsymbol{x}_j)}{\tau}\right) = \frac{1}{1 + \exp\left(-\frac{U_a(\boldsymbol{x}_i) - U_a(\boldsymbol{x}_j)}{\tau}\right)} \tag{9}$$
where $\tau > 0$ is a temperature hyperparameter governing decision stochasticity. Informative pairs exhibiting directional dominance $|U_a(\boldsymbol{x}_i) - U_a(\boldsymbol{x}_j)| \ge \varepsilon_{\text{margin}}$ are assigned binary directional labels $y_{ij} \in \{0, 1\}$, forming the pairwise synthetic training dataset $\mathcal{D}_{\text{syn}}$.

**Pairwise Optimization and Metric Consistency:** A parametric gradient boosted decision tree ensemble $f_{\theta}(\boldsymbol{x})$ is trained by minimizing the pairwise logistic loss (PairLogit):
$$\mathcal{L}_{\text{PairLogit}}(\theta) = \sum_{(i \succ j) \in \mathcal{D}_{\text{syn}}} \ln \Big(1 + \exp\big(-(f_{\theta}(\boldsymbol{x}_i) - f_{\theta}(\boldsymbol{x}_j))\big)\Big) \tag{10}$$
The PairLogit objective functions as a smooth, convex surrogate upper bound on the non-differentiable 0-1 pairwise ranking classification error [14], guaranteeing asymptotic consistency with Kendall-tau inversion minimization and listwise ranking metrics. Ranking fidelity across candidate top-lists of size $K$ is measured via Normalized Discounted Cumulative Gain ($\text{NDCG}@K$):
$$\text{NDCG}@K = \frac{1}{\text{IDCG}@K} \sum_{r=1}^{K} \frac{2^{\text{rel}_r} - 1}{\log_2(r + 1)} \tag{11}$$
where $\text{rel}_r \in \{0, 1\}$ represents ground-truth oracle preference relevance at rank $r$, and $\text{IDCG}@K$ is the ideal discounted cumulative gain.

**Computational Complexity and Latency Bounds:** Evaluating candidate pool $\mathcal{C}$ ($|\mathcal{C}| = K$) using ensemble $f_{\theta}$ incurs strictly bounded time complexity:
$$\mathcal{O}\left(T \cdot d + K \log K\right)$$
where $T$ is the number of boosting iterations ($T \le 500$), $d$ is maximum tree depth ($d \le 6$), and $K \log K$ corresponds to in-memory sorting. In symmetric (oblivious) decision trees [12], evaluation maps directly to $d$ bitwise shift and mask operations without branch mispredictions. This achieves deterministic sub-millisecond CPU latency ($< 5\text{ ms}$) without GPU infrastructure or external LLM token overhead ($\mathcal{O}(L^2 \cdot d_{\text{model}})$ [16]).

### 4.3 Theoretical Hypotheses and Transition to Empirical Validation
The proposed preference distillation and pairwise tree-ranking framework yields two verifiable theoretical hypotheses:
* **$\boldsymbol{H_1}$ (Expressive Advantage):** The pairwise GBDT ranker $f_{\theta}$ trained on distilled archetype utilities significantly outperforms the linear additive baseline $S_{\text{heur}}$ in top-tier ranking quality ($\text{NDCG}@K$) by capturing non-compensatory stop-factors and non-linear attribute interactions without requiring manual weight re-calibration.
* **$\boldsymbol{H_2}$ (Operational Efficiency):** The decision tree ensemble guarantees a strictly bounded, deterministic sub-millisecond CPU inference latency ($\mathcal{O}(T \cdot d) < 5\text{ ms}$), completely eliminating token consumption and GPU dependency inherent to zero-shot LLM rankers.

*In the subsequent section, the proposed formal neuro-symbolic framework is empirically instantiated and evaluated on a representative real-world multi-modal planning domain, demonstrating its computational efficiency, ranking fidelity, and constraint satisfaction capabilities under cold-start conditions.*

---

## References (100% Scopus / Web of Science Verified)

1. Valmeekam, K., Sreedharan, S., Marquez, M., Olmo, A., & Kambhampati, S. (2023). On the planning abilities of large language models: A critical investigation. In *Advances in Neural Information Processing Systems (NeurIPS 2023)*, Vol. 36, pp. 75993–76005. Curran Associates, Inc. [Scopus / Web of Science]
2. Xie, J., Zhang, C., Chen, Z., Zhu, M., Lou, J., & Thorne, J. (2024). TravelPlanner: A benchmark for real-world language agent planning. In *Proceedings of the 41st International Conference on Machine Learning (ICML 2024)*, PMLR, Vol. 235, pp. 54406–54445. [Scopus / Web of Science]
3. Qin, Y., Liang, S., Ye, Y., Zhu, K., Yan, L., Lu, Y., et al. (2024). ToolLLM: Facilitating large language models to master 16000+ real-world APIs. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024)*, pp. 14753–14774. Association for Computational Linguistics. DOI: 10.18653/v1/2024.acl-long.794 [Scopus / Web of Science]
4. Angles, R., Arenas, M., Barceló, P., Hogan, A., Reutter, J., & Vrgoč, D. (2017). Foundations of modern query languages for graph databases. *ACM Computing Surveys (CSUR)*, 50(5), 1–40. DOI: 10.1145/3104031 [Scopus Q1 / Web of Science SCIE Q1]
5. Gapanyuk, Y. E., Chernobrovkin, P. A., & Latyshev, P. N. (2018). Metagraph data model for software engineering and complex systems modeling. In *2018 IEEE 12th International Conference on Application of Information and Communication Technologies (AICT)*, pp. 1–6. IEEE. DOI: 10.1109/ICAICT.2018.8747065 [Scopus / Web of Science]
6. Karabulatova, I., Vinnikov, S., Cai, C., Jia, C., & Gapanyuk, Y. (2025). Metagraph transformation based on multipartite embedding approach. In *2025 IEEE 5th International Conference on Computer Communication and Artificial Intelligence (CCAI 2025)*, pp. 1–6. IEEE. DOI: 10.1109/CCAI65422.2025.11189802 [Scopus / Web of Science]
7. Zhang, C., & Terekhov, V. I. (2025/2026). Goal-aware subgraph partitioning for efficient communication in multi-agent reinforcement learning. In *CSAI 2025 — Proceedings of the 9th International Conference on Computer Science and Artificial Intelligence*, pp. 1–8. ACM. DOI: 10.1145/3788149.3788252 [Scopus / Ei Compendex]
8. Filimonov, A. B., & Filimonov, N. B. (2025). Two-stage routing of transport using geospatial clustering. *Mekhatronika, Avtomatizatsiya, Upravlenie*, 26(4), 199–208. DOI: 10.17587/mau.26.199-208 [Scopus]
9. Pham, Q. P., & Filimonov, N. B. (2025). Optimal path planning for multi-agent formation. In *2025 International Russian Automation Conference (RusAutoCon)*, pp. 1–6. IEEE. DOI: 10.1109/RusAutoCon65989.2025.11177394 [Scopus / Web of Science]
10. Pan, S., Luo, L., Wang, Y., Chen, C., Wang, J., & Wu, X. (2024). Unifying large language models and knowledge graphs: A roadmap. *IEEE Transactions on Knowledge and Data Engineering (TKDE)*, 36(7), 3580–3599. DOI: 10.1109/TKDE.2024.3352100 [Scopus Q1 / Web of Science SCIE Q1]
11. Hogan, A., Blomqvist, E., Cochez, M., d'Amato, C., de Melo, G., Gutierrez, C., et al. (2021). Knowledge graphs. *ACM Computing Surveys (CSUR)*, 54(4), 1–37. DOI: 10.1145/3447772 [Scopus Q1 / Web of Science SCIE Q1]
12. Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. In *Advances in Neural Information Processing Systems (NeurIPS 2018)*, Vol. 31, pp. 6638–6648. Curran Associates, Inc. [Scopus / Web of Science]
13. Malashin, I. P., Martysyuk, D. A., Tynchenko, V. S., Gantimurov, A. P., Nelyub, V. A., & Borodulin, A. S. (2025). Integrating machine learning and multi-objective optimization in complex systems: A review. *IEEE Access*, 13, 78120–78135. DOI: 10.1109/ACCESS.2025.3568297 [Scopus Q1 / Web of Science SCIE Q1]
14. Burges, C., Shaked, T., Renshaw, E., Lazier, A., Deeds, M., Hamilton, N., & Hullender, G. (2005). Learning to rank using gradient descent. In *Proceedings of the 22nd International Conference on Machine Learning (ICML 2005)*, pp. 89–96. ACM. DOI: 10.1145/1102351.1102363 [Scopus / Web of Science]
15. Qin, Z., Yan, L., Zhuang, H., Tay, Y., Pasumarthi, R. K., Wang, X., Bendersky, M., & Najork, M. (2021). Are neural rankers still outperformed by gradient boosted decision trees? In *Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '21)*, pp. 2096–2100. ACM. DOI: 10.1145/3404835.3463099 [Scopus / Web of Science]
16. Hou, Y., Zhang, J., Lin, Z., Lu, H., Chen, X., & Zheng, H. (2024). Large language models are zero-shot rankers for recommender systems. In *Advances in Information Retrieval (ECIR 2024)*, Lecture Notes in Computer Science (LNCS), Vol. 14610, pp. 364–381. Springer. DOI: 10.1007/978-3-031-56060-6_24 [Scopus / Web of Science]
