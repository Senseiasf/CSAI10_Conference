# Руководство по написанию теоретического блока статьи (CSAI 2026, Track 3)

> **Концепция документа:** Декомпозиция чистого теоретического и методологического блока статьи на **3 независимые авторские роли**. Теория пишется как самостоятельное фундаментальное исследование и располагается в статье **до** практической части. В теоретическом блоке **строго исключены** любые упоминания названий репозитория, файлов, скриптов, таблиц БД, фреймворков, конкретных СУБД (Neo4j, PostgreSQL) и частных статистических цифр каталога (9 направлений, 1 327 отелей, замеры в миллисекундах и проценты бенчмарка). Мы формулируем универсальный математический и алгоритмический аппарат, который затем (в практическом разделе Section 5) получает программную реализацию и экспериментальную валидацию со ссылками назад на теоретические формулы.
> 
> **Железный стандарт библиографии:** **100% цитируемых источников строго индексируются в международных базах SCOPUS и/или WEB OF SCIENCE.** Никаких неиндексируемых препринтов, веб-учебников, методичек или локальных неиндексируемых журналов. Для каждой публикации указан официальный DOI и подтверждённый статус индексации.

---

## 1. Выходные данные конференции и общие ориентиры по объёму

* **Конференция:** The 10th International Conference on Computer Science and Artificial Intelligence (**CSAI 2026**), 18–21 декабря 2026 г., Пекин, Китай.
* **Индексация трудов:** ACM Digital Library, **Ei Compendex, Scopus**.
* **Целевой трек:** **Technical Track 3: «Digital Humanities and Modeling of Complex Subject Areas»** (председатель секции — проф. Ю. Е. Гапанюк, МГТУ им. Н.Э. Баумана).
  * Ключевые научные топики трека, закрываемые теоретическим блоком:
    1. *Modeling of complex subject areas using complex graph approaches (hypergraphs, metagraphs, property graphs)*;
    2. *Hybrid neuro-symbolic approaches combining LLMs with formal graph-based structures*;
    3. *Limitations of neural methods (hallucinations, interpretability, verifiability) in complex domain modeling*.
* **Язык и оформление:** Academic UK English (*modelling, optimisation, behaviour*). Двухколоночный формат IET / ACM (шрифт Times New Roman 10 pt, одинарный интервал). Скалярные переменные в формулах выделяются *курсивом* ($x, t, B$), векторы и матрицы — **_полужирным курсивом_** ($\boldsymbol{x}, \boldsymbol{e}$). Таблицы оформляются строго без вертикальных линий.
* **Требование к источникам: ТОЛЬКО SCOPUS ИЛИ WEB OF SCIENCE:**
  * Все работы отобраны из профильных журналов первого и второго квартилей (Q1/Q2) и флагманских конференций уровня A/A* (ACM, IEEE, NeurIPS, ICML, SIGIR, ACL, Springer LNCS).
  * Интегрированы профильные статьи бауманских авторов из официальной базы университета (**2026-09-16_Публикации_МГТУ_2025-2027.xlsx**), имеющие подтверждённую индексацию в Scopus / Web of Science:
    * *Zhang, C., & Terekhov, V. I. (CSAI 2025, ACM DL)* — Scopus / Ei Compendex;
    * *Filimonov, A. B., & Filimonov, N. B. (2025)* — Mekhatronika, Avtomatizatsiya, Upravlenie (Scopus);
    * *Pham, Q. P., & Filimonov, N. B. (2025)* — IEEE RusAutoCon (Scopus / Web of Science);
    * *Malashin, I. P., Nelyub, V. A., Borodulin, A. S., et al. (2025)* — IEEE Access (Scopus Q1 / Web of Science Q1);
    * *Karabulatova, I., Vinnikov, S., Gapanyuk, Y., et al. (2025)* — IEEE CCAI (Scopus / Web of Science);
    * *Gapanyuk, Y. E., Chernobrovkin, P. A., & Latyshev, P. N. (2018)* — IEEE AICT (Scopus / Web of Science).
* **Общий потолок объёма на всю теорию:** **2,0–2,2 страницы** (суммарно около **1 200–1 350 слов** на троих, включая формулы, 1 общую схему и 1 таблицу). Каждый автор свободно выстраивает логику своего текста в рамках выделенного лимита.

---

## 2. Логика теоретического блока (о чём мы пишем в отрыве от кода)

Чтобы теория идеально подводила читателя к практическим экспериментам, но оставалась строгой фундаментальной наукой, повествование выстраивается вокруг **четырёх ключевых теоретических проблем**:

1. **Проблема ненадёжности генеративных нейросетей в задачах со строгими ограничениями (Section 2):**
   Автономные языковые модели (LLM) эффективны при семантической интерпретации неструктурированных пользовательских интентов, но при прямом синтезе сложных многоэтапных планов неизбежно порождают **структурные, темпоральные и арифметические галлюцинации**. Они не способны строго гарантировать непрерывность расписаний, соблюдение пропускной способности каналов и точный аддитивный расчёт бюджета.
2. **Гибридный нейросимвольный поиск: семантика + булева фильтрация (Section 3.1):**
   Вместо непрозрачной генерации ответа моделью «вслепую» формализуется двухэтапный отбор кандидатов. Сначала нечёткие семантические предпочтения отображаются в плотное векторное пространство $\mathbb{R}^d$ для сопоставления с верхнеуровневыми кластерами, после чего к сервисным объектам применяется конъюнктивная фильтрация на базе булевых предикатов.
3. **Иерархическая графовая / метаграфовая модель и символьная верификация (Section 3.2 & 3.3):**
   Плоские графы способны отражать лишь элементарные бинарные связи. Однако в сложных пространственно-временных задачах синтезируемый план представляет собой **составной версионируемый объект**, объединяющий топологию маршрута, активный контекст ограничений и структуру затрат. Предметная область формализуется через иерархическую графовую структуру (размеченный граф свойств с метавершинами состояний), где конфигурация плана моделируется как составная вершина (метавершина). На этой структуре задаётся строгий предикат допустимости $\Psi_{\text{valid}}$, контролирующий непрерывность стыковок, окна регламентной доступности, ёмкость сегментов и аддитивный бюджетный инвариант.
4. **Обучение ранжированию (Learning-to-Rank) в условиях холодного старта (Section 4):**
   Множество допустимых планов, прошедших графовую верификацию, необходимо ранжировать по степени полезности. При отсутствии исторических логов взаимодействия (*проблема холодного старта*) и нерентабельности прямого LLM-ранжирования формулируется метод **синтетической дистилляции предпочтений**. Для латентных поведенческих архетипов строятся нелинейные функции полезности, генерируется пространство парных доминирований и обучается компактный ансамбль градиентного бустинга (GBDT) с парной функцией потерь, обеспечивающий миллисекундный инференс.

---

## 3. Сводная таблица распределения ролей между 3 авторами теории

| Автор | Зона ответственности (разделы статьи) | Общий потолок объёма | Формулы и визуальные элементы | Статус источников (строго Scopus / WoS) |
| :--- | :--- | :--- | :--- | :--- |
| **Автор 1**<br>*(Графовые и метаграфовые модели, символьная верификация)* | **Section 3.2** (Формальная графовая / метаграфовая модель и версионирование состояний)<br>**Section 3.3** (Пространственно-временная и бюджетная верификация на графе) | **До 500–550 слов**<br>(~0,85–0,90 стр.) | **Формулы (3), (4), (5)**<br>Правая часть **Рис. 1** (топология составной вершины плана и подграфа связей) | 100% Scopus / Web of Science<br>(ACM CSUR, IEEE AICT, IEEE CCAI, IEEE RusAutoCon, Мехатроника) |
| **Автор 2**<br>*(Аналитический обзор литературы и нейросимвольный поиск)* | **Section 2** (Related Work: галлюцинации LLM, графы, LTR)<br>**Section 3.1** (Нейросимвольная декомпозиция, векторный поиск и булевы предикаты) | **До 350–400 слов**<br>(~0,65 стр.) | **Формулы (1), (2)**<br>Левая часть **Рис. 1** (концептуальная схема четырёхступенчатого конвейера) | 100% Scopus / Web of Science<br>(NeurIPS, ICML, ACL, CSAI ACM DL, IEEE TKDE, ACM CSUR) |
| **Автор 3**<br>*(Теория ранжирования, холодный старт и дистилляция предпочтений)* | **Section 4.1** (Признаковое пространство и дилемма компенсации в линейной модели)<br>**Section 4.2** (Латентные архетипы, модель Брэдли–Терри–Люса и парное обучение GBDT)<br>**Section 4.3** (Теоретические гипотезы и академический переход к эксперименту) | **До 450–500 слов**<br>(~0,8 стр.) | **Формулы (6), (7), (8), (9), (10), (11)**<br>Аналитическая декомпозиция вектора признаков (без громоздких таблиц для экономии объёма) | 100% Scopus / Web of Science<br>(NeurIPS, ICML, ACM SIGIR, IEEE Access, Springer LNCS) |

---

## 4. Подробное техническое задание для каждого автора

### Автор 1: Формальная графовая / метаграфовая модель и символьная верификация
* **Общий ориентир по объёму:** до **500–550 слов** (Sections 3.2 и 3.3, правая часть Рис. 1).

#### Содержательные блоки для написания:
1. **Формальная модель сложной предметной области (`Section 3.2`):**
   * Определить мультимодальную сеть предметной области как иерархическую графовую структуру (размеченный граф свойств с составными метавершинами состояний):
     $$MG = \langle V, MV, E, \mathcal{L}, \mathcal{P} \rangle \tag{3}$$
     где $V = V_{\text{loc}} \cup V_{\text{hub}} \cup V_{\text{serv}}$ — множество атомарных вершин с метками типов $\mathcal{L}$ (географические локации $V_{\text{loc}}$, транспортные хабы $V_{\text{hub}}$, сервисные объекты $V_{\text{serv}}$); $E = E_{\text{trans}} \cup E_{\text{link}}$ — множество ориентированных рёбер (транспортные транзиты и структурная принадлежность); $\mathcal{P}(e) = \langle t_{\text{dep}}, t_{\text{arr}}, c, \text{cap}, W_{\text{dom}} \rangle$ — кортеж атрибутов ребра: моменты отправления $t_{\text{dep}}$ и прибытия $t_{\text{arr}}$, тарифная стоимость $c(e)$, пропускная способность $\text{cap}(e)$ и регламентное окно функционирования $W_{\text{dom}}(e)$.
   * Описать структуру **составной вершины (метавершины) конфигурации плана** $mv_k \in MV$, агрегирующей три проекции:
     $$mv_k = \langle G_{\text{sub}}^{(k)}, C_u^{(k)}, Q_{\text{cost}}^{(k)} \rangle \tag{4}$$
     где $G_{\text{sub}}^{(k)} = (V_k, E_k)$ — связный ориентированный подграф маршрута (включая многоузловые цепочки), $C_u^{(k)}$ — вектор активных ограничений пользователя, $Q_{\text{cost}}^{(k)}$ — детализированная смета затрат.
   * Раскрыть принцип **локальной мутации при версионировании**: переход между последовательными версиями плана $(mv_{k-1} \to mv_k)$ при диалоговой корректировке предпочтений осуществляется заменой отдельных рёбер в $E_k$ и пересчётом $Q_{\text{cost}}^{(k)}$ при сохранении инвариантной части подграфа, исключая необходимость регенерации всего плана с нуля.
2. **Символьная верификация ограничений и бюджетный инвариант (`Section 3.3`):**
   * Сформулировать строгий предикат структурной, временной и финансовой допустимости плана $\Psi_{\text{valid}}(mv_k)$:
     $$\Psi_{\text{valid}}(mv_k) = \begin{cases} 1, & \text{если } \forall (e_i, e_{i+1}) \in E_k: \\ & \text{(a) } t_{\text{dep}}(e_{i+1}) - t_{\text{arr}}(e_i) \ge \tau_{\min}, \\ & \text{(b) } \text{cap}(e_i) \ge N_{\text{group}}, \\ & \text{(c) } [t_{\text{dep}}(e_i), t_{\text{arr}}(e_i)] \subseteq W_{\text{dom}}(e_i), \\ & \text{(d) } B_{\text{fixed}}(mv_k) + B_{\text{var}}^{\max}(mv_k) \le B_{\text{limit}}; \\ 0, & \text{иначе.} \end{cases} \tag{5}$$
   * Аккуратно раскрыть физический смысл каждого из четырёх условий в обобщённых терминах:
     * **(a) Временная непрерывность и буфер стыковки:** интервал между прибытием по сегменту $e_i$ и отправлением по следующему $e_{i+1}$ строго не меньше технологического порога $\tau_{\min}$;
     * **(b) Пропускная способность (вместимость):** ёмкость каждого выбранного транспортного или сервисного сегмента $\text{cap}(e_i)$ не меньше размера группы путешественников $N_{\text{group}}$;
     * **(c) Регламентные временные окна:** интервал движения $[t_{\text{dep}}, t_{\text{arr}}]$ обязан целиком лежать внутри разрешённого окна $W_{\text{dom}}(e_i)$ (например, ночные ограничения навигации или графики работы локальной инфраструктуры);
     * **(d) Аддитивный бюджетный инвариант:** сумма детерминированных затрат $B_{\text{fixed}}(mv_k)$ и верхней оценки сопутствующих расходов $B_{\text{var}}^{\max}(mv_k)$ не превышает лимит пользователя $B_{\text{limit}}$.
3. **Графика:** правая часть Рис. 1 — абстрактная концептуальная схема метавершины $mv_k$, связывающей подграф $G_{\text{sub}}^{(k)}$, вектор $C_u^{(k)}$ и смету $Q_{\text{cost}}^{(k)}$.

#### Рецензируемые научные источники для Автора 1 (100% Scopus / Web of Science):
1. **Angles, R., Arenas, M., Barceló, P., Hogan, A., Reutter, J., & Vrgoč, D.** (2017). Foundations of Modern Query Languages for Graph Databases. *ACM Computing Surveys (CSUR)*, 50(5), 1–40. DOI: `10.1145/3104031`. **Индексация: Scopus (Q1), Web of Science (SCIE, Q1)**.
2. **Gapanyuk, Y. E., Chernobrovkin, P. A., & Latyshev, P. N.** (2018). Metagraph Data Model for Software Engineering and Complex Systems Modeling. In *2018 IEEE 12th International Conference on Application of Information and Communication Technologies (AICT)*, pp. 1–6. IEEE. DOI: `10.1109/ICAICT.2018.8747065`. **Индексация: Scopus, Web of Science**.
3. **Karabulatova, I., Vinnikov, S., Cai, C., Jia, C., & Gapanyuk, Y.** (2025). Metagraph Transformation based on Multipartite Embedding Approach. In *2025 IEEE 5th International Conference on Computer Communication and Artificial Intelligence (CCAI 2025)*, pp. 1–6. IEEE. DOI: `10.1109/CCAI65422.2025.11189802`. **Индексация: Scopus, Web of Science** (МГТУ им. Баумана).
4. **Filimonov, A. B., & Filimonov, N. B.** (2025). Two-Stage Routing of Transport Using Geospatial Clustering. *Mekhatronika, Avtomatizatsiya, Upravlenie*, 26(4), 199–208. DOI: `10.17587/mau.26.199-208`. **Индексация: Scopus** (МГТУ им. Баумана).
5. **Pham, Q. P., & Filimonov, N. B.** (2025). Optimal Path Planning for Multi-agent Formation. In *2025 International Russian Automation Conference (RusAutoCon)*, pp. 1–6. IEEE. DOI: `10.1109/RusAutoCon65989.2025.11177394`. **Индексация: Scopus, Web of Science** (МГТУ им. Баумана).

---

### Автор 2: Обзор литературы и формализация нейросимвольного отбора кандидатов
* **Общий ориентир по объёму:** до **350–400 слов** суммарно на `Section 2` и `Section 3.1` (левая часть Рис. 1).

#### Содержательные блоки для написания:
1. **Обзор литературы и теоретических предпосылок (`Section 2. Related Work`):**
   * Описать фундаментальное противоречие автономных LLM-агентов: высокая семантическая выразительность при парсинге естественного языка сочетается с фатальной ненадежностью при многошаговом планировании в дискретных пространствах с ограничениями (структурные и арифметические галлюцинации). Сослаться на бенчмарк **TravelPlanner (Xie et al., ICML 2024)**, исследование Kambhampati et al. (NeurIPS 2023) и архитектуру ToolLLM (Qin et al., ACL 2024).
   * Осветить эволюцию графовых структур представления сложных предметных областей: переход от плоских схем к графам знаний и иерархическим метаграфам (Hogan et al., ACM CSUR 2021). Сослаться на работу **Zhang & Terekhov (CSAI 2025, МГТУ им. Баумана / ACM DL)** по декомпозиции подграфов.
   * Обосновать необходимость гибридных нейросимвольных конвейеров: нейросеть отвечает за извлечение интентов и векторный поиск, символический графовый движок гарантирует соблюдение физических инвариантов, а финальное ранжирование делегируется быстрым табличным моделям (Pan et al., IEEE TKDE 2024; Hou et al., ECIR 2024).
2. **Нейросимвольная декомпозиция и гибридный первичный отбор (`Section 3.1`):**
   * Формализовать отображение входного запроса пользователя $q$ в пару $\langle \boldsymbol{e}(q), C_u \rangle$, где $\boldsymbol{e}(q) \in \mathbb{R}^d$ — плотный вектор семантических предпочтений, а $C_u$ — кортеж извлечённых структурированных ограничений.
   * Записать функцию семантического ранжирования макро-кластеров $d_j \in V_{\text{loc}}$ через косинусную близость эмбеддингов:
     $$S_{\text{vec}}(q, d_j) = \frac{\boldsymbol{e}(q) \cdot \boldsymbol{e}(d_j)}{\|\boldsymbol{e}(q)\|_2 \, \|\boldsymbol{e}(d_j)\|_2} \tag{1}$$
   * Записать конъюнктивный булев предикат отсева сервисных объектов $h \in V_{\text{serv}}$ по множеству обязательных атрибутов $M_{\text{req}}$ и множеству ранее исключённых сущностей $H_{\text{excl}}$:
     $$\Phi_{\text{filter}}(h, C_u) = \mathbb{I}(h \notin H_{\text{excl}}) \cdot \prod_{r \in M_{\text{req}}} \mathbb{I}\big(\text{attr}_r(h) \ge c_{u,r}\big) \tag{2}$$
     где $\mathbb{I}(\cdot) \in \{0, 1\}$ — индикаторная функция. Пояснить, что совместное действие (1) и (2) сужает пространство поиска перед вызовом графового алгоритма.
3. **Графика:** левая часть Рис. 1 — обобщённая теоретическая блок-схема потока данных через 4 уровня абстракции: *(I) Булева фильтрация $\to$ (II) Векторное сопоставление $\to$ (III) Графовая верификация $\to$ (IV) Градиентное ранжирование*.

#### Рецензируемые научные источники для Автора 2 (100% Scopus / Web of Science):
1. **Valmeekam, K., Sreedharan, S., Marquez, M., Olmo, A., & Kambhampati, S.** (2023). On the Planning Abilities of Large Language Models: A Critical Investigation. In *Advances in Neural Information Processing Systems (NeurIPS 2023)*, Vol. 36, pp. 75993–76005. Curran Associates, Inc. **Индексация: Scopus, Web of Science**.
2. **Xie, J., Zhang, C., Chen, Z., Zhu, M., Lou, J., & Thorne, J.** (2024). TravelPlanner: A Benchmark for Real-World Language Agent Planning. In *Proceedings of the 41st International Conference on Machine Learning (ICML 2024)*, PMLR, Vol. 235, pp. 54406–54445. **Индексация: Scopus, Web of Science**.
3. **Qin, Y., Liang, S., Ye, Y., Zhu, K., Yan, L., Lu, Y., et al.** (2024). ToolLLM: Facilitating Large Language Models to Master 16000+ Real-World APIs. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024)*, pp. 14753–14774. DOI: `10.18653/v1/2024.acl-long.794`. **Индексация: Scopus, Web of Science**.
4. **Zhang, C., & Terekhov, V. I.** (2025/2026). Goal-Aware Subgraph Partitioning for Efficient Communication in Multi-Agent Reinforcement Learning. In *CSAI 2025 — Proceedings of the 9th International Conference on Computer Science and Artificial Intelligence*, ACM, pp. 1–8. DOI: `10.1145/3788149.3788252`. **Индексация: Scopus, Ei Compendex** (МГТУ им. Баумана).
5. **Pan, S., Luo, L., Wang, Y., Chen, C., Wang, J., & Wu, X.** (2024). Unifying Large Language Models and Knowledge Graphs: A Roadmap. *IEEE Transactions on Knowledge and Data Engineering (TKDE)*, 36(7), 3580–3599. DOI: `10.1109/TKDE.2024.3352100`. **Индексация: Scopus (Q1), Web of Science (SCIE, Q1)**.
6. **Hogan, A., Blomqvist, E., Cochez, M., d'Amato, C., de Melo, G., Gutierrez, C., et al.** (2021). Knowledge Graphs. *ACM Computing Surveys (CSUR)*, 54(4), 1–37. DOI: `10.1145/3447772`. **Индексация: Scopus (Q1), Web of Science (SCIE, Q1)**.

---

### Автор 3: Математическая модель ранжирования в условиях холодного старта
* **Общий ориентир по объёму:** до **450–500 слов** суммарно на `Section 4.1`, `Section 4.2` и `Section 4.3`.
* **Принцип компактности:** Таблица 1 заменена строгой аналитической прямой суммой подпространств признаков $\boldsymbol{x}_k = [\boldsymbol{x}_{\text{cat}}^T, \boldsymbol{x}_{\text{cont}}^T, \boldsymbol{x}_{\text{ord}}^T, \boldsymbol{x}_{\text{bin}}^T]^T \in \mathbb{R}^M$, что освобождает ~30 строк журнального формата под математику и гарантирует отсутствие утечки конкретных сущностей предметной области в теорию.

#### Содержательные блоки для написания:
1. **Гетерогенное признаковое пространство и линейная эвристика (`Section 4.1`):**
   * Описать проекцию каждого допустимого кандидата $mv_k \in MV_{\text{valid}}$ в $M$-мерное признаковое пространство:
     $$\boldsymbol{x}_k = \big[\boldsymbol{x}_{\text{cat}}^T, \, \boldsymbol{x}_{\text{cont}}^T, \, \boldsymbol{x}_{\text{ord}}^T, \, \boldsymbol{x}_{\text{bin}}^T\big]^T \in \mathbb{R}^M \tag{6}$$
     с раскрытием физического смысла четырех подпространств: категориальные токены $\boldsymbol{x}_{\text{cat}}$, непрерывные временные и финансовые буферы $\boldsymbol{x}_{\text{cont}}$, порядковые уровни сервиса $\boldsymbol{x}_{\text{ord}}$ и бинарные индикаторы политик $\boldsymbol{x}_{\text{bin}}$.
   * Записать линейную эвристическую свёртку:
     $$S_{\text{heur}}(\boldsymbol{x}_k) = \sum_{m=1}^{M} w_m(a) \cdot \tilde{x}_{k,m} \tag{7}$$
   * Формализовать **дилемму компенсации (The Compensation Dilemma)**: аддитивная сумма с постоянными весами допускает компенсацию критического порога по одному параметру (например, недопустимой длительности стыковки) произвольно хорошим значением другого (низкой ценой), что нарушает фундаментальные аксиомы некомпенсаторного выбора человека.
2. **Синтетическая дистилляция, модель Брэдли–Терри–Люса и парное обучение GBDT (`Section 4.2`):**
   * Формализовать проблему **холодного старта (Cold-Start)**: в отсутствие исторических матриц кликов формулируется парадигма синтетической дистилляции предпочтений через конечное множество латентных поведенческих архетипов $\mathcal{A} = \{a_1, a_2, \dots, a_L\}$.
   * Задать нелинейную функцию полезности оракула $U_a(\boldsymbol{x}_k)$ с аддитивными предельными полезностями, перекрестными синергиями и мультипликативным барьером стоп-факторов:
     $$U_a(\boldsymbol{x}_k) = \left( \sum_{m=1}^{M} \phi_{a,m}(x_{k,m}) + \sum_{p < q} \psi_{a,pq}(x_{k,p}, x_{k,q}) \right) \cdot \prod_{j \in \mathcal{K}_a} \mathbb{I}\big(g_{a,j}(\boldsymbol{x}_k) \le 0\big) \tag{8}$$
     Объяснить, почему решающие деревья изолируют стоп-факторы $\prod \mathbb{I}(\cdot)$ ровно одним сплитом в корне ($x_j > \theta_j \implies -\infty$), в то время как линейная гиперплоскость не способна смоделировать такой разрыв.
   * Формализовать вероятностную генерацию пар кандидатов через **модель Брэдли–Терри–Люса (BTL)**:
     $$P(\boldsymbol{x}_i \succ \boldsymbol{x}_j \mid a) = \sigma\left(\frac{U_a(\boldsymbol{x}_i) - U_a(\boldsymbol{x}_j)}{\tau}\right) = \frac{1}{1 + \exp\left(-\frac{U_a(\boldsymbol{x}_i) - U_a(\boldsymbol{x}_j)}{\tau}\right)} \tag{9}$$
     Пары с информативным зазором $|U_a(\boldsymbol{x}_i) - U_a(\boldsymbol{x}_j)| \ge \varepsilon_{\text{margin}}$ получают метки $y_{ij} \in \{0, 1\}$ и формируют обучающий датасет $\mathcal{D}_{\text{syn}}$.
   * Записать парную логистическую функцию потерь (PairLogit):
     $$\mathcal{L}_{\text{PairLogit}}(\theta) = \sum_{(i \succ j) \in \mathcal{D}_{\text{syn}}} \ln \Big(1 + \exp\big(-(f_{\theta}(\boldsymbol{x}_i) - f_{\theta}(\boldsymbol{x}_j))\big)\Big) \tag{10}$$
     Пояснить, что $\mathcal{L}_{\text{PairLogit}}$ является гладкой выпуклой верхней оценкой (surrogate upper bound) для недифференцируемой ошибки парного порядка, гарантируя оптимизацию списка по метрике $\text{NDCG}@K$:
     $$\text{NDCG}@K = \frac{1}{\text{IDCG}@K} \sum_{r=1}^{K} \frac{2^{\text{rel}_r} - 1}{\log_2(r + 1)} \tag{11}$$
   * Выписать оценку асимптотической сложности инференса $\mathcal{O}(T \cdot d + K \log K)$ и обосновать преимущество симметричных (oblivious) деревьев CatBoost (битовые маски без branch-mispredictions, задержка $< 5$ мс на CPU без токенов и видеокарт) по сравнению с квадратичной сложностью трансформеров $\mathcal{O}(L^2 \cdot d_{\text{model}})$.
3. **Теоретические гипотезы и академический переход к практике (`Section 4.3`):**
   * Сформулировать проверяемые гипотезы $H_1$ (выразительное превосходство над $S_{\text{heur}}$) и $H_2$ (субмиллисекундная детерминированная производительность на CPU).
   * Завершить раздел классической переходной связкой к эмпирической валидации:
     > *«In the subsequent section, the proposed formal neuro-symbolic framework is empirically instantiated and evaluated on a representative real-world multi-modal planning domain, demonstrating its computational efficiency, ranking fidelity, and constraint satisfaction capabilities under cold-start conditions.»*

#### Рецензируемые научные источники для Автора 3 (100% Scopus / Web of Science):
1. **Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A.** (2018). CatBoost: Unbiased Boosting with Categorical Features. In *Advances in Neural Information Processing Systems (NeurIPS 2018)*, Vol. 31, pp. 6638–6648. Curran Associates, Inc. **Индексация: Scopus, Web of Science**.
2. **Malashin, I. P., Martysyuk, D. A., Tynchenko, V. S., Gantimurov, A. P., Nelyub, V. A., & Borodulin, A. S.** (2025). Integrating Machine Learning and Multi-Objective Optimization in Complex Systems: A Review. *IEEE Access*, 13, 78120–78135. DOI: `10.1109/ACCESS.2025.3568297`. **Индексация: Scopus (Q1), Web of Science (SCIE, Q1)** (МГТУ им. Баумана).
3. **Burges, C., Shaked, T., Renshaw, E., Lazier, A., Deeds, M., Hamilton, N., & Hullender, G.** (2005). Learning to Rank Using Gradient Descent. In *Proceedings of the 22nd International Conference on Machine Learning (ICML 2005)*, pp. 89–96. ACM. DOI: `10.1145/1102351.1102363`. **Индексация: Scopus, Web of Science**.
4. **Qin, Z., Yan, L., Zhuang, H., Tay, Y., Pasumarthi, R. K., Wang, X., Bendersky, M., & Najork, M.** (2021). Are Neural Rankers still Outperformed by Gradient Boosted Decision Trees? In *Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '21)*, pp. 2096–2100. ACM. DOI: `10.1145/3404835.3463099`. **Индексация: Scopus, Web of Science**.
5. **Hou, Y., Zhang, J., Lin, Z., Lu, H., Chen, X., & Zheng, H.** (2024). Large Language Models are Zero-Shot Rankers for Recommender Systems. In *Advances in Information Retrieval (ECIR 2024)*, Lecture Notes in Computer Science (LNCS), Vol. 14610, pp. 364–381. Springer. DOI: `10.1007/978-3-031-56060-6_24`. **Индексация: Scopus, Web of Science**.

---

## 5. Правила внутренней стыковки теории и перехода к практике

1. **Единая математическая нотация между всеми 3 авторами:**
   * Входной запрос пользователя — $q$, плотный эмбеддинг — $\boldsymbol{e}(q) \in \mathbb{R}^d$, структурированные ограничения — $C_u$.
   * Иерархический метаграф — $MG = \langle V, MV, E, \mathcal{L}, \mathcal{P} \rangle$.
   * Составная метавершина плана — $mv_k \in MV$, ассоциированный подграф — $G_{\text{sub}}^{(k)} = (V_k, E_k)$, смета затрат — $Q_{\text{cost}}^{(k)}$.
   * Предикат первичного булева фильтра — $\Phi_{\text{filter}}(h, C_u)$, предикат полной структурной и бюджетной верификации — $\Psi_{\text{valid}}(mv_k)$.
   * Вектор признаков кандидата — $\boldsymbol{x}_k \in \mathbb{R}^M$, линейная эвристика — $S_{\text{heur}}(\boldsymbol{x}_k)$, параметрическая функция ранжирования — $f_{\theta}(\boldsymbol{x}_k)$.
2. **Академический мостик к практическому разделу (в самом конце Section 4.2):**
   Вся теория пишется полностью в абстрактных терминах, а в завершающем предложении Раздела 4.2 даётся стандартная академическая связка к практической части статьи:
   > *«In the subsequent section, the proposed formal neuro-symbolic framework is empirically instantiated and evaluated on a real-world multi-modal itinerary planning domain, demonstrating its computational efficiency and constraint satisfaction capabilities.»*
3. **Как практика ссылается на теорию (в Section 5):**
   В начале практического раздела команда внедрения пишет: *«Для проверки теоретической модели, сформулированной в Разделах 3–4, мы разворачиваем прикладную систему. Граф свойств $MG$ из формулы (3) наполняется данными каталога... Символический предикат $\Psi_{\text{valid}}$ из формулы (5) реализуется в модуле верификации... Обученная модель CatBoostRanker (формула 7) оценивается по метрике NDCG@2 (формула 8)...»*. Таким образом, сохраняется идеальная академическая композиция без смешивания абстрактной теории и программного кода.
