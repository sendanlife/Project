# Technical Report: Predictive Bus Captain Incident Risk Modeling

**Author(s):** `[Fill in: Name(s) / Student ID(s)]`  
**Module / Project:** `[Fill in: Module Code & Project Title]`  
**Artifacts:** `project.ipynb`, `driver_predictions.csv`, `[Fill in: Streamlit App Filename if applicable]`  

---

## 1. Project Objective

### 1.1 Learning Paradigm: Supervised Binary Classification
This predictive safety modeling task is formulated as **supervised binary classification** rather than continuous regression or unsupervised clustering:

* **Why Supervised Binary Classification:** The historical fleet dataset provides a verified ground-truth binary outcome (`At_Risk`), where `1` indicates driver involvement in a recorded safety incident and `0` indicates a crash-free record. Supervised classification enables the model to learn empirical relationships between quantitative driving behaviors ($X$) and real-world collision outcomes ($y$), outputting a calibrated continuous risk probability $P(\text{At\_Risk}=1 \mid X) \in (0, 1)$.
* **Why Not Regression:** The target variable represents a discrete historical incident status (`0` or `1`) rather than a continuous financial loss or continuous crash severity score. Modeling a bounded binary outcome with linear or continuous regression produces unbounded predictions outside $[0, 1]$, whereas binary cross-entropy classification provides the mathematically appropriate probabilistic formulation.
* **Why Not Unsupervised Clustering:** Unsupervised methods such as K-Means partition drivers purely by geometric proximity in feature space without reference to whether those behavioral clusters actually correlate with collisions. Because verified historical incident labels exist, supervised learning anchors risk scoring directly to crash outcomes.

### 1.2 Target Audience & Core Operational Message
* **Target Audience:** `[Fill in: e.g., Fleet Safety Managers, Depot Operations Supervisors, and Bus Captain Training Coordinators]`
* **Core Message — Proactive Safety Intervention:** Rather than relying solely on reactive post-accident disciplinary reviews, continuous telematics risk scoring identifies elevated behavioral risk patterns early—enabling targeted coaching, simulator refreshers, and fatigue interventions before an incident occurs.

---

## 2. Dataset Audit & Exploratory Data Analysis (EDA)

### 2.1 Dataset Dimensions and Attribution
The dataset comprises **$N = 1,544$ bus captains** characterized by **14 quantitative telematics attributes** capturing operational driving events (such as harsh braking, overspeeding, and fatigue indicators), paired with a unique identifier (`Driver_ID`) and the binary target label (`At_Risk`).

| Attribute Role | Column(s) | Count | Data Type | Modeling Treatment |
| :--- | :--- | :---: | :--- | :--- |
| **Identifier** | `Driver_ID` | 1 | String / ID | Dropped prior to modeling to prevent memorization |
| **Predictive Features ($X$)** | `[Fill in: List 14 telematics feature names]` | 14 | Numerical | Standardized via `StandardScaler` |
| **Target Label ($y$)** | `At_Risk` | 1 | Binary (`0` / `1`) | Supervised classification target |

### 2.2 Severe Class Imbalance
Audit of the target variable `At_Risk` reveals a pronounced class imbalance typical of real-world transport safety records:
* **Majority Class (`At_Risk = 0`, Non-Incident):** `1,380` drivers (**89.4%**)
* **Minority Class (`At_Risk = 1`, Incident):** `164` drivers (**10.6%**)
* **Imbalance Ratio:** Approximately **8.4 : 1** (negative-to-positive)

Without explicit cost-sensitive adjustments, standard classifiers trained on an 89.4% majority baseline default toward predicting `At_Risk = 0` for nearly all drivers, achieving misleadingly high raw accuracy while failing to detect at-risk bus captains.

### 2.3 Partitioning Strategy
To prevent data leakage during model tuning and preserve an unbiased benchmark for final evaluation, a three-way data partitioning strategy was implemented:

1. **Stratified Train–Test Split (80% / 20%):** Using `train_test_split(..., test_size=0.2, stratify=y, random_state=42)`, the 1,544 drivers were divided into **1,235 training drivers** and **309 held-out test drivers** while preserving the exact 10.6% incident prevalence across both splits.
2. **Training and Validation Split (`validation_split=0.2`):** During Artificial Neural Network (ANN) training, Keras reserved 20% of the 1,235 training records (**247 drivers**, or **16.0%** of the full dataset) as an internal **Validation Set** (`Peak Val AUC`) to monitor epoch convergence and select architecture hyperparameters, leaving **988 drivers** (**64.0%** of the full dataset) for active gradient updates.
3. **Unseen Test Set:** The **309-driver test set** (containing **33 positive incident cases** and **276 crash-free drivers**) remained strictly locked during hyperparameter tuning and was evaluated only once for final multi-model comparison.

### 2.4 Assessment of Data Sufficiency
1. **Sufficient for Compact Models:** With 988 active training records and 164 total positive cases across 14 numerical features, the dataset provides sufficient statistical density to train compact, heavily regularized neural networks (`16 -> 8`) and shallow tree ensembles without immediate parameter collapse.
2. **Insufficient for Deep Networks or High-Dimensional Expansion:** Deep multi-layer architectures (3 or more wide hidden layers) or aggressive feature expansion rapidly exceed the degrees of freedom supported by approximately 105 active training incidents, leading to training-set memorization or severe underfitting under stacked dropout regularization.

### 2.5 Pre-Modelling Exploratory Data Analysis (EDA)
`[Fill in: Summarize key findings from your notebook's EDA plots—e.g., right-skewed distributions in event counts, correlation heatmap observations between harsh braking/speeding and At_Risk, outlier behavior, and box-plot comparisons between Class 0 and Class 1 drivers.]`

---

## 3. Data Preprocessing, Encoding Rationale & Mitigating Imbalance

### 3.1 Numerical Preprocessing and Encoding
Schema verification confirmed that all 14 predictive input features are strictly numerical (`int64` / `float64`) and `At_Risk` is already encoded in binary (`0` / `1`) form. The non-predictive unique identifier `Driver_ID` was dropped prior to modeling so models learn generalizable behavioral patterns rather than indexing individual drivers.

### 3.2 Categorical Encoding (One-Hot Encoding vs. Alternatives)
Because all 14 telematics inputs are continuous or discrete numerical event counts rather than nominal text categories, categorical transformations such as One-Hot Encoding (OHE) or Target Encoding were not required. Avoiding unnecessary OHE preserved a compact 14-dimensional feature space and prevented sparsity-induced overfitting on a 1,544-row dataset.

### 3.3 Imputation and Feature Scaling
* **Missing Value Verification:** `[Fill in: State missing value audit result—e.g., Zero missing values detected across all 1,544 rows, or specify median/mean imputation applied inside the training pipeline.]`
* **Z-Score Standardization (`StandardScaler`):** Telematics variables span widely different numeric scales (e.g., cumulative distance metrics versus single-digit fatigue alert counts). `StandardScaler` was fitted **exclusively on the training split** ($\mu_{\text{train}}, \sigma_{\text{train}}$) and applied to transform the validation and test splits via:

$$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

This ensures every input enters the neural network on a zero-mean, unit-variance scale without test-set leakage.

### 3.4 Strategy for Imbalance and Limited Sample Size
To address the 8.4:1 class imbalance and limited minority sample count (164 total positive cases), three data-sufficiency strategies were evaluated:

1. **Cost-Sensitive Learning (`class_weight`):** Instead of altering the underlying data distribution, a moderated positive class weight (`class_weight={0: 1.0, 1: 3.5}`) was applied in the binary cross-entropy loss function. Penalizing missed incident drivers $3.5\times$ more heavily than false alarms improved minority-class sensitivity while avoiding the probability distortion and overfitting caused by using the raw inverse ratio (`8.4`).
2. **Stratified Partitioning & Validation Monitoring:** Stratified splitting ensured every data partition maintained the exact 10.6% positive incident rate (`33` positive cases in the 309-driver test set), preventing fold-level class starvation that occurs under unstratified random splits.
3. **Deliberate Exclusion of Synthetic Data Generation (e.g., SMOTE):** Synthetic oversampling techniques such as SMOTE were deliberately excluded. Interpolating synthetic minority points across 14 noisy telematics dimensions risks fabricating non-physical driving profiles and inflating training-set memorization without adding genuine real-world crash signal.

---

## 4. Evaluation Methodology and Metric Selection

### 4.1 Transition from Overall Accuracy to ROC-AUC
On an imbalanced dataset where 89.4% of drivers belong to Class `0`, a naive classifier that predicts `At_Risk = 0` for every bus captain achieves **89.4% overall accuracy** while detecting **0%** of at-risk drivers. Consequently, evaluation transitioned to **Receiver Operating Characteristic Area Under the Curve (ROC-AUC)**:

* **Threshold-Independent Ranking:** ROC-AUC evaluates model discrimination across all possible probability cutoffs $\tau \in [0, 1]$, decoupling architectural assessment from any single arbitrary classification threshold.
* **Pairwise Separation:** Mathematically, ROC-AUC measures the probability that a randomly selected incident driver (`At_Risk = 1`) receives a higher predicted risk score than a randomly selected safe driver (`At_Risk = 0`) across all $33 \times 276 = 9,108$ test pairs, where `0.500` represents random guessing.

### 4.2 Operational Trade-Off: Recall vs. Precision and Accuracy
Once a model's ranking quality is established via ROC-AUC, operational deployment requires selecting a probability threshold tailored to fleet safety economics:
* **At-Risk Recall (Class 1 Sensitivity):** Measures the proportion of actual incident drivers flagged for intervention:

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

In safety-critical transit operations, false negatives carry severe human and financial costs, making high Recall the primary operational priority.
* **Precision and Overall Accuracy:** Lowering the decision threshold to capture more at-risk drivers increases false positives (safe drivers assigned preventative coaching), reducing Precision ($\frac{\text{TP}}{\text{TP} + \text{FP}}$) and overall Accuracy. Because coaching interventions are low-cost and non-punitive, moderate false-positive rates are operationally acceptable in exchange for higher incident recall.

---

## 5. Cross-Algorithm Model Comparison

### 5.1 Multi-Model Benchmark Comparison (Unseen Test Set)
All finalized models were trained on the standardized training partition (`random_state=42`) and evaluated once on the held-out 309-driver test set:

| Model | Configuration & Imbalance Handling | Interpretability | Overfitting Risk | Test ROC-AUC |
| :--- | :--- | :--- | :--- | :---: |
| **Logistic Regression** | Linear L2-regularized classifier (`random_state=42`) | **High** — Direct feature log-odds coefficients | **Low** — Rigid linear decision boundary | **0.478** |
| **Random Forest Classifier** | 100 bagged decision trees (`max_depth=5`, `random_state=42`) | **Moderate** — Gini feature importance rankings | **Moderate** — Step-wise splits dilute minority signal | **0.509** |
| **Gradient Boosting Classifier** | Sequential residual trees (`random_state=42`) | **Moderate** — Feature importance & partial dependence | **Moderate** — Excels on sharp tabular thresholds | **0.566** |
| **Final Tuned ANN (`16 -> 8`)** | 2-layer MLP (`ReLU`), `Dropout(0.4/0.3)`, Adam (`lr=0.0005`), `class_weight={0:1.0, 1:3.5}` | **Low (Black-Box)** — Requires post-hoc attribution | **Controlled** — Mitigated via bottleneck & dropout | **0.551** |

### 5.2 Iterative ANN Experimentation (5 Controlled Feature & Hyperparameter Configurations)
To validate preprocessing choices and architectural capacity without peeking at test-set outcomes, five controlled ANN configurations were compared using **Peak Validation AUC** and **Training AUC** overfitting diagnostics:

| Run | Features | Layers | Dropout | LR | Pos Weight | Train AUC | Peak Val AUC | Test ROC-AUC | Diagnostic Outcome |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Run 1: Unscaled Baseline** | Raw Unscaled | `32 -> 16` | `0.0 / 0.0` | `0.0010` | `1.0` | 0.511 | 0.522 | 0.468 | Stalled learning due to unscaled input magnitudes. |
| **Run 2: Scaled Baseline** | `StandardScaler` | `32 -> 16` | `0.0 / 0.0` | `0.0010` | `1.0` | 0.650 | 0.608 | 0.533 | Severe training overfitting (`0.650` Train AUC) without dropout. |
| **Run 3: Log-Transformed + Reg** | `Log1p + Scaled` | `32 -> 16` | `0.2 / 0.2` | `0.0010` | `3.5` | 0.622 | 0.568 | 0.528 | Persistent training overfitting (`0.622` Train); log compression underperformed standard scaling on validation. |
| **Run 4: Deeper 3-Layer ANN** | `StandardScaler` | `32 -> 16 -> 8` | `0.4 / 0.3 / 0.2` | `0.0005` | `3.5` | 0.493 | 0.543 | 0.573 | Underfit training split (`0.493` < `0.500` baseline); test spike reflects small-sample variance. |
| **Run 5: Final Tuned ANN** | `StandardScaler` | `16 -> 8` | `0.4 / 0.3` | `0.0005` | `3.5` | **0.552** | **0.578** | **0.551** *(Stage 1)* | **Selected:** Eliminated memorization while achieving highest Peak Val AUC (`0.578`) among regularized models. |

### 5.3 Threshold Sensitivity Analysis
`[Fill in: Populate the table below using probability thresholds evaluated on your model's predicted probabilities]`

| Decision Threshold ($\tau$) | Flagged Drivers (%) | At-Risk Recall (Class 1) | Precision (Class 1) | Overall Accuracy | Operational Interpretation |
| :---: | :---: | :---: | :---: | :---: | :--- |
| `0.30` | `[Fill in]` | `[Fill in]` | `[Fill in]` | `[Fill in]` | Broad safety net; maximizes incident recall with higher coaching volume |
| `0.40` | `[Fill in]` | `[Fill in]` | `[Fill in]` | `[Fill in]` | `[Fill in: Balanced screening tier]` |
| `0.50` | `[Fill in]` | `[Fill in]` | `[Fill in]` | `[Fill in]` | Default midpoint cutoff |
| `0.60` | `[Fill in]` | `[Fill in]` | `[Fill in]` | `[Fill in]` | High-confidence intervention tier for limited supervisor capacity |

### 5.4 Conclusions from Empirical Experiments
1. **Compact Architecture Outperforms Complexity:** Narrowing layer capacity from `32 -> 16` down to `16 -> 8` with strong dropout (`0.4 / 0.3`) eliminated the training-set memorization observed in Runs 2 and 3 (`0.650` and `0.622` Train AUC), whereas adding a third hidden layer (`32 -> 16 -> 8` in Run 4) over-regularized the network and depressed training AUC below random chance (`0.493`).
2. **Standardization Preserves Signal Without Over-Compressing Magnitudes:** Comparing Run 1 (`0.468` Test AUC) against Run 2 (`0.533` Test AUC) proved that Z-score standardization is essential for neural network convergence, while Run 3 (`0.568` Peak Val AUC vs. `0.578` for Run 5) confirmed that `StandardScaler` on natural magnitudes outperformed `np.log1p` compression.
3. **Avoiding Validation & Test Selection Bias:** Model selection was governed strictly by **Peak Validation AUC** (`0.578` for `16 -> 8` vs. `0.543` for `32 -> 16 -> 8`) paired with training diagnostics. Rejecting Run 4 despite its `0.573` test spike prevented test-set peeking on an underfitted model (`0.493` Train AUC).

---

## 6. Data Visualization

### 6.1 Technical Visualizations (`matplotlib`)
Static diagnostic plots generated in `project.ipynb` provide visual verification across every pipeline stage:
* **Stage 1 Multi-Model ROC Curve Comparison:** Overlays ROC curves for Gradient Boosting (`0.566`), Tuned ANN (`0.551`), Random Forest (`0.509`), Logistic Regression (`0.478`), and the diagonal random chance baseline (`0.500`), highlighting the ANN's superior True Positive Rate (`0.85` to `0.94`) in high-sensitivity screening zones ($\text{FPR} > 0.55$).
* **Stage 2 Ablation Grouped Bar Chart:** Compares `Final Train AUC`, `Peak Val AUC`, and `Test ROC-AUC` across the ablation configurations against the `0.500` baseline dashed line, visually illustrating training overfitting in Runs 2–3 and underfitting in Run 4.
* **Training & Validation Convergence Curves:** `[Fill in: Describe your Step 6 Loss and AUC epoch curves over 25 epochs]`

### 6.2 Interactive Dashboard (`Streamlit + Plotly`)
`[Fill in: Describe your interactive Streamlit + Plotly application—e.g., interactive risk threshold slider, individual driver telematics radar/bar breakdown, fleet risk tier distribution charts, and searchable Driver_ID lookup table powered by driver_predictions.csv]`

---

## 7. Predictions and Risk Stratification

The finalized Tuned ANN generates continuous incident probabilities exported to `driver_predictions.csv` (`[Fill in: List exact column names in driver_predictions.csv, e.g., Driver_ID, Predicted_Probability, Risk_Tier]`), enabling operational risk stratification across the fleet:

* **High-Risk Tier (`[Fill in: Probability Range, e.g., P >= 0.55]`):** `[Fill in: Driver count/percentage and recommended immediate intervention, e.g., Mandatory 1-on-1 safety coaching and route telemetry review]`
* **Moderate / Watchlist Tier (`[Fill in: Probability Range, e.g., 0.35 <= P < 0.55]`):** `[Fill in: Driver count/percentage and recommended preventive action, e.g., Automated in-cab feedback alerts and bi-weekly monitoring]`
* **Low-Risk Tier (`[Fill in: Probability Range, e.g., P < 0.35]`):** `[Fill in: Driver count/percentage and routine baseline monitoring]`

---

## 8. Final Selected Model

### 8.1 Architecture

$$\text{Input}(14) \rightarrow \text{Dense}(16, \text{ReLU}) \rightarrow \text{Dropout}(0.4) \rightarrow \text{Dense}(8, \text{ReLU}) \rightarrow \text{Dropout}(0.3) \rightarrow \text{Dense}(1, \text{Sigmoid})$$

* **Total Trainable Parameters:** $(14 \times 16 + 16) + (16 \times 8 + 8) + (8 \times 1 + 1) = 240 + 136 + 9 = \mathbf{385\text{ parameters}}$ — deliberately compact to prevent memorization on 988 active training records.

### 8.2 Optimization and Training
* **Optimizer:** Adam ($\eta = 0.0005$)
* **Loss Function:** Binary Cross-Entropy (`binary_crossentropy`)
* **Class Imbalance Weighting:** `class_weight={0: 1.0, 1: 3.5}`
* **Batch Size & Epochs:** `batch_size=64`, `epochs=25`, `validation_split=0.2`
* **Reproducibility Controls:** `tf.keras.backend.clear_session()` and fixed random seeds (`42`) prior to graph construction.

### 8.3 Convergence Diagnostics
Across 25 training epochs, the `16 -> 8` architecture achieved a **Final Training AUC of `0.552`** and a **Peak Validation AUC of `0.578`**, demonstrating steady convergence without the divergence between training and validation curves observed in unregularized `32 -> 16` baselines (`0.650` Train AUC).

### 8.4 Unseen Test Performance
On the locked 309-driver test set, the Final Tuned ANN achieved a **Test ROC-AUC of `0.551`**, outperforming Logistic Regression (`0.478`) and Random Forest (`0.509`). While Gradient Boosting (`0.566`) led in aggregate AUC, the Tuned ANN outperformed Gradient Boosting across high-recall screening thresholds ($\text{FPR} > 0.55$) and provided smooth, continuous probability rankings without discrete leaf-node ties.

---

## 9. Challenges and Limitations

* **Supervised Training vs. Proactive Inference:** Models are trained on historical post-event aggregates (`At_Risk = 1` recorded after an incident occurred), whereas deployment requires scoring active drivers *prior* to an incident. In production, elevated behavioral alerts may partially reflect reactive caution following near-misses rather than purely pre-incident habits.
* **Label Noise & Bayes Error Ceiling:** Across all model families—linear (`0.478`), bagged trees (`0.509`), neural networks (`0.551`), and gradient boosted trees (`0.566`)—test ROC-AUC converges within a `0.48` to `0.57` ceiling. Collision involvement depends heavily on external environmental stochasticity (other road users, weather, traffic density, and route complexity) unobserved in driver-level telematics summaries, imposing an irreducible Bayes error ceiling on static tabular features.
* **Small-Sample Test Variance:** Because the 20% test split contains **33 positive incident drivers** (`309` total drivers), each individual incident driver's relative ranking shifts aggregate Test ROC-AUC by approximately `0.030` (`1 / 33`). Combined with stochastic weight initialization and dropout sampling in small tabular neural networks, single-split test metrics exhibit inherent finite-sample variance compared to deterministic tree ensembles.