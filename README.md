# panda-quantum-classifier

**Robustness to Noisy Telemetry, Tested Where the Field's Own Papers Say It Would Matter: A Feature-Corruption Stress Test of Quantum Classifiers on IoT Intrusion Data**

Project ID: `F08-I3` · Archetype: `stress_test` · Full machine-readable spec: [`project_data.json`](project_data.json)

---

## Summary

A near-identical Gaussian-noise / imbalance robustness comparison of quantum classifiers against classical baselines already exists — but only on five generic UCI toy datasets (iris, wine quality, breast cancer, UCI-HAR, Pima diabetes). This project reruns the same corruption axis, fixed to the field's own three QML method families, on **two real network-intrusion datasets** instead.

**Task:** network intrusion detection (binary and multiclass), scored at test time under increasing input feature corruption.

**Why it's worth publishing:** IoT network telemetry is intermittently noisy and lossy by nature — the deployment premise every QML-for-IoT-security paper in this scan states as its target. Testing robustness on Iris does not tell anyone whether a quantum classifier's claimed resilience survives on the traffic data it is actually proposed for.

## Methods (3 arms)

All three ship in the single public, Apache-2.0 [`qiskit-machine-learning`](https://github.com/qiskit-community/qiskit-machine-learning) repo — 0 of 3 reimplemented.

| Arm | Implementation | Reference paper |
|---|---|---|
| **QSVC** — Quantum Support Vector Classifier (quantum kernel) | `algorithms/classifiers/qsvc.py`, `kernels/fidelity_quantum_kernel.py` (`PegasosQSVC` also available) | [QML for NIDS: a Systematic Literature Review](https://www.semanticscholar.org/paper/f28f64cb03b89f1bfb753d32d2d4b1d8471cb76d) (Nicesio, 2023) |
| **VQC** — Variational Quantum Classifier | `algorithms/classifiers/vqc.py` | [Network Anomaly Detection Using QNNs on Noisy Quantum Computers](https://www.semanticscholar.org/paper/024b06315a81807a1da31ca1aabbe0468a244112) (Kukliansky, 2024, IEEE TQE) |
| **QNN** — hybrid quantum-classical `EstimatorQNN` / `SamplerQNN` | `neural_networks/estimator_qnn.py`, `sampler_qnn.py` | [QML-IDS: Quantum ML Intrusion Detection System](https://arxiv.org/abs/2410.16308) (Abreu, 2024, ISCC) |

Qiskit is chosen over PennyLane for the QNN arm because PennyLane's `pyproject.toml` declares `requires-python>=3.12`, a possible mismatch with Colab's default runtime; `qiskit-machine-learning` states no such floor and ships all three arms in one repo.

## Datasets (2)

| Dataset | Source | License | Notes |
|---|---|---|---|
| **NSL-KDD** | [`Mireu-Lab/NSL-KDD`](https://huggingface.co/datasets/Mireu-Lab/NSL-KDD) | GPL-3.0 | 151,165 rows × 41 cols (38 numeric + `protocol_type`/`service`/`flag` categorical + class label), 72.6 MB. Schema probe-confirmed, 0 nulls across sampled columns. |
| **N-BaIoT** | [`codymlewis/nbaiot`](https://huggingface.co/datasets/codymlewis/nbaiot) | CC-BY-4.0 | ~1.7 GB download / ~3.1 GB decompressed. Real-device IoT botnet traffic (9 devices, Mirai/BASHLITE) — a different modality from flow-record datasets. **Schema is NOT probe-confirmed** (HF datasets-server reports "no viewer"); verify `load_dataset` returns rows and record the real schema before building preprocessing on it. |

Both are freely downloadable via the `datasets` library — no login, form, DUA, or credentialing course.

## Constraint axis

**Test-time input feature-corruption severity** — additive Gaussian noise on the PCA-reduced feature vector, applied *only at evaluation*:

| Level | Noise std |
|---|---|
| clean | 0σ |
| low | 0.1× per-feature std |
| medium | 0.3× per-feature std |
| high | 0.6× per-feature std |

This mirrors the corruption mechanism of the closest precedent found in the novelty search, but applied to real network-intrusion telemetry instead of generic UCI tables — directly testing the deployment condition (noisy/lossy IoT channels) the field's own papers cite as their motivation.

### Held fixed

- One trained model per method/dataset/seed, trained on **clean data only**; corruption is applied at test time only ("fix the methods, vary the corruption").
- PCA budget fixed at **8 components/qubits** (F08-I1's middle level), so this study does not re-litigate the budget axis.
- Preprocessing pipeline and train/test split identical to F08-I1 / F08-I2.
- The per-feature standard deviation setting low/medium/high is estimated **once** from the clean training split and reused identically across all three methods, so no method faces an easier or harder corruption by construction.

### Confounds ruled out

- **Quantum circuit / shot noise** — all circuits run on an exact, noiseless statevector simulator, so the only noise source is the explicit Gaussian perturbation applied to the classical feature vector before encoding.
- **Differing feature scales** — noise is a multiple of each feature's own clean-split std, not an absolute value, so each level means the same relative perturbation everywhere.
- **Training-time adaptation** — corruption is inference-only on a frozen model, matching the stress-test archetype rather than adversarial training.

## Metrics

- **macro-F1 at each corruption level** — imbalance-aware, consistent with F08-I1/I2 so results compare across the pack.
- **Relative degradation (Δ macro-F1 from that method/dataset/seed's own clean score)** — the point is to rank by *degradation*, not by peak; existing single-point papers only ever report the peak.

## Compute budget

3 methods × 2 datasets × 3 seeds = **18 training runs** × ~30 min ≈ **9.0 GPU-hours** against a ~60 h ceiling (~51 h headroom). The 4 corruption levels add only extra forward-pass scoring on the already-trained models (< 1 min per run), giving 72 scoring passes with no extra training.

Peak VRAM ~2 GB · disk ~6 GB · fits free Colab. Subsampled to ≤ 1,300 rows per run; N-BaIoT is downloaded once and cached, never used in full.

## Expected findings

- One method may have the best clean-data score but the **steepest degradation curve**, inverting the ranking the field would report from clean-only evaluation.
- **Null result:** degradation curves may run near-parallel, meaning no method is more robust than another and the field's clean-data rankings hold up.
- N-BaIoT and NSL-KDD may show **different degradation orderings**, showing robustness claims transplant no better across IoT-security datasets than accuracy claims do.

A null result is publishable here.

## Novelty

> No study stress-tests QSVC/VQC/QNN under input feature-noise injection specifically on IoT network-intrusion data, as opposed to generic UCI tabular data or quantum-circuit/hardware noise.

The [closest hit](https://www.semanticscholar.org/paper/71dd01e6f2bf5e05f5dbff1e4b84c1b6edd865e2) — *Robust evaluation of classical and quantum machine learning under noise, imbalance, feature reduction and explainability* (2025) — runs almost exactly this design (Gaussian feature-noise injection, SMOTE/ADASYN imbalance, ANOVA feature selection, QSVM/QkNN/VQC vs. classical baselines) but exclusively on five generic UCI datasets, none of them network-security data. Other hits concern quantum *circuit*/hardware noise (a different corruption source) or non-quantum IDS robustness. Notably, this near-miss surfaced only via a broad, non-dataset-name-specific query — a query naming an IoT-security dataset would have missed it entirely.

## Milestones

1. **Read & digest** — internalize the design: three field-standard methods trained once on clean NSL-KDD and N-BaIoT, then scored under increasing additive Gaussian noise on the PCA-reduced features.
2. **Download data** — pull both datasets; for N-BaIoT specifically, confirm `load_dataset` returns rows and inspect the real column schema *before* building preprocessing. If it fails, pick a fallback (e.g. ToN_IoT, already probed) **in week 1, not week 10**.
3. **Preprocess** — same encoding / scaling / PCA-to-8-components / 70-30 split / 3-seed protocol as F08-I1 and F08-I2 (budget fixed, not swept).
4. **Train** — QSVC, VQC, and QNN once per dataset per seed on clean training data: 18 training runs.
5. **Score under corruption** — at test time only, add i.i.d. Gaussian noise (mean 0, std = level × per-feature training-split std) to the PCA-reduced test features and score with the already-trained model: 3 × 2 × 4 × 3 = 72 scoring passes, no extra training.
6. **Report** — macro-F1 and Δ macro-F1 per corruption level; rank methods by **degradation slope** across the 4 levels, not by clean-level peak.

## Risks

- **N-BaIoT schema unconfirmed** — no queryable schema on HF; verify the load path before committing to it as dataset 2, with ToN_IoT identified as a week-1 fallback.
- **Noise levels are an a priori choice** — 0.1/0.3/0.6× std is reasonable but not validated against what real IoT packet loss / telemetry noise looks like statistically. Treat the exact levels as adjustable pending a quick literature check; the overall design does not change.
- **Seed variance** — macro-F1 on rare attack subtypes could be volatile with only 3 seeds; ~50 h of unused headroom allows adding seeds if early runs show high variance.

## Reading list

1. [Robust Evaluation of Classical and Quantum ML Under Noise, Imbalance, and Feature Reduction](https://www.semanticscholar.org/paper/71dd01e6f2bf5e05f5dbff1e4b84c1b6edd865e2) — the direct precedent this project relocates to real network-intrusion data.
2. [Quantum Machine Learning for Network Intrusion Detection Systems: a Systematic Literature Review](https://www.semanticscholar.org/paper/f28f64cb03b89f1bfb753d32d2d4b1d8471cb76d) — establishes QSVC as the dominant quantum-kernel method in QML-IDS.
3. [Network Anomaly Detection Using Quantum Neural Networks on Noisy Quantum Computers](https://www.semanticscholar.org/paper/024b06315a81807a1da31ca1aabbe0468a244112) — VQC robustness under NISQ *hardware* noise; contrast with the input-feature corruption tested here.
4. [QML-IDS: Quantum Machine Learning Intrusion Detection System](https://arxiv.org/abs/2410.16308) — the EstimatorQNN/SamplerQNN hybrid used as the third arm.
5. [Post-Quantum Cryptosystems and IoT Constraints](https://arxiv.org/abs/2402.00790) — frames resource-constrained IoT hardware as the deployment context where noisy telemetry is unavoidable.
