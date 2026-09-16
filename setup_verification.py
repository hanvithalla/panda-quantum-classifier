"""
setup_verification.py -- Milestone 1 technical result for F08-I3.

Proves the environment can actually run what README.md / project_data.json
specify: the three Qiskit method arms (QSVC, VQC, QNN) at the fixed 8-qubit /
8-PCA-component budget on a noiseless statevector simulator, plus data access
to NSL-KDD and N-BaIoT.

Every check is executed, not asserted: the quantum arms are fitted on a tiny
synthetic 8-component matrix, and dataset access is confirmed by pulling a real
row rather than only reading metadata.

Usage:
    python setup_verification.py                 # prints and writes setup_log.txt
    python setup_verification.py --log out.txt   # custom log path
    python setup_verification.py --no-log        # stdout only
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
import traceback
import urllib.request
from datetime import datetime, timezone

# Project constants -- must match project_data.json / README.md
N_QUBITS = 8  # PCA components == qubits, fixed (F08-I1 middle level)
NSL_KDD_REPO = "Mireu-Lab/NSL-KDD"
NBAIOT_REPO = "codymlewis/nbaiot"
NBAIOT_UPSTREAM = (
    "https://archive.ics.uci.edu/static/public/442/"
    "detection+of+iot+botnet+attacks+n+baiot.zip"
)
TON_IOT_FALLBACK = "codymlewis/TON_IoT_network"  # week-1 fallback named in the spec

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

_results: list[tuple[str, str, str]] = []  # (status, check name, detail)
_out_lines: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    _out_lines.append(msg)


def section(title: str) -> None:
    log("")
    log(f"--- {title} ---")


def record(status: str, name: str, detail: str = "") -> None:
    """status in {PASS, FAIL, WARN}."""
    _results.append((status, name, detail))
    log(f"[{status}] {name}" + (f": {detail}" if detail else ""))


def check_versions() -> None:
    section("Environment")
    log(f"Timestamp (UTC)  : {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    log(f"Platform         : {platform.platform()}")
    log(f"Python           : {sys.version.split()[0]} ({sys.executable})")

    section("Core library versions")
    required = [
        ("qiskit", "qiskit"),
        ("qiskit_machine_learning", "qiskit-machine-learning"),
        ("datasets", "datasets"),
        ("pandas", "pandas"),
        ("sklearn", "scikit-learn"),
        ("numpy", "numpy"),
    ]
    import importlib

    for module_name, pkg_name in required:
        try:
            mod = importlib.import_module(module_name)
            record("PASS", f"import {pkg_name}", getattr(mod, "__version__", "version n/a"))
        except Exception as exc:  # noqa: BLE001
            record("FAIL", f"import {pkg_name}", f"{type(exc).__name__}: {exc}")


def check_method_arms() -> None:
    """The three arms must all resolve from the single qiskit-machine-learning repo."""
    section("Method arms (all from qiskit-machine-learning)")
    arms = {
        "QSVC + FidelityQuantumKernel (quantum kernel arm)":
            "from qiskit_machine_learning.algorithms.classifiers import QSVC;"
            "from qiskit_machine_learning.kernels import FidelityQuantumKernel",
        "VQC (variational arm)":
            "from qiskit_machine_learning.algorithms.classifiers import VQC",
        "EstimatorQNN / SamplerQNN (hybrid QNN arm)":
            "from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN",
    }
    for name, stmt in arms.items():
        try:
            exec(compile(stmt, "<arm>", "exec"), {})  # noqa: S102
            record("PASS", name, "importable")
        except Exception as exc:  # noqa: BLE001
            record("FAIL", name, f"{type(exc).__name__}: {exc}")


def check_qubit_budget() -> None:
    """Actually run the 8-component / 8-qubit pipeline end to end, noiselessly."""
    section(f"Project constraint: PCA components == qubits == {N_QUBITS}")
    try:
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        rng = np.random.RandomState(0)
        x_raw = rng.randn(24, 30)          # stand-in for a 30-feature flow record
        y = rng.randint(0, 2, 24)
        x_pca = PCA(n_components=N_QUBITS, random_state=0).fit_transform(
            StandardScaler().fit_transform(x_raw)
        )
        assert x_pca.shape[1] == N_QUBITS
        record("PASS", f"PCA -> {N_QUBITS} components", f"shape {x_pca.shape}")
    except Exception as exc:  # noqa: BLE001
        record("FAIL", f"PCA -> {N_QUBITS} components", f"{type(exc).__name__}: {exc}")
        return

    try:
        from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap

        feature_map = ZZFeatureMap(feature_dimension=N_QUBITS, reps=1)
        ansatz = RealAmplitudes(num_qubits=N_QUBITS, reps=1)
        ok = feature_map.num_qubits == ansatz.num_qubits == N_QUBITS
        record(
            "PASS" if ok else "FAIL",
            f"{N_QUBITS}-qubit circuits build",
            f"ZZFeatureMap={feature_map.num_qubits}q, RealAmplitudes={ansatz.num_qubits}q",
        )
    except Exception as exc:  # noqa: BLE001
        record("FAIL", f"{N_QUBITS}-qubit circuits build", f"{type(exc).__name__}: {exc}")
        return

    # QSVC: fit a quantum kernel on 16 points, predict 8.
    try:
        from qiskit_machine_learning.algorithms.classifiers import QSVC
        from qiskit_machine_learning.kernels import FidelityQuantumKernel

        t0 = time.time()
        qsvc = QSVC(quantum_kernel=FidelityQuantumKernel(feature_map=feature_map))
        qsvc.fit(x_pca[:16], y[:16])
        preds = qsvc.predict(x_pca[16:])
        record(
            "PASS",
            "QSVC smoke fit/predict",
            f"{len(preds)} predictions in {time.time() - t0:.1f}s (noiseless statevector)",
        )
    except Exception as exc:  # noqa: BLE001
        record("FAIL", "QSVC smoke fit/predict", f"{type(exc).__name__}: {exc}")

    # VQC: 3 optimizer iterations is enough to prove the training loop runs.
    try:
        from qiskit_machine_learning.algorithms.classifiers import VQC
        from qiskit_machine_learning.optimizers import COBYLA

        t0 = time.time()
        vqc = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=COBYLA(maxiter=3))
        vqc.fit(x_pca[:16], y[:16])
        record(
            "PASS",
            "VQC smoke fit",
            f"3 COBYLA iters in {time.time() - t0:.1f}s, ansatz={vqc.ansatz.num_qubits}q",
        )
    except Exception as exc:  # noqa: BLE001
        record("FAIL", "VQC smoke fit", f"{type(exc).__name__}: {exc}")

    # QNN: one forward pass through the hybrid circuit.
    try:
        import numpy as np
        from qiskit_machine_learning.neural_networks import EstimatorQNN

        circuit = feature_map.compose(ansatz)
        t0 = time.time()
        qnn = EstimatorQNN(
            circuit=circuit,
            input_params=list(feature_map.parameters),
            weight_params=list(ansatz.parameters),
        )
        out = qnn.forward(x_pca[:4], np.zeros(qnn.num_weights))
        record(
            "PASS",
            "EstimatorQNN forward pass",
            f"output {out.shape}, {qnn.num_weights} weights, {time.time() - t0:.1f}s",
        )
    except Exception as exc:  # noqa: BLE001
        record("FAIL", "EstimatorQNN forward pass", f"{type(exc).__name__}: {exc}")

    # The confound rule: no circuit/shot noise may enter. Confirm exact primitives.
    try:
        from qiskit.primitives import StatevectorEstimator, StatevectorSampler

        est, smp = StatevectorEstimator(), StatevectorSampler()
        record(
            "PASS",
            "Noiseless statevector primitives available",
            f"{type(est).__name__}, {type(smp).__name__} (exact, no shot noise)",
        )
    except Exception as exc:  # noqa: BLE001
        record("FAIL", "Noiseless statevector primitives available", f"{type(exc).__name__}: {exc}")


def check_nsl_kdd() -> None:
    section(f"Dataset 1: NSL-KDD ({NSL_KDD_REPO})")
    try:
        from datasets import load_dataset, load_dataset_builder

        builder = load_dataset_builder(NSL_KDD_REPO)
        features = list(builder.info.features.keys()) if builder.info.features else []
        record(
            "PASS",
            "NSL-KDD metadata",
            f"{len(features)} columns, first 6: {features[:6]}",
        )

        t0 = time.time()
        row = next(iter(load_dataset(NSL_KDD_REPO, split="train", streaming=True)))
        label_keys = [k for k in row if k.lower() in ("class", "label")]
        record(
            "PASS",
            "NSL-KDD real row fetched (streaming)",
            f"{len(row)} fields in {time.time() - t0:.1f}s; "
            f"label column {label_keys} = {[row[k] for k in label_keys]}",
        )
        numeric = sum(1 for v in row.values() if isinstance(v, (int, float)))
        log(f"      categorical/numeric split in sampled row: "
            f"{len(row) - numeric} non-numeric / {numeric} numeric")
        log(f"      sample: {dict(list(row.items())[:5])}")
    except Exception as exc:  # noqa: BLE001
        record("FAIL", "NSL-KDD access", f"{type(exc).__name__}: {str(exc)[:200]}")


def check_nbaiot() -> None:
    section(f"Dataset 2: N-BaIoT ({NBAIOT_REPO})")
    # project_data.json flagged this as unverified ("datasets-server has no viewer").
    hf_ok = False
    try:
        from datasets import load_dataset_builder

        builder = load_dataset_builder(NBAIOT_REPO)
        features = list(builder.info.features.keys()) if builder.info.features else []
        hf_ok = True
        record("PASS", "N-BaIoT via datasets library", f"{len(features)} columns")
    except Exception as exc:  # noqa: BLE001
        record(
            "WARN",
            "N-BaIoT via datasets library",
            f"{type(exc).__name__}: {str(exc)[:160]}",
        )

    if not hf_ok:
        log("      -> The HF repo is a loader SCRIPT (nbaiot.py), not data files;")
        log("         datasets>=4 removed script support, so load_dataset() cannot be used.")
        log("         The script itself just downloads the UCI archive, so the")
        log("         supported path is to fetch that archive directly.")
        try:
            req = urllib.request.Request(
                NBAIOT_UPSTREAM, method="HEAD", headers={"User-Agent": "Mozilla/5.0"}
            )
            resp = urllib.request.urlopen(req, timeout=60)  # noqa: S310
            record(
                "PASS",
                "N-BaIoT upstream (UCI archive) reachable",
                f"HTTP {resp.status} -- viable direct-download path",
            )
        except Exception as exc:  # noqa: BLE001
            record(
                "FAIL",
                "N-BaIoT upstream (UCI archive) reachable",
                f"{type(exc).__name__}: {str(exc)[:160]}",
            )

        try:
            from huggingface_hub import HfApi

            HfApi().dataset_info(TON_IOT_FALLBACK)
            record(
                "PASS",
                f"Week-1 fallback reachable ({TON_IOT_FALLBACK})",
                "available if N-BaIoT ingestion proves too costly",
            )
        except Exception as exc:  # noqa: BLE001
            record(
                "WARN",
                f"Week-1 fallback reachable ({TON_IOT_FALLBACK})",
                f"{type(exc).__name__}: {str(exc)[:120]}",
            )


def summarise() -> int:
    section("Summary")
    passed = [r for r in _results if r[0] == "PASS"]
    warned = [r for r in _results if r[0] == "WARN"]
    failed = [r for r in _results if r[0] == "FAIL"]
    log(f"{len(passed)} passed, {len(warned)} warning(s), {len(failed)} failed "
        f"out of {len(_results)} checks.")

    for status, name, detail in warned + failed:
        log(f"  {status}: {name} -- {detail}")

    log("")
    if failed:
        log("STATUS: NOT READY -- resolve the failures above before Milestone 2.")
        return 1
    if warned:
        log("STATUS: READY (with caveats)")
        log("  Toolchain and the 8-qubit budget are verified end to end; all three")
        log("  method arms run on a noiseless statevector simulator, and NSL-KDD")
        log("  loads. N-BaIoT needs the documented direct-download path instead of")
        log("  load_dataset() -- this confirms the risk already recorded in")
        log("  project_data.json rather than a new blocker.")
        return 0
    log("STATUS: READY")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default="setup_log.txt", help="path for the log file")
    parser.add_argument("--no-log", action="store_true", help="print only, write no log")
    args = parser.parse_args()

    log("=" * 72)
    log("F08-I3 -- Milestone 1 environment verification")
    log("Quantum classifier robustness to noisy IoT telemetry (QSVC / VQC / QNN)")
    log("=" * 72)

    exit_code = 1
    try:
        check_versions()
        check_method_arms()
        check_qubit_budget()
        check_nsl_kdd()
        check_nbaiot()
        exit_code = summarise()
    except Exception:  # noqa: BLE001
        log("")
        log("UNHANDLED ERROR:")
        log(traceback.format_exc())
        log("STATUS: NOT READY")

    if not args.no_log:
        with open(args.log, "w", encoding="utf-8") as handle:
            handle.write("\n".join(_out_lines) + "\n")
        print(f"\n(log written to {args.log})")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
