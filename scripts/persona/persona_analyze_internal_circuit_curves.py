#!/usr/bin/env python3
"""Aggregate repaired upstream/downstream internal curves and controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT


def _bootstrap_metrics(frame, *, samples: int, seed: int) -> dict:
    import numpy as np

    values = {
        name: frame[name].to_numpy(dtype=float)
        for name in ("clean", "corrupt", "circuit_only", "circuit_ablated")
    }
    point_denominator = float(
        values["clean"].mean() - values["corrupt"].mean()
    )
    if point_denominator <= 1e-8:
        raise ValueError("Internal clean-minus-corrupt denominator is not positive.")
    point = {
        "faithfulness": float(
            (values["circuit_only"].mean() - values["corrupt"].mean())
            / point_denominator
        ),
        "necessity": float(
            (values["clean"].mean() - values["circuit_ablated"].mean())
            / point_denominator
        ),
    }
    rng = np.random.default_rng(seed)
    bootstrap = {name: [] for name in point}
    for _ in range(samples):
        index = rng.integers(0, len(frame), size=len(frame))
        denominator = float(
            values["clean"][index].mean()
            - values["corrupt"][index].mean()
        )
        if denominator <= 1e-8:
            continue
        bootstrap["faithfulness"].append(
            float(
                (
                    values["circuit_only"][index].mean()
                    - values["corrupt"][index].mean()
                )
                / denominator
            )
        )
        bootstrap["necessity"].append(
            float(
                (
                    values["clean"][index].mean()
                    - values["circuit_ablated"][index].mean()
                )
                / denominator
            )
        )
    result = dict(point)
    for name, estimates in bootstrap.items():
        estimates = np.asarray(estimates, dtype=float)
        valid_fraction = len(estimates) / samples
        result[f"{name}_bootstrap_valid_fraction"] = valid_fraction
        result[f"{name}_ci_low"] = (
            float(np.quantile(estimates, 0.025))
            if valid_fraction >= 0.95
            else float("nan")
        )
        result[f"{name}_ci_high"] = (
            float(np.quantile(estimates, 0.975))
            if valid_fraction >= 0.95
            else float("nan")
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/qwen_sycophancy.json"
    )
    parser.add_argument("--upstream-run", required=True)
    parser.add_argument("--downstream-run", required=True)
    parser.add_argument("--upstream-k-values", type=int, nargs="+", required=True)
    parser.add_argument("--downstream-k-values", type=int, nargs="+", required=True)
    parser.add_argument(
        "--upstream-evaluation-template",
        default="source_to_sink_k{k}_n32_v1",
    )
    parser.add_argument(
        "--downstream-evaluation-template",
        default="source_to_sink_k{k}_n32_v1",
    )
    parser.add_argument("--upstream-selected-k", type=int, default=None)
    parser.add_argument("--downstream-selected-k", type=int, default=None)
    parser.add_argument("--control-name", default="topology_stratified_v2")
    parser.add_argument(
        "--control-evaluation-template",
        default="control_topology_v2_seed{seed}_k{k}_n32",
    )
    parser.add_argument(
        "--control-seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4)
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--output-name", default="repaired_internal_circuit_curves_v1"
    )
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive.")
    for values in (args.upstream_k_values, args.downstream_k_values):
        if any(k <= 0 for k in values) or len(values) != len(set(values)):
            raise ValueError("Curve k values must be unique and positive.")
    if len(args.control_seeds) != len(set(args.control_seeds)):
        raise ValueError("Control seeds must be unique.")

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from refusal_circuit.paths import write_json
    from refusal_circuit.persona import sha256_file

    config = json.loads(args.config.read_text(encoding="utf-8"))
    result_root = ROOT / config["paths"]["results_root"]
    output = result_root / args.output_name
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} already contains results.")
    output.mkdir(parents=True, exist_ok=True)

    scopes = {
        "upstream": {
            "root": (
                result_root
                / "circuit_discovery_upstream_contrastive"
                / args.upstream_run
            ),
            "k_values": args.upstream_k_values,
            "selected_k": args.upstream_selected_k,
            "eligible_key": "eligible_edges",
            "count_key": "n_pairs",
            "circuit_name": "eapig_top_{k}.json",
            "evaluation_template": args.upstream_evaluation_template,
        },
        "downstream": {
            "root": (
                result_root
                / "circuit_discovery_steering_path"
                / args.downstream_run
            ),
            "k_values": args.downstream_k_values,
            "selected_k": args.downstream_selected_k,
            "eligible_key": "eligible_downstream_edges",
            "count_key": "n_prompts",
            "circuit_name": "steering_path_eapig_top_{k}.json",
            "evaluation_template": args.downstream_evaluation_template,
        },
    }

    curve_rows = []
    control_rows = []
    artifacts = {}
    for scope, spec in scopes.items():
        metadata_path = spec["root"] / "eapig_run_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "complete":
            raise ValueError(f"Incomplete {scope} discovery.")
        eligible = int(metadata[spec["eligible_key"]])
        artifacts[str(metadata_path.relative_to(ROOT))] = sha256_file(
            metadata_path
        )
        for k in spec["k_values"]:
            circuit_path = (
                spec["root"]
                / "circuits_source_to_sink"
                / spec["circuit_name"].format(k=k)
            )
            evaluation_root = (
                spec["root"]
                / "evaluation"
                / spec["evaluation_template"].format(k=k)
            )
            aggregate_path = evaluation_root / "aggregate.json"
            instances_path = evaluation_root / "instance_metrics.csv"
            circuit = json.loads(circuit_path.read_text(encoding="utf-8"))
            aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
            if (
                circuit.get("circuit_family") != "source_to_sink"
                or int(circuit.get("retained_edges", 0)) <= 0
                or not circuit.get("connectivity", {}).get(
                    "all_edges_on_source_to_sink_paths"
                )
                or aggregate.get("status") != "complete"
                or aggregate.get("truncation") is not False
                or int(aggregate.get("requested_k", -1)) != k
                or aggregate.get("circuit")
                != str(circuit_path.relative_to(ROOT))
                or aggregate.get("circuit_sha256")
                != sha256_file(circuit_path)
            ):
                raise ValueError(f"Invalid repaired {scope} cell at k={k}.")
            instances = pd.read_csv(instances_path)
            uncertainty = _bootstrap_metrics(
                instances,
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed + k,
            )
            if (
                not np.isclose(
                    uncertainty["faithfulness"],
                    float(aggregate["faithfulness"]),
                    atol=1e-5,
                )
                or not np.isclose(
                    uncertainty["necessity"],
                    float(aggregate["necessity"]),
                    atol=1e-5,
                )
            ):
                raise ValueError(f"Aggregate metrics disagree at {scope} k={k}.")
            retained = int(circuit["retained_edges"])
            curve_rows.append(
                {
                    "scope": scope,
                    "requested_k": k,
                    "retained_edges": retained,
                    "eligible_edges": eligible,
                    "retained_fraction": retained / eligible,
                    "n": int(aggregate[spec["count_key"]]),
                    "clean": float(aggregate["means"]["clean"]),
                    "corrupt": float(aggregate["means"]["corrupt"]),
                    "circuit_only": float(
                        aggregate["means"]["circuit_only"]
                    ),
                    "circuit_ablated": float(
                        aggregate["means"]["circuit_ablated"]
                    ),
                    **uncertainty,
                    "circuit": str(circuit_path.relative_to(ROOT)),
                    "evaluation": str(aggregate_path.relative_to(ROOT)),
                }
            )
            for path in (circuit_path, aggregate_path, instances_path):
                artifacts[str(path.relative_to(ROOT))] = sha256_file(path)

        selected_k = spec["selected_k"]
        if selected_k is None:
            continue
        if selected_k not in spec["k_values"]:
            raise ValueError(
                f"Selected {scope} k must be included in its curve."
            )
        for seed in args.control_seeds:
            circuit_path = (
                spec["root"]
                / "controls"
                / args.control_name
                / "source_to_sink"
                / f"seed_{seed}"
                / spec["circuit_name"].format(k=selected_k)
            )
            evaluation_root = (
                spec["root"]
                / "evaluation"
                / args.control_evaluation_template.format(
                    seed=seed, k=selected_k
                )
            )
            aggregate_path = evaluation_root / "aggregate.json"
            instances_path = evaluation_root / "instance_metrics.csv"
            circuit = json.loads(circuit_path.read_text(encoding="utf-8"))
            aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
            if (
                circuit.get("method")
                != "topology_stratified_score_permutation_control"
                or aggregate.get("status") != "complete"
                or aggregate.get("circuit")
                != str(circuit_path.relative_to(ROOT))
                or aggregate.get("circuit_sha256")
                != sha256_file(circuit_path)
            ):
                raise ValueError(
                    f"Invalid {scope} topology control seed {seed}."
                )
            instances = pd.read_csv(instances_path)
            uncertainty = _bootstrap_metrics(
                instances,
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed + selected_k + seed + 100000,
            )
            control_rows.append(
                {
                    "scope": scope,
                    "requested_k": selected_k,
                    "seed": seed,
                    "control_frontier_k": int(
                        circuit["control_frontier_k"]
                    ),
                    "target_retained_edges": int(
                        circuit["target_retained_edges"]
                    ),
                    "retained_edges": int(circuit["retained_edges"]),
                    "retained_size_difference": int(
                        circuit["retained_size_difference"]
                    ),
                    **uncertainty,
                    "circuit": str(circuit_path.relative_to(ROOT)),
                    "evaluation": str(aggregate_path.relative_to(ROOT)),
                }
            )
            for path in (circuit_path, aggregate_path, instances_path):
                artifacts[str(path.relative_to(ROOT))] = sha256_file(path)

    curves = pd.DataFrame(curve_rows).sort_values(
        ["scope", "retained_edges"]
    )
    controls = pd.DataFrame(control_rows)
    curves.to_csv(output / "internal_curves.csv", index=False)
    controls.to_csv(output / "topology_controls.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))
    for scope, frame in curves.groupby("scope"):
        axes[0].plot(
            frame["retained_edges"], frame["faithfulness"], "o-", label=scope
        )
        axes[1].plot(
            frame["retained_edges"], frame["necessity"], "o-", label=scope
        )
    axes[0].set(
        title="Internal faithfulness",
        xlabel="Retained edges",
        ylabel="Normalized faithfulness",
    )
    axes[1].set(
        title="Internal necessity",
        xlabel="Retained edges",
        ylabel="Normalized necessity",
    )
    for axis in axes:
        axis.set_xscale("log")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "internal_curves.pdf", bbox_inches="tight")
    fig.savefig(output / "internal_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "upstream_run": args.upstream_run,
        "downstream_run": args.downstream_run,
        "upstream_k_values": args.upstream_k_values,
        "downstream_k_values": args.downstream_k_values,
        "upstream_selected_k": args.upstream_selected_k,
        "downstream_selected_k": args.downstream_selected_k,
        "control_name": args.control_name,
        "control_evaluation_template": args.control_evaluation_template,
        "control_seeds": args.control_seeds,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "tables": ["internal_curves.csv", "topology_controls.csv"],
        "figures": ["internal_curves.pdf", "internal_curves.png"],
        "artifacts_sha256": dict(sorted(artifacts.items())),
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
