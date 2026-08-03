#!/usr/bin/env python3
"""Unified entry point for refusal-circuit experiments.

The repository still keeps the original one-off scripts under
``scripts/legacy/`` for reproducibility.  New runs should prefer this file:

    python scripts/rq.py list
    python scripts/rq.py rq2.prepare-polyrefuse --languages en de zh th
    python scripts/rq.py rq2.run-eapig --language en --num-prompts 64

Arguments after the command are passed through unchanged.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "scripts" / "legacy"


COMMANDS: dict[str, str] = {
    # RQ1 / Gemma single-language refusal circuit.
    "rq1.smoke": "rq1_smoke_test.py",
    "rq1.prepare-data": "rq1_prepare_dataset.py",
    "rq1.extract-direction": "rq1_extract_refusal_direction.py",
    "rq1.validate-direction": "rq1_validate_refusal_direction.py",
    "rq1.run-eapig": "rq1_run_eapig.py",
    "rq1.evaluate": "rq1_evaluate_faithfulness.py",
    "rq1.behavior": "rq1_behavioral_sanity.py",
    "rq1.circuit-closure": "rq1_check_circuit_closure.py",
    "rq1.figures": "rq1_make_figures.py",
    "rq1.debug-residual": "rq1_debug_residual_reconstruction.py",
    "rq1.debug-gradient": "rq1_debug_terminal_gradient.py",
    # RQ2 / multilingual refusal directions and circuits.
    "rq2.prepare-polyrefuse": "rq2_prepare_polyrefuse.py",
    "rq2.extract-direction": "rq2_extract_multilingual_direction.py",
    "rq2.compare-directions": "rq2_compare_multilingual_directions.py",
    "rq2.direction-transfer": "rq2_evaluate_direction_transfer.py",
    "rq2.make-shared-metrics": "rq2_make_shared_metrics.py",
    "rq2.precheck-metric": "rq2_precheck_shared_metric.py",
    "rq2.run-eapig": "rq1_run_eapig.py",
    "rq2.run-eapig-logit": "rq_run_eapig_logit_metric.py",
    "rq2.rebuild-circuits": "rq2_rebuild_circuits_from_scores.py",
    "rq2.rebuild-logit-circuits": "rq2_rebuild_logit_circuits_from_scores.py",
    "rq2.evaluate-local": "rq1_evaluate_faithfulness.py",
    "rq2.evaluate-transfer": "rq2_evaluate_cross_language_circuit.py",
    "rq2.make-intersections": "rq2_make_intersection_circuits.py",
    "rq2.make-consensus": "rq2_make_consensus_circuits.py",
    "rq2.compare-circuits": "rq2_compare_multilingual_circuits.py",
    "rq2.behavior": "rq2_behavioral_validation.py",
    "rq2.behavior-logit": "rq2_behavioral_validation_logit_circuit.py",
    "rq2.wildguard": "rq2_wildguard_behavioral_validation.py",
    "rq2.wildguard-smoke": "rq2_wildguard_smoke_test.py",
    "rq2.inspect-refusal-starters": "rq2_inspect_refusal_starters.py",
    "rq2.debug-wildguard-harmless": "rq2_debug_wildguard_harmless.py",
    "rq2.plots": "rq2_make_presentation_plots.py",
    # Frozen Qwen refusal Phase 2 (new code; old jobs remain under legacy/).
    "rq2.phase2-setup": "phase2/phase2_refusal_setup.py",
    "rq2.phase2-structure": "phase2/phase2_cross_size_structure.py",
    "rq2.phase2-metrics": "phase2/phase2_refusal_metrics_sweep.py",
    "rq2.phase2-behavior": "phase2/phase2_refusal_behavior_sweep.py",
    "rq2.phase2-analyze": "phase2/phase2_refusal_analysis.py",
    "rq2.phase2-random-controls": "phase2/phase2_random_controls.py",
    "rq2.phase2-freeze": "phase2/phase2_freeze_selection.py",
    # Persona Vectors replication and causal-circuit experiments.
    "persona.fetch": "persona/persona_fetch_artifacts.py",
    "persona.audit": "persona/persona_audit_protocol.py",
    "persona.audit-generation": "persona/persona_audit_generation.py",
    "persona.generate": "persona/persona_generate_responses.py",
    "persona.judge": "persona/persona_judge_responses.py",
    "persona.judge-layer-sweep": "persona/persona_judge_layer_sweep.py",
    "persona.extract": "persona/persona_extract_response_vector.py",
    "persona.check-layer": "persona/persona_check_layer_mapping.py",
    "persona.steer": "persona/persona_steer_responses.py",
    "persona.select-layer": "persona/persona_select_layer.py",
    "persona.run-upstream-contrastive-eapig": "persona/persona_run_upstream_contrastive_eapig.py",
    "persona.freeze-shared-circuit-dataset": "persona/persona_freeze_shared_circuit_dataset.py",
    "persona.rebuild-nested-circuits": "persona/persona_rebuild_nested_circuits.py",
    "persona.analyze-causal-curves": "persona/persona_analyze_causal_curves.py",
    "persona.evaluate-upstream-contrastive-circuit": "persona/persona_evaluate_upstream_contrastive_circuit.py",
    "persona.behavior-upstream-contrastive-circuit": "persona/persona_behavior_upstream_contrastive_circuit.py",
    "persona.behavior-connected-circuits": "persona/persona_behavior_connected_circuits.py",
    "persona.vector-diagnostic": "persona/persona_vector_intervention_diagnostic.py",
    "persona.run-steering-eapig": "persona/persona_run_steering_eapig.py",
    "persona.evaluate-steering-circuit": "persona/persona_evaluate_steering_circuit.py",
    "persona.behavior-steering-circuit": "persona/persona_behavior_steering_circuit.py",
    # Cross-RQ diagnostics / reporting.
    "rq.causal-site-sweep": "rq_causal_site_sweep.py",
    "results.summary": "results_summary",
}


DESCRIPTIONS: dict[str, str] = {
    "rq1.smoke": "Load the target model through the experiment model path.",
    "rq1.prepare-data": "Prepare RQ1 harmful/harmless splits.",
    "rq1.extract-direction": "Extract and validate the Gemma refusal direction.",
    "rq1.run-eapig": "Score projection-target EAP/EAP-IG edges.",
    "rq1.evaluate": "Evaluate faithfulness and necessity for saved circuits.",
    "rq1.behavior": "Run RQ1 behavioral sanity checks without saving completions.",
    "rq2.prepare-polyrefuse": "Prepare PolyRefuse splits with first-token filtering.",
    "rq2.extract-direction": "Extract one language-specific PolyRefuse direction.",
    "rq2.make-shared-metrics": "Write English/common-mean shared metric files.",
    "rq2.run-eapig": "Score projection-target multilingual circuits.",
    "rq2.run-eapig-logit": "Score logit-target refusal circuits.",
    "rq2.evaluate-local": "Evaluate a language-local circuit on held-out prompts.",
    "rq2.evaluate-transfer": "Evaluate a frozen source-language circuit on a target language.",
    "rq2.make-intersections": "Build pairwise/strict/majority shared-core circuits.",
    "rq2.make-consensus": "Build consensus circuits from averaged edge scores.",
    "rq2.compare-circuits": "Compute IoU, overlap, weighted Jaccard, and score correlations.",
    "rq2.behavior": "Run first-token / generation substring behavioral validation.",
    "rq2.wildguard": "Run WildGuard behavioral validation without saving text.",
    "rq2.plots": "Regenerate RQ2 presentation plots.",
    "rq2.phase2-setup": "Audit sources, make new splits, or derive masks from frozen scores.",
    "rq2.phase2-structure": "Measure same-size and cross-size multilingual circuit overlap.",
    "rq2.phase2-metrics": "Sweep projection and first-token metrics with baselines computed once.",
    "rq2.phase2-behavior": "Generate and judge refusal behavior across circuit sizes.",
    "rq2.phase2-analyze": "Bootstrap behavior curves and identify preregistered candidate k values.",
    "rq2.phase2-random-controls": "Build uniform/topology-stratified masks matched to selected k.",
    "rq2.phase2-freeze": "Hash and freeze reviewed k, masks, judge, and data before confirmation.",
    "persona.fetch": "Fetch and hash official Persona Vectors artifacts at the frozen commit.",
    "persona.audit": "Audit the frozen sycophancy protocol, data split, hashes, and runtime gates.",
    "persona.audit-generation": "Validate and freeze the S1 extraction-generation manifest.",
    "persona.generate": "Generate the official positive/negative persona extraction responses.",
    "persona.judge": "Score persona responses with the official trait and coherence rubrics.",
    "persona.judge-layer-sweep": "Judge the frozen Qwen persona layer/steering-coefficient grid.",
    "persona.extract": "Filter paired responses and extract the response-average persona vector.",
    "persona.check-layer": "Check the HF-to-TransformerLens persona layer mapping.",
    "persona.steer": "Generate held-out responses under response-time persona steering.",
    "persona.select-layer": "Aggregate a judged all-layer steering sweep and select informative layers.",
    "persona.run-upstream-contrastive-eapig": "Discover an upstream persona circuit from matched positive/negative prompt states.",
    "persona.freeze-shared-circuit-dataset": "Freeze shared upstream/downstream circuit discovery and evaluation samples.",
    "persona.rebuild-nested-circuits": "Derive nested upstream or downstream circuits from frozen edge scores.",
    "persona.analyze-causal-curves": "Aggregate final causal necessity and behavioral-sufficiency curves.",
    "persona.evaluate-upstream-contrastive-circuit": "Evaluate an upstream persona circuit on held-out positive/negative prompt pairs.",
    "persona.behavior-upstream-contrastive-circuit": "Validate the natural-contrast upstream persona circuit during generation.",
    "persona.behavior-connected-circuits": "Test vector rescue and connected upstream/downstream persona circuits.",
    "persona.vector-diagnostic": "Test positive/negative steering and projection removal on high-trait prompts.",
    "persona.run-steering-eapig": "Discover a downstream behavioral circuit along a persona-steering coefficient path.",
    "persona.evaluate-steering-circuit": "Evaluate steering-path circuit mediation on held-out response logits.",
    "persona.behavior-steering-circuit": "Validate a steering-path circuit during free generation.",
    "rq.causal-site-sweep": "Sweep local projection-ablation sites.",
    "results.summary": "Summarize committed RQ1/RQ2 results into Markdown.",
}


def _print_commands() -> int:
    width = max(len(command) for command in COMMANDS)
    for command in sorted(COMMANDS):
        description = DESCRIPTIONS.get(command, "")
        print(f"{command:<{width}}  {description}")
    return 0


def _run_legacy(script_name: str, forwarded: list[str]) -> int:
    if script_name == "results_summary":
        module_path = ROOT / "src" / "refusal_circuit" / "results_summary.py"
        command = [sys.executable, str(module_path), *forwarded]
    else:
        # New commands may live in a focused scripts/ subdirectory. Legacy
        # command names still resolve exactly as before.
        new_script_path = ROOT / "scripts" / script_name
        script_path = new_script_path if new_script_path.exists() else LEGACY / script_name
        if not script_path.exists():
            raise SystemExit(f"Unknown or missing legacy script: {script_path}")
        command = [sys.executable, str(script_path), *forwarded]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = env.get("PYTHONUNBUFFERED", "1")
    return subprocess.call(command, cwd=ROOT, env=env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)
    parser.add_argument("command", nargs="?", help="Experiment command. Use `list` to inspect choices.")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to the selected command.")
    parsed = parser.parse_args(argv)

    if parsed.command in (None, "list", "--list"):
        return _print_commands()
    if parsed.command not in COMMANDS:
        print(f"Unknown command: {parsed.command}\n", file=sys.stderr)
        _print_commands()
        return 2
    return _run_legacy(COMMANDS[parsed.command], parsed.args)


if __name__ == "__main__":
    raise SystemExit(main())
