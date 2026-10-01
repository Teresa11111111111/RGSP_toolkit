"""Command-line entry points; config files carry radians and intensity efficiencies."""
import argparse
import json
import sys
from pathlib import Path

from .core import client_phase_update, random_hiding, validate_qubits
from .io import (compile_config, export_plan, keys, new_output_dir, parse_client,
                 write_json, write_modes)


def load_json(path):
    def unique(pairs):
        obj = {}
        for k, v in pairs:
            if k in obj:
                raise ValueError(f"duplicate JSON key: {k}")
            obj[k] = v
        return obj
    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique,
                      parse_constant=reject_constant)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rgsp", description="RGSP phase and loss-compensation calculator")
    parser.add_argument("--version", action="version", version="rgsp-toolkit 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [("compile", "Compile source amplitudes, client phases and private corrections"),
                            ("client", "Generate one client's local phase table independently")]:
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("config", type=Path)
        cmd.add_argument("--out", type=Path, required=True, help="New output directory; never overwrites")
    random = sub.add_parser("random-mask", help="Generate fresh private eight-angle hiding and Z bits")
    random.add_argument("--qubits", type=int, required=True)
    random.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "compile":
            config = load_json(args.config)
            plan = compile_config(config)
            out = export_plan(plan, config, args.out)
            print(f"Wrote {1 << plan.qubits} modes for {len(plan.client_updates_rad)} client(s) to {out}")
            print(f"Predicted all-outcome probability: {plan.survival_probability:.8g}")
        elif args.command == "client":
            config = load_json(args.config)
            keys(config, {"qubits", "client"}, {"qubits", "client"})
            n = validate_qubits(config["qubits"])
            phase = client_phase_update(n, parse_client(config["client"]))
            out = new_output_dir(args.out)
            write_modes(out / "client_phases.csv", n, phase)
            write_json(out / "private_client_config.json", config)
            print(f"Wrote local phase table to {out}")
        else:
            mask = random_hiding(args.qubits)
            out = new_output_dir(args.out)
            write_json(out / "private_mask.json", mask)
            print(f"Wrote private mask to {out}; insert fields into your local client configuration")
        return 0
    except (ValueError, TypeError, OSError) as exc:
        print(f"rgsp: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
