"""kerneltrace command line."""
from __future__ import annotations

import argparse
import json
import sys


def cmd_validate(args) -> int:
    import yaml

    from .core.spec import load_claim, load_experiment

    for path in args.paths:
        try:
            with open(path, encoding="utf-8") as f:
                is_experiment = "experiment_id" in (yaml.safe_load(f) or {})
            if not is_experiment:
                c = load_claim(path)
                print(f"ok  claim {c.claim_id}: {len(c.predictions)} predictions, "
                      f"{len(c.invariances)} invariances, status {c.status.value}")
            else:
                s, c = load_experiment(path)
                print(f"ok  experiment {s.experiment_id} -> claim {c.claim_id}")
        except Exception as exc:  # report every file, then fail
            print(f"FAIL {path}: {exc}")
            return 1
    return 0


def cmd_env(args) -> int:
    from .core.manifest import environment

    env = environment()
    env.pop("packages")
    print(json.dumps(env, indent=2))
    return 0


def cmd_run(args) -> int:
    from .runner import run_experiment

    root = run_experiment(args.spec, args.out)
    print(root)
    return 0


def cmd_replay(args) -> int:
    from .runner import replay_bundle

    out = replay_bundle(args.bundle, args.atol)
    print(json.dumps(out, indent=2))
    return 0 if out["reproduced"] else 2


def cmd_preflight(args) -> int:
    from .runner import Experiment

    exp = Experiment(args.spec, args.out)
    exp.setup()
    exp.build_tasks()
    exp.baseline()
    den = exp.denominator()
    print(json.dumps(dict(gates={k: v for k, v in exp.gates.items()}, denominator=den), indent=2))
    return 0 if all(ok for ok, _ in exp.gates.values()) else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="kerneltrace", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("run", help="run an experiment spec and write an evidence bundle")
    s.add_argument("spec")
    s.add_argument("--out", default="runs")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("replay", help="re-run a bundle's reference measurement and compare")
    s.add_argument("bundle")
    s.add_argument("--atol", type=float, default=1e-5)
    s.set_defaults(fn=cmd_replay)

    s = sub.add_parser("preflight", help="task validity and baseline gates only")
    s.add_argument("spec")
    s.add_argument("--out", default="runs")
    s.set_defaults(fn=cmd_preflight)

    s = sub.add_parser("validate", help="schema-check claim cards and experiment specs")
    s.add_argument("paths", nargs="+")
    s.set_defaults(fn=cmd_validate)

    s = sub.add_parser("env", help="print the environment manifest")
    s.set_defaults(fn=cmd_env)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
