# structurezyme/cli.py
import argparse
import sys
from pathlib import Path

from .config import RunConfig, PathsConfig, load_config
from .hosts import apply_host_defaults
from .manifest import Manifest
from .paths import RunLayout
from .runner import Runner


def _template() -> RunConfig:
    return RunConfig(paths=PathsConfig(output_root="", boltz_cache_dir=""))


def cmd_init(args) -> int:
    _template().write(args.output)
    print(f"Wrote template config to {args.output}")
    return 0


def cmd_run(args) -> int:
    cfg = load_config(args.config)
    cfg = apply_host_defaults(cfg, args.host)
    if args.force:
        cfg.runtime.force = args.force.split(",")
    Runner(cfg).run()
    return 0


def cmd_resume(args) -> int:
    layout = RunLayout(args.run_dir)
    cfg = load_config(layout.config_path)
    Runner(cfg).run()
    return 0


def cmd_step(args) -> int:
    layout = RunLayout(args.run_dir)
    cfg = load_config(layout.config_path)
    cfg.runtime.force = list(set(cfg.runtime.force) | {args.name})
    Runner(cfg).run()
    return 0


def cmd_status(args) -> int:
    layout = RunLayout(args.run_dir)
    manifest_path = layout.manifest_path
    if not manifest_path.is_file():
        print(f"No manifest at {manifest_path}", file=sys.stderr)
        return 1
    m = Manifest.load(manifest_path)
    for name, rec in m.steps.items():
        wt = f"{rec.wall_time_s:.1f}s" if rec.wall_time_s else "-"
        print(f"{name:20s} {rec.status:20s} {wt}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="structurezyme")
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("init")
    pi.add_argument("--output", required=True)
    pi.set_defaults(func=cmd_init)

    pr = sub.add_parser("run")
    pr.add_argument("--config", required=True)
    pr.add_argument("--host", default=None)
    pr.add_argument("--force", default=None)
    pr.set_defaults(func=cmd_run)

    ps = sub.add_parser("resume")
    ps.add_argument("--run-dir", required=True)
    ps.set_defaults(func=cmd_resume)

    pst = sub.add_parser("step")
    pst.add_argument("name")
    pst.add_argument("--run-dir", required=True)
    pst.set_defaults(func=cmd_step)

    pstat = sub.add_parser("status")
    pstat.add_argument("--run-dir", required=True)
    pstat.set_defaults(func=cmd_status)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
