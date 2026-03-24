"""
CLI entry point — ``python -m cognitive_memory``.

Subcommands:
    store <text>       Encode and store a memory strand
    query <text>       Query the memory system
    stats              Print genome/graph/entity statistics
    consolidate        Run memory consolidation (sleep)
    forget             Run intelligent forgetting
    export             Dump genome as JSON to stdout
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import Config
from .memory import MemorySystem


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cognitive_memory",
        description="Cognitive memory system for AI agents",
    )
    parser.add_argument(
        "--genome", default=None, help="Path to genome.json"
    )
    parser.add_argument(
        "--graph", default=None, help="Path to graph.json"
    )
    parser.add_argument(
        "--entities", default=None, help="Path to entities.json"
    )
    parser.add_argument(
        "--model", default=None, help="Claude model to use"
    )

    sub = parser.add_subparsers(dest="command")

    # store
    p_store = sub.add_parser("store", help="Store a memory")
    p_store.add_argument("text", help="Raw text to encode and store")

    # query
    p_query = sub.add_parser("query", help="Query the memory system")
    p_query.add_argument("text", help="Query text")

    # stats
    sub.add_parser("stats", help="Print system statistics")

    # consolidate
    sub.add_parser("consolidate", help="Run memory consolidation")

    # forget
    p_forget = sub.add_parser("forget", help="Run intelligent forgetting")
    p_forget.add_argument(
        "--min-age", type=int, default=None,
        help="Minimum strand age in seconds (default: 30 days)",
    )
    p_forget.add_argument(
        "--min-activations", type=int, default=None,
        help="Minimum activation count to protect a strand (default: 0)",
    )

    # export
    sub.add_parser("export", help="Export genome as JSON")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    # Build config from env + CLI flags
    cli_overrides = {}
    if args.genome:
        cli_overrides["genome_path"] = args.genome
    if args.graph:
        cli_overrides["graph_path"] = args.graph
    if args.entities:
        cli_overrides["entities_path"] = args.entities
    if args.model:
        cli_overrides["model"] = args.model

    config = Config.from_env(**cli_overrides)

    if args.command == "stats":
        system = MemorySystem(config=config)
        stats = system.stats()
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return 0

    if args.command == "export":
        system = MemorySystem(config=config)
        data = [s.to_dict() for s in system.genome.all_strands()]
        json.dump(data, sys.stdout, indent=2)
        print()
        return 0

    if args.command == "store":
        system = MemorySystem(config=config)
        strand = system.store(args.text)
        if strand is None:
            print("Duplicate — already stored.")
        else:
            print(f"Stored strand {strand.strand_id}")
            print(f"  Trace: {strand.trace}")
            print(f"  Entities: {strand.get_entity_instance_ids()}")
        return 0

    if args.command == "query":
        system = MemorySystem(config=config)
        result = system.query(args.text)
        print(result["answer"])
        print(f"\n--- {len(result['activated'])} strands activated, "
              f"{result['tokens_used']} tokens used ---")
        return 0

    if args.command == "consolidate":
        system = MemorySystem(config=config)
        result = system.consolidate()
        print(f"Consolidated {result['consolidated']} strands "
              f"across {result['groups_processed']} groups")
        return 0

    if args.command == "forget":
        system = MemorySystem(config=config)
        kwargs = {}
        if args.min_age is not None:
            kwargs["min_age_seconds"] = args.min_age
        if args.min_activations is not None:
            kwargs["min_activations"] = args.min_activations
        result = system.forget(**kwargs)
        print(f"Forgotten {result['forgotten']} strands")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
