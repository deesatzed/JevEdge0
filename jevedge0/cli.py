"""JevEdge0 command line.

    jevedge0 serve      run the workbench (UI + API) over a loaded Edge0
    jevedge0 decide     one typed decision on the terminal
    jevedge0 compare    dual-channel decision with the guardrail verdict
    jevedge0 ask        ask the agent (tools enabled)
    jevedge0 ingest     add documents to the knowledge base
    jevedge0 search     search the knowledge base
    jevedge0 bench      compare decision methods on a JSONL set
"""

from __future__ import annotations

import argparse
import json
import os
import sys

DEFAULT_MODEL_ENV = "EDGE0_35B_MODEL"


def _resolve_model(args) -> str:
    """Find the checkpoint: --model, the tier env var, or models/<tier>."""
    if getattr(args, "model", None):
        return os.path.expanduser(args.model)
    env = os.environ.get(DEFAULT_MODEL_ENV)
    if env and os.path.isdir(env):
        return env
    for candidate in ("models/edge0-35b", "models/edge0-8b"):
        if os.path.isdir(candidate):
            return candidate
    raise SystemExit(
        "no checkpoint found. Pass --model /path/to/checkpoint or set "
        f"{DEFAULT_MODEL_ENV}.")


def _load_engine(args):
    from edge0 import AutoEngine
    model_dir = _resolve_model(args)
    name = getattr(args, "name", None)
    print(f"[jevedge0] loading {model_dir}", file=sys.stderr)
    return AutoEngine.from_pretrained(model_dir, name=name)


def _queue_server(args):
    from edge0.server.chat import QueueServer
    engine = _load_engine(args)
    return QueueServer(engine, model_name=getattr(args, "name", None)
                       or engine.name)


def _workbench(args, load_embedder: bool = True):
    from jevedge0.app import Workbench
    folders = [os.path.expanduser(f) for f in (args.allow or [])]
    return Workbench(_queue_server(args), home=args.home,
                     allowed_folders=folders or None,
                     load_embedder=load_embedder)


def _read_row(args) -> dict:
    if args.file:
        with open(os.path.expanduser(args.file)) as fh:
            return json.load(fh)
    if not (args.state and args.criterion and args.option):
        raise SystemExit(
            "provide --file, or --state, --criterion and at least two "
            "--option id=description pairs")
    options = []
    for item in args.option:
        if "=" not in item:
            raise SystemExit(f"option must be id=description: {item!r}")
        oid, description = item.split("=", 1)
        options.append({"id": oid.strip(), "description": description.strip()})
    return {"id": args.id or "decision", "state": args.state,
            "question": args.criterion, "options": options}


def _print_decision(record: dict) -> None:
    print(f"\nchoice: {record['choice']}")
    pairs = list(zip(record["option_ids"], record["probabilities"]))
    width = max(len(o) for o, _ in pairs)
    for option_id, probability in pairs:
        bar = "█" * round(probability * 30)
        marker = "→" if option_id == record["choice"] else " "
        print(f"  {marker} {option_id:<{width}}  {probability * 100:6.2f}%  {bar}")
    print(f"\nmargin {record['margin']:.3f}   entropy {record['entropy']:.3f}"
          f"   {record['input_tokens']} tokens"
          f"   {record['forward_seconds']:.2f}s")
    stability = record.get("stability")
    if stability:
        state = ("stable" if stability["stable_across_permutations"]
                 else "UNSTABLE")
        print(f"permutations: {state} over {stability['trials']} orderings "
              f"(max spread {stability['max_probability_spread']:.3f})")
    print(f"\nnote: {record['probability_status']}")


# ---- commands -------------------------------------------------------------

def cmd_decide(args) -> int:
    from jevedge0.decision import Edge0DecisionScorer
    from jevedge0.decision.stability import score_with_stability

    row = _read_row(args)
    engine = _load_engine(args)
    try:
        scorer = Edge0DecisionScorer(engine)
        record = (score_with_stability(scorer, row, trials=args.trials)
                  if args.trials > 1 else scorer.score(row))
        if args.json:
            print(json.dumps(record, indent=2, default=str))
        else:
            _print_decision(record)
    finally:
        engine.close()
    return 0


def cmd_compare(args) -> int:
    from jevedge0.comparator.dual import Comparator
    from jevedge0.decision import Edge0DecisionScorer
    from jevedge0.orchestrator.engine_client import InProcessClient

    row = _read_row(args)
    server = _queue_server(args)
    try:
        client = InProcessClient(server)
        comparator = Comparator(Edge0DecisionScorer(server.engine),
                                client.chat, stability_trials=args.trials)
        result = comparator.run(row)
        if args.json:
            print(json.dumps(result, indent=2, default=str))
            return 0
        _print_decision(result["decision"])
        judgment = result["judgment"]
        print(f"\njudgment channel: {judgment['choice'] or 'UNPARSED'}")
        print(f"agreement: {result['agreement']}")
        print(f"\nverdict: {result['action'].upper()} — {result['summary']}")
        for reason in result["reasons"]:
            print(f"  · {reason}")
        if args.show_reasoning:
            print(f"\n--- judgment reasoning ---\n{judgment['reasoning']}")
    finally:
        server.engine.close()
    return 0


def cmd_ask(args) -> int:
    workbench = _workbench(args)
    try:
        result = workbench.agent.run(args.question)
        for step in result["steps"]:
            print(f"[tool] {step['tool']} → {step['status']}", file=sys.stderr)
        print(result["answer"])
    finally:
        workbench.queue_server.engine.close()
    return 0


def cmd_ingest(args) -> int:
    from jevedge0.rag.knowledge import KnowledgeService
    from jevedge0.store.db import Store

    # Ingestion needs the embedder, not the 35B model.
    store = Store(os.path.join(os.path.expanduser(args.home), "jevedge0.db"))
    knowledge = KnowledgeService(store)
    path = os.path.expanduser(args.path)
    if os.path.isdir(path):
        report = knowledge.ingest_folder(path, args.collection)
        print(f"ingested {report['ingested']} documents "
              f"({report['chunks']} chunks), {report['duplicates']} duplicates")
        for failure in report["failed"]:
            print(f"  failed: {failure['file']}: {failure['error']}",
                  file=sys.stderr)
    else:
        report = knowledge.ingest(path, args.collection)
        print(f"{report['status']}: {report['filename']} "
              f"({report['chunks']} chunks)")
    return 0


def cmd_search(args) -> int:
    from jevedge0.rag.knowledge import KnowledgeService
    from jevedge0.store.db import Store

    store = Store(os.path.join(os.path.expanduser(args.home), "jevedge0.db"))
    knowledge = KnowledgeService(store)
    results = knowledge.search(args.query, top_k=args.top_k,
                               collection=args.collection)
    if not results:
        print("no matches")
        return 0
    for index, result in enumerate(results, start=1):
        page = f" p.{result['page']}" if result["page"] else ""
        print(f"\n[{index}] {result['filename']}{page}  "
              f"score {result.get('rerank_score', 0):.3f}")
        if result["heading"]:
            print(f"    {result['heading']}")
        print(f"    {result['text'][:300].strip()}...")
    return 0


def cmd_serve(args) -> int:
    from jevedge0.app import build_router
    from jevedge0.server.http import serve

    workbench = _workbench(args)
    router = build_router(workbench)
    health = workbench.health()
    print(f"[jevedge0] model      {health['model']}", file=sys.stderr)
    print(f"[jevedge0] home       {health['home']}", file=sys.stderr)
    embeddings = health["embeddings"]
    print(f"[jevedge0] embeddings "
          f"{embeddings['model'] if embeddings['available'] else 'UNAVAILABLE: ' + str(embeddings['error'])}",
          file=sys.stderr)
    print(f"[jevedge0] tools      {', '.join(health['tools'])}",
          file=sys.stderr)
    print(f"\n  http://{args.host}:{args.port}\n", file=sys.stderr)
    try:
        serve(router, host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("\n[jevedge0] stopped", file=sys.stderr)
    finally:
        workbench.queue_server.engine.close()
    return 0


def cmd_bench(args) -> int:
    from jevedge0.bench.runner import run_benchmark
    return run_benchmark(args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jevedge0", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_model_args(p):
        p.add_argument("--model", default=None,
                       help="checkpoint directory (default: $EDGE0_35B_MODEL "
                            "or models/edge0-35b)")
        p.add_argument("--name", default=None, help="tier name override")

    def add_row_args(p):
        p.add_argument("--file", default=None, help="JSON file with the row")
        p.add_argument("--state", default=None, help="the evidence")
        p.add_argument("--criterion", default=None, help="the question")
        p.add_argument("--option", action="append", default=None,
                       metavar="ID=DESCRIPTION", help="repeatable")
        p.add_argument("--id", default=None)
        p.add_argument("--json", action="store_true")

    p = sub.add_parser("decide", help="run one typed decision")
    add_model_args(p); add_row_args(p)
    p.add_argument("--trials", type=int, default=1,
                   help="option orderings to test (>1 enables stability)")
    p.set_defaults(fn=cmd_decide)

    p = sub.add_parser("compare", help="dual-channel decision + guardrail")
    add_model_args(p); add_row_args(p)
    p.add_argument("--trials", type=int, default=4)
    p.add_argument("--show-reasoning", action="store_true")
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("ask", help="ask the tool-using agent")
    add_model_args(p)
    p.add_argument("question")
    p.add_argument("--home", default="~/.jevedge0")
    p.add_argument("--allow", action="append", default=None,
                   help="folder the tools may read (repeatable)")
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("ingest", help="add documents to the knowledge base")
    p.add_argument("path")
    p.add_argument("--collection", default="default")
    p.add_argument("--home", default="~/.jevedge0")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("search", help="search the knowledge base")
    p.add_argument("query")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--collection", default=None)
    p.add_argument("--home", default="~/.jevedge0")
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("serve", help="run the workbench UI and API")
    add_model_args(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--home", default="~/.jevedge0")
    p.add_argument("--allow", action="append", default=None)
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("bench", help="compare decision methods")
    add_model_args(p)
    p.add_argument("--input", required=True, help="JSONL decision rows")
    p.add_argument("--output", required=True, help="results JSON (create-only)")
    p.add_argument("--methods", default="edge0-logit,edge0-judgment",
                   help="comma-separated: edge0-logit, edge0-judgment, "
                        "comparator")
    p.add_argument("--trials", type=int, default=4)
    p.add_argument("--labels", default=None,
                   help="JSONL of {id, label} for accuracy scoring")
    p.set_defaults(fn=cmd_bench)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
