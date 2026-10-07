"""Submit / inspect a queue of population runs on the deployed `vitals-chain`
app (remote/modal_app_chain.py). The queue runs on Modal; the laptop only
submits and, later, pulls.

    python3 remote/call_chain.py submit --name <name> [--wait-for fc-... ...] --steps-file steps.json
    python3 remote/call_chain.py status <name>
    python3 remote/call_chain.py pull <name>        # pull every finished step that is not local yet

steps.json: [{"key": "soft_ramp_wan22", "argv": ["--model", "wan22", "--scenario", "soft_ramp", ...]}, ...]
"""
import argparse
import json
import pathlib
import subprocess
import sys

import modal

ROOT = pathlib.Path(__file__).resolve().parents[1]
APP_NAME = "vitals-chain"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("submit"); s.add_argument("--name", required=True); s.add_argument("--wait-for", nargs="*", default=[])
    s.add_argument("--steps-file", required=True); s.add_argument("--wait-for-chains", nargs="*", default=[])
    st = sub.add_parser("status"); st.add_argument("name")
    pl = sub.add_parser("pull"); pl.add_argument("name"); pl.add_argument("--force", action="store_true")
    a = p.parse_args()
    if a.command == "submit":
        steps = json.loads(pathlib.Path(a.steps_file).read_text())
        f = modal.Function.from_name(APP_NAME, "run_chain")
        fc = f.spawn(a.name, steps, a.wait_for, [], a.wait_for_chains)
        print(f"spawned chain {a.name!r}: {fc.object_id}, {len(steps)} step(s), waiting on {len(a.wait_for)} call(s), {len(a.wait_for_chains)} chain(s)")
        return
    state = modal.Function.from_name(APP_NAME, "chain_status").remote(a.name)
    if a.command == "status":
        if "error" in state:
            print(state["error"]); return
        print(f"chain {state['name']}: started {state['started']}" + (f", completed {state['completed']}" if state.get("completed") else ""))
        if state.get("wait_for"):
            print(f"  waiting on: {state['wait_for']}")
        for d in state["done"]:
            print(f"  {d['status']:6s} {d['key']}  n={d.get('n')} failed={d.get('failed_seeds')}  {d.get('finished', '')}" + (f"  {d['error']}" if d.get("error") else ""))
        if state.get("current"):
            print(f"  RUNNING {state['current']['key']}  {state['current']['call_id']}  since {state['current']['submitted']}")
        for s_ in state["pending"]:
            if not state.get("current") or s_["key"] != state["current"]["key"]:
                print(f"  queued  {s_['key']}")
        return
    if a.command == "pull":
        for d in state["done"]:
            if d["status"] != "done":
                continue
            scen_model = d["key"]
            local = ROOT / "results" / f"l0_demo_{scen_model}.json"
            if local.exists() and not a.force:
                continue
            cmd = [sys.executable, str(ROOT / "remote" / "call_orchestrate.py"), "pull", "--only", scen_model] + (["--force"] if a.force else [])
            r = subprocess.run(cmd, capture_output=True, text=True)
            print(f"pull {scen_model}: {'ok' if r.returncode == 0 else 'FAILED'} {(r.stdout or r.stderr).strip().splitlines()[-1][:120] if (r.stdout or r.stderr).strip() else ''}")


if __name__ == "__main__":
    main()
