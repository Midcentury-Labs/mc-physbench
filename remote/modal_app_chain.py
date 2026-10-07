"""`vitals-chain`: run a QUEUE of population submits on Modal itself, so the
laptop is out of the scheduling loop (2026-10-06: the laptop slept three
times and every local chain's poll stalled for hours while Modal sat
finished). Its own app -- it looks the orchestrator up by name, so
deploying it never touches `vitals-orchestrate` or anything in flight.

A chain is a list of steps; each step is the argv for
run_model_population.py (exactly what `call_orchestrate.py submit` takes).
Steps run strictly one after another: spawn `run_population`, wait for
it, record the result, next. `wait_for` is a list of orchestrator call
ids that must finish before the first step starts (how a chain queues
behind runs that were submitted by hand). Modal caps a function at 24 h,
so the runner re-spawns itself with the remaining steps before its budget
runs out; the ledger (a JSON on the results Volume, chains/<name>.json)
carries the state across those hops.

    modal deploy remote/modal_app_chain.py
    python3 remote/call_chain.py submit --name cog_rest --wait-for fc-... --steps-file steps.json
    python3 remote/call_chain.py status cog_rest
"""
import json
import pathlib
import time

import modal

APP_NAME = "vitals-chain"
ORCH_APP = "vitals-orchestrate"
BUDGET_S = 22 * 3600          # re-spawn before Modal's 24 h ceiling
POLL_S = 120

app = modal.App(APP_NAME)
results_volume = modal.Volume.from_name("vitals-results", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12")

LEDGER_DIR = pathlib.Path("/results/chains")


def _ledger_path(name):
    return LEDGER_DIR / f"{name}.json"


def _write(name, state):
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    _ledger_path(name).write_text(json.dumps(state, indent=1))
    results_volume.commit()


def _read(name):
    results_volume.reload()
    p = _ledger_path(name)
    return json.loads(p.read_text()) if p.exists() else None


@app.function(image=image, volumes={"/results": results_volume}, timeout=24 * 3600, cpu=1, memory=512)
def run_chain(name: str, steps: list, wait_for: list = None, done: list = None, wait_for_chains: list = None):
    """steps: list of {"key": "<scenario>_<model>", "argv": [...]}; runs them in order."""
    t0 = time.time()
    done = list(done or [])
    # RESUME (2026-10-06): Modal restarts a preempted container with the
    # ORIGINAL input, which would replay the whole queue. If a ledger for this
    # name already exists and is not complete, continue from it instead:
    # keep its done list and pending list, and if a step was in flight,
    # adopt that call rather than submitting it again.
    prior = _read(name)
    if prior and not prior.get("completed") and (prior.get("done") or prior.get("current")):
        state = prior
        state["wait_for"] = []
        state["hop_started"] = time.strftime("%Y-%m-%d %H:%M:%S")
        state.setdefault("notes", []).append(f"resumed from ledger at {state['hop_started']}")
    else:
        state = dict(name=name, wait_for=list(wait_for or []), wait_for_chains=list(wait_for_chains or []), pending=list(steps), done=done, current=None,
                     started=time.strftime("%Y-%m-%d %H:%M:%S"), hop_started=time.strftime("%Y-%m-%d %H:%M:%S"))
    _write(name, state)
    run_population = modal.Function.from_name(ORCH_APP, "run_population")

    # queue behind OTHER CHAINS (2026-10-07: parallel chains, with the wan step
    # alone at the end): poll their ledgers until each is completed.
    for other in list(state.get("wait_for_chains") or []):
        while True:
            st = _read(other)
            if st is None or st.get("completed"):
                break
            time.sleep(POLL_S)
        state["wait_for_chains"].remove(other)
        _write(name, state)

    # queue behind hand-submitted runs
    for cid in list(state["wait_for"]):
        fc = modal.FunctionCall.from_id(cid)
        while True:
            try:
                fc.get(timeout=0)
                break
            except TimeoutError:
                time.sleep(POLL_S)
            except Exception as e:           # the awaited run failed; nothing to wait for any more
                state.setdefault("notes", []).append(f"wait_for {cid}: {type(e).__name__}")
                break
        state["wait_for"].remove(cid)
        _write(name, state)

    while state["pending"]:
        if time.time() - t0 > BUDGET_S:          # hand the remainder to a fresh container
            nxt = run_chain.spawn(name, state["pending"], [], state["done"])
            state["rehop"] = nxt.object_id
            _write(name, state)
            return state
        step = state["pending"][0]
        cur = state.get("current")
        if cur and cur.get("key") == step["key"] and cur.get("call_id"):
            fc = modal.FunctionCall.from_id(cur["call_id"])          # adopt the in-flight call after a resume
        else:
            fc = run_population.spawn(step["argv"])
            state["current"] = dict(key=step["key"], call_id=fc.object_id, submitted=time.strftime("%Y-%m-%d %H:%M:%S"))
        _write(name, state)
        try:
            while True:
                try:
                    res = fc.get(timeout=0)
                    break
                except TimeoutError:
                    time.sleep(POLL_S)
            n = len(res.get("seeds", [])) if isinstance(res, dict) else None
            outcome = dict(key=step["key"], call_id=fc.object_id, status="done", n=n,
                           failed_seeds=(res.get("failed_seeds") if isinstance(res, dict) else None),
                           finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        except Exception as e:
            outcome = dict(key=step["key"], call_id=fc.object_id, status="failed", error=repr(e)[:300],
                           finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        state["done"].append(outcome)
        state["pending"].pop(0)
        state["current"] = None
        _write(name, state)
    state["completed"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write(name, state)
    return state


@app.function(image=image, volumes={"/results": results_volume}, timeout=300)
def chain_status(name: str) -> dict:
    return _read(name) or {"error": f"no chain named {name!r}"}
