"""
Client for the AISysProj server (protocol version 1).

Usage:
    python run_agent.py path/to/config.json           # normal, parallel runs
    python run_agent.py path/to/config.json --debug   # one game at a time

The config files (ss26.1.2.1.json ... ss26.1.2.8.json) come from the team
repository; agent.py must sit next to this script.  Standard library only.

Reminder: for the rhombus environment (ss26.1.2.1), flip BOARD_TYPE to
"rhombus" in agent.py.  Leave it "star" for every other environment.
"""

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.request

from agent import get_move

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("fauhalma")

REQUIRED_CONFIG_KEYS = ("url", "env", "agent", "pwd")
RETRY_DELAY = 3  # seconds to wait when the server is busy or unreachable


def load_config(path: str) -> dict:
    try:
        with open(path) as f:
            config = json.load(f)
    except FileNotFoundError:
        sys.exit(f"Config file not found: {path}")
    except json.JSONDecodeError as e:
        sys.exit(f"Config file {path} is not valid JSON: {e}")

    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        sys.exit(f"Config file {path} is missing keys: {', '.join(missing)}")
    return config


def answer_requests(action_requests: list) -> list:
    """Run the agent on every percept the server sent us."""
    actions = []
    for request in action_requests:
        percept = request["percept"]  # the position dict {"A": [...], ...}
        try:
            action = get_move(percept)
        except Exception:
            # One crashing percept shouldn't take down the other parallel
            # runs.  The affected run will presumably time out and count
            # as a loss, so log enough to reproduce the bug afterwards.
            log.exception(
                f"get_move failed on run {request.get('run')}, percept {percept!r}"
            )
            continue
        actions.append(
            {"run": request["run"], "act_no": request["act_no"], "action": action}
        )
    return actions


def play(config_path: str, parallel_runs: bool = True) -> None:
    config = load_config(config_path)
    endpoint = f"{config['url']}/act/{config['env']}"
    log.info(f"Playing on {endpoint} as agent '{config['agent']}'")

    actions: list = []  # the first request sends none, just to fetch percepts
    while True:
        body = json.dumps(
            {
                "protocol_version": 1,
                "agent": config["agent"],
                "pwd": config["pwd"],
                "actions": actions,
                "parallel_runs": parallel_runs,
                "client": "fauhalma-stdlib-client",
            }
        ).encode()

        request = urllib.request.Request(
            endpoint,
            data=body,
            method="PUT",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request) as response:
                data = json.loads(response.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 503:
                # The server asks us to back off; perfectly normal when
                # it is under load, so just try again in a moment.
                log.warning(f"Server busy - retrying in {RETRY_DELAY}s")
                time.sleep(RETRY_DELAY)
                continue
            # Anything else (wrong password, malformed request, ...) will
            # not fix itself by retrying, so surface it and stop.
            log.error(f"HTTP {e.code}: {e.read().decode(errors='replace')}")
            return
        except urllib.error.URLError as e:
            # Flaky network.  Keep trying — an overnight rating run should
            # not die because the wifi blipped once.
            log.error(f"Connection error: {e} - retrying in {RETRY_DELAY}s")
            time.sleep(RETRY_DELAY)
            continue
        except json.JSONDecodeError as e:
            log.error(f"Server sent malformed JSON ({e}) - retrying in {RETRY_DELAY}s")
            time.sleep(RETRY_DELAY)
            continue

        for message in data.get("messages", []):
            log.info(f"[{message.get('type')}] {message.get('content')}")

        actions = answer_requests(data.get("action_requests", []))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the FAUhalma agent against the AISysProj server."
    )
    parser.add_argument("config", help="path to the environment config JSON")
    parser.add_argument(
        "--debug", action="store_true", help="play one game at a time"
    )
    args = parser.parse_args()

    try:
        play(args.config, parallel_runs=not args.debug)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
