# FAUhalma Agent — AISysProj SS26, Assignment 2

A FAUhalma agent (player A) using a **paranoid alpha-beta** search over a
distance-to-home heuristic. Handles star and rhombus boards, 2- and 3-player.

## Dependencies
- **Python 3.10+**, standard library only (no external packages).

## How to run on a server environment
1. Keep `agent.py` and `run_agent.py` in the same folder, together with the
   config file for the environment (from your FAU team repo).
2. For the **rhombus** environment (`ss26.1.2.1`) set `BOARD_TYPE = "rhombus"`
   in `agent.py`; leave it `"star"` for all other environments.
3. Run:
   ```bash
   python run_agent.py ss26.1.2.4.json            # plays on that environment
   python run_agent.py ss26.1.2.1.json --debug    # one game at a time (debug)
   ```
4. Watch your rating on https://aisysproj.kwarc.info.

To trade strength for speed, edit `SEARCH_DEPTH` and `TOP_K_MOVES` at the top
of `agent.py` (larger = stronger but slower per move).

## Repository structure
```
agent.py       # board model, legal moves, heuristic, paranoid search, get_move()
run_agent.py   # AISysProj server client (calls get_move each turn)
README.md
solution-summary.md
```

## Notes
- The agent is always player `A`. `get_move(position)` returns a JSON move
  (`[[x, y], ...]`); hop chains include all intermediate cells.
- The four blocked centre cells `{(0,0), (-1,2), (2,-1), (-1,-1)}` are
  cross-verified against the SS24 specification and the example hop chain.
