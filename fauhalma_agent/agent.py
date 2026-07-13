"""
FAUhalma agent for AISysProj SS26, assignment 2.

The server client (run_agent.py) calls get_move() once per turn:

    get_move(position) -> [[x, y], [x, y], ...]

`position` maps player names to peg coordinates, for example
{"A": [[0, -4], ...], "B": [...], "C": [...]} — "C" only exists in
3-player games, and we are always player "A".  The returned list is the
path of a single move: two entries for a plain step, more for a hop
chain (the server wants every intermediate landing spelled out).

The strategy in one paragraph: model the board (star or rhombus),
generate every legal move (steps, hop chains, the swap rule), score
positions by how far our pegs still have to travel, and pick a move
with a small "paranoid" alpha-beta search that treats all opponents as
one adversary trying to minimise our score.  Standard library only.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import permutations

Coord = tuple[int, int]
Move = list[Coord]  # first entry is the peg's cell, the rest is its path
Position = dict[str, list[list[int]]]  # raw JSON shape from the server
Occupied = dict[Coord, str]  # occupied cell -> player sitting on it

# The FAUhalma axes are drawn skewed, but if we read coordinates as a
# standard axial hex grid these six steps are exactly the neighbours.
DIRECTIONS: tuple[Coord, ...] = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, -1),
    (-1, 1),
)

# Centre cells no peg may ever occupy.  The spec never lists them
# outright: (-1, 2) has to be blocked for the assignment's own example
# hop chain to be legal, and the rest follow from 120-degree symmetry
# plus the centre itself.
BLOCKED_CENTRE: set[Coord] = {(0, 0), (-1, 2), (2, -1), (-1, -1)}

# Which board we are playing on.  Only environment ss26.1.2.1 uses the
# rhombus; every other environment (where most of the points live) is
# the full star, so that is the default.
BOARD_TYPE = "star"  # or "rhombus"

# Strength/speed knobs.  SEARCH_DEPTH is plies of lookahead and
# TOP_K_MOVES is how many candidate moves survive at each search node.
# The server enforces a time budget per move, so don't get greedy here.
SEARCH_DEPTH = 2
TOP_K_MOVES = 12


@dataclass(frozen=True)
class Board:
    cells: set[Coord]  # every cell that exists on this board
    blocked: set[Coord]  # permanently forbidden centre cells
    home: dict[str, set[Coord]]  # target corner, per player
    start: dict[str, set[Coord]]  # starting corner, per player


# ---------------------------------------------------------------- helpers
def as_coord(pair: Sequence[int]) -> Coord:
    return int(pair[0]), int(pair[1])


def as_json_move(move: Move) -> list[list[int]]:
    return [[x, y] for x, y in move]


def step(cell: Coord, direction: Coord, distance: int = 1) -> Coord:
    return cell[0] + direction[0] * distance, cell[1] + direction[1] * distance


def occupied_map(position: Position) -> Occupied:
    """Map every occupied cell to the player whose peg sits on it."""
    occupied: Occupied = {}
    for player, pegs in position.items():
        for peg in pegs:
            occupied[as_coord(peg)] = player
    return occupied


# ------------------------------------------------------------------ board
def star_cells() -> set[Coord]:
    # Work in cube coordinates (a, b, c) with a + b + c = 0.  The star is
    # the radius-3 hexagon plus six corner triangles, which boils down to:
    # every coordinate within +/-6, and at most one of them allowed to
    # leave the +/-3 hexagon.  73 cells in total.
    cells: set[Coord] = set()
    for x in range(-6, 7):
        for y in range(-6, 7):
            a, b, c = x, y, -x - y
            if max(abs(a), abs(b), abs(c)) > 6:
                continue
            coords_outside_hexagon = sum(1 for v in (a, b, c) if abs(v) > 3)
            if coords_outside_hexagon <= 1:
                cells.add((x, y))
    return cells


def rhombus_cells() -> set[Coord]:
    # The rhombus board is the star with the four side corners cut off,
    # leaving only A's and B's triangles attached to the hexagon.
    return {(x, y) for (x, y) in star_cells() if abs(x) <= 3 and abs(-x - y) <= 3}


def build_board(position: Position) -> Board:
    if BOARD_TYPE == "star":
        cells = star_cells()
    elif BOARD_TYPE == "rhombus":
        cells = rhombus_cells()
    else:
        # Catch typos here rather than losing rated games to a board that
        # silently fell back to the wrong shape.
        raise ValueError(
            f"BOARD_TYPE must be 'star' or 'rhombus', not {BOARD_TYPE!r}"
        )

    # Corners are easiest to write as a bound on one cube coordinate:
    # y >= 4 is the top triangle (A's home), y <= -4 the bottom one
    # (where A starts).  The other players' corners are the same idea
    # rotated by 120 degrees.
    home = {"A": {c for c in cells if c[1] >= 4}}
    start = {"A": {c for c in cells if c[1] <= -4}}

    if "C" in position:
        # Three players sit on alternating corners of the star.
        home["B"] = {c for c in cells if (-c[0] - c[1]) >= 4}
        start["B"] = {c for c in cells if (-c[0] - c[1]) <= -4}
        home["C"] = {c for c in cells if c[0] >= 4}
        start["C"] = {c for c in cells if c[0] <= -4}
    else:
        # Two players: B is simply A mirrored.
        home["B"] = start["A"]
        start["B"] = home["A"]

    return Board(cells, set(BLOCKED_CENTRE), home, start)


# ------------------------------------------------------------- move rules
def on_board(cell: Coord, board: Board) -> bool:
    return cell in board.cells and cell not in board.blocked


def can_step_to(cell: Coord, player: str, occupied: Occupied, board: Board) -> bool:
    """Can a plain (non-hop) step land on this cell?

    Normally the cell has to be empty.  The one exception is the swap
    rule: stepping onto an opponent's peg that sits inside our own home
    trades places with it — without this an opponent could squat in our
    home forever and deny us the win.
    """
    if not on_board(cell, board):
        return False
    if cell not in occupied:
        return True
    return cell in board.home.get(player, set()) and occupied[cell] != player


def can_hop_over(cell: Coord, occupied: Occupied, board: Board) -> bool:
    # Anything that "fills" the midpoint can be hopped over: a peg of any
    # colour, or a blocked centre cell (the assignment's example chain
    # hops across a blocked cell, so those definitely count).
    return cell in occupied or cell in board.blocked


# ------------------------------------------------------------ legal moves
def step_moves(peg: Coord, player: str, occupied: Occupied, board: Board) -> list[Move]:
    moves: list[Move] = []
    for direction in DIRECTIONS:
        target = step(peg, direction)
        if can_step_to(target, player, occupied, board):
            moves.append([peg, target])
    return moves


def hop_moves(peg: Coord, occupied: Occupied, board: Board) -> list[Move]:
    """All hop chains starting at `peg`, one chain per reachable target.

    A hop jumps over an occupied (or blocked) neighbour onto the free
    cell directly behind it, and hops may be chained.  Note that we only
    allow the swap rule on plain steps: a hop landing must be genuinely
    empty (which is also why, unlike step_moves, this function does not
    care who is moving).  If the server were more permissive we would
    merely miss a few options, whereas sending an illegal move forfeits
    the game, so it pays to stay conservative here.
    """
    chains: list[Move] = []

    def explore(current: Coord, path: Move, visited: set[Coord]) -> None:
        for direction in DIRECTIONS:
            over = step(current, direction)
            landing = step(current, direction, 2)
            if not can_hop_over(over, occupied, board):
                continue
            if landing in visited:
                continue  # already been there; also stops hop cycles
            if not on_board(landing, board) or landing in occupied:
                continue
            chain = path + [landing]
            chains.append(chain)
            explore(landing, chain, visited | {landing})

    explore(peg, [peg], {peg})

    # Different chains frequently end on the same cell, and for the game
    # they are interchangeable.  Keep only the shortest chain per target;
    # without this the branching factor explodes and the search starves.
    best_per_target: dict[Coord, Move] = {}
    for chain in chains:
        target = chain[-1]
        if target not in best_per_target or len(chain) < len(best_per_target[target]):
            best_per_target[target] = chain
    return list(best_per_target.values())


def legal_moves(position: Position, player: str, board: Board) -> list[Move]:
    occupied = occupied_map(position)
    moves: list[Move] = []
    for raw_peg in position.get(player, []):
        peg = as_coord(raw_peg)
        moves.extend(step_moves(peg, player, occupied, board))
        moves.extend(hop_moves(peg, occupied, board))
    return moves


# --------------------------------------------------------- applying moves
def apply_move(position: Position, move: Move, player: str) -> Position:
    """Return the position after `player` makes `move`; input is untouched."""
    result: Position = {p: [peg[:] for peg in pegs] for p, pegs in position.items()}
    origin = as_coord(move[0])
    target = as_coord(move[-1])

    own_pegs = [as_coord(p) for p in result[player]]
    try:
        moved = own_pegs.index(origin)
    except ValueError:
        raise ValueError(
            f"{player} has no peg at {origin}, so move {move} is bogus"
        )
    result[player][moved] = [target[0], target[1]]

    # If the target held an enemy peg, this was a swap: the enemy peg
    # moves back to the cell we just vacated.
    target_owner = occupied_map(position).get(target)
    if target_owner is not None and target_owner != player:
        enemy_pegs = [as_coord(p) for p in result[target_owner]]
        result[target_owner][enemy_pegs.index(target)] = [origin[0], origin[1]]
    return result


def is_won(position: Position, player: str, board: Board) -> bool:
    home = board.home[player]
    return all(as_coord(peg) in home for peg in position.get(player, []))


# -------------------------------------------------------------- heuristic
def hex_distance(a: Coord, b: Coord) -> int:
    # Standard cube distance, folded down to axial coordinates.
    dx, dy = a[0] - b[0], a[1] - b[1]
    return max(abs(dx), abs(dy), abs(dx + dy))


def distance_home(cell: Coord, player: str, board: Board) -> int:
    return min(hex_distance(cell, home_cell) for home_cell in board.home[player])


def cheapest_home_assignment(pegs: list[Coord], home_cells: set[Coord]) -> int:
    """Minimum total distance to bring every peg to a *distinct* home cell.

    Summing each peg's distance to its nearest home cell looks tempting,
    but it lets two pegs chase the same cell, and the agent then shuffles
    back and forth in front of its home at the end of the game.  An exact
    assignment fixes that, and brute force is fine at this size: a home
    corner has only 6 cells, so we try at most 6! = 720 pairings.  (This
    would need a real assignment algorithm for full-size Halma.)
    """
    travelling = [p for p in pegs if p not in home_cells]
    if not travelling:
        return 0

    # An enemy peg squatting in our home still counts as an open cell,
    # since the swap rule lets us evict it with a plain step.
    open_cells = [h for h in home_cells if h not in pegs]

    cheapest = None
    for assignment in permutations(open_cells, len(travelling)):
        cost = sum(hex_distance(p, cell) for p, cell in zip(travelling, assignment))
        if cheapest is None or cost < cheapest:
            cheapest = cost
    # cheapest is only ever None if there are fewer open home cells than
    # travelling pegs, which a legal position cannot produce — but a
    # heuristic should degrade, not crash.
    return cheapest if cheapest is not None else 0


def evaluate(position: Position, player: str, board: Board) -> float:
    """Score a position from `player`'s point of view; higher is better."""
    home = board.home[player]
    start = board.start.get(player, set())
    pegs = [as_coord(p) for p in position.get(player, [])]

    # Pegs already home dominate everything else: 1000 outweighs any
    # possible sum of the distance terms, so the search never trades a
    # finished peg for prettier positioning of the rest.
    pegs_home = sum(1 for p in pegs if p in home)
    travel_cost = cheapest_home_assignment(pegs, home)

    # The game only ends when the LAST peg arrives, so the straggler gets
    # an extra nudge — otherwise the search happily optimises the total
    # while one peg rots in the start corner.  Same idea behind the small
    # penalty for pegs that never left the start.
    outside = [p for p in pegs if p not in home]
    straggler = max((distance_home(p, player, board) for p in outside), default=0)
    pegs_in_start = sum(1 for p in pegs if p in start)

    return 1000 * pegs_home - travel_cost - 3 * straggler - 4 * pegs_in_start


# ----------------------------------------------------------------- search
def ordered_moves(
    position: Position, player: str, perspective: str, board: Board
) -> list[Move]:
    """The TOP_K_MOVES most promising moves for `player`, judged by how
    the resulting position looks for `perspective` (i.e. for us).

    The ordering matters twice: it gives alpha-beta earlier cutoffs, and
    it *is* the pruning, because everything below the cut is dropped.
    """
    scored = []
    for move in legal_moves(position, player, board):
        outcome = evaluate(apply_move(position, move, player), perspective, board)
        scored.append((outcome, move))
    # Our own turn wants the best outcome first; an opponent's turn wants
    # the moves that hurt us most first.
    scored.sort(key=lambda pair: pair[0], reverse=(player == perspective))
    return [move for _, move in scored[:TOP_K_MOVES]]


def paranoid_value(
    position: Position,
    me: str,
    board: Board,
    depth: int,
    alpha: float,
    beta: float,
    our_turn: bool,
) -> float:
    """Alpha-beta value where all opponents are merged into one adversary.

    Proper n-player search (max^n) barely prunes at all.  The classic
    workaround is to be paranoid: assume the whole table cooperates
    against us.  That turns the game back into a two-player zero-sum one,
    where alpha-beta is sound again.  On the minimising ply we simply let
    the "adversary" pick from every move of every opponent.
    """
    if depth == 0 or is_won(position, me, board):
        return evaluate(position, me, board)

    if our_turn:
        best = float("-inf")
        for move in ordered_moves(position, me, me, board):
            child = apply_move(position, move, me)
            value = paranoid_value(
                child, me, board, depth - 1, alpha, beta, our_turn=False
            )
            best = max(best, value)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best

    worst = float("inf")
    for opponent in position:
        if opponent == me:
            continue
        for move in ordered_moves(position, opponent, me, board):
            child = apply_move(position, move, opponent)
            value = paranoid_value(
                child, me, board, depth - 1, alpha, beta, our_turn=True
            )
            worst = min(worst, value)
            beta = min(beta, worst)
            if beta <= alpha:
                return worst
    return worst


# ------------------------------------------------------- public interface
def get_move(position: Position) -> list[list[int]]:
    """Pick a move for player A.  This is the function the client calls."""
    me = "A"
    board = build_board(position)

    moves = legal_moves(position, me, board)
    if not moves:
        # Extremely unlikely (pegs are rarely all boxed in at once), but
        # if it happens we want a loud error with the position in it, not
        # a silent timeout on the server.
        raise RuntimeError(
            f"no legal move for player A in position {position!r}"
        )

    # If any single move wins on the spot, take it and skip the search.
    for move in moves:
        if is_won(apply_move(position, move, me), me, board):
            return as_json_move(move)

    best_move = None
    best_value = float("-inf")
    for move in ordered_moves(position, me, me, board):
        child = apply_move(position, move, me)
        value = paranoid_value(
            child, me, board, SEARCH_DEPTH - 1,
            alpha=float("-inf"), beta=float("inf"), our_turn=False,
        )
        if value > best_value:
            best_move, best_value = move, value

    if best_move is None:
        # Only reachable if every searched line came back -inf.  Fall
        # back to the greedy one-ply choice rather than return nothing.
        def one_ply_score(m: Move) -> float:
            return evaluate(apply_move(position, m, me), me, board)

        best_move = max(moves, key=one_ply_score)
    return as_json_move(best_move)


# Some client scripts look for other entry-point names; cover them too.
def choose_move(position: Position) -> list[list[int]]:
    return get_move(position)


def move(position: Position) -> list[list[int]]:
    return get_move(position)
