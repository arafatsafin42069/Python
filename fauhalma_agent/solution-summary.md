# Solution Summary — FAUhalma Agent

## Representation
The skewed FAUhalma axes are treated as a standard **axial hex grid**: the six
step directions are the six hex neighbours and distance is the cube/hex distance
`max(|dx|, |dy|, |dx+dy|)`. The board is generated programmatically — a cell
`(a,b,c)=(x,y,-x-y)` is valid iff every coordinate is within ±6 and at most one
exceeds ±3 (central hexagon + six corner triangles = 73 cells); the rhombus board
is the same with the four side corners removed. Each player's home is the
opposite corner, written as a one-line bound on a cube coordinate, and the agent
adapts homes/starts to 2- vs 3-player positions automatically.

## Move generation
Simple moves step a peg to an adjacent free cell. Hop chains are found by DFS:
a hop requires the midpoint to hold a peg or a blocked centre cell and the
landing to be free, and chains are **deduplicated by final target** to avoid the
branching-factor blow-up. The swap rule is supported for simple moves into the
mover's own home.

## Heuristic
Each out-of-home peg is matched to a distinct home cell by an exact minimum-cost
assignment; the score rewards pegs already home, penalises the total assigned
distance, and adds an extra penalty on the single slowest peg (the game is only
won once every peg arrives) and on pegs still stuck in the start corner.

## Search
A **paranoid alpha-beta** search: our moves maximise the heuristic while all
opponents are collapsed into one adversary that minimises our score, which
restores ordinary alpha-beta pruning in the multi-player game. Top-k move
ordering at each node keeps the high branching factor manageable; an
immediate-win shortcut and a greedy fallback guarantee a legal move.
`SEARCH_DEPTH` and `TOP_K_MOVES` are the main strength/speed knobs.

## Possible improvements
Stronger heuristic terms (central-column hop potential, filling deep home cells
first), a deeper search with better move ordering, or MCTS as an alternative.
