"""Reference solvers used for par (scoring) and tests.

``solve_standard`` wraps Kociemba's solver for 3x3x3 states (optional
dependency; falls back to ``None``). ``solve_2x2`` is a pure-Python reference
solver for 2x2x2 states that is always available: A* over the corner-cubie
state (permutation + orientation) guided by two small pattern databases
(corner permutation, 8! = 40320 states; corner orientation, 3^8 = 6561
states). Both return move lists in the standard Singmaster semantics used by
``rubikbench.cube`` (unprimed = clockwise), where ``U2`` counts as one move
(half-turn metric).

Corner orientation follows the standard cubie convention: a piece's
orientation is the slot (0, 1, 2) that its U/D face occupies among the three
face classes (U/D, R/L, F/B) of the position it sits in. A solved cube has
orientation 0 everywhere. Face turns act on the three slots with a
position-dependent permutation (a turn can transpose the U/D and F/B
classes), so the per-move orientation tables are measured empirically from
the geometric facelet move tables instead of being derived by hand.
"""

from __future__ import annotations

import heapq
import math

GODS_NUMBER: int = 20   # 3x3x3, half-turn metric
GODS_NUMBER_2X2: int = 14  # 2x2x2, half-turn metric

try:  # pragma: no cover - exercised through tests
    import kociemba as _kociemba

    HAS_KOCIEMBA = True
except Exception:  # noqa: BLE001 - pragma: no cover; optional dependency
    _kociemba = None
    HAS_KOCIEMBA = False


def solve_standard(facelet_string: str) -> list[str] | None:
    """Return a solution for a 3x3x3 facelet state in standard Singmaster
    semantics, or ``None`` if kociemba is unavailable or the state is invalid."""
    if not HAS_KOCIEMBA:
        return None
    try:
        raw = _kociemba.solve(facelet_string)
    except ValueError:
        return None
    if not raw.strip():
        return []
    return raw.split()


def par_moves(facelet_string: str, is_solved: bool = False, default: int = GODS_NUMBER) -> int:
    """Best-effort reference ("par") move count for a 3x3x3 state.

    Uses kociemba's near-optimal solver when available; otherwise falls back to
    God's number (20 moves).
    """
    if is_solved:
        return 0
    solution = solve_standard(facelet_string)
    if solution is None:
        return default
    return len(solution)


# ---------------------------------------------------------------------------
# 2x2x2 reference solver (pure Python, always available)

#: Canonical corner-position order: (R+, L-)(F+, B-)(U+, D-) by sign triple of
#: the sticker positions on the unit cube.
_CORNER_2X2_ORDER = (
    (True, True, True), (False, True, True), (False, False, True), (True, False, True),
    (True, True, False), (False, True, False), (False, False, False), (True, False, False),
)

_FACT = tuple(math.factorial(i) for i in range(8))
_TWIST_RANGE = 3 ** 8
_MOVE_ORDER = tuple(f + s for f in "URFDLB" for s in ("", "2", "'"))
_SOLVED_2X2 = "".join(f * 4 for f in "URFDLB")
_CLASS_ORDER = {"R": 1, "L": 1, "F": 2, "B": 2, "U": 0, "D": 0}

_corner_pos_facelets: tuple[tuple[int, ...], ...] | None = None  # position -> facelets
_corner_class_order: tuple[tuple[int, ...], ...] | None = None   # position -> facelets in U/D, R/L, F/B order
_home_of: dict[frozenset[str], int] | None = None    # piece color triple -> home position
_perm_tables: dict[str, tuple[int, ...]] | None = None  # move -> position permutation
_move_ops: dict[str, tuple[tuple[int, int, int, int], ...]] | None = None  # move -> (q, map0, map1, map2) per p
_perm_db: tuple[int, ...] | None = None
_twist_db: tuple[int, ...] | None = None
_unperm_cache: list[tuple[int, ...]] = []
_perm_index_cache: dict[tuple[int, ...], int] = {}
_twist_from: tuple[tuple[int, ...], ...] = ()
_twist_to: dict[tuple[int, ...], int] = {}


def _perm_index(perm: tuple[int, ...]) -> int:
    """Rank a permutation of 8 elements (Lehmer code), cached."""
    try:
        return _perm_index_cache[perm]
    except KeyError:
        rank = 0
        for i in range(8):
            smaller = sum(1 for j in range(i + 1, 8) if perm[j] < perm[i])
            rank += smaller * _FACT[7 - i]
        _perm_index_cache[perm] = rank
        return rank


def _unperm(index: int) -> tuple[int, ...]:
    """Unrank a permutation of 8 elements, cached."""
    while len(_unperm_cache) <= index:
        pos = len(_unperm_cache)
        i = pos
        elements = list(range(8))
        perm: list[int] = []
        while elements:
            k = i // _FACT[len(elements) - 1]
            i %= _FACT[len(elements) - 1]
            perm.append(elements.pop(k))
        _unperm_cache.append(tuple(perm))
    return _unperm_cache[index]


def _twist_int(twists: tuple[int, ...]) -> int:
    """Encode 8 trits into an int (position 0 is the most significant trit)."""
    return _twist_to[twists]


def _un_twist_int(value: int) -> tuple[int, ...]:
    """Inverse of ``_twist_int``."""
    return _twist_from[value]


def _orientation_of(facelet_string: str, pos_id: int) -> int:
    """Slot (0, 1, 2) of the U/D-colored sticker in the position's canonical
    (U/D, R/L, F/B) face-class order."""
    assert _corner_class_order is not None
    colors = [facelet_string[i] for i in _corner_class_order[pos_id]]
    return next(k for k in range(3) if colors[k] in "UD")


def _corner_state(facelet_string: str) -> tuple[int, int] | None:
    """``(perm_rank, twist_rank)`` for a 2x2x2 facelet state, or None when the
    color triples do not describe the 8 legal corner pieces."""
    if _corner_pos_facelets is None:
        _init_2x2()
    assert _corner_pos_facelets is not None and _home_of is not None
    perm: list[int] = []
    twists: list[int] = []
    for pos_id in range(8):
        colors = [facelet_string[i] for i in _corner_pos_facelets[pos_id]]
        home = _home_of.get(frozenset(colors))
        if home is None:
            return None
        perm.append(home)
        twists.append(_orientation_of(facelet_string, pos_id))
    return _perm_index(tuple(perm)), _twist_int(tuple(twists))


def _init_2x2() -> None:
    """Build corner metadata, move tables, orientation maps, and the two DBs."""
    global _corner_pos_facelets, _corner_class_order, _home_of
    global _perm_tables, _move_ops, _perm_db, _twist_db, _twist_from
    if not _twist_to:
        twist_list = [None] * _TWIST_RANGE
        for i in range(_TWIST_RANGE):
            twists = [0] * 8
            v = i
            for j in range(7, -1, -1):
                twists[j] = v % 3
                v //= 3
            t = tuple(twists)
            twist_list[i] = t
            _twist_to[t] = i
        _twist_from = tuple(twist_list)
    from .cube import _sticker_positions, moves_for

    pos = _sticker_positions(2)
    face_of: dict[int, str] = {}
    for idx, p in pos.items():
        axis = max(range(3), key=lambda i: abs(p[i]))
        faces_on_axis = ("RL", "FB", "UD")[axis]
        face_of[idx] = faces_on_axis[0] if p[axis] > 0 else faces_on_axis[1]

    corner_of: dict[tuple[bool, bool, bool], list[int]] = {}
    for idx, p in pos.items():
        corner_of.setdefault(tuple(c > 0 for c in p), []).append(idx)
    assert all(len(v) == 3 for v in corner_of.values()) and len(corner_of) == 8

    facelets_of: list[tuple[int, ...]] = []
    class_of: list[tuple[int, ...]] = []
    for key in _CORNER_2X2_ORDER:
        indices = tuple(sorted(corner_of[key]))
        facelets_of.append(indices)
        class_of.append(tuple(
            i for i in sorted(indices, key=lambda i: _CLASS_ORDER[face_of[i]])
        ))
    _corner_pos_facelets = tuple(facelets_of)
    _corner_class_order = tuple(class_of)

    home_of: dict[frozenset[str], int] = {}
    for pos_id, indices in enumerate(facelets_of):
        home_of[frozenset(face_of[i] for i in indices)] = pos_id
    _home_of = home_of

    facelet_of: dict[int, int] = {}
    for pos_id, indices in enumerate(facelets_of):
        for idx in indices:
            facelet_of[idx] = pos_id

    tables: dict[str, tuple[int, ...]] = {}
    for move, perm in moves_for(2).items():
        tables[move] = tuple(facelet_of[perm[facelets_of[p][0]]] for p in range(8))
    _perm_tables = tables

    # --- orientation maps, measured empirically --------------------------
    # For each move and destination position p, map each possible orientation
    # (0, 1, 2) of the piece that arrives at p (from q = tables[move][p]) to
    # its orientation after the move. Probe states put the home piece of q at
    # q with the requested orientation, then the geometric facelet move is
    # applied and the resulting orientation at p is read off. Faces turns may
    # transpose the U/D and F/B slot classes, so these maps are not simple
    # additive offsets; the empirical measurement keeps them exact.
    orient_maps: dict[str, tuple[tuple[int, int, int], ...]] = {}
    for move in _MOVE_ORDER:
        perm = moves_for(2)[move]
        maps: list[tuple[int, int, int]] = []
        for p in range(8):
            q = tables[move][p]
            home_class_faces = [face_of[i] for i in class_of[q]]
            images: list[int] = []
            for o in range(3):
                placed = list(_SOLVED_2X2)
                for slot_pos, idx in enumerate(class_of[q]):
                    placed[idx] = home_class_faces[(slot_pos - o) % 3]
                after = [placed[perm[i]] for i in range(24)]
                images.append(_orientation_of("".join(after), p))
            maps.append(tuple(images))
        orient_maps[move] = tuple(maps)

    move_ops: dict[str, tuple[tuple[int, int, int, int], ...]] = {}
    for move in _MOVE_ORDER:
        move_ops[move] = tuple(
            (p, tables[move][p], *orient_maps[move][p]) for p in range(8)
        )
    _move_ops = move_ops

    # --- perm DB: BFS over 40320 permutations, distance to the identity ---
    dist = [-1] * 40320
    dist[0] = 0
    frontier = [0]
    while frontier:
        nxt: list[int] = []
        for rank in frontier:
            state = _unperm(rank)
            d = dist[rank]
            for table in tables.values():
                new_state = tuple(state[q] for q in table)
                new_rank = _perm_index(new_state)
                if dist[new_rank] == -1:
                    dist[new_rank] = d + 1
                    nxt.append(new_rank)
        frontier = nxt
    _perm_db = tuple(dist)

    # --- twist DB: BFS over 3^8 orientation states (pieces pinned), where a
    # move applies the same orientation maps the full cube would experience.
    twist_moves: list[tuple[tuple[int, int, int, int], ...]] = [
        move_ops[m] for m in _MOVE_ORDER
    ]
    tdist = [-1] * _TWIST_RANGE
    tdist[0] = 0
    frontier = [0]
    while frontier:
        nxt: list[int] = []
        for value in frontier:
            twists = _twist_from[value]
            d = tdist[value]
            for ops in twist_moves:
                new_twists = list(twists)
                for p, q, m0, m1, m2 in ops:
                    new_twists[p] = (m0, m1, m2)[twists[q]]
                new_value = _twist_to[tuple(new_twists)]
                if tdist[new_value] == -1:
                    tdist[new_value] = d + 1
                    nxt.append(new_value)
        frontier = nxt
    _twist_db = tuple(tdist)


def _apply_move(perm_rank: int, twist_value: int, move: str) -> tuple[int, int]:
    """Apply a move to an encoded corner-cubie state."""
    assert _move_ops is not None
    ops = _move_ops[move]
    old_perm = _unperm(perm_rank)
    old_twists = _twist_from[twist_value]
    new_perm = [0] * 8
    new_twists = [0] * 8
    for p, q, m0, m1, m2 in ops:
        new_perm[p] = old_perm[q]
        new_twists[p] = (m0, m1, m2)[old_twists[q]]
    return _perm_index(tuple(new_perm)), _twist_to[tuple(new_twists)]


def solve_2x2(facelet_string: str, max_depth: int = 20) -> list[str] | None:
    """Solve a 2x2x2 facelet state; optimal in the half-turn metric.

    Returns ``None`` when the input is not a legal 2x2x2 state or when no
    solution is found within ``max_depth`` moves (should never happen for a
    legal state: God's number is 14).
    """
    if len(facelet_string) != 24 or not set(facelet_string) <= set("URFDLB"):
        return None
    start = _corner_state(facelet_string)
    if start is None:
        return None
    if _perm_db is None:
        _init_2x2()
    assert _perm_db is not None and _twist_db is not None

    start_key = start[0] * _TWIST_RANGE + start[1]
    if start_key == 0:
        return []

    def heuristic(perm_rank: int, twist_value: int) -> int:
        return max(_perm_db[perm_rank], _twist_db[twist_value])

    g_score = {start_key: 0}
    came_from: dict[int, tuple[int, str]] = {}
    open_heap: list[tuple[int, int, int]] = [(heuristic(*start), 0, start_key)]
    while open_heap:
        _, g, key = heapq.heappop(open_heap)
        if g != g_score.get(key):
            continue
        if key == 0:
            moves: list[str] = []
            while key in came_from:
                prev, move = came_from[key]
                moves.append(move)
                key = prev
            moves.reverse()
            return moves
        if g >= max_depth:
            continue
        perm_rank, twist_value = divmod(key, _TWIST_RANGE)
        prev_face = came_from[key][1][0] if key in came_from else ""
        for move in _MOVE_ORDER:
            if move[0] == prev_face:
                continue
            n_perm, n_twist = _apply_move(perm_rank, twist_value, move)
            n_key = n_perm * _TWIST_RANGE + n_twist
            ng = g + 1
            if ng >= g_score.get(n_key, 10**9):
                continue
            g_score[n_key] = ng
            came_from[n_key] = (key, move)
            heapq.heappush(open_heap, (ng + heuristic(n_perm, n_twist), ng, n_key))
    return None


def par_moves_2x2(facelet_string: str, default: int = GODS_NUMBER_2X2) -> int:
    """Reference ("par") move count for a 2x2x2 state, or the default when the
    state cannot be solved."""
    solution = solve_2x2(facelet_string)
    if solution is None:
        return default
    return len(solution)
