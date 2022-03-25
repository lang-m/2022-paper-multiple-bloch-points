"""Helper method for fast strip initialisation."""
import itertools


def create_pattern(n, method, type1="i", **kwargs):
    """Create patterns of Bloch point arrangements.

    Patterns are strings consisting of `i`, `o`, and `x`.
    Each character represents one Bloch point position, head-to-head as `i`,
    tail-to-tail as `o`. A position without Bloch point is denoted as `x`.

    Currently, this method does not return patterns that contain `x`.

    Parameters
    ----------
    n: int, List[int]
        Single number of Bloch points or list of numbers.

    method : str
        Method to use for pattern creation. Can be one of
        - `same-type`: same-type Bloch points
        - `alternating`: alternating opposite-type Bloch points
        - `all`: all possible configurations
        - `energy-levels`: all configurations resulting in different energies
        - `selection`: custom number of type changes.

    type1 : str, optional
        Bloch-point type at position 1. Defaults to `i` (head-to-head).
        Options:
        - head-to-head: `i`, `hh`
        - tail-to-tail: `o`, `tt`

    **kwargs : Any, optional
       Additional arguments required for certain metods:
       - `energy-levels`: `bcx: bool = False` to take boundary conditions
          into account.
       - `selection`: `typechanges: List[List[int]]` list of number of type
          changes

    Returns
    -------
    List[str]
        List of strings of Bloch point patterns.
    """
    if type1 not in ["i", "o", "hh", "tt"]:
        raise ValueError('Bloch point of type "{type1}" unknown.')
    if type1 == "hh":
        type1 = "i"
    elif type1 == "tt":
        type1 = "o"

    if method == "same-type":
        return _pattern_same_type(n, type1)
    elif method == "alternating":
        return _pattern_alternating_opposite_type(n, type1)
    elif method == "all":
        return _pattern_all_configs(n)
    elif method == "energy-levels":
        return _pattern_energetically_different(n, type1, **kwargs)
    elif method == "selection":
        return _pattern_n_typechanges(n, **kwargs, type1=type1)
    else:
        raise ValueError(f'Method "{method}" is unknown.')


def _pattern_same_type(n, type1="i"):
    res = []
    if isinstance(n, int):
        res.append(type1 * n)
    else:
        for i in n:
            res.append(type1 * i)
    return res


def _pattern_alternating_opposite_type(n, type1="i"):
    if type1 in ["i"]:
        start = 0
    else:
        start = 1
    res = []
    if isinstance(n, int):
        res.append(("io" * n)[start :start + n])
    else:
        for i in n:
            res.append(("io" * i)[start :start + i])
    return res


def _pattern_all_configs(n):
    return map(lambda x: "".join(x), itertools.product("io", repeat=n))


def _pattern_energetically_different(n, type1="i", bcx=False):
    if bcx:
        raise NotImplementedError
    if isinstance(n, int):
        n = [n]
    return _pattern_n_typechanges(n=n, typechanges=[range(i) for i in n], type1=type1)


def _pattern_n_typechanges(n, typechanges, type1="i"):
    res = []
    if isinstance(n, int):
        for tc in typechanges[0]:
            res.append(_pattern_from_typechanges(n, tc, type1))
    else:
        if len(n) != len(typechanges):
            msg = (
                "Length of the outer list of typechanges does not match"
                "Bloch point numbers"
            )
            raise ValueError(msg)
        for i, tcs in zip(n, typechanges):
            for tc in tcs:
                res.append(_pattern_from_typechanges(i, tc, type1))
    return res


def _pattern_from_typechanges(n, typechanges, type1):
    if typechanges >= n:
        msg = "Number of typechanges must be smaller than Bloch point number"
        raise ValueError(msg)
    type2 = "o" if type1 == "i" else "i"
    same = type1 * n
    alternating = (type2 + type1) * n
    return same[: n - typechanges] + alternating[:typechanges]


def pattern_ascii(characters):
    """Pattern representing the binary representation of the given characters.

    Head-to-head represents a 1 ('i'), tail-to-tail a 0 ('o').

    Parameters
    ----------
    characters : str
        String  to represent with Bloch points.

    Returns
    -------
    str
        Bloch point pattern encoded with `i` and `o`.
    """
    res = []
    for c in characters:
        binary = bin(ord(c))
        binary = binary[2:].zfill(8)  # string starts with '0b'
        for b in binary:
            if b == "1":
                res.append("i")
            else:
                res.append("o")
    return "".join(res)


def pattern_to_config(pattern, UP=(0, 0, 1), DOWN=(0, 0, -1), ELSE=None):
    """Translate a pattern string into subregions."""
    res = dict()
    res["subdivisions"] = len(pattern)
    res["positions"] = [i + 1 for i, p in enumerate(pattern) if p in "io"]
    res["subdivide_free"] = "i" in pattern and "o" in pattern
    patterndict = {"default": UP if 'i' in pattern else DOWN}
    for i, p in enumerate(pattern):
        if p == "i":
            patterndict[f"fixed{i+1}top"] = DOWN
        elif p == "o":
            patterndict[f"fixed{i+1}top"] = UP
            patterndict[f"fixed{i+1}bottom"] = DOWN
            patterndict[f"free{i+1}bottom"] = DOWN
            patterndict[f"free{i+1}top"] = DOWN
        elif ELSE:
            patterndict[f"free{i+1}bottom"] = ELSE
            patterndict[f"free{i+1}top"] = ELSE
    res["m_values"] = [pattern, patterndict]
    return res
