#!/usr/bin/env python
"""The stabilized P1-P1 Stokes immersed element (Unfitted.h StokesIFEP1), reached through proteus.cutclip.stokes_ife_p1.

On a triangle cut by a straight interface Gamma = {phi = 0}, u, v and p are linear on each side (A = {phi < 0},
viscosity mu_a; B = {phi > 0}, mu_b) and glued by the interface conditions: the 9 nodal values (read from the side
that owns the vertex), [u] = J at the two ends of Gamma, [sigma n] = j_t at its midpoint, [div u] = 0 and
[grad p] = 0, with sigma = -p I + mu (grad u + grad u^T), n from A into B and [q] = q_B - q_A. The element has 13
basis columns: 9 shape functions (u, v, p at the three vertices) and 4 correction functions (J_u, J_v, jt_x, jt_y).

The checks run on random physical triangles and cuts, element sizes 1e-3 to 1, viscosity ratios 1e-4 to 1e4:
  1. delta property; 2. every column satisfies the interface conditions it was built from; 3. equal viscosities
  give the standard P1 hats; 4. the pressure shape functions are P1 hats with zero velocity; 5. partition of unity;
  6. piecewise-linear two-fluid Stokes solutions are rebuilt exactly from their nodal values and interface data;
  7. the "balloon" (zero velocity, pressure step) is rebuilt, and needs the interface data; 8. special cuts.
And against the construction it comes from: the notebook IFEM_2D_Stokes.ipynb (cut_cell_system, k = 1), written
again here symbolically with sympy on the reference triangle, must give the same 13 columns.

Errors are relative to the size of each column: at high contrast on a small cell a velocity shape function carries a
pressure step of size ~[mu]/h (up to 1e7 here), so absolute round-off grows with it.
"""
import numpy as np
import pytest

from proteus import cutclip

NDATA, A, B = 13, 0, 1
JU, JV, JTX, JTY = 9, 10, 11, 12
RATIOS = [1e-4, 1e-3, 0.1, 1.0, 10.0, 1e3, 1e4]


def nodal(f, j):
    """column of the shape function of field f (0, 1, 2 = u, v, p) at vertex j"""
    return 3 * f + j


def build(x, phi, mua, mub, pts):
    return cutclip.stokes_ife_p1(np.asarray(x, float), np.asarray(phi, float), mua, mub,
                                 np.asarray(pts, float).reshape(-1, 2))


def random_triangle(rng, h):
    """a triangle of size ~h somewhere in [-10, 10]^2, not too flat"""
    while True:
        x = rng.uniform(-10, 10, 2) + h * rng.uniform(-1, 1, (3, 2))
        e = np.roll(x, -1, axis=0) - x
        area = 0.5 * abs(e[0, 0] * e[2, 1] - e[0, 1] * e[2, 0])
        if area > 0.1 * max(np.hypot(*e.T)) ** 2:
            return x


def random_points(rng, x, m):
    a, c = rng.uniform(size=(2, m))
    flip = a + c > 1
    a[flip], c[flip] = 1 - a[flip], 1 - c[flip]
    return x[0] + np.outer(a, x[1] - x[0]) + np.outer(c, x[2] - x[0])


def random_cut(rng, x):
    """a linear level set that changes sign on the triangle"""
    while True:
        g, p = rng.uniform(-1, 1, 2), random_points(rng, x, 1)[0]
        phi = (x - p) @ g
        if phi.min() < 0 < phi.max():
            return phi


def traction(grad, value_p, mu, n):
    """sigma n of one side: grad[f] = gradient of field f (u, v), value_p = pressure"""
    G = np.array([grad[0], grad[1]])                     # G[i, j] = du_i / dx_j
    return -value_p * n + mu * (G + G.T) @ n


class Piecewise:
    """a two-fluid field that is linear on each side: c[side, field] = (c0, cx, cy)"""

    def __init__(self, c):
        self.c = np.asarray(c, float)

    def value(self, side, field, pts):
        pts = np.atleast_2d(pts)
        return self.c[side, field, 0] + pts @ self.c[side, field, 1:]

    def data(self, x, phi, mua, mub, cell):
        """nodal values from the owning side, [u] at Gamma, [sigma n] at the midpoint (the element's data vector)"""
        d = np.zeros(NDATA)
        for f in range(3):
            for j in range(3):
                d[nodal(f, j)] = self.value(cell["owner"][j], f, x[j])[0]
        E, M, n = cell["gamma"][0], cell["mid"], cell["normal"]
        d[JU], d[JV] = (self.value(B, f, E)[0] - self.value(A, f, E)[0] for f in (0, 1))
        t = [traction(self.c[s, :2, 1:], self.value(s, 2, M)[0], mu, n) for s, mu in ((A, mua), (B, mub))]
        d[JTX], d[JTY] = t[1] - t[0]
        return d


def linear_family(rng, cell):
    """an exact piecewise-linear two-fluid Stokes solution (the notebook's linear_family): a kinked shear along
    Gamma, a linear divergence-free flow, a velocity jump J and a pressure step Pi; grad p is the same on both sides"""
    n = cell["normal"]
    t, E = np.array([-n[1], n[0]]), cell["gamma"][0]
    s0, G, p0, Pi = rng.normal(size=4)
    cs, J, a0 = rng.normal(size=2), rng.normal(size=2), rng.normal(size=2)
    m0, m1, m2 = rng.normal(size=3)
    M = np.array([[m0, m1], [m2, -m0]])
    c = np.zeros((2, 3, 3))
    for s in (A, B):
        # (s0 + c_s xi) t + a0 + M x (+ J on B), with xi = n.(x - E) the signed distance from Gamma
        for i in (0, 1):
            c[s, i] = [(s0 - cs[s] * n @ E) * t[i] + a0[i] + (J[i] if s == B else 0.0),
                       cs[s] * n[0] * t[i] + M[i, 0], cs[s] * n[1] * t[i] + M[i, 1]]
        c[s, 2] = [p0 + (Pi if s == B else 0.0), G * t[0], G * t[1]]
    return Piecewise(c)


def cells(seed=20261128, trials=60):
    """(x, phi, mua, mub) for random cut cells: sizes 1e-3 to 1, all ratios"""
    rng = np.random.default_rng(seed)
    for trial in range(trials):
        h = 10.0 ** (-3 + trial % 4)                       # 1e-3, 1e-2, 0.1, 1
        for q in RATIOS:
            x = random_triangle(rng, h)
            mua = 10.0 ** (trial % 3 - 1)
            yield rng, x, random_cut(rng, x), mua, q * mua


def column_size(cell):
    """largest value of each column at the vertices, either side, any field (at least 1): cell values at x"""
    return np.maximum(1.0, np.abs(cell["values"][..., :3]).max(axis=(1, 2, 3)))


def delta_property_error():
    err = 0.0
    for rng, x, phi, mua, mub in cells():
        cell = build(x, phi, mua, mub, x)
        size = column_size(cell)
        for j in range(NDATA):
            for f in range(3):
                for v in range(3):
                    target = 1.0 if j == nodal(f, v) else 0.0
                    err = max(err, abs(cell["values"][j, cell["owner"][v], f, v] - target) / size[j])
    return err


def test_delta_property():
    err = delta_property_error()
    assert err < 1e-11, err


def interface_conditions_error():
    err = 0.0
    for rng, x, phi, mua, mub in cells():
        geo = build(x, phi, mua, mub, x)
        cell = build(x, phi, mua, mub, np.vstack([geo["gamma"], geo["mid"]]))
        n, g, h = cell["normal"], cell["gradients"], np.ptp(x, axis=0).max()
        for j in range(NDATA):
            size = max(1.0, np.abs(g[j]).max() * h)
            for e in (0, 1):
                for f, col in ((0, JU), (1, JV)):
                    jump = cell["values"][j, B, f, e] - cell["values"][j, A, f, e]
                    err = max(err, abs(jump - (j == col)) / size)
            t = [traction(g[j, s, :2], cell["values"][j, s, 2, 2], mu, n) for s, mu in ((A, mua), (B, mub))]
            tsize = max(1.0, max(mua, mub) * size / h)
            err = max(err, abs(t[1][0] - t[0][0] - (j == JTX)) / tsize, abs(t[1][1] - t[0][1] - (j == JTY)) / tsize)
            div = g[j, :, 0, 0] + g[j, :, 1, 1]
            err = max(err, abs(div[B] - div[A]) * h / size, np.abs(g[j, B, 2] - g[j, A, 2]).max() * h / size)
    return err


def test_interface_conditions():
    err = interface_conditions_error()
    assert err < 1e-9, err


def standard(x, mua, mub, pts):
    """the standard P1-P1 element: an element Gamma does not cut"""
    return build(x, np.ones(3), mua, mub, pts)


def equal_viscosities_give_the_standard_element_error():
    err = 0.0
    for rng, x, phi, mua, mub in cells():
        pts = random_points(rng, x, 5)
        eq, std = build(x, phi, mua, mua, pts), standard(x, mua, mua, pts)
        for q in range(5):
            err = max(err, np.abs(eq["values"][:9, eq["side"][q], :, q] - std["values"][:9, A, :, q]).max())
    return err


def test_equal_viscosities_give_the_standard_element():
    err = equal_viscosities_give_the_standard_element_error()
    assert err < 1e-9, err


def pressure_shape_functions_are_hats_error():
    err = 0.0
    for rng, x, phi, mua, mub in cells():
        pts = random_points(rng, x, 3)
        cell, std = build(x, phi, mua, mub, pts), standard(x, mua, mub, pts)
        for v in range(3):
            j = nodal(2, v)
            err = max(err, np.abs(cell["values"][j] - std["values"][j, A][None]).max())
    return err


def test_pressure_shape_functions_are_hats():
    err = pressure_shape_functions_are_hats_error()
    assert err < 1e-9, err


def partition_of_unity_error():
    err = 0.0
    for rng, x, phi, mua, mub in cells():
        pts = random_points(rng, x, 3)
        cell = build(x, phi, mua, mub, np.vstack([x, pts]))
        size = column_size(cell)
        for F in range(3):
            cols = [nodal(F, v) for v in range(3)]
            for q in range(3, 6):
                total = cell["values"][cols, cell["side"][q], :, q].sum(axis=0)
                err = max(err, np.abs(total - np.eye(3)[F]).max() / size[cols].max())
    return err


def test_partition_of_unity():
    err = partition_of_unity_error()
    assert err < 1e-11, err


def rebuild(cell, d, q):
    """sum_j d_j column_j, all fields, at point q on its own side"""
    return d @ cell["values"][:, cell["side"][q], :, q]


def linear_solutions_are_rebuilt_exactly_error():
    err = 0.0
    for rng, x, phi, mua, mub in cells():
        pts = random_points(rng, x, 4)
        geo = build(x, phi, mua, mub, x)
        cell = build(x, phi, mua, mub, pts)
        for _ in range(3):
            F = linear_family(rng, geo)
            d = F.data(x, phi, mua, mub, geo)
            big = max(1.0, np.abs(d).max())
            for q in range(4):
                s = cell["side"][q]
                exact = np.array([F.value(s, f, pts[q])[0] for f in range(3)])
                err = max(err, np.abs(rebuild(cell, d, q) - exact).max() / big)
    return err


def test_linear_solutions_are_rebuilt_exactly():
    err = linear_solutions_are_rebuilt_exactly_error()
    assert err < 1e-8, err


def balloon():
    """zero velocity, p = 0 on A and 1 on B (a static drop held by a normal interface force)"""
    F = Piecewise(np.zeros((2, 3, 3)))
    F.c[B, 2, 0] = 1.0
    return F


def balloon_error():
    """the balloon is rebuilt with the interface data (its traction jump is -n)"""
    err = 0.0
    for rng, x, phi, mua, mub in cells(trials=20):
        geo = build(x, phi, mua, mub, x)
        pts = random_points(rng, x, 3)
        cell = build(x, phi, mua, mub, pts)
        d = balloon().data(x, phi, mua, mub, geo)
        for q in range(3):
            s = cell["side"][q]
            err = max(err, np.abs(rebuild(cell, d, q) - [0, 0, s == B]).max())
    return err


def balloon_step_from_nodal_data():
    """without the interface data a pressure step cannot appear: with mu_a = mu_b the nodal data alone give none"""
    x, phi = np.array([[0., 0.], [0., 1.], [1., 0.]]), np.array([-0.5, 0.4, 0.3])
    geo = build(x, phi, 1.0, 1.0, x)
    cell = build(x, phi, 1.0, 1.0, geo["mid"][None])
    d = balloon().data(x, phi, 1.0, 1.0, geo)
    d[9:] = 0.0
    return abs(d @ (cell["values"][:, B, 2, 0] - cell["values"][:, A, 2, 0]))


def test_balloon():
    err = balloon_error()
    assert err < 1e-9, err
    step = balloon_step_from_nodal_data()
    assert step < 1e-12, step


# (description, error function, tolerance): the checks above, for the tests notebook to print
CHECKS = [
    ("1. delta property (relative to each column's size)", delta_property_error, 1e-11),
    ("2. every column satisfies its interface conditions", interface_conditions_error, 1e-9),
    ("3. equal viscosities give the standard P1 hats", equal_viscosities_give_the_standard_element_error, 1e-9),
    ("4. pressure shape functions are P1 hats, zero velocity", pressure_shape_functions_are_hats_error, 1e-9),
    ("5. partition of unity (relative)", partition_of_unity_error, 1e-11),
    ("6. linear two-fluid solutions rebuilt exactly", linear_solutions_are_rebuilt_exactly_error, 1e-8),
    ("7. balloon rebuilt with the interface data", balloon_error, 1e-9),
    ("7b. no pressure step from nodal data alone (mu_a = mu_b)", balloon_step_from_nodal_data, 1e-12),
]


@pytest.mark.parametrize("phi", [[0.0, 1.0, -1.0], [-1e-6, 1.0, 1.0], [1.0, -1e-6, -1e-6]],
                         ids=["through_a_vertex", "near_a_vertex", "near_an_edge"])
def test_special_cuts(phi):
    x = np.array([[0., 0.], [0., 1.], [1., 0.]])
    cell = build(x, phi, 1.0, 100.0, x)
    assert cell["cut"]
    for j in range(NDATA):
        for f in range(3):
            for v in range(3):
                assert abs(cell["values"][j, cell["owner"][v], f, v] - (j == nodal(f, v))) < 1e-9


def test_touching_is_not_cut():
    x = np.array([[0., 0.], [0., 1.], [1., 0.]])
    assert not build(x, [0.0, 1.0, 2.0], 1.0, 100.0, x)["cut"]


def notebook_columns(X0, Y0, mua, mub, points):
    """The notebook's construction (IFEM_2D_Stokes.ipynb, cut_cell_system with k = 1): reference triangle (0,0),
    (0,1), (1,0), Gamma from (X0, 0) to (0, Y0), side a holds vertex 0. Returns values[column, side, field, point]."""
    sp = pytest.importorskip("sympy")
    x, y = sp.symbols("x y")
    vertices = [(0, 0), (0, 1), (1, 0)]
    Q, unknowns = {}, []
    for s in "ab":
        cu, cv, cp = sp.symbols(f"u{s}0:3"), sp.symbols(f"v{s}0:3"), sp.symbols(f"p{s}0:3")
        Q[s] = [c[0] + c[1] * x + c[2] * y for c in (cu, cv, cp)]
        unknowns += [*cu, *cv, *cp]
    names = ["U0", "U1", "U2", "V0", "V1", "V2", "P0", "P1", "P2", "Ju", "Jv", "jt0x", "jt0y"]
    d = dict(zip(names, sp.symbols(names)))
    n = np.array([Y0, X0]) / np.hypot(X0, Y0)
    owner = lambda P: "a" if P[0] / X0 + P[1] / Y0 < 1 else "b"
    at = lambda f, P: f.subs({x: P[0], y: P[1]})
    mu = {"a": mua, "b": mub}

    def sigma_n(s, P):
        u, v, p = Q[s]
        G = sp.Matrix([[u.diff(x), u.diff(y)], [v.diff(x), v.diff(y)]])
        return [at(t, P) for t in (-p * sp.eye(2) + mu[s] * (G + G.T)) * sp.Matrix(n)]

    rows = []
    for i, kind in enumerate("UVP"):
        rows += [at(Q[owner(P)][i], P) - d[f"{kind}{j}"] for j, P in enumerate(vertices)]
    rows += [at(Q["b"][i] - Q["a"][i], P) - d[J] for i, J in ((0, "Ju"), (1, "Jv")) for P in [(X0, 0), (0, Y0)]]
    ta, tb = sigma_n("a", (X0 / 2, Y0 / 2)), sigma_n("b", (X0 / 2, Y0 / 2))
    rows += [tb[c] - ta[c] - d["jt0" + "xy"[c]] for c in (0, 1)]
    rows += [(Q["b"][0].diff(x) + Q["b"][1].diff(y)) - (Q["a"][0].diff(x) + Q["a"][1].diff(y))]
    rows += [Q["b"][2].diff(w) - Q["a"][2].diff(w) for w in (x, y)]
    Amat, b = sp.linear_eq_to_matrix(rows, unknowns)
    C = np.linalg.solve(np.array(Amat.tolist(), float), np.array(b.jacobian(list(d.values())).tolist(), float))
    values = np.zeros((NDATA, 2, 3, len(points)))
    for j in range(NDATA):
        sub = dict(zip(unknowns, C[:, j]))
        for si, s in enumerate("ab"):
            for f in range(3):
                fn = sp.lambdify((x, y), Q[s][f].subs(sub), "numpy")
                values[j, si, f] = [float(fn(*P)) for P in points]
    return values


@pytest.mark.parametrize("X0, Y0, mua, mub", [(0.62, 0.37, 1.0, 10.0), (0.4, 0.4, 1.0, 1.0),
                                              (0.05, 0.9, 1000.0, 1.0), (0.95, 0.1, 1.0, 1000.0)])
def test_matches_the_notebook(X0, Y0, mua, mub):
    x = np.array([[0., 0.], [0., 1.], [1., 0.]])
    phi = x[:, 0] / X0 + x[:, 1] / Y0 - 1.0
    points = [(0.1, 0.1), (0.2, 0.5), (0.6, 0.3)]
    ours = build(x, phi, mua, mub, points)["values"]
    ref = notebook_columns(X0, Y0, mua, mub, points)
    scale = np.maximum(1.0, np.abs(ref).max(axis=(1, 2, 3)))
    err = (np.abs(ours - ref).max(axis=(1, 2, 3)) / scale).max()
    assert err < 1e-11, f"largest relative difference from the notebook construction: {err:.2e}"
