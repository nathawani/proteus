// Python binding for proteus/cutclip.h (C44): exact planar clipping of tetrahedra and triangles and their moments.
// Used as the oracle in test/ADR3D and by the fit-versus-clip checks; not called from the ADR kernel. Also the test
// binding of the immersed bases in Unfitted.h: the scalar IFE (ifem_basis_2d/3d) and the Stokes P1-P1 element
// (stokes_ife_p1, tested in test/IFEM/test_stokes_ife_p1.py).
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <memory>
#include <vector>
#include "cutclip.h"
#include "equivalent_polynomials.h"
#include "Unfitted.h"

namespace py = pybind11;
using namespace proteus::cutclip;
using Arr = py::array_t<double, py::array::c_style | py::array::forcecast>;

static Tet toTet(const Arr &a)
{
  if (a.ndim() != 2 || a.shape(0) != 4 || a.shape(1) != 3)
    throw std::invalid_argument("tet must be (4,3)");
  Tet t;
  for (int i = 0; i < 4; ++i)
    for (int c = 0; c < 3; ++c)
      t[i][c] = a.at(i, c);
  return t;
}
static Tri toTri(const Arr &a)
{
  if (a.ndim() != 2 || a.shape(0) != 3 || a.shape(1) != 3)
    throw std::invalid_argument("tri must be (3,3)");
  Tri t;
  for (int i = 0; i < 3; ++i)
    for (int c = 0; c < 3; ++c)
      t[i][c] = a.at(i, c);
  return t;
}
static std::vector<Plane> toPlanes(const Arr &a)
{
  if (a.ndim() != 2 || a.shape(1) != 4)
    throw std::invalid_argument("planes must be (m,4): a.x + b > 0 is kept");
  std::vector<Plane> p(a.shape(0));
  for (py::ssize_t i = 0; i < a.shape(0); ++i)
    for (int c = 0; c < 4; ++c)
      p[i][c] = a.at(i, c);
  return p;
}
template <class P, std::size_t N>
static Arr piecesToArray(const std::vector<std::array<std::array<double, 3>, N>> &v)
{
  Arr out({(py::ssize_t)v.size(), (py::ssize_t)N, (py::ssize_t)3});
  for (std::size_t i = 0; i < v.size(); ++i)
    for (std::size_t k = 0; k < N; ++k)
      for (int c = 0; c < 3; ++c)
        out.mutable_at(i, k, c) = v[i][k][c];
  return out;
}

// The real equivalent_polynomials::Simplex with the two-sided IFEM basis (the object ADR.h holds per
// element), evaluated at up to 64 reference points of one simplex. Test hook for the 2D and 3D basis
// (C44 phase 2): returns the cut classification, the Heaviside/Dirac fits, VA/VB and their gradients and the
// unit flux-jump lift at each point, exactly as the kernel reads them through set_quad().
template <int nSpace>
static py::dict ifemBasis(const Arr &nodes, const Arr &phi, const Arr &pts, double ma, double mb)
{
  constexpr int nN = nSpace + 1, NQ = 64;
  if (nodes.ndim() != 2 || nodes.shape(0) != nN || nodes.shape(1) != 3)
    throw std::invalid_argument("nodes must be (nSpace+1, 3)");
  if (phi.ndim() != 1 || phi.shape(0) != nN)
    throw std::invalid_argument("phi must be (nSpace+1,)");
  if (pts.ndim() != 2 || pts.shape(1) != 3 || pts.shape(0) > NQ)
    throw std::invalid_argument("pts must be (m <= 64, 3) reference points");
  const int m = (int)pts.shape(0);
  using S = proteus::unfitted::IFEFunctions<nSpace, nN, 3, NQ, NQ>;
  std::unique_ptr<S> sx(new S());
  std::vector<double> xi(NQ * 3, 0.0);
  for (int q = 0; q < m; ++q)
    for (int c = 0; c < 3; ++c)
      xi[q * 3 + c] = pts.at(q, c);
  std::vector<double> ph(nN), nd(nN * 3);
  for (int i = 0; i < nN; ++i)
  {
    ph[i] = phi.at(i);
    for (int c = 0; c < 3; ++c)
      nd[i * 3 + c] = nodes.at(i, c);
  }
  const int icase = sx->calculate(ph.data(), nd.data(), xi.data(), ma, mb, 0.0, false, false);
  py::dict out;
  out["icase"] = icase;
  out["edge"] = sx->exact.edge;
  out["corner"] = sx->exact.corner;
  out["inside_out"] = sx->exact.inside_out;
  py::array_t<double> H({m}), ImH({m}), D({m}), va({m, nN}), vb({m, nN}), ga({m, nN, 3}), gb({m, nN, 3}), psi({m, 2, 4});
  py::array_t<double> normal({nSpace});
  if (icase == 0)
  {
    for (int c = 0; c < nSpace; ++c)
      normal.mutable_at(c) = sx->get_normal()[c];
    for (int q = 0; q < m; ++q)
    {
      sx->set_quad(q);
      H.mutable_at(q) = sx->H(0., 0.);
      ImH.mutable_at(q) = sx->ImH(0., 0.);
      D.mutable_at(q) = sx->D(0., 0.);
      for (int i = 0; i < nN; ++i)
      {
        va.mutable_at(q, i) = sx->VA(i);
        vb.mutable_at(q, i) = sx->VB(i);
        ga.mutable_at(q, i, 0) = sx->VA_x(i);
        ga.mutable_at(q, i, 1) = sx->VA_y(i);
        ga.mutable_at(q, i, 2) = sx->VA_z(i);
        gb.mutable_at(q, i, 0) = sx->VB_x(i);
        gb.mutable_at(q, i, 1) = sx->VB_y(i);
        gb.mutable_at(q, i, 2) = sx->VB_z(i);
      }
      const double pa[4] = {sx->PSIA(), sx->PSIA_x(), sx->PSIA_y(), sx->PSIA_z()};
      const double pb[4] = {sx->PSIB(), sx->PSIB_x(), sx->PSIB_y(), sx->PSIB_z()};
      for (int c = 0; c < 4; ++c)
      {
        psi.mutable_at(q, 0, c) = pa[c];
        psi.mutable_at(q, 1, c) = pb[c];
      }
    }
  }
  out["normal"] = normal;
  out["H"] = H;
  out["ImH"] = ImH;
  out["D"] = D;
  out["va"] = va;
  out["vb"] = vb;
  out["va_grad"] = ga;
  out["vb_grad"] = gb;
  out["psi"] = psi;
  return out;
}

// The stabilized P1-P1 Stokes immersed element (Unfitted.h StokesIFEP1) on a physical triangle. Every data column j
// (9 shape functions, then the correction functions J_u, J_v, jt_x, jt_y) on both sides, at the points pts:
// values[j, side, field, q] (field 0, 1, 2 = u, v, p; side 0 = A = {phi < 0}, 1 = B) and the constant gradients
// gradients[j, side, field, :]; plus the side of each point, the owner of each vertex and the cut itself.
static py::dict stokesIFEP1(const Arr &nodes, const Arr &phi, double mua, double mub, const Arr &pts)
{
  using IFE = proteus::unfitted::StokesIFEP1;
  if (nodes.ndim() != 2 || nodes.shape(0) != 3 || nodes.shape(1) != 2)
    throw std::invalid_argument("nodes must be (3, 2)");
  if (phi.ndim() != 1 || phi.shape(0) != 3)
    throw std::invalid_argument("phi must be (3,)");
  if (pts.ndim() != 2 || pts.shape(1) != 2)
    throw std::invalid_argument("pts must be (m, 2)");
  double x[3][2], ph[3];
  for (int i = 0; i < 3; ++i)
  {
    ph[i] = phi.at(i);
    x[i][0] = nodes.at(i, 0);
    x[i][1] = nodes.at(i, 1);
  }
  IFE b;
  b.build(x, ph, mua, mub);
  const int m = (int)pts.shape(0);
  py::array_t<double> values({IFE::nData, 2, 3, m}), gradients({IFE::nData, 2, 3, 2});
  py::array_t<int> side({m}), owner({3});
  for (int q = 0; q < m; ++q)
  {
    const double pt[2] = {pts.at(q, 0), pts.at(q, 1)};
    side.mutable_at(q) = b.sideOf(pt);
    for (int j = 0; j < IFE::nData; ++j)
      for (int s = 0; s < 2; ++s)
        for (int f = 0; f < 3; ++f)
          values.mutable_at(j, s, f, q) = b.value(j, s, f, pt);
  }
  for (int j = 0; j < IFE::nData; ++j)
    for (int s = 0; s < 2; ++s)
      for (int f = 0; f < 3; ++f)
      {
        double g[2];
        b.gradient(j, s, f, g);
        gradients.mutable_at(j, s, f, 0) = g[0];
        gradients.mutable_at(j, s, f, 1) = g[1];
      }
  for (int i = 0; i < 3; ++i)
    owner.mutable_at(i) = b.owner(i);
  py::dict out;
  out["cut"] = b.isCut();
  out["values"] = values;
  out["gradients"] = gradients;
  out["side"] = side;
  out["owner"] = owner;
  if (b.isCut())
  {
    py::array_t<double> ends({2, 2}), mid({2}), normal({2});
    for (int c = 0; c < 2; ++c)
    {
      ends.mutable_at(0, c) = b.gammaEnd(0)[c];
      ends.mutable_at(1, c) = b.gammaEnd(1)[c];
      mid.mutable_at(c) = b.gammaMid()[c];
      normal.mutable_at(c) = b.normal()[c];
    }
    out["gamma"] = ends;
    out["mid"] = mid;
    out["normal"] = normal;
  }
  return out;
}

PYBIND11_MODULE(cutclip, m)
{
  m.doc() = "Exact planar clipping of a tetrahedron/triangle and its polynomial moments (C44 oracle)";
  m.def("clip_tet", [](Arr tet, Arr planes) { return piecesToArray<Tet, 4>(clipTet(toTet(tet), toPlanes(planes))); });
  m.def("clip_tri", [](Arr tri, Arr planes) { return piecesToArray<Tri, 3>(clipTri(toTri(tri), toPlanes(planes))); });
  m.def("cut_surface", [](Arr tet, Arr plane, Arr others) {
    Arr pl = plane;
    if (pl.ndim() != 1 || pl.shape(0) != 4)
      throw std::invalid_argument("plane must be (4,)");
    Plane h{pl.at(0), pl.at(1), pl.at(2), pl.at(3)};
    return piecesToArray<Tri, 3>(cutSurface(toTet(tet), h, toPlanes(others)));
  });
  m.def("tet_moments", [](Arr tet, Arr planes, int degree) {
    return moments(clipTet(toTet(tet), toPlanes(planes)), degree);
  });
  m.def("tri_moments", [](Arr tri, Arr planes, int degree) {
    return moments(clipTri(toTri(tri), toPlanes(planes)), degree);
  });
  m.def("surface_moments", [](Arr tet, Arr plane, Arr others, int degree) {
    Arr pl = plane;
    Plane h{pl.at(0), pl.at(1), pl.at(2), pl.at(3)};
    return moments(cutSurface(toTet(tet), h, toPlanes(others)), degree);
  });
  m.def("ifem_basis_2d", &ifemBasis<2>, py::arg("nodes"), py::arg("phi"), py::arg("pts"), py::arg("ma"), py::arg("mb"));
  m.def("ifem_basis_3d", &ifemBasis<3>, py::arg("nodes"), py::arg("phi"), py::arg("pts"), py::arg("ma"), py::arg("mb"));
  m.def("monomials", [](int degree) { return monomialIndex(degree); });
  m.def("stokes_ife_p1", &stokesIFEP1, py::arg("nodes"), py::arg("phi"), py::arg("mua"), py::arg("mub"),
        py::arg("pts"));
}
