#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
make_figures.py
===============

Figure and verification code for the article

    S. Ghassemi Tabrizi, "Hidden symmetry at the diabolical points of a
    biaxial spin" (Physical Review B, under review; arXiv:2607.26308).

Default run: writes the five figures of the article into the directory of
this file,

    fig_chains.pdf    Fig. 1  the two severed chains          (schematic, J=3)
    fig_lattice.pdf   Fig. 2  the DP lattice                  (J=3)
    fig_stars.pdf     Fig. 3  the Majorana-star count         (schematic)
    fig_detM.pdf      Fig. 4  smallest |det M| over the lattice
    fig_cutline.pdf   Fig. 5  spectrum along a cut line       (J=3, k2/k1=0.3)

    python3 make_figures.py

--verify: writes no figure; recomputes the quantitative claims of Sec. VI,
Table II, Table S1 of the Supplemental Material, the thresholds of SM
Sec. S5, and the Fe8 census, field values and values of C at which the
ground-pair DPs leave the hard axis (Sec. VII), and compares them with the
manuscript source (see README.md for the exact list):

    python3 make_figures.py --verify --tex /path/to/Biaxial_DP_PRB_v10.tex

The source of the Supplemental Material, <stem>_supp.tex, must lie in the
same directory.  Equation, section and table numbers in this file refer to
the revised manuscript (v10).

Requirements: numpy and matplotlib for the figures; scipy and mpmath in
addition for --verify.

Model
-----
    H(h) = k1 Sx^2 + k2 Sy^2 - h.S ,      k1 > k2 > 0,  spin J,  N = 2J+1
    c = sqrt(k2/k1),  s = sqrt(1 - k2/k1),  c^2 + s^2 = 1
    Z_pm = +- s Sx + c Sz            (two rotated spin projections)
    P_m  = 1_[-J,m](Z_+)             (spectral cut of Z_+, rank p = J+m+1)
    Q_n  = 1_[-J,n](Z_-)             (spectral cut of Z_-, rank q = J+n+1)
    B_mn = P_m Q_n (1 - P_m)         (pairing operator, rank f)
    f(m,n) = min{p, q, N-p, N-q}     (number of doublets at the lattice point)
    DP lattice:  h_x = k1 s (m-n),  h_y = 0,  h_z = -k1 c (m+n+1),
                 m, n in {-J, ..., J-1}

Determinism
-----------
No random numbers are drawn anywhere in this file (the only "arbitrary"
constant is the fixed unitary U_TEST used as a basis-change probe in the
accuracy diagnostic, defined explicitly below).  SOURCE_DATE_EPOCH is pinned
before matplotlib is imported so that the PDF byte stream -- which otherwise
carries a creation timestamp -- is reproducible: two runs give byte-identical
PDFs, and the log prints the SHA-256 of every file it writes.

Numerical safeguards
--------------------
Ranks are obtained from a singular-value cutoff, the classification margin
is monitored, and rank == f(m,n) is asserted at every sampled point; a
failure raises and stops the run.  Degeneracies are classified with a grey
zone between two thresholds, and a spacing inside the grey zone aborts the
run.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import re
import sys
import time

# Pin the PDF creation date *before* importing matplotlib so that repeated
# runs produce byte-identical files.
os.environ.setdefault("SOURCE_DATE_EPOCH", "1700000000")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon

# --------------------------------------------------------------------------
# Global configuration
# --------------------------------------------------------------------------

OUT = os.path.dirname(os.path.abspath(__file__))
FIGURES = ["fig_chains.pdf", "fig_lattice.pdf", "fig_cutline.pdf",
           "fig_detM.pdf", "fig_stars.pdf"]

EPS = np.finfo(float).eps

# Numerical-rank cutoff, *relative* to the largest singular value (defect c).
# Measured separation over J <= 4 and k2/k1 in [0.005, 0.995]:
#   largest spurious (round-off) singular value   2.5e-15 * max(sigma_1, 1)
#   smallest genuine singular value               1.8e-11 * max(sigma_1, 1)
# RANK_RTOL sits close to the geometric mean of those two bounds, i.e. about
# 2.5 decades of margin on each side.  The margins are re-measured at run time
# and printed; MIN_RANK_MARGIN below is enforced.
RANK_RTOL = 1.0e-13
MIN_RANK_MARGIN = 10.0        # required ratio sigma_k / cutoff and cutoff / sigma_{k+1}

# Degeneracy detection: two adjacent levels form a doublet if their relative
# gap is below DOUBLET_RTOL.  Observed for J <= 4 on the grids used here:
# largest gap inside a doublet ~ 7e-16, smallest gap between distinct levels
# ~ 4e-5, so the threshold sits in a nine-decade-wide empty band.  Any gap
# landing inside the "gray zone" below is treated as a hard failure.
DOUBLET_RTOL = 1.0e-7
GREY_ZONE = (1.0e-10, 1.0e-6)   # (max tolerated doublet gap, min tolerated separation)

# Fixed 2x2 unitary used to re-evaluate det M in a rotated basis of the
# doublet.  det M is invariant under SU(2) changes of basis inside the
# doublet, so the discrepancy is a direct estimate of the numerical error.
_TH = 0.7
U_TEST = np.array([[np.cos(_TH), -np.sin(_TH) * np.exp(0.9j)],
                   [np.sin(_TH) * np.exp(-0.9j), np.cos(_TH)]])

# Colours of the old scripts, deliberately unchanged (see defect (a)).
CL, CR, CA, CG = "#1f4e79", "#a03020", "#0f7b6c", "#909090"
COL = 3.35                      # PRB single-column width, inches

PLT_STYLE = {
    "font.size": 7.5, "axes.labelsize": 8, "axes.linewidth": 0.6,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    "legend.fontsize": 6.8,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "pdf.compression": 6,
}


class CheckFailure(RuntimeError):
    """Raised when a numerical check that the manuscript relies on fails."""


def fail(msg: str) -> None:
    """Abort loudly.  Used for every check the manuscript's claims rest on."""
    raise CheckFailure(msg)


# --------------------------------------------------------------------------
# 1.  Spin algebra and the model
# --------------------------------------------------------------------------

def spin_ops(J):
    """(Sx, Sy, Sz, r) in the Sz eigenbasis, r = (-J, -J+1, ..., J)."""
    d = int(round(2 * J)) + 1
    r = np.array([-J + i for i in range(d)], dtype=float)
    Sz = np.diag(r).astype(complex)
    Sp = np.zeros((d, d), dtype=complex)
    for i in range(d - 1):
        rr = r[i]
        Sp[i + 1, i] = np.sqrt((J - rr) * (J + rr + 1))     # <r+1|S+|r>
    Sm = Sp.conj().T
    return (Sp + Sm) / 2, (Sp - Sm) / (2j), Sz, r


def H_of(J, k1, k2, h, ops=None):
    """H = k1 Sx^2 + k2 Sy^2 - h.S."""
    Sx, Sy, Sz, _ = ops if ops is not None else spin_ops(J)
    return k1 * Sx @ Sx + k2 * Sy @ Sy - (h[0] * Sx + h[1] * Sy + h[2] * Sz)


def cut_projector(Op, rank_p):
    """Spectral projector on the lowest `rank_p` eigenvalues of Hermitian Op.

    The old scripts selected eigenvalues with `w <= cut + 1e-7`, which relies
    on the eigenvalues of Z_pm being exactly the half-integers.  They are (Z_pm
    is a rotation of Sz, since c^2+s^2 = 1), but selecting by *count* removes
    the threshold from the code path entirely and can never return the wrong
    rank.  The eigenvalue check is done once per (J, k2/k1) in lattice_scan().
    """
    w, V = np.linalg.eigh(Op)
    Vs = V[:, :rank_p]
    return Vs @ Vs.conj().T


def det_M(S, vecs):
    """det of the 3x3 cone matrix M of the doublet spanned by vecs[:, :2].

    M_{a,i} = coefficients of the traceless part of the 2x2 perturbation
    <u_a| dH/dh_i |u_b> in the Pauli basis, evaluated in the *original* field
    frame (h_x, h_y, h_z), so that the frame independence claimed in the
    manuscript is tested rather than assumed.  det M is invariant under a
    change of orthonormal basis inside the doublet (a U(2) rotation acts on M
    as a proper rotation of its rows), which test_detM_invariance() exploits
    as an error estimate.
    """
    M = np.zeros((3, 3))
    p, q = vecs[:, 0], vecs[:, 1]
    for a in range(3):
        off = p.conj() @ S[a] @ q
        M[0, a] = -off.real
        M[1, a] = off.imag
        M[2, a] = -0.5 * ((p.conj() @ S[a] @ p).real - (q.conj() @ S[a] @ q).real)
    return np.linalg.det(M)


# --------------------------------------------------------------------------
# 2.  Robust numerical rank  (defect c)
# --------------------------------------------------------------------------

def robust_rank(A, rtol=RANK_RTOL):
    """Numerical rank of A with a cutoff scaled by the largest singular value.

    Returns (rank, margin_above, margin_below, singular_values) where
        margin_above = sigma_rank      / cutoff        (should be >> 1)
        margin_below = cutoff / sigma_{rank+1}         (should be >> 1)

    Two points matter and are the reason the old fixed `tol=1e-8` failed:

    * The cutoff must scale with the size of the problem, not be an absolute
      number.  Here every matrix whose rank is taken is built from *orthonormal
      projectors*, so the round-off floor is set by unit-norm inputs even when
      the matrix itself is minuscule (near a uniaxial limit ||B|| can be 1e-9
      while the round-off in forming it is still ~1e-17).  The scale used is
      therefore max(sigma_1, 1).
    * A pure "largest gap in the sorted spectrum" rule is *not* safe here: the
      genuine singular values decay geometrically (ratios of ~1e-3 per step
      near a uniaxial limit) and the trailing round-off values occasionally
      contain one anomalously small entry, so the largest log-gap can land
      either one step too early or at the very end of the spectrum.  Both
      failure modes were observed.  The scaled cutoff below is unambiguous,
      and the two margins returned certify that for every point actually used.
    """
    sv = np.linalg.svd(A, compute_uv=False)          # descending
    n = sv.size
    scale = max(float(sv[0]), 1.0)
    cutoff = rtol * scale
    k = int(np.count_nonzero(sv > cutoff))
    above = (sv[k - 1] / cutoff) if k > 0 else np.inf
    below = (cutoff / sv[k]) if (k < n and sv[k] > 0.0) else np.inf
    return k, float(above), float(below), sv


# --------------------------------------------------------------------------
# 3.  Lattice scan with all checks  (defects c, e)
# --------------------------------------------------------------------------

def lattice_scan(J, k1, k2, want_dets=True, accuracy_probe=False):
    """Diagonalise H at every DP lattice point of (J, k1, k2) and check it.

    Every claim the manuscript makes point-by-point is asserted here:
        [H, P_m] = [H, Q_n] = 0
        rank[P_m, Q_n] = 2 f(m,n)
        rank B_mn      =   f(m,n)
        number of doublets = f(m,n), no multiplicity above two
        det M < 0 for every doublet                      <-- defect (e)
    A violation raises CheckFailure; nothing is silently tolerated.

    Returns a dict with the per-point rows and the worst-case diagnostics.
    """
    if not (k1 > k2 > 0):
        fail(f"model requires k1 > k2 > 0, got k1={k1}, k2={k2}")
    ratio = k2 / k1
    c, s = np.sqrt(ratio), np.sqrt(1.0 - ratio)
    N = int(round(2 * J)) + 1
    Sx, Sy, Sz, rvals = spin_ops(J)
    S = [Sx, Sy, Sz]
    Zp, Zm = s * Sx + c * Sz, -s * Sx + c * Sz
    I = np.eye(N)

    # The projectors are spectral cuts of Z_pm, so the eigenvalues of Z_pm had
    # better be the half-integers -J..J.  (Manuscript: "checked to eight
    # digits before the projectors were formed".)
    dev = 0.0
    for Z in (Zp, Zm):
        dev = max(dev, float(np.max(np.abs(np.linalg.eigvalsh(Z) - rvals))))
    if dev > 1e-8:
        fail(f"J={J} k2/k1={ratio}: eigenvalues of Z_pm deviate from -J..J by {dev:.2e}")

    wP, VP = np.linalg.eigh(Zp)
    wQ, VQ = np.linalg.eigh(Zm)

    rows = []
    diag = dict(max_comm=0.0, min_rank_margin=np.inf, max_pair_gap=0.0,
                min_sep_gap=np.inf, max_det_relerr=0.0, zpm_dev=dev)

    for m in np.arange(-J, J, 1.0):
        for n in np.arange(-J, J, 1.0):
            p, q = int(round(J + m + 1)), int(round(J + n + 1))
            f = min(p, q, N - p, N - q)
            # the two closed forms for f used in the manuscript must agree
            f_alt = int(round((N - abs(m - n) - abs(m + n + 1)) / 2))
            if f != f_alt:
                fail(f"J={J} (m,n)=({m},{n}): f mismatch {f} vs {f_alt}")

            U, W = VP[:, :p], VQ[:, :q]
            P, Q = U @ U.conj().T, W @ W.conj().T
            B = P @ Q @ (I - P)
            C = P @ Q - Q @ P

            h = np.array([k1 * s * (m - n), 0.0, -k1 * c * (m + n + 1)])
            H = H_of(J, k1, k2, h, ops=(Sx, Sy, Sz, rvals))

            # --- (i) the two projectors commute with H -----------------------
            cP = float(np.linalg.norm(H @ P - P @ H))
            cQ = float(np.linalg.norm(H @ Q - Q @ H))
            diag["max_comm"] = max(diag["max_comm"], cP, cQ)
            if max(cP, cQ) > 1e-10 * max(1.0, float(np.linalg.norm(H))):
                fail(f"J={J} k2/k1={ratio} (m,n)=({m},{n}): "
                     f"[H,P]={cP:.2e}, [H,Q]={cQ:.2e} do not vanish")

            # --- (ii) ranks, with the robust cutoff --------------------------
            rkB, aB, bB, svB = robust_rank(B)
            rkC, aC, bC, svC = robust_rank(C)
            diag["min_rank_margin"] = min(diag["min_rank_margin"], aB, bB, aC, bC)
            if rkB != f:
                fail(f"J={J} k2/k1={ratio} (m,n)=({m},{n}): rank B = {rkB}, "
                     f"expected f = {f}; singular values "
                     + " ".join(f"{x:.2e}" for x in svB))
            if rkC != 2 * f:
                fail(f"J={J} k2/k1={ratio} (m,n)=({m},{n}): rank[P,Q] = {rkC}, "
                     f"expected 2f = {2 * f}; singular values "
                     + " ".join(f"{x:.2e}" for x in svC))
            if min(aB, bB, aC, bC) < MIN_RANK_MARGIN:
                fail(f"J={J} k2/k1={ratio} (m,n)=({m},{n}): rank classification "
                     f"margin {min(aB, bB, aC, bC):.2f} below {MIN_RANK_MARGIN}; "
                     "double precision no longer resolves this rank")

            # --- (iii) doublets ---------------------------------------------
            w, V = np.linalg.eigh(H)
            gaps = np.diff(w) / np.maximum(1.0, np.abs(w[:-1]))
            pair_idx, i = [], 0
            while i < N - 1:
                if gaps[i] < DOUBLET_RTOL:
                    pair_idx.append(i)
                    diag["max_pair_gap"] = max(diag["max_pair_gap"], float(gaps[i]))
                    # a triplet would show up as two consecutive small gaps
                    if i + 1 < N - 1 and gaps[i + 1] < DOUBLET_RTOL:
                        fail(f"J={J} k2/k1={ratio} (m,n)=({m},{n}): eigenvalue of "
                             "multiplicity > 2 detected")
                    i += 2
                else:
                    diag["min_sep_gap"] = min(diag["min_sep_gap"], float(gaps[i]))
                    i += 1
            if len(pair_idx) != f:
                fail(f"J={J} k2/k1={ratio} (m,n)=({m},{n}): {len(pair_idx)} doublets, "
                     f"expected f = {f}")
            # no gap may sit in the ambiguous band between the two populations
            amb = gaps[(gaps > GREY_ZONE[0]) & (gaps < GREY_ZONE[1])]
            if amb.size:
                fail(f"J={J} k2/k1={ratio} (m,n)=({m},{n}): relative level gap "
                     f"{amb.min():.2e} lies in the ambiguous band {GREY_ZONE}")

            # --- (iv) det M --------------------------------------------------
            dets = []
            if want_dets:
                for i0 in pair_idx:
                    pair = V[:, i0:i0 + 2]
                    d = det_M(S, pair)
                    if not (d < 0.0):                       # defect (e)
                        fail(f"J={J} k2/k1={ratio} (m,n)=({m},{n}): det M = {d:.6e} "
                             "is not strictly negative")
                    dets.append(float(d))
                    if accuracy_probe:
                        d2 = det_M(S, pair @ U_TEST)        # SU(2)-rotated basis
                        diag["max_det_relerr"] = max(
                            diag["max_det_relerr"], abs(d2 - d) / abs(d))

            rows.append(dict(m=float(m), n=float(n), X=float(m - n),
                             Z=float(-(m + n + 1)), p=p, q=q, f=f,
                             ndoub=len(pair_idx), rkPQ=rkC, rkB=rkB,
                             commP=cP, commQ=cQ, dets=dets))

    alldets = [d for r in rows for d in r["dets"]]
    # global consistency: sum_{m,n} f = (2/3) J(J+1)(2J+1)
    tot = sum(r["f"] for r in rows)
    pred = 2 * J * (J + 1) * (2 * J + 1) / 3
    if abs(tot - pred) > 1e-9:
        fail(f"J={J}: sum f = {tot} != (2/3)J(J+1)(2J+1) = {pred}")

    return dict(J=J, k1=k1, k2=k2, ratio=ratio, N=N, rows=rows, dets=alldets,
                ndp=len(alldets), tot=tot, pred=pred,
                min_absdet=(min(abs(d) for d in alldets) if alldets else None),
                max_det=(max(alldets) if alldets else None),
                ok=True, **diag)


def check_point(res):
    """The 'all checks' flag of the manuscript's Table II -- including the sign
    test on det M that the old make_data.py left out (defect e)."""
    rows = res["rows"]
    return (all(r["f"] == r["ndoub"] for r in rows)
            and all(r["rkPQ"] == 2 * r["f"] for r in rows)
            and all(r["rkB"] == r["f"] for r in rows)
            and all(d < 0.0 for d in res["dets"])          # <-- the missing test
            and abs(res["tot"] - res["pred"]) < 1e-9)


# --------------------------------------------------------------------------
# 4.  Scientific-notation helper and its unit tests  (defect d)
# --------------------------------------------------------------------------

def sci(x, digits=1):
    r"""TeX scientific notation, e.g. -9.5\!\times\!10^{-8}.

    The old make_tables.py version was

        def sci(x):
            s = f"{abs(x):.1e}"; mant, ex = s.split("e")
            return "-" + mant + r"\!\times\!10^{" + str(int(ex)) + "}"

    which took the absolute value and then pasted a minus sign in front, so a
    *positive* determinant would have been typeset as negative -- exactly the
    quantity whose sign the paper's central claim is about.
    """
    if not np.isfinite(x):
        raise ValueError(f"sci() needs a finite number, got {x!r}")
    mant, ex = f"{abs(float(x)):.{digits}e}".split("e")
    sign = "-" if float(x) < 0.0 else ""
    return sign + mant + r"\!\times\!10^{" + str(int(ex)) + "}"


def test_sci():
    """Unit test for sci(); returns the number of assertions made."""
    cases = [
        (-9.5e-8, 1, r"-9.5\!\times\!10^{-8}"),     # negative stays negative
        (4.8e-11, 1, r"4.8\!\times\!10^{-11}"),     # positive is NOT negated
        (-1.0, 1, r"-1.0\!\times\!10^{0}"),
        (0.0, 1, r"0.0\!\times\!10^{0}"),           # zero is not signed
        (-1.25e-3, 2, r"-1.25\!\times\!10^{-3}"),
        (1.0e5, 1, r"1.0\!\times\!10^{5}"),
        (-4.5e-2, 1, r"-4.5\!\times\!10^{-2}"),
    ]
    for x, d, want in cases:
        got = sci(x, d)
        assert got == want, f"sci({x!r}, {d}) = {got!r}, expected {want!r}"
    # the old implementation is wrong precisely on the positive case
    old = lambda v: "-" + f"{abs(v):.1e}".split("e")[0] + r"\!\times\!10^{" \
                    + str(int(f"{abs(v):.1e}".split("e")[1])) + "}"
    assert old(4.8e-11) != sci(4.8e-11), "regression test for defect (d) is vacuous"
    for bad in (np.nan, np.inf):
        try:
            sci(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"sci({bad}) should have raised")
    return len(cases) + 3


def test_detM_invariance():
    """det M must not depend on the basis chosen inside the doublet."""
    res = lattice_scan(2.0, 1.0, 0.3, accuracy_probe=True)
    err = res["max_det_relerr"]
    assert err < 1e-9, f"det M is basis dependent at the {err:.2e} level"
    return err


def test_scale_invariance():
    """det M depends on k2/k1 only, never on the overall energy scale.

    The manuscript quotes its det M numbers "in units with k1 = 1"; this test
    establishes that the qualifier is empty -- H -> lambda H leaves the
    eigenvectors, hence M, hence det M, untouched.
    """
    a = sorted(lattice_scan(3.0, 1.0, 0.30)["dets"])
    b = sorted(lattice_scan(3.0, 7.0, 2.10)["dets"])         # lambda = 7
    d = sorted(lattice_scan(3.0, 0.13, 0.039)["dets"])       # lambda = 0.13
    e1 = float(np.max(np.abs(np.array(a) - np.array(b))))
    e2 = float(np.max(np.abs(np.array(a) - np.array(d))))
    assert max(e1, e2) < 1e-12, f"det M is scale dependent: {e1:.2e}, {e2:.2e}"
    return e1, e2


# --------------------------------------------------------------------------
# 5.  Figures
# --------------------------------------------------------------------------

def fig_chains(path, J=3.0, m_cut=0, n_cut=-1):
    """Fig. 1 -- the same Hamiltonian as two chains, severed at two bonds.

    Greyscale fix (defect a): the sites below/above the cut were distinguished
    only by their marker edge color.  They now also differ in shape --
    circles for the range of the projector, squares for its kernel -- so the
    figure survives black-and-white printing.  Colours are unchanged.
    """
    N = int(round(2 * J)) + 1
    fig, ax = plt.subplots(figsize=(COL, 1.95))
    ax.set_axis_off()

    def draw_chain(y, cut, label, tag):
        xs = np.arange(N) * 1.0
        mid = None
        for i in range(N - 1):
            r = -J + i
            if r == cut:                       # the severed bond: dotted + scissors
                mid = (xs[i] + xs[i + 1]) / 2
                ax.plot([xs[i] + 0.22, xs[i + 1] - 0.22], [y, y], color=CG,
                        lw=0.8, ls=(0, (1.1, 1.1)), zorder=1)
                for sgn in (1, -1):
                    ax.plot([mid - 0.10, mid + 0.10], [y - sgn * 0.16, y + sgn * 0.16],
                            color="black", lw=1.0, zorder=5)
            else:
                ax.plot([xs[i] + 0.18, xs[i + 1] - 0.18], [y, y], color="black",
                        lw=0.9, zorder=1)
        for i in range(N):
            r = -J + i
            in_range = r <= cut
            col = CL if in_range else CR
            mk = "o" if in_range else "s"      # <-- second, color-free channel
            ms = 7.2 if in_range else 6.4      # equal visual weight for the square
            ax.plot([xs[i]], [y], mk, ms=ms, color="white", mec=col, mew=1.1,
                    zorder=3)
        ax.text(-0.85, y, label, ha="right", va="center", fontsize=7.5)
        ax.text(mid, y + 0.34, tag, ha="center", va="bottom", fontsize=6.4)

    draw_chain(0.90, m_cut, r"$Z_+$ basis", r"bond $m\!\leftrightarrow\!m{+}1$")
    draw_chain(0.00, n_cut, r"$Z_-$ basis", r"bond $n\!\leftrightarrow\!n{+}1$")

    xr = N - 0.55
    ax.annotate("", xy=(xr, 0.05), xytext=(xr, 0.85),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color="black"))
    ax.text(xr + 0.18, 0.45, r"rotate by $2\theta$", ha="left", va="center",
            fontsize=6.6, rotation=90)

    ax.plot([], [], "o", ms=6.5, color="white", mec=CL, mew=1.1,
            label=r"$\mathrm{Ran}\,P_m$ / $\mathrm{Ran}\,Q_n$")
    ax.plot([], [], "s", ms=5.8, color="white", mec=CR, mew=1.1,
            label=r"$\ker P_m$ / $\ker Q_n$")
    ax.legend(loc="lower center", bbox_to_anchor=(0.42, -0.30), ncol=2,
              frameon=False, handletextpad=0.35, columnspacing=1.1, fontsize=6.4)

    ax.set_xlim(-2.6, N + 0.9)
    ax.set_ylim(-0.55, 1.55)
    fig.savefig(path)
    plt.close(fig)
    return dict(J=J, m=m_cut, n=n_cut, N=N)


def fig_lattice(path, J=3.0):
    """Fig. 2 -- the DP lattice in scaled coordinates X = m-n, Z = -(m+n+1).

    Greyscale fix (defect a): the two families of severing loci were both thin
    solid lines separated only by color.  They are now solid (P_m, blue) and
    dashed (Q_n, red).  NOTE for the caption, which currently says "Blue lines
    ... red lines ...": the encoding is now redundant, and wording such as
    "solid (blue) lines ... dashed (red) lines" would match the figure.
    """
    N = int(round(2 * J)) + 1
    fig, ax = plt.subplots(figsize=(COL, 3.15))
    lim = 2 * J + 0.55
    tt = np.array([-lim, lim])
    for m in np.arange(-J, J, 1.0):            # X - Z = 2m+1   (cut of Z_+)
        ax.plot(tt, tt - (2 * m + 1), color=CL, lw=0.5, alpha=0.55,
                ls="-", zorder=1)
    for n in np.arange(-J, J, 1.0):            # X + Z = -(2n+1) (cut of Z_-)
        ax.plot(tt, -(2 * n + 1) - tt, color=CR, lw=0.5, alpha=0.55,
                ls=(0, (3.4, 1.8)), zorder=1)
    fmax = 0
    for m in np.arange(-J, J, 1.0):
        for n in np.arange(-J, J, 1.0):
            X, Z = m - n, -(m + n + 1)
            p, q = int(round(J + m + 1)), int(round(J + n + 1))
            f = min(p, q, N - p, N - q)
            fmax = max(fmax, f)
            ax.plot([X], [Z], "o", ms=9.0, color="white", mec=CA, mew=0.8, zorder=3)
            ax.text(X, Z, str(f), ha="center", va="center", fontsize=6, zorder=4)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$X=h_x/(k_1 s)$")
    ax.set_ylabel(r"$Z=h_z/(k_1 c)$")
    ax.set_xticks(range(-6, 7, 3))
    ax.set_yticks(range(-6, 7, 3))
    ax.legend(handles=[Line2D([], [], color=CL, lw=0.9, ls="-",
                              label=r"$X-Z=2m+1$"),
                       Line2D([], [], color=CR, lw=0.9, ls=(0, (3.4, 1.8)),
                              label=r"$X+Z=-2n-1$")],
              loc="lower center", bbox_to_anchor=(0.5, -0.34), frameon=False,
              ncol=2, handlelength=2.0, columnspacing=1.0)
    fig.savefig(path)
    plt.close(fig)
    return dict(J=J, npts=int((2 * J) ** 2), fmax=fmax)


def fig_cutline(path, J=3.0, k1=1.0, k2=0.3, m=0.0, nw=1200):
    """Fig. 5 -- spectrum along the cut line m = 0.

    The two chain segments are already drawn solid vs dashed (the caption says
    "solid and dashed"); the dash pattern is made longer here so it is
    unmistakable at column width, and the line widths differ slightly, giving
    a third greyscale channel.  Colours unchanged.
    """
    N = int(round(2 * J)) + 1
    c, s = np.sqrt(k2 / k1), np.sqrt(1 - k2 / k1)
    a = k1 * c * s
    Sx, Sy, Sz, rv = spin_ops(J)
    Zp = s * Sx + c * Sz
    _, Vz = np.linalg.eigh(Zp)
    u = a * (2 * m + 1)
    cut = int(round(m + J))                 # p - 1, last site of the lower segment
    ws = np.linspace(-6.5, 6.5, nw)
    EL = np.zeros((nw, cut + 1))
    ER = np.zeros((nw, N - cut - 1))
    max_imag = 0.0
    for i, w in enumerate(ws):
        hx, hz = c * u + s * w, -s * u + c * w
        Hb = Vz.conj().T @ H_of(J, k1, k2, np.array([hx, 0.0, hz]),
                                ops=(Sx, Sy, Sz, rv)) @ Vz
        max_imag = max(max_imag, float(np.max(np.abs(Hb.imag))))
        Hb = Hb.real
        # the severed bond makes Hb block diagonal along the whole line
        offblock = float(np.max(np.abs(Hb[:cut + 1, cut + 1:])))
        if offblock > 1e-10:
            fail(f"cut line m={m}: blocks are coupled by {offblock:.2e}")
        EL[i] = np.linalg.eigvalsh(Hb[:cut + 1, :cut + 1])
        ER[i] = np.linalg.eigvalsh(Hb[cut + 1:, cut + 1:])
    if max_imag > 1e-12:
        fail(f"cut line m={m}: H is not real in the Z_+ basis ({max_imag:.2e})")

    fig, ax = plt.subplots(figsize=(COL, 2.25))
    for j in range(EL.shape[1]):
        ax.plot(ws, EL[:, j], color=CL, lw=1.0, ls="-",
                label=r"$P_m\mathcal{H}$" if j == 0 else None)
    for j in range(ER.shape[1]):
        ax.plot(ws, ER[:, j], color=CR, lw=0.8, ls=(0, (4.5, 2.0)),
                label=r"$(1-P_m)\mathcal{H}$" if j == 0 else None)
    for n in np.arange(-J, J, 1.0):
        wl = s * (k1 * s * (m - n)) + c * (-k1 * c * (m + n + 1))
        ax.axvline(wl, color=CG, lw=0.4, alpha=0.6, zorder=0)
    ax.set_xlabel(r"$w=s\,h_x+c\,h_z$")
    ax.set_ylabel(r"$E$")
    ax.set_xlim(ws[0], ws[-1])
    ax.legend(frameon=False, loc="upper center", ncol=2, handlelength=2.0,
              columnspacing=1.0, borderaxespad=0.15)
    fig.savefig(path)
    plt.close(fig)
    p = cut + 1
    return dict(J=J, k2_over_k1=k2 / k1, m=m, p=p, crossings=p * (N - p),
                nw=nw, max_offblock=offblock)


def fig_detM(path, Js=(1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0),
             ratios=(0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.96)):
    """Fig. 4 -- smallest |det M| over the whole lattice vs k2/k1.

    Defect (b): the old version stopped at J = 3, although the text quotes the
    J = 4 number (4.8e-11 at k2/k1 = 0.05).  J = 7/2 and J = 4 are included
    now, which extends the vertical range by three decades.

    Every lattice point entering this figure goes through lattice_scan(), so
    all rank/multiplicity/sign checks are enforced on the plotted data itself.
    """
    ratios = np.asarray(ratios, dtype=float)
    marks = ["o", "s", "^", "D", "v", "P", "X"]
    styles = ["-", "--", "-.", ":", "-", "--", "-."]
    fig, ax = plt.subplots(figsize=(COL, 2.4))
    table = {}
    diag = dict(min_rank_margin=np.inf, max_comm=0.0)
    for i, JJ in enumerate(Js):
        mins = []
        for rr in ratios:
            res = lattice_scan(JJ, 1.0, float(rr))
            if not check_point(res):
                fail(f"J={JJ} k2/k1={rr}: consolidated check failed")
            mins.append(res["min_absdet"])
            diag["min_rank_margin"] = min(diag["min_rank_margin"],
                                          res["min_rank_margin"])
            diag["max_comm"] = max(diag["max_comm"], res["max_comm"])
        table[JJ] = np.array(mins)
        lab = (rf"$J={int(JJ)}$" if float(JJ).is_integer()
               else rf"$J={int(2 * JJ)}/2$")
        ax.semilogy(ratios, mins, marker=marks[i % len(marks)], ms=2.8, lw=0.8,
                    ls=styles[i % len(styles)], label=lab)
    ax.set_xlabel(r"$k_2/k_1$")
    ax.set_ylabel(r"$\min_{\rm DP}|\det M|$")
    ax.legend(frameon=False, ncol=4, handlelength=1.9, columnspacing=0.8,
              handletextpad=0.4, loc="lower center", borderaxespad=0.2)
    # headroom above the curves and below the smallest point, so that the
    # two-row legend in the (empty) lower centre cannot cover a data point
    lo = min(v.min() for v in table.values())
    hi = max(v.max() for v in table.values())
    ax.set_ylim(10 ** (np.log10(lo) - 1.3), 10 ** (np.log10(hi) + 1.6))
    ax.grid(True, which="both", lw=0.25, alpha=0.3)
    fig.savefig(path)
    plt.close(fig)
    return dict(Js=list(Js), ratios=ratios.tolist(), table=table, **diag)


def fig_stars(path):
    """Fig. 3 -- the Majorana-star counting argument (schematic, no data).

    Every star is a genuine spherical polygon, not a marker glyph.  Its outline
    is laid out in the tangent plane at the star's own position and carried to
    the sphere by the exponential map, point by point, with each edge sampled
    so that the edge itself follows the surface.  The result is then projected
    orthographically.  A star therefore curves with the ball, and one near the
    limb is foreshortened along the correct direction instead of merely being
    rotated.  The sphere is shaded (Lambertian, light from the upper left) and
    seen from 20 degrees above the equator, so the equator projects to an
    ellipse.  All stars are filled; the three groups differ by color.
    """
    AZ, RD, AM = "#0d7fc4", "#c0392b", "#e2a32b"     # azure, red, amber

    fig, ax = plt.subplots(figsize=(COL, 1.62))
    ax.set_aspect("equal")
    ax.set_axis_off()

    R, NG = 1.0, 420
    # ---- shaded ball -----------------------------------------------------
    g = np.linspace(-R, R, NG)
    X, Y = np.meshgrid(g, g)
    r2 = X ** 2 + Y ** 2
    inside = r2 <= R ** 2
    Z = np.sqrt(np.clip(R ** 2 - r2, 0.0, None))
    light = np.array([-0.42, 0.52, 0.74])
    light = light / np.linalg.norm(light)
    shade = np.clip((X * light[0] + Y * light[1] + Z * light[2]) / R, 0.0, 1.0)
    img = np.where(inside, 0.30 + 0.70 * shade ** 0.9, np.nan)
    ax.imshow(img, extent=[-R, R, -R, R], origin="lower",
              cmap=LinearSegmentedColormap.from_list(
                  "ball", ["#5b7fa6", "#eaf1f8"]),
              vmin=0.0, vmax=1.0, zorder=0, interpolation="bilinear")
    ax.add_patch(Circle((0, 0), R, fill=False, lw=0.7, color="#5d6b7a",
                        zorder=4))

    # ---- viewing rotation ------------------------------------------------
    TILT = np.radians(20.0)
    CT, ST = np.cos(TILT), np.sin(TILT)

    def view(v):
        """Rotate about the horizontal screen axis; +z is toward the viewer."""
        x, y, z = v
        return np.array([x, CT * y + ST * z, -ST * y + CT * z])

    def unit(v):
        v = np.asarray(v, float)
        return v / np.linalg.norm(v)

    # ---- graticule -------------------------------------------------------
    t = np.linspace(0, 2 * np.pi, 400)
    def draw_curve(px, py, pz, lw, col):
        pts = np.array([view((a, b, c)) for a, b, c in zip(px, py, pz)])
        vis = pts[:, 2] > 0
        ax.plot(np.where(vis, pts[:, 0], np.nan),
                np.where(vis, pts[:, 1], np.nan),
                lw=lw, color=col, alpha=0.55, zorder=2)
    draw_curve(R * np.cos(t), np.zeros_like(t), R * np.sin(t), 0.4, "#ffffff")
    for lam in (-1.0, -0.5, 0.0, 0.5, 1.0):
        draw_curve(R * np.cos(t) * np.cos(lam), R * np.sin(t),
                   R * np.cos(t) * np.sin(lam), 0.35, "#ffffff")

    # ---- a star that actually lies on the sphere -------------------------
    def tangent_basis(c):
        ref = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(ref, c))) > 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        e1 = unit(np.cross(ref, c))
        return e1, np.cross(c, e1)

    def star_on_sphere(c, col, r_out=0.17, ratio=0.42, phase=np.pi / 2,
                       nseg=7, z=6):
        """Draw a five-pointed star whose whole outline lies on the sphere.

        The outline is defined in the tangent plane at c by its polar
        coordinates (rho, phi) and mapped to the sphere by
            p(rho, phi) = c cos(rho) + (cos(phi) e1 + sin(phi) e2) sin(rho),
        the exponential map, so rho is a true angular distance along the
        surface.  Each of the ten edges is sampled nseg times, so the edges
        are surface curves too, not chords.
        """
        c = unit(c)
        e1, e2 = tangent_basis(c)
        r_in = ratio * r_out
        corners = [((r_out if k % 2 == 0 else r_in), phase + k * np.pi / 5)
                   for k in range(10)]
        pts3 = []
        for k in range(10):
            r0, p0 = corners[k]
            r1, p1 = corners[(k + 1) % 10]
            a0 = np.array([r0 * np.cos(p0), r0 * np.sin(p0)])
            a1 = np.array([r1 * np.cos(p1), r1 * np.sin(p1)])
            for tt in np.linspace(0.0, 1.0, nseg, endpoint=False):
                a = a0 + (a1 - a0) * tt
                rho = float(np.hypot(a[0], a[1]))
                phi = float(np.arctan2(a[1], a[0]))
                u = np.cos(phi) * e1 + np.sin(phi) * e2
                pts3.append(c * np.cos(rho) + u * np.sin(rho))
        seen = np.array([view(p) for p in pts3])
        if np.min(seen[:, 2]) <= 0.0:          # would wrap round the limb
            return
        ax.add_patch(Polygon(seen[:, :2], closed=True, facecolor=col,
                             edgecolor=col, lw=0.3, joinstyle="round",
                             zorder=z))

    def along(c, axis, ang):
        """Move from c by the angle ang along the surface, toward axis."""
        c, a = unit(c), unit(axis)
        a = unit(a - np.dot(a, c) * c)
        return c * np.cos(ang) + a * np.sin(ang)

    c1 = unit([-0.50, 0.50, 0.71])
    for d in (-0.23, 0.23):
        star_on_sphere(along(c1, [1.0, 0.30, 0.0], d), AZ)
    c2 = unit([0.50, 0.50, 0.71])
    for d in (-0.23, 0.23):
        star_on_sphere(along(c2, [1.0, -0.30, 0.0], d), RD)

    for cf in ([-0.42, -0.74, 0.52], [0.46, -0.71, 0.53]):
        v = unit(cf)
        star_on_sphere(v, AM, r_out=0.15, z=5)
        for sgn in (+1, -1):
            w, vv = view(along(v, [1.0, 0.0, 0.0], sgn * 0.34)), view(v)
            if w[2] > 0.02 and vv[2] > 0.02:
                ax.annotate("", xy=(w[0], w[1]), xytext=(vv[0], vv[1]),
                            arrowprops=dict(arrowstyle="->", lw=0.45,
                                            color=AM, shrinkA=5.0,
                                            shrinkB=0.5), zorder=5)

    # ---- labels, beside their clusters, no leader lines ------------------
    ax.text(-1.10, 1.24, r"$-\hat n_+$", fontsize=7.0, color=AZ,
            ha="center", va="center")
    ax.text(-1.10, 1.02, r"$J-m$ pinned", fontsize=6.3, color=AZ,
            ha="center", va="center")
    ax.text(1.10, 1.24, r"$-\hat n_-$", fontsize=7.0, color=RD,
            ha="center", va="center")
    ax.text(1.10, 1.02, r"$J-n$ pinned", fontsize=6.3, color=RD,
            ha="center", va="center")
    ax.text(0.0, -1.32, r"$m+n$ free", ha="center", fontsize=6.3, color=AM)

    ax.set_xlim(-1.72, 1.72)
    ax.set_ylim(-1.46, 1.40)
    fig.savefig(path)
    plt.close(fig)
    return dict(schematic=True)


# --------------------------------------------------------------------------
# 6.  Studies quoted in the report
# --------------------------------------------------------------------------

def study_monotonicity(Js=(1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0),
                       ratios=None):
    """Test the Fig. 4 caption claim: min |det M| decreases monotonically with J.

    Returns (ratios, table, violations) with violations a list of
    (k2/k1, J_low, J_high, value_low, value_high).
    """
    if ratios is None:
        ratios = np.round(np.arange(0.01, 0.9901, 0.01), 10)
    ratios = np.asarray(ratios, dtype=float)
    table = np.zeros((len(Js), len(ratios)))
    worst_margin = np.inf
    for i, J in enumerate(Js):
        for j, rr in enumerate(ratios):
            res = lattice_scan(J, 1.0, float(rr))
            if not check_point(res):
                fail(f"J={J} k2/k1={rr}: consolidated check failed")
            table[i, j] = res["min_absdet"]
            worst_margin = min(worst_margin, res["min_rank_margin"])
    violations = []
    for j, rr in enumerate(ratios):
        for i in range(len(Js) - 1):
            if table[i + 1, j] >= table[i, j]:
                violations.append((float(rr), Js[i], Js[i + 1],
                                   float(table[i, j]), float(table[i + 1, j])))
    return ratios, table, violations, worst_margin


def study_quoted_numbers():
    """Re-check the two det M numbers quoted in the text."""
    out = []
    for J, rr, quoted in [(3.0, 0.05, 9.5e-8), (4.0, 0.05, 4.8e-11)]:
        res = lattice_scan(J, 1.0, rr, accuracy_probe=True)
        out.append(dict(J=J, ratio=rr, quoted=quoted,
                        min_absdet=res["min_absdet"], max_det=res["max_det"],
                        ndp=res["ndp"], relerr=res["max_det_relerr"],
                        ok=check_point(res)))
    return out


# --------------------------------------------------------------------------
# 7.  Provenance log helpers
# --------------------------------------------------------------------------

def sha256(path, n=12):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def jlabel(J):
    return f"{int(J)}" if float(J).is_integer() else f"{int(round(2 * J))}/2"


# ==========================================================================
# 8.  --verify : reproduce, print and cross-check every number the
#     manuscript asserts in Sec. VI, SM Sec. S5, Table II and Table S1.
#
#     Nothing below is reached by the default (figure-generating) run, so the
#     five PDFs remain byte-identical.
# ==========================================================================

TEX_NAME = "Biaxial_DP_PRB_v10.tex"

# The sweep of Sec. VI, verbatim.
VERIFY_JS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
VERIFY_RATIOS = (0.05, 0.3, 0.9)

# Table S1 is the complete lattice data for this point of the sweep.
TABLE2_J, TABLE2_RATIO = 2.0, 0.3

# High-precision spot checks of Eq. (27) against Eq. (24).  At least one
# lattice point at J = 4 and one at J = 3, as the text claims.  The first
# entry is the point that carries the worst double-precision residual.
HP_DPS = 60
HP_POINTS = ((4.0, 0.05, 0.0, 0.0),
             (4.0, 0.30, -1.0, 1.0),
             (3.0, 0.05, 0.0, 0.0),
             (3.0, 0.30, 0.0, -1.0))

_WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
            "twelve": 12}


class Mismatch(list):
    """Collector for manuscript-vs-computation disagreements."""

    def check(self, what, manuscript, computed, ok=None):
        if ok is None:
            ok = (manuscript == computed)
        if not ok:
            self.append((what, manuscript, computed))
        return ok


def _mant_exp(x, digits=1):
    """('5.5', -14) for 5.529e-14 -- the rounding the manuscript prints with."""
    mant, ex = f"{abs(float(x)):.{digits}e}".split("e")
    return mant, int(ex)


# --------------------------------------------------------------------------
# 8a.  The Z_+ frame: severed chain, L/R doublet partners, factorized det M (27)
# --------------------------------------------------------------------------

def zplus_frame(J, k2_over_k1):
    r"""(S_x, S_y, S_z) expressed in the |r;+> eigenbasis of Z_+.

    (X_+, Y, Z_+) of Eq. (4) satisfy the same su(2) algebra as (S_x, S_y, S_z)
    -- [X_+,Y] = i Z_+ and cyclic -- and Z_+ is diagonal with entries -J..J in
    its own eigenbasis, so in that basis the triple (X_+, Y, Z_+) is
    numerically the same set of matrices as spin_ops(J).  Inverting Eq. (4),

        S_x = c X_+ + s Z_+,    S_y = Y,    S_z = -s X_+ + c Z_+ .
    """
    c, s = np.sqrt(k2_over_k1), np.sqrt(1.0 - k2_over_k1)
    Xp, Y, Zp, r = spin_ops(J)
    S = [c * Xp + s * Zp, Y, -s * Xp + c * Zp]
    return S, Xp, Y, Zp, r, float(c), float(s)


def lr_doublets(J, k1, k2, m, n):
    r"""Doublets of H_{mn} resolved into their L/R partners, plus Eq. (27).

    In the Z_+ eigenbasis H_{mn} is real symmetric and the bond m<->m+1 is
    severed (Result 1), so H is block diagonal with blocks

        L = span{|r>}_{r<=m}          R = span{|r>}_{r>=m+1} ,

    and the two partners of a doublet are one eigenvector of each block --
    exactly the |L>, |R> of Sec. V.  Diagonalizing the two blocks
    separately therefore produces the doublet basis in which Eq. (27) is
    stated, with no degenerate-subspace ambiguity at all.

    Returns (rows, diagnostics).  Each row carries
        det_closed  -- Eq. (27),  -(l_m^2 a^2 b^2 / 8)(<Z_+>_R - <Z_+>_L)
        det_lr      -- Eq. (24) evaluated in the same (|L>,|R>) basis
    """
    N = int(round(2 * J)) + 1
    S, Xp, Y, Zp, rvals, c, s = zplus_frame(J, k2 / k1)
    hx, hy, hz = k1 * s * (m - n), 0.0, -k1 * c * (m + n + 1)
    H = k1 * S[0] @ S[0] + k2 * S[1] @ S[1] - (hx * S[0] + hy * S[1] + hz * S[2])

    max_imag = float(np.max(np.abs(H.imag)))
    if max_imag > 1e-12:
        fail(f"J={J} k2/k1={k2/k1} (m,n)=({m},{n}): H is not real in the Z_+ "
             f"basis ({max_imag:.2e})")
    Hr = H.real
    scale = max(1.0, float(np.max(np.abs(Hr))))

    cut = int(round(J + m))                    # last index of the L block
    off = float(np.max(np.abs(Hr[:cut + 1, cut + 1:])))
    if off > 1e-12 * scale:
        fail(f"J={J} k2/k1={k2/k1} (m,n)=({m},{n}): the bond m<->m+1 is not "
             f"severed, off-block norm {off:.2e}")

    wL, VL = np.linalg.eigh(Hr[:cut + 1, :cut + 1])
    wR, VR = np.linalg.eigh(Hr[cut + 1:, cut + 1:])
    ell_m = float(np.sqrt((J - m) * (J + m + 1)))

    rows = []
    for i, eL in enumerate(wL):
        for j, eR in enumerate(wR):
            if abs(eL - eR) > 1e-9 * scale:
                continue
            L = np.zeros(N); L[:cut + 1] = VL[:, i]
            R = np.zeros(N); R[cut + 1:] = VR[:, j]
            alpha, beta = float(L[cut]), float(R[cut + 1])       # <m|L>, <m+1|R>
            zL = float(np.sum(rvals * L ** 2))
            zR = float(np.sum(rvals * R ** 2))
            if not (zL <= m + 1e-9 <= m + 1 - 1e-9 <= zR):        # Eq. (26)
                fail(f"J={J} (m,n)=({m},{n}): <Z_+>_L={zL}, <Z_+>_R={zR} "
                     "violate the ordering of Eq. (26)")
            d_cl = -(ell_m ** 2 * alpha ** 2 * beta ** 2 / 8.0) * (zR - zL)
            d_lr = float(det_M(S, np.stack([L, R], axis=1).astype(complex)))
            rows.append(dict(E=0.5 * (eL + eR), alpha=alpha, beta=beta,
                             ell_m=ell_m, zL=zL, zR=zR, zgap=zR - zL,
                             det_closed=d_cl, det_lr=d_lr))
    rows.sort(key=lambda t: t["E"])
    return rows, dict(offblock=off, max_imag=max_imag)


def min_sigma_B(J, k2_over_k1):
    """Smallest *nonvanishing* singular value of B_mn over the lattice of J.

    "Nonvanishing" means the f(m,n)-th singular value, f being the exact rank
    of Eq. (15); everything past it is round-off.  This is the quantity that
    decides whether an absolute rank cutoff is safe (SM Sec. S5).
    """
    N = int(round(2 * J)) + 1
    c, s = np.sqrt(k2_over_k1), np.sqrt(1.0 - k2_over_k1)
    Sx, _Sy, Sz, _r = spin_ops(J)
    _, VP = np.linalg.eigh(s * Sx + c * Sz)
    _, VQ = np.linalg.eigh(-s * Sx + c * Sz)
    I = np.eye(N)
    best = (np.inf, None)
    for m in np.arange(-J, J, 1.0):
        for n in np.arange(-J, J, 1.0):
            p, q = int(round(J + m + 1)), int(round(J + n + 1))
            f = min(p, q, N - p, N - q)
            U, W = VP[:, :p], VQ[:, :q]
            P, Q = U @ U.conj().T, W @ W.conj().T
            sv = np.linalg.svd(P @ Q @ (I - P), compute_uv=False)
            if sv[f - 1] < best[0]:
                best = (float(sv[f - 1]), (float(m), float(n)))
    return best


def direct_doublets(J, k1, k2, m, n):
    """Eq. (24) from a full diagonalisation of H in the S_z basis.

    This is the code path the shipped lattice_scan() uses: the doublet basis
    is whatever LAPACK returns for the degenerate pair, and the 3x3 matrix of
    Eq. (24) is formed and its determinant taken.  det M is invariant under
    U(2) rotations inside the doublet, so the result must agree with Eq. (27);
    the discrepancy is the round-off of the direct evaluation.
    """
    N = int(round(2 * J)) + 1
    Sx, Sy, Sz, rv = spin_ops(J)
    S = [Sx, Sy, Sz]
    c, s = np.sqrt(k2 / k1), np.sqrt(1.0 - k2 / k1)
    h = np.array([k1 * s * (m - n), 0.0, -k1 * c * (m + n + 1)])
    H = H_of(J, k1, k2, h, ops=(Sx, Sy, Sz, rv))
    w, V = np.linalg.eigh(H)
    width = float(w[-1] - w[0])
    gaps_lvl = np.diff(w) / np.maximum(1.0, np.abs(w[:-1]))   # as in lattice_scan
    gaps_wid = np.diff(w) / max(width, 1e-300)                # as SM Sec. S5 reads
    out, pair_gaps, sep_gaps, i = [], [], [], 0
    while i < N - 1:
        if gaps_lvl[i] < DOUBLET_RTOL:
            pair_gaps.append((float(gaps_lvl[i]), float(gaps_wid[i])))
            out.append((0.5 * (w[i] + w[i + 1]), float(det_M(S, V[:, i:i + 2]))))
            i += 2
        else:
            sep_gaps.append((float(gaps_lvl[i]), float(gaps_wid[i])))
            i += 1
    return out, pair_gaps, sep_gaps


# --------------------------------------------------------------------------
# 8b.  60-digit arithmetic
# --------------------------------------------------------------------------

def _ensure_mpmath():
    try:
        import mpmath                                     # noqa: F401
    except ImportError:
        fail("--verify needs mpmath for the 60-digit checks; install it with "
             "'python3 -m pip install mpmath' (see requirements.txt)")
    return mpmath


def highprec_point(mp, J, k1, k2, m, n, dps=HP_DPS):
    r"""Eq. (27) vs Eq. (24) at (J, k2/k1, m, n) in `dps`-digit arithmetic.

    H is rebuilt from Eq. (1) -- not from the tridiagonal shortcut -- in the
    Z_+ basis, where S_y = i A with A real antisymmetric, so
    H = k1 S_x^2 - k2 A^2 - h_x S_x - h_z S_z is real symmetric.  The closed
    form uses the L/R block partners; the direct determinant uses eigenvectors
    of the *full* H, so the two evaluations share nothing but the matrix.
    """
    old = mp.mp.dps
    mp.mp.dps = dps + 15                      # guard digits, reported at `dps`
    try:
        Jm, k1m, k2m = mp.mpf(J), mp.mpf(k1), mp.mpf(k2)
        mm, nn = mp.mpf(m), mp.mpf(n)
        N = int(round(2 * J)) + 1
        c, s = mp.sqrt(k2m / k1m), mp.sqrt(1 - k2m / k1m)
        r = [-Jm + i for i in range(N)]
        X = mp.zeros(N, N); A = mp.zeros(N, N); Z = mp.zeros(N, N)
        for i in range(N):
            Z[i, i] = r[i]
        for i in range(N - 1):
            ell = mp.sqrt((Jm - r[i]) * (Jm + r[i] + 1))
            X[i, i + 1] = X[i + 1, i] = ell / 2          # <r+1|X_+|r> = l_r/2
            A[i + 1, i] = -ell / 2                        # S_y = i A
            A[i, i + 1] = ell / 2
        SX = c * X + s * Z                                # S_x
        SZ = -s * X + c * Z                               # S_z
        hx = k1m * s * (mm - nn)
        hz = -k1m * c * (mm + nn + 1)
        H = k1m * (SX * SX) - k2m * (A * A) - hx * SX - hz * SZ
        # symmetrise away the last bit of matrix-product round-off
        for i in range(N):
            for j in range(i + 1, N):
                v = (H[i, j] + H[j, i]) / 2
                H[i, j] = H[j, i] = v

        tol = mp.mpf(10) ** (-(dps + 5))
        cut = int(round(J + m))
        HL = H[0:cut + 1, 0:cut + 1]
        HR = H[cut + 1:N, cut + 1:N]
        EL, VL = mp.eigsy(HL)
        ER, VR = mp.eigsy(HR)
        EF, VF = mp.eigsy(H)
        ell_m = mp.sqrt((Jm - mm) * (Jm + mm + 1))

        out = []
        for i in range(len(EL)):
            for j in range(len(ER)):
                if abs(EL[i] - ER[j]) > tol * max(1, abs(EL[i])):
                    continue
                L = mp.matrix(N, 1); R = mp.matrix(N, 1)
                for t in range(cut + 1):
                    L[t] = VL[t, i]
                for t in range(N - cut - 1):
                    R[cut + 1 + t] = VR[t, j]
                alpha, beta = L[cut], R[cut + 1]
                zL = sum(r[t] * L[t] ** 2 for t in range(N))
                zR = sum(r[t] * R[t] ** 2 for t in range(N))
                d_cl = -(ell_m ** 2 * alpha ** 2 * beta ** 2 / 8) * (zR - zL)

                E0 = (EL[i] + ER[j]) / 2
                idx = [t for t in range(N)
                       if abs(EF[t] - E0) < tol * max(1, abs(E0))]
                if len(idx) != 2:
                    fail(f"high precision J={J} (m,n)=({m},{n}): the full "
                         f"spectrum shows {len(idx)} levels at E={mp.nstr(E0,10)}"
                         ", expected a doublet")
                P = mp.matrix(N, 1); Q = mp.matrix(N, 1)
                for t in range(N):
                    P[t] = VF[t, idx[0]]
                    Q[t] = VF[t, idx[1]]

                def bil(Op, a, b):
                    return sum(a[t] * Op[t, u] * b[u]
                               for t in range(N) for u in range(N))

                M = mp.zeros(3, 3)
                for col, Op in ((0, SX), (2, SZ)):        # real symmetric
                    M[0, col] = -bil(Op, P, Q)
                    M[1, col] = 0
                    M[2, col] = -(bil(Op, P, P) - bil(Op, Q, Q)) / 2
                # S_y = i A with A real antisymmetric and P, Q real:
                # <P|S_y|Q> is purely imaginary, <P|S_y|P> = <Q|S_y|Q> = 0
                M[0, 1] = 0
                M[1, 1] = bil(A, P, Q)
                M[2, 1] = 0
                d_dir = mp.det(M)
                rel = abs(d_cl - d_dir) / abs(d_dir)
                out.append((mp.nstr(d_cl, 30), mp.nstr(d_dir, 30),
                            float(mp.log10(rel)) if rel > 0 else -mp.inf,
                            mp.nstr(rel, 4), float(E0)))
        out.sort(key=lambda t: t[4])
        return out
    finally:
        mp.mp.dps = old


# --------------------------------------------------------------------------
# 8c.  Reading the manuscript (read-only)
# --------------------------------------------------------------------------

def _tex_rows(tex, label):
    """Body rows of the tabular carrying \\label{label}, as lists of cells."""
    i = tex.index("\\label{" + label + "}")
    j = tex.index("\\begin{tabular}", i)
    k = tex.index("\\end{tabular}", j)
    body = tex[j:k]
    body = body[body.index("\\hline") + len("\\hline"):]
    rows = []
    for chunk in body.split("\\\\"):
        chunk = chunk.strip()
        if not chunk:
            continue
        rows.append([cell.strip() for cell in chunk.split("&")])
    return rows


def _plain(cell):
    """Drop the enclosing math delimiters and the thin spaces, nothing else.

    Braces are *kept*: the determinant cells are of the form
    ``$-4.5\\!\\times\\!10^{-2}$`` and must survive verbatim so that they can be
    compared character by character with what sci() emits.
    """
    s = cell.strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1]
    return s.replace("\\,", "").strip()


def _plain_int(cell):
    """An integer cell such as ``$+1$``, ``$3/2$`` or ``$-3$``."""
    return _plain(cell).replace("{", "").replace("}", "")


def parse_manuscript(path):
    """Every assertion of Sec. VI, the Supplemental Material and the tables.

    The numerical method and the point-by-point data live in the Supplemental
    Material, so a file <stem>_supp.tex next to the manuscript is read as well.
    """
    import os
    with open(path, "r", encoding="utf-8") as fh:
        raw = fh.read()
    stem = path[:-4] if path.endswith(".tex") else path
    for cand in (stem + "_supp.tex", stem + "_supplement.tex"):
        if os.path.exists(cand):
            with open(cand, "r", encoding="utf-8") as fh:
                raw += "\n" + fh.read()
    flat = re.sub(r"\s+", " ", raw)
    man = {"path": path, "sha256": sha256(path)}

    def grab(name, pattern, cast=str, flags=0):
        mo = re.search(pattern, flat, flags)
        if mo is None:
            fail(f"could not find the manuscript's statement of '{name}' "
                 f"(pattern {pattern!r}) in {path}")
        g = mo.groups()
        man[name] = cast(*g) if len(g) > 1 else cast(g[0])
        return man[name]

    def powten(mant, ex):
        return float(mant) * 10.0 ** int(ex)

    # ---- Sec. VI ---------------------------------------------------------
    grab("ratios", r"k_2/k_1\\in\\\{([0-9.,]+)\\\}",
         lambda s: tuple(float(v) for v in s.split(",")))
    grab("comm_max",
         r"largest Frobenius norm of \$([0-9.]+)\\times10\^\{(-?\d+)\}\$",
         powten)
    grab("detM_relerr",
         r"relative error below \$(?:([0-9.]+)\\times)?10\^\{(-?\d+)\}\$",
         lambda mant, ex: (float(mant) if mant else 1.0) * 10.0 ** int(ex))
    grab("hp_digits", r"in \$(\d+)\$-digit arithmetic", int)

    # ---- SM Sec. S5 ------------------------------------------------------
    grab("zpm_digits", r"checked against \$-J,\\dots,J\$ to (\w+) digits",
         lambda w: _WORDNUM[w])
    grab("doublet_below", r"degenerate below \$10\^\{(-?\d+)\}\$",
         lambda e: 10.0 ** int(e))
    grab("distinct_above", r"as distinct above \$10\^\{(-?\d+)\}\$",
         lambda e: 10.0 ** int(e))
    grab("rank_rtol",
         r"singular-value cutoff of\s+\$10\^\{(-?\d+)\}\$", lambda e: 10.0 ** int(e))
    mo = re.search(r"on the axis, at \$h_x=([0-9.]+)\$,\s*\$([0-9.]+)\$,"
                   r"\s*\$([0-9.]+)\$ and \$([0-9.]+)\$~K", flat)
    if mo:
        man["gp_on"] = [float(g) for g in mo.groups()]
    mo = re.search(r"\\pm([0-9.]+)\)\$, \$\(([0-9.]+),\\pm([0-9.]+)\)\$ and\s*"
                   r"\$\(([0-9.]+),\\pm([0-9.]+)\)\$~K", flat)
    mo2 = re.search(r"\$\(h_x,h_y\)=\(([0-9.]+),\\pm([0-9.]+)\)\$", flat)
    if mo and mo2:
        man["gp_off"] = [(float(mo2.group(1)), float(mo.group(1))),
                         (float(mo.group(2)), float(mo.group(3))),
                         (float(mo.group(4)), float(mo.group(5)))]

    # Sec. VII: values of C at which a hard-axis crossing reverses its charge
    # and emits a mirror pair, and the bound on the delay to the annihilation
    mo = re.search(r"near \$C=([0-9.]+)\$, \$([0-9.]+)\$ and \$([0-9.]+)\\,\\mu\$K",
                   flat)
    if mo:
        man["c_flip"] = [g for g in mo.groups()]
    mo = re.search(r"less than \$([0-9.]+)\\,\\mu\$K before the annihilation", flat)
    if mo:
        man["c_delay"] = float(mo.group(1))

    grab("margin_min", r"requiring both to exceed\s+(\w+)",
         lambda w: float(_WORDNUM[w]))
    grab("margin_worst", r"worst of the two margins is \$(\d+)\$", float)
    grab("sigma_min",
         r"smallest retained singular value is\s+\$([0-9.]+)\\times10\^\{(-?\d+)\}\$",
         powten)
    grab("sigma_min_where",
         r"smallest retained singular value is \$[0-9.]+\\times10\^\{-?\d+\}\$,"
         r" at \$J=(\d+)\$ and \$k_2/k_1=([0-9.]+)\$",
         lambda j, r_: (float(j), float(r_)))
    grab("abs_cutoff", r"older threshold of \$10\^\{(-?\d+)\}\$",
         lambda e: 10.0 ** int(e))

    # ---- Table II ---------------------------------------------------------
    # Since v13 the table prints seven columns.  The "sum f" and closed-form
    # columns were dropped because they repeated the "doublets" column, and
    # the per-row checkmark became a caption sentence.  Both claims are still
    # verified below -- the equalities numerically, the all-passed claim
    # against the caption text -- so nothing that used to be checked has
    # stopped being checked.
    man["tab1"] = []
    for cells in _tex_rows(raw, "tab:summary"):
        if len(cells) != 7:
            fail(f"Table II: expected 7 columns, got {len(cells)}: {cells}")
        man["tab1"].append(dict(
            J=_plain_int(cells[0]), N=int(_plain_int(cells[1])),
            pts=int(_plain_int(cells[2])), doublets=int(_plain_int(cells[3])),
            det=[_plain(cells[4]), _plain(cells[5]), _plain(cells[6])]))
    m_cap = re.search(r"\\caption\{Global checks\.(.*?)\}\s*\\label\{tab:summary\}",
                      raw, re.S)
    if not m_cap:
        fail("Table II: caption not found")
    man["tab1_caption"] = " ".join(m_cap.group(1).split())

    # ---- Table S1 --------------------------------------------------------
    man["tab2"] = []
    for cells in _tex_rows(raw, "tab:J2"):
        if len(cells) != 11:
            fail(f"Table S1: expected 11 columns, got {len(cells)}: {cells}")
        ints = [int(_plain_int(c)) for c in cells[:10]]
        dets = [d.strip() for d in _plain(cells[10]).split(",\\;")]
        man["tab2"].append(dict(
            m=ints[0], n=ints[1], X=ints[2], Z=ints[3], p=ints[4], q=ints[5],
            f=ints[6], obs=ints[7], rkPQ=ints[8], rkB=ints[9], dets=dets))
    return man


# --------------------------------------------------------------------------
# 8d.  The verification run
# --------------------------------------------------------------------------

def axis_census(J=10.0, k1=0.338, k2=0.246, Cs=(0.0, 29e-6), npts=20000,
                hmax=8.0):
    """Count the degeneracies on the positive half of each anisotropy axis.

    A pi rotation about the field axis commutes with H even when the
    fourth-order term is present, so its two eigen-sectors can be diagonalized
    separately and a degeneracy between them is exactly a sign change of the
    difference of the two sector spectra.  That makes the count exact rather
    than threshold-dependent.  Returns {(axis, C): number of degeneracies}.
    """
    from scipy.linalg import expm
    Sx, Sy, Sz, rvals = spin_ops(J)
    # S_+ exactly as spin_ops builds it: <r+1|S_+|r> on the subdiagonal of the
    # ascending-r basis.  Getting this slot wrong silently replaces S_+ by
    # something that is neither S_+ nor S_-, so it is asserted here.
    d = Sx.shape[0]
    Sp = np.zeros((d, d), dtype=complex)
    for i in range(d - 1):
        rr = rvals[i]
        Sp[i + 1, i] = np.sqrt((J - rr) * (J + rr + 1))
    Sm = Sp.conj().T
    if np.max(np.abs(Sz @ Sp - Sp @ Sz - Sp)) > 1e-9 * max(1.0, J):
        fail("axis_census: the constructed S_+ does not satisfy [Sz,S+] = S+")
    if np.max(np.abs((Sp + Sm) / 2 - Sx)) > 1e-9 * max(1.0, J):
        fail("axis_census: (S_+ + S_-)/2 does not reproduce Sx")
    H0 = k1 * Sx @ Sx + k2 * Sy @ Sy
    # Ref. [LiGarg] writes the quartic term as -C(S_+^4 + S_-^4) with C > 0;
    # the sign matters, and with the opposite one the h_x census is not 31.
    Q = -(Sp @ Sp @ Sp @ Sp + Sm @ Sm @ Sm @ Sm)
    out = {}
    for axis, A in (("z", Sz), ("x", Sx)):
        R = expm(1j * np.pi * A)
        R = (R + R.conj().T) / 2
        w, V = np.linalg.eigh(R)
        Pp, Pm_ = V[:, w > 0], V[:, w < 0]
        for C in Cs:
            HC = H0 + C * Q
            prev, cross = None, 0
            for h in np.linspace(1e-7, hmax, npts):
                H = HC - h * A
                a = np.linalg.eigvalsh(Pp.conj().T @ H @ Pp).real
                b = np.linalg.eigvalsh(Pm_.conj().T @ H @ Pm_).real
                S = np.sign(a[:, None] - b[None, :])
                if prev is not None:
                    cross += int(np.sum(S != prev))
                prev = S
            out[(axis, C)] = cross
    return out


def ground_pair_fe8(J=10.0, k1=0.338, k2=0.246, C=29e-6, npts=200000,
                    hmax=4.0):
    """Where the ground-pair DPs of Fe8 sit once the quartic term is on.

    Returns (on_axis, off_axis): the surviving crossings of the two lowest
    levels on the positive h_x axis, and the ones that have left it, refined
    to ~1e-9.  These are the numbers the Discussion quotes, and they are the
    only ones in the paper that no other check touches.
    """
    from scipy.linalg import expm
    from scipy.optimize import minimize
    Sx, Sy, Sz, rvals = spin_ops(J)
    d = Sx.shape[0]
    Sp = np.zeros((d, d), dtype=complex)
    for i in range(d - 1):
        rr = rvals[i]
        Sp[i + 1, i] = np.sqrt((J - rr) * (J + rr + 1))
    Sm = Sp.conj().T
    H0 = k1 * Sx @ Sx + k2 * Sy @ Sy - C * (Sp @ Sp @ Sp @ Sp + Sm @ Sm @ Sm @ Sm)
    R = expm(1j * np.pi * Sx)
    R = (R + R.conj().T) / 2
    w, V = np.linalg.eigh(R)
    Pp, Pm_ = V[:, w > 0], V[:, w < 0]
    prev, on = None, []
    hs = np.linspace(1e-7, hmax, npts)
    for h in hs:
        H = H0 - h * Sx
        a = np.linalg.eigvalsh(Pp.conj().T @ H @ Pp).real
        b = np.linalg.eigvalsh(Pm_.conj().T @ H @ Pm_).real
        sg = np.sign(a[0] - b[0])
        if prev is not None and sg != prev:
            lo, hi = hprev, h
            for _ in range(60):
                mid = (lo + hi) / 2
                Hm = H0 - mid * Sx
                am = np.linalg.eigvalsh(Pp.conj().T @ Hm @ Pp).real
                bm = np.linalg.eigvalsh(Pm_.conj().T @ Hm @ Pm_).real
                if np.sign(am[0] - bm[0]) == prev:
                    lo = mid
                else:
                    hi = mid
            on.append((lo + hi) / 2)
        prev, hprev = sg, h
    off = []
    for guess in ((2.12, 0.43), (2.62, 0.90), (3.15, 1.38)):
        gap = lambda v: float(np.diff(np.linalg.eigvalsh(
            H0 - v[0] * Sx - v[1] * Sy))[0])
        r = minimize(gap, guess, method="Nelder-Mead",
                     options=dict(xatol=1e-10, fatol=1e-16, maxiter=8000))
        if r.fun > 1e-10:
            fail(f"ground_pair_fe8: no degeneracy near {guess}, gap {r.fun:.2e}")
        off.append((r.x[0], r.x[1]))
    return on, off


def fe8_axis_events(J=10.0, k1=0.338, k2=0.246, Cmax=29e-6, dC=0.5e-6,
                    nh=4000, hmax=4.0):
    """How the ground-pair crossings of Fe8 leave the positive hard axis.

    Sec. VII states that two crossings of the two lowest levels on the h_x
    axis can disappear only together, that both carry chi = -1 at C = 0, and
    that one of them therefore first reverses its charge, emitting a mirror
    pair of DPs into the plane h_z = 0, before the two annihilate.  On the
    axis the two crossing states lie in different sectors of the pi rotation
    about x, and the charge of a crossing is the sign of
        slope * Im(<+|S_y|-> <-|S_z|+>),   slope = d(E_+ - E_-)/dh_x,
    a gauge-invariant product (Result 4 fixes the common sign at C = 0).

    Returns a list of events (C_flip, C_ann, x, (hx_off, hy_off)): the value
    of C at which one crossing of the merging pair reverses its charge, the
    value at which the pair annihilates, where on the axis that happens, and
    an off-axis degeneracy in the plane h_z = 0 found just after it.
    """
    from scipy.optimize import brentq, minimize, minimize_scalar
    Sx, Sy, Sz, rvals = spin_ops(J)
    d = Sx.shape[0]
    Sp = np.zeros((d, d))
    for i in range(d - 1):
        rr = rvals[i]
        Sp[i + 1, i] = np.sqrt((J - rr) * (J + rr + 1))
    Sm = Sp.T
    Sxr, Szr = Sx.real, Sz.real
    iSy = (Sp - Sm) / 2                      # i S_y, real antisymmetric
    if np.max(np.abs(1j * Sy - iSy)) > 1e-12 * max(1.0, J):
        fail("fe8_axis_events: i*Sy is not (S_+ - S_-)/2")
    Q4 = np.linalg.matrix_power(Sp, 4) + np.linalg.matrix_power(Sm, 4)
    H0 = (k1 * Sx @ Sx + k2 * Sy @ Sy).real
    # sectors of R_x(pi) = exp(-i pi S_x): parity of the S_x eigenvalue
    wx, Vx = np.linalg.eigh(Sxr)
    par = np.round(np.cos(np.pi * wx)).astype(int)
    if set(par) != {-1, 1}:
        fail("fe8_axis_events: the pi rotation about x does not split the space")
    Vp, Vm = Vx[:, par == 1], Vx[:, par == -1]
    blocks = [(V.T @ H0 @ V, V.T @ Q4 @ V, np.diag(V.T @ Sxr @ V), V)
              for V in (Vp, Vm)]

    def lowest(h, C):
        out = []
        for A, B, sx, V in blocks:
            e, u = np.linalg.eigh(A - C * B - h * np.diag(sx))
            out.append((e[0], V @ u[:, 0]))
        return out

    def delta(h, C):
        # eigvalsh on exactly the matrices that roots() diagonalizes in a
        # batch, so that both see the same signs at the grid points
        e = [np.linalg.eigvalsh(A - C * B - h * np.diag(sx))[0]
             for A, B, sx, _ in blocks]
        return e[0] - e[1]

    def sigma(h, C):
        (_, vp), (_, vm) = lowest(h, C)
        slope = -(vp @ Sxr @ vp - vm @ Sxr @ vm)          # Hellmann-Feynman
        g = (vp @ iSy @ vm) * (vm @ Szr @ vp)              # Im<+|Sy|-><-|Sz|+>
        return int(np.sign(slope * g))

    hs = np.linspace(1e-4, hmax, nh)

    def roots(C):
        vals = []
        for A, B, sx, _ in blocks:
            M = (A - C * B)[None, :, :] - hs[:, None, None] * np.diag(sx)[None]
            vals.append(np.linalg.eigvalsh(M)[:, 0])
        D = vals[0] - vals[1]
        idx = np.nonzero(np.sign(D[:-1]) != np.sign(D[1:]))[0]
        return [brentq(delta, hs[i], hs[i + 1], args=(C,), xtol=1e-14)
                for i in idx]

    Cs = np.arange(0.0, Cmax + 0.5 * dC, dC)
    R = [roots(C) for C in Cs]
    counts = [len(r) for r in R]
    s0 = {sigma(x, 0.0) for x in R[0]}
    if len(s0) != 1:
        fail("fe8_axis_events: the hard-axis crossings at C = 0 do not share "
             "one charge, contrary to Result 4")
    s0 = s0.pop()
    if any(b > a or (a - b) % 2 for a, b in zip(counts, counts[1:])):
        fail(f"fe8_axis_events: crossing count not non-increasing in steps "
             f"of two: {counts}")

    events = []
    for k in range(len(Cs) - 1):
        for _ in range((counts[k] - counts[k + 1]) // 2):
            Ca, Cb, xs = Cs[k], Cs[k + 1], R[k]
            gaps = np.diff(xs)
            j = int(np.argmin(gaps))
            x1, x2 = xs[j], xs[j + 1]
            pad = 0.25 * min([g for i, g in enumerate(gaps) if i != j]
                             + [x1, hmax - x2])
            lo, hi = x1 - pad, x2 + pad
            s_end = np.sign(delta(lo, Ca))
            if np.sign(delta(hi, Ca)) != s_end:
                fail("fe8_axis_events: window around a merging pair is not clean")

            def extremum(C):
                r = minimize_scalar(lambda x: s_end * delta(x, C),
                                    bounds=(lo, hi), method="bounded",
                                    options=dict(xatol=1e-12))
                return r.x, r.fun                  # < 0: the pair still exists

            C_ann = brentq(lambda C: extremum(C)[1], Ca, Cb, xtol=1e-15)

            def pair_charges(C):
                xe, _ = extremum(C)
                a = brentq(delta, lo, xe, args=(C,), xtol=1e-15)
                b = brentq(delta, xe, hi, args=(C,), xtol=1e-15)
                return sigma(a, C), sigma(b, C)

            # step back from the annihilation until the pair is resolved well
            # above round-off (|Delta| at the extremum > 1e-12 K)
            back = 1e-13
            while extremum(C_ann - back)[1] > -1e-12:
                back *= 2
                if back > C_ann - Ca:
                    fail("fe8_axis_events: merging pair never resolved")
            C_lo, C_hi = Ca, C_ann - back
            if len(set(pair_charges(C_lo))) != 1:
                fail(f"fe8_axis_events: pair near h_x = {x1:.3f} K has opposite "
                     f"charges already at C = {C_lo * 1e6:.2f} uK")
            if len(set(pair_charges(C_hi))) != 2:
                fail(f"fe8_axis_events: pair near h_x = {x1:.3f} K still has "
                     f"equal charges {back * 1e6:.1e} uK before it merges")
            for _ in range(50):
                mid = 0.5 * (C_lo + C_hi)
                if len(set(pair_charges(mid))) == 1:
                    C_lo = mid
                else:
                    C_hi = mid
            C_flip = 0.5 * (C_lo + C_hi)
            # the emitted mirror pair: a degeneracy in the plane h_z = 0 at
            # h_y != 0, just after the annihilation
            C_after = C_ann + 2e-9
            xe, _ = extremum(C_after)
            Hc = H0 - C_after * Q4
            gap = lambda v: float(np.diff(np.linalg.eigvalsh(
                Hc - v[0] * Sx - v[1] * Sy)[:2])[0])
            r = minimize(gap, (xe, 5e-3), method="Nelder-Mead",
                         options=dict(xatol=1e-12, fatol=1e-16, maxiter=8000))
            if r.fun > 1e-10 or abs(r.x[1]) < 1e-5:
                fail(f"fe8_axis_events: no mirror pair off the axis near "
                     f"h_x = {xe:.4f} K after C = {C_ann * 1e6:.4f} uK "
                     f"(gap {r.fun:.2e} at h_y = {r.x[1]:.2e})")
            events.append((C_flip, C_ann, xe, (r.x[0], abs(r.x[1]))))
            counts[k] -= 2
            R[k] = [x for i, x in enumerate(xs) if i not in (j, j + 1)]
    return events, s0


def run_verify(tex_path):
    t0 = time.time()
    mis = Mismatch()
    man = parse_manuscript(tex_path)

    print("=" * 78)
    print("make_figures.py --verify -- every number asserted by the manuscript")
    print("=" * 78)
    print(f"manuscript    : {man['path']}")
    print(f"                sha256:{man['sha256']}  (read only, never written)")
    print(f"host python   : {platform.python_version()}   numpy {np.__version__}")
    print(f"sweep         : J = " + ", ".join(jlabel(J) for J in VERIFY_JS)
          + "   x   k2/k1 = " + ", ".join(str(r) for r in VERIFY_RATIOS))
    mis.check("Sec. VI anisotropy list", tuple(man["ratios"]), VERIFY_RATIOS)
    mis.check("SM Sec. S5 rank cutoff (RANK_RTOL)", man["rank_rtol"], RANK_RTOL)
    mis.check("SM Sec. S5 required margin (MIN_RANK_MARGIN)",
              man["margin_min"], MIN_RANK_MARGIN)
    mis.check("SM Sec. S5 degenerate-below threshold",
              man["doublet_below"], GREY_ZONE[0])
    mis.check("SM Sec. S5 distinct-above threshold",
              man["distinct_above"], GREY_ZONE[1])

    # ---------------------------------------------------------------- 1 ----
    print()
    print("-" * 78)
    print("[1] SWEEP OF SEC. VI -- every lattice point of every (J, k2/k1)")
    print("-" * 78)
    print("     J  k2/k1   pts  doublets  sum f  2/3J(J+1)(2J+1)   "
          "max det M      min |det M|   all checks")

    scan, npoints, ndoublets = {}, 0, 0
    g_comm = (0.0, None)
    g_margin = (np.inf, None)
    g_sigma = (np.inf, None)
    g_relerr = (0.0, None)
    g_relerr_lr = (0.0, None)
    g_pairgap_lvl = g_pairgap_wid = 0.0
    g_sepgap_lvl = g_sepgap_wid = np.inf
    g_zpm = 0.0

    for J in VERIFY_JS:
        N = int(round(2 * J)) + 1
        for rr in VERIFY_RATIOS:
            res = lattice_scan(J, 1.0, float(rr))
            ok = check_point(res)
            if not ok:
                fail(f"J={J} k2/k1={rr}: consolidated check failed")
            scan[(J, rr)] = res
            npoints += len(res["rows"])
            ndoublets += res["ndp"]
            g_zpm = max(g_zpm, res["zpm_dev"])
            if res["max_comm"] > g_comm[0]:
                g_comm = (res["max_comm"], (J, rr))
            if res["min_rank_margin"] < g_margin[0]:
                g_margin = (res["min_rank_margin"], (J, rr))

            # smallest genuine singular value of B_mn over this lattice
            sB, smn = min_sigma_B(J, float(rr))
            if sB < g_sigma[0]:
                g_sigma = (sB, (J, rr, smn[0], smn[1]))

            # Eq. (27) against Eq. (24), point by point
            for row in res["rows"]:
                m, n, f = row["m"], row["n"], row["f"]

                lr, _d = lr_doublets(J, 1.0, float(rr), m, n)
                dd, pg, sg = direct_doublets(J, 1.0, float(rr), m, n)
                if len(lr) != f or len(dd) != f:
                    fail(f"J={J} k2/k1={rr} (m,n)=({m},{n}): {len(lr)} L/R "
                         f"doublets and {len(dd)} direct doublets, expected "
                         f"f = {f}")
                for a, b in pg:
                    g_pairgap_lvl = max(g_pairgap_lvl, a)
                    g_pairgap_wid = max(g_pairgap_wid, b)
                for a, b in sg:
                    g_sepgap_lvl = min(g_sepgap_lvl, a)
                    g_sepgap_wid = min(g_sepgap_wid, b)
                dd.sort(key=lambda t: t[0])
                for k, (row_lr, (_E, d_direct)) in enumerate(zip(lr, dd)):
                    e1 = abs(row_lr["det_closed"] - d_direct) / abs(d_direct)
                    e2 = (abs(row_lr["det_closed"] - row_lr["det_lr"])
                          / abs(row_lr["det_lr"]))
                    if e1 > g_relerr[0]:
                        g_relerr = (e1, (J, rr, m, n, row_lr["det_closed"],
                                         d_direct))
                    if e2 > g_relerr_lr[0]:
                        g_relerr_lr = (e2, (J, rr, m, n))

        for rr in VERIFY_RATIOS:
            res = scan[(J, rr)]
            print(f"  {jlabel(J):>4}  {rr:<5}  {len(res['rows']):>4}  "
                  f"{res['ndp']:>8}  {res['tot']:>5}  {res['pred']:>15.0f}   "
                  f"{res['max_det']:>13.6e}  {res['min_absdet']:>11.4e}   "
                  f"{'OK' if check_point(res) else 'FAILED'}")

    print(f"\n  lattice points visited : {npoints}   doublets examined : "
          f"{ndoublets}")
    print(f"  eigenvalues of Z_pm vs -J..J : max deviation {g_zpm:.2e} "
          f"(manuscript: correct to {man['zpm_digits']} digits)")
    mis.check("SM Sec. S5: Z_pm eigenvalues good to N digits",
              f">= {man['zpm_digits']} digits", f"{g_zpm:.2e}",
              ok=(g_zpm < 10.0 ** (-man["zpm_digits"])))
    print(f"  doublet detection, gap/|E| : in-doublet <= {g_pairgap_lvl:.2e}, "
          f"distinct >= {g_sepgap_lvl:.2e}")
    print(f"  doublet detection, gap/width: in-doublet <= {g_pairgap_wid:.2e}, "
          f"distinct >= {g_sepgap_wid:.2e}   <- SM Sec. S5's scaling")
    mis.check("SM Sec. S5: every in-doublet gap below the quoted threshold",
              f"< {man['doublet_below']:.0e}", f"{g_pairgap_wid:.2e}",
              ok=(g_pairgap_wid < man["doublet_below"]))
    mis.check("SM Sec. S5: every distinct gap above the quoted threshold",
              f"> {man['distinct_above']:.0e}", f"{g_sepgap_wid:.2e}",
              ok=(g_sepgap_wid > man["distinct_above"]))

    # ---------------------------------------------------------------- 2 ----
    print()
    print("-" * 78)
    print("[2] COMMUTATORS  ||[H,P_m]||_F , ||[H,Q_n]||_F")
    print("-" * 78)
    cm, cw = g_comm
    print(f"  largest over the sweep : {cm:.4e}   at J={jlabel(cw[0])}, "
          f"k2/k1={cw[1]}")
    print(f"  manuscript (Sec. VI)   : {man['comm_max']:.1e}")
    mant, ex = _mant_exp(cm)
    m_mant, m_ex = _mant_exp(man["comm_max"])
    mis.check("Sec. VI: largest ||[H,P]||_F over the sweep",
              f"{m_mant}e{m_ex}", f"{mant}e{ex}")

    # ---------------------------------------------------------------- 3 ----
    print()
    print("-" * 78)
    print("[3] EQ. (27) AGAINST THE DIRECT DETERMINANT EQ. (24)")
    print("-" * 78)
    er, ew = g_relerr
    print(f"  double precision, Eq. (27) vs full-H eigenvectors:")
    print(f"    largest relative error : {er:.4e}")
    at = (f"at J={jlabel(ew[0])}, k2/k1={ew[1]}, "
          f"(m,n)=({ew[2]:+.0f},{ew[3]:+.0f})")
    print(f"    {at}   Eq.(27) = {ew[4]:.12e}")
    print(f"    {'':{len(at)}}   Eq.(24) = {ew[5]:.12e}")
    print(f"    manuscript (Sec. VI)   : below {man['detM_relerr']:.1e}")
    mis.check("Sec. VI: Eq. (27) vs Eq. (24), double precision",
              f"< {man['detM_relerr']:.1e}", f"{er:.4e}",
              ok=(er < man["detM_relerr"]))
    print(f"  same comparison in the (|L>,|R>) doublet basis, where Eq. (24)")
    print(f"  suffers no degenerate-subspace round-off:")
    print(f"    largest relative error : {g_relerr_lr[0]:.4e}   "
          f"(at J={jlabel(g_relerr_lr[1][0])}, k2/k1={g_relerr_lr[1][1]})")
    print("    -> the residual above is round-off in the direct determinant,")
    print("       as the manuscript states, not an error in the formula.")

    mp = _ensure_mpmath()
    print(f"  {man['hp_digits']}-digit arithmetic (mpmath {mp.__version__}):")
    hp_worst = 0.0
    hp_seen_J = set()
    for (J, rr, m, n) in HP_POINTS:
        out = highprec_point(mp, J, 1.0, float(rr), m, n, dps=man["hp_digits"])
        hp_seen_J.add(J)
        for (scl, sdi, log10rel, srel, _E) in out:
            digits = (man["hp_digits"] if log10rel == -np.inf
                      else min(man["hp_digits"], int(-log10rel)))
            hp_worst = max(hp_worst, 10.0 ** log10rel if log10rel != -np.inf
                           else 0.0)
            tag = (f"J={jlabel(J):>3} k2/k1={rr:<5} "
                   f"(m,n)=({m:+.0f},{n:+.0f})")
            print(f"    {tag}  Eq.(27) = {scl}")
            print(f"    {'':{len(tag)}}  Eq.(24) = {sdi}")
            print(f"    {'':{len(tag)}}  rel.err = {srel}  "
                  f"({digits} digits agree)")
    print(f"    worst relative error over the high-precision points: "
          f"{hp_worst:.3e}")
    mis.check("Sec. VI: Eq. (27) exact in high precision",
              f"agree to {man['hp_digits']} digits",
              f"worst rel. err {hp_worst:.3e}",
              ok=(hp_worst < 10.0 ** (-man["hp_digits"] + 5)))
    for need in (4.0, 3.0):
        mis.check(f"Sec. VI: high-precision check covers J={jlabel(need)}",
                  "present", "present" if need in hp_seen_J else "absent")

    # ---------------------------------------------------------------- 4 ----
    print()
    print("-" * 78)
    print("[4] RANK-CUTOFF MARGINS (APPENDIX G)")
    print("-" * 78)
    print(f"  cutoff                        : {RANK_RTOL:.0e} * max(sigma_1, 1)"
          f"   (manuscript {man['rank_rtol']:.0e})")
    print(f"  required margin on each side  : {MIN_RANK_MARGIN:g}"
          f"                (manuscript {man['margin_min']:g})")
    print("  per (J, k2/k1) worst margin   :")
    for J in VERIFY_JS:
        cells = "  ".join(f"{rr}: {scan[(J, rr)]['min_rank_margin']:8.2f}"
                          for rr in VERIFY_RATIOS)
        print(f"    J={jlabel(J):>3}   {cells}")
    gm, gw = g_margin
    print(f"  worst margin over the sweep   : {gm:.4f}   at J={jlabel(gw[0])}, "
          f"k2/k1={gw[1]}")
    print(f"  manuscript (SM Sec. S5)       : {man['margin_worst']:.0f}")
    margin_ok = mis.check(
        "SM Sec. S5: worst rank-classification margin over the sweep",
        f"{man['margin_worst']:.0f}", f"{gm:.2f}",
        ok=(round(gm) == round(man["margin_worst"])))
    if not margin_ok:
        # Where does the manuscript's number come from?  The margin is a pure
        # round-off quantity (cutoff / largest discarded singular value), so it
        # is governed by the size of the anisotropy grid that is swept, not by
        # the physics.  Probe the two other grids this script uses.
        print("  the quoted value is not attained on the sweep of Sec. VI; "
              "probing the")
        print("  other anisotropy grids this script scans:")
        for tag, grid in (("Fig. 4 grid (10 ratios)",
                           fig_detM.__defaults__[1]),
                          ("Fig. 4 caption study (99 ratios)",
                           tuple(np.round(np.arange(0.01, 0.9901, 0.01), 10)))):
            worst = (np.inf, None)
            for J in VERIFY_JS:
                for rr in grid:
                    r_ = lattice_scan(J, 1.0, float(rr), want_dets=False)
                    if r_["min_rank_margin"] < worst[0]:
                        worst = (r_["min_rank_margin"], (J, float(rr)))
            print(f"    {tag:<34}: {worst[0]:8.4f}  at J="
                  f"{jlabel(worst[1][0])}, k2/k1={worst[1][1]:g}"
                  + ("   <- rounds to the quoted value"
                     if round(worst[0]) == round(man["margin_worst"]) else ""))
    sg, sw = g_sigma
    print(f"  smallest nonvanishing sigma(B_mn) : {sg:.4e}   at "
          f"J={jlabel(sw[0])}, k2/k1={sw[1]}, (m,n)=({sw[2]:+.0f},{sw[3]:+.0f})")
    print(f"  manuscript (SM Sec. S5)           : {man['sigma_min']:.1e} at "
          f"J={man['sigma_min_where'][0]:.0f}, "
          f"k2/k1={man['sigma_min_where'][1]}")
    s_mant, s_ex = _mant_exp(sg)
    q_mant, q_ex = _mant_exp(man["sigma_min"])
    mis.check("SM Sec. S5: smallest nonvanishing singular value of B_mn",
              f"{q_mant}e{q_ex}", f"{s_mant}e{s_ex}")
    mis.check("SM Sec. S5: location of that singular value",
              f"J={man['sigma_min_where'][0]:.0f}, "
              f"k2/k1={man['sigma_min_where'][1]}",
              f"J={sw[0]:.0f}, k2/k1={sw[1]}",
              ok=(sw[0] == man["sigma_min_where"][0]
                  and sw[1] == man["sigma_min_where"][1]))
    print(f"  SM Sec. S5 further claims that the absolute cutoff "
          f"{man['abs_cutoff']:.0e} of the old code")
    print("  misclassifies ranks 'just outside the grid used here'.  Walking "
          "the largest J")
    print("  down in anisotropy (all these points have exact rank f, so any "
          "sigma below")
    print(f"  {man['abs_cutoff']:.0e} is a genuine misclassification):")
    Jmax = max(VERIFY_JS)
    first_bad = None
    for rr in (0.05, 0.04, 0.03, 0.02, 0.01):
        sB, smn = min_sigma_B(Jmax, rr)
        bad = sB < man["abs_cutoff"]
        if bad and first_bad is None:
            first_bad = (rr, sB, smn)
        print(f"    J={jlabel(Jmax):>3}  k2/k1={rr:<5}  min sigma(B_mn) = "
              f"{sB:.3e}   at (m,n)=({smn[0]:+.0f},{smn[1]:+.0f})"
              + ("   <- an absolute 1e-8 cutoff calls this rank zero"
                 if bad else ""))
    mis.check("SM Sec. S5: an absolute 1e-8 cutoff fails just outside the grid",
              f"fails below some k2/k1 < {VERIFY_RATIOS[0]}",
              ("fails first at k2/k1=%g" % first_bad[0]) if first_bad
              else "never fails on the probed ratios",
              ok=(first_bad is not None))

    # ---------------------------------------------------------------- 5 ----
    print()
    print("-" * 78)
    print("[5] TABLE II -- GLOBAL CHECKS  (as the table typesets it)")
    print("-" * 78)
    hdr = (f"  {'J':>4} {'N':>3} {'pts':>4} {'doub':>5} {'sum f':>6} "
           f"{'2/3J(J+1)(2J+1)':>16}  {'max det M (0.05)':>26} "
           f"{'(0.3)':>26} {'(0.9)':>26}  checks")
    print(hdr)
    for phrase in ("equal at every anisotropy both to",
                   "All rank, multiplicity and sign checks passed in every row"):
        if phrase not in man["tab1_caption"]:
            fail(f"Table II caption no longer claims: {phrase!r} -- the "
                 f"corresponding check below would be vacuous")
    if len(man["tab1"]) != len(VERIFY_JS):
        mis.check("Table II: number of rows", len(VERIFY_JS), len(man["tab1"]))
    for idx, J in enumerate(VERIFY_JS):
        N = int(round(2 * J)) + 1
        pts = int(round((2 * J) ** 2))
        nds = {rr: scan[(J, rr)]["ndp"] for rr in VERIFY_RATIOS}
        if len(set(nds.values())) != 1:
            fail(f"J={J}: doublet count differs between anisotropies: {nds}")
        ndoub = nds[VERIFY_RATIOS[0]]
        sumf = int(scan[(J, VERIFY_RATIOS[0])]["tot"])
        closed = int(round(2 * J * (J + 1) * (2 * J + 1) / 3))
        dets = [sci(scan[(J, rr)]["max_det"]) for rr in VERIFY_RATIOS]
        allok = all(check_point(scan[(J, rr)]) for rr in VERIFY_RATIOS)
        print(f"  {jlabel(J):>4} {N:>3} {pts:>4} {ndoub:>5} {sumf:>6} "
              f"{closed:>16}  {dets[0]:>26} {dets[1]:>26} {dets[2]:>26}  "
              f"{'checkmark' if allok else 'FAILED'}")
        if idx >= len(man["tab1"]):
            continue
        row = man["tab1"][idx]
        tag = f"Table II row J={jlabel(J)}"
        mis.check(f"{tag}: J", row["J"], jlabel(J))
        mis.check(f"{tag}: N", row["N"], N)
        mis.check(f"{tag}: pts", row["pts"], pts)
        mis.check(f"{tag}: doublets", row["doublets"], ndoub)
        # The caption asserts that the printed "doublets" column equals both
        # sum f(m,n) and the closed form; those columns are no longer printed,
        # so the equalities are checked here against the printed value.
        mis.check(f"{tag}: caption claim doublets = sum f", ndoub, sumf)
        mis.check(f"{tag}: caption claim doublets = 2/3 J(J+1)(2J+1)",
                  ndoub, closed)
        for jj, rr in enumerate(VERIFY_RATIOS):
            mis.check(f"{tag}: max det M at k2/k1={rr}", row["det"][jj],
                      dets[jj])
        # The per-row checkmark column became the caption sentence "All rank,
        # multiplicity and sign checks passed in every row"; assert it.
        mis.check(f"{tag}: all checks passed", allok, True)

    # ---------------------------------------------------------------- 6 ----
    print()
    print("-" * 78)
    print(f"[6] TABLE S1 -- COMPLETE LATTICE DATA FOR J={jlabel(TABLE2_J)}, "
          f"k2/k1={TABLE2_RATIO}")
    print("-" * 78)
    print("     m    n    X    Z    p    q    f  obs  rank[P,Q]  rank B   det M")
    res2 = scan[(TABLE2_J, TABLE2_RATIO)]
    rows2 = sorted(res2["rows"], key=lambda r: (r["Z"], r["X"]))
    if len(man["tab2"]) != len(rows2):
        mis.check("Table S1: number of rows", len(man["tab2"]), len(rows2))
    for idx, r in enumerate(rows2):
        dets = [f"{d:.4f}" for d in sorted(r["dets"])]
        i2 = lambda v: int(round(v)) + 0            # kills the -0 of %+.0f
        print(f"  {i2(r['m']):>+4} {i2(r['n']):>+4} {i2(r['X']):>+4} "
              f"{i2(r['Z']):>+4} {r['p']:>4} {r['q']:>4} {r['f']:>4} "
              f"{r['ndoub']:>4} {r['rkPQ']:>10} {r['rkB']:>7}   "
              + ", ".join(dets))
        if idx >= len(man["tab2"]):
            continue
        mrow = man["tab2"][idx]
        tag = (f"Table S1 row {idx + 1} "
               f"(m,n)=({i2(r['m']):+d},{i2(r['n']):+d})")
        for key, val in (("m", i2(r["m"])), ("n", i2(r["n"])),
                         ("X", i2(r["X"])), ("Z", i2(r["Z"])),
                         ("p", r["p"]), ("q", r["q"]), ("f", r["f"]),
                         ("obs", r["ndoub"]), ("rkPQ", r["rkPQ"]),
                         ("rkB", r["rkB"])):
            mis.check(f"{tag}: {key}", mrow[key], val)
        mis.check(f"{tag}: det M", ", ".join(mrow["dets"]), ", ".join(dets))

    # ------------------------------------------------- fourth-order census --
    print()
    print("-" * 78)
    print("[7] ON-AXIS CENSUS UNDER THE FOURTH-ORDER TERM  C (S_+^4 + S_-^4)")
    print("-" * 78)
    cens = axis_census()
    for (axis, C), cnt in sorted(cens.items()):
        print(f"  h_{axis} axis, positive half, C = {C:9.1e} K :  "
              f"{cnt:3d} degeneracies, counted with multiplicity")
    print("  (J = 10, k1 = 0.338 K, k2 = 0.246 K; the quartic term is")
    print("   -C (S_+^4 + S_-^4) with C = 29 uK, the sign and value of Ref. [LiGarg];")
    print("   counted exactly as sign changes between the two sectors of the")
    print("   pi rotation about the field axis, grid-converged)")
    for key, want in (("census_z", 55), ("census_x", 31)):
        got = cens[("z", 29e-6)] if key == "census_z" else cens[("x", 29e-6)]
        mis.check(f"Sec. VII: {key} at C = 29 uK", man.get(key, want), got)

    # -------------------------------------------- Fe8 ground-pair fields --
    print()
    print("-" * 78)
    print("[8] GROUND-PAIR DPs OF Fe8 UNDER THE QUARTIC TERM")
    print("-" * 78)
    on, off = ground_pair_fe8()
    print("  on the positive h_x axis : "
          + ", ".join(f"{h:.3f}" for h in on) + "  K")
    for hx, hy in off:
        print(f"  off axis                : ({hx:.3f}, +/-{hy:.3f})  K")
    man_on = man.get("gp_on")
    if man_on is not None:
        mis.check("Sec. VII: number of surviving on-axis ground-pair DPs",
                  len(man_on), len(on))
        for k, (want, got) in enumerate(zip(man_on, on), 1):
            mis.check(f"Sec. VII: on-axis ground-pair field {k}",
                      f"{want:.3f}", f"{got:.3f}")
    man_off = man.get("gp_off")
    if man_off is not None:
        for k, (want, got) in enumerate(zip(man_off, off), 1):
            mis.check(f"Sec. VII: off-axis ground-pair field {k} (h_x)",
                      f"{want[0]:.3f}", f"{got[0]:.3f}")
            mis.check(f"Sec. VII: off-axis ground-pair field {k} (h_y)",
                      f"{want[1]:.3f}", f"{got[1]:.3f}")

    # ------------------------------- how the ground-pair DPs leave the axis --
    print()
    print("-" * 78)
    print("[9] HOW THE GROUND-PAIR DPs OF Fe8 LEAVE THE HARD AXIS AS C GROWS")
    print("-" * 78)
    events, s0 = fe8_axis_events()
    print("  at C = 0 all crossings on the positive h_x axis carry one charge")
    print("  (chi = -1, Result 4); for each pair that disappears as C grows:")
    for C_flip, C_ann, x, (hx, hy) in events:
        print(f"  one crossing reverses its charge at C = {C_flip * 1e6:8.5f} uK,"
              f" the pair annihilates")
        print(f"    at C = {C_ann * 1e6:8.5f} uK near h_x = {x:.4f} K "
              f"(delay {(C_ann - C_flip) * 1e6:.5f} uK); just after, a DP in")
        print(f"    the plane h_z = 0 sits at (h_x, h_y) = ({hx:.4f}, +/-{hy:.4f}) K")
    print("  (J = 10, k1 = 0.338 K, k2 = 0.246 K, quartic term -C (S_+^4 + S_-^4);")
    print("   charges from the sign of slope * Im(<+|S_y|-><-|S_z|+>) between the")
    print("   two sectors of the pi rotation about x)")
    mis.check("Sec. VII: number of pairs leaving the positive hard axis",
              len(man.get("c_flip", [None] * 3)), len(events))
    if "c_flip" in man:
        for k, (want, (C_flip, _, _, _)) in enumerate(zip(man["c_flip"], events), 1):
            mis.check(f"Sec. VII: C of charge reversal {k} (uK)", want,
                      f"{C_flip * 1e6:.{len(want.split('.')[1])}f}")
    if "c_delay" in man:
        worst = max(C_ann - C_flip for C_flip, C_ann, _, _ in events) * 1e6
        mis.check("Sec. VII: delay from emission to annihilation below "
                  f"{man['c_delay']} uK", f"< {man['c_delay']}", f"{worst:.5f}",
                  ok=worst < man["c_delay"])

    # ---------------------------------------------------------------- v ----
    print()
    print("=" * 78)
    if mis:
        print(f"VERDICT: {len(mis)} MISMATCH(ES) between the manuscript and "
              "this computation")
        print("=" * 78)
        for what, manuscript, computed in mis:
            print(f"  {what}")
            print(f"      manuscript : {manuscript}")
            print(f"      computed   : {computed}")
        print("=" * 78)
        print(f"finished in {time.time() - t0:.1f} s -- FAILED")
        return 1
    print("VERDICT: no mismatches -- every number the manuscript asserts is "
          "reproduced")
    print("=" * 78)
    print(f"finished in {time.time() - t0:.1f} s")
    return 0


# --------------------------------------------------------------------------
# 9.  main
# --------------------------------------------------------------------------

def main(argv=None):
    t0 = time.time()
    plt.rcParams.update(PLT_STYLE)

    print("=" * 74)
    print("make_figures.py -- figures and verification, revised manuscript (v10)")
    print("=" * 74)
    print(f"host python   : {platform.python_version()} ({sys.executable})")
    print(f"numpy         : {np.__version__}")
    print(f"matplotlib    : {matplotlib.__version__}")
    print(f"output dir    : {OUT}")
    print(f"SOURCE_DATE_EPOCH pinned to {os.environ['SOURCE_DATE_EPOCH']} "
          "(byte-reproducible PDFs)")
    print("randomness    : none (no RNG is used anywhere in this file)")
    print(f"rank criterion: cutoff = {RANK_RTOL:.0e} * max(sigma_1, 1), "
          f"required margin {MIN_RANK_MARGIN:g} on each side")
    print(f"doublet crit. : relative level gap < {DOUBLET_RTOL:.0e}, "
          f"ambiguous band {GREY_ZONE} rejected")
    print("-" * 74)

    # ---- unit / invariance tests ------------------------------------------
    n_sci = test_sci()
    print(f"[test] sci()                       : {n_sci} assertions passed")
    err_basis = test_detM_invariance()
    print(f"[test] det M basis invariance      : max rel. dev {err_basis:.2e}")
    e1, e2 = test_scale_invariance()
    print(f"[test] det M scale invariance      : max abs dev {max(e1, e2):.2e} "
          "(det M depends on k2/k1 only)")

    # ---- figures -----------------------------------------------------------
    print("-" * 74)
    meta = {}
    p = os.path.join(OUT, "fig_chains.pdf")
    meta["fig_chains.pdf"] = fig_chains(p)
    print(f"[Fig. 1] fig_chains.pdf   J=3, (m,n)=(0,-1), schematic; "
          "circles = Ran, squares = ker")

    p = os.path.join(OUT, "fig_lattice.pdf")
    meta["fig_lattice.pdf"] = fig_lattice(p)
    print(f"[Fig. 2] fig_lattice.pdf  J=3, {meta['fig_lattice.pdf']['npts']} lattice "
          "points; solid = P_m loci, dashed = Q_n loci")

    p = os.path.join(OUT, "fig_cutline.pdf")
    meta["fig_cutline.pdf"] = fig_cutline(p)
    mc = meta["fig_cutline.pdf"]
    print(f"[Fig. 5] fig_cutline.pdf  J=3, k2/k1={mc['k2_over_k1']}, m={mc['m']:.0f}, "
          f"{mc['nw']} field points, p(N-p)={mc['crossings']} crossings; "
          "solid/dashed blocks")

    p = os.path.join(OUT, "fig_detM.pdf")
    meta["fig_detM.pdf"] = fig_detM(p)
    md = meta["fig_detM.pdf"]
    print(f"[Fig. 4] fig_detM.pdf     J = " + ", ".join(jlabel(J) for J in md["Js"])
          + f"; {len(md['ratios'])} anisotropy ratios "
          f"[{md['ratios'][0]}, {md['ratios'][-1]}]")
    print(f"         all lattice points checked; min rank margin "
          f"{md['min_rank_margin']:.2e}, max |[H,P]| {md['max_comm']:.2e}")

    p = os.path.join(OUT, "fig_stars.pdf")
    meta["fig_stars.pdf"] = fig_stars(p)
    print("[Fig. 3] fig_stars.pdf    schematic; filled vs open stars for the "
          "two pinned clusters")

    # ---- the numbers quoted in the text ------------------------------------
    print("-" * 74)
    print("quoted det M numbers (text, Sec. VI and Table II):")
    for q in study_quoted_numbers():
        print(f"  J={jlabel(q['J']):>3}  k2/k1={q['ratio']:<5} "
              f"min|det M| = {q['min_absdet']:.4e}   (quoted {q['quoted']:.1e})   "
              f"max det M = ${sci(q['max_det'])}$   "
              f"{q['ndp']} doublets, checks {'OK' if q['ok'] else 'FAILED'}")

    # ---- caption claim: monotonicity in J ----------------------------------
    print("-" * 74)
    ratios, table, viol, margin = study_monotonicity()
    print(f"Fig. 4 caption claim, min|det M| decreasing in J, tested on "
          f"{len(ratios)} ratios in [{ratios[0]:.2f}, {ratios[-1]:.2f}] "
          f"x 7 values of J:")
    if not viol:
        print("  holds at every sampled point")
    else:
        print(f"  FAILS at {len(viol)} of {len(ratios) * 6} (ratio, J-step) pairs:")
        for v in viol:
            print(f"    k2/k1={v[0]:.2f}  J={jlabel(v[1])} -> {jlabel(v[2])}: "
                  f"{v[3]:.4e} -> {v[4]:.4e}  (increase by "
                  f"{100 * (v[4] / v[3] - 1):.1f}%)")
    print(f"  worst rank-classification margin over the whole scan: {margin:.2e}")

    # ---- files -------------------------------------------------------------
    print("-" * 74)
    print("files written:")
    for name in FIGURES:
        fp = os.path.join(OUT, name)
        if not os.path.exists(fp) or os.path.getsize(fp) == 0:
            fail(f"{name} was not written or is empty")
        with open(fp, "rb") as fh:
            head = fh.read(5)
        if head != b"%PDF-":
            fail(f"{name} is not a PDF")
        print(f"  {name:<18} {os.path.getsize(fp):>7d} bytes   sha256:{sha256(fp)}")
    print("-" * 74)
    print(f"all checks passed in {time.time() - t0:.1f} s")
    return 0


def build_parser():
    p = argparse.ArgumentParser(
        prog="make_figures.py",
        description="Generate the five figures of the PRB manuscript "
                    "(default), or verify every number the manuscript asserts.")
    p.add_argument("--verify", action="store_true",
                   help="do not write any figure; instead reproduce and print "
                        "the sweep of Sec. VI, the diagnostics of SM Sec. S5 "
                        "and both tables, and cross-check them against the "
                        "numbers hard-coded in the .tex source")
    p.add_argument("--tex", default=os.path.join(OUT, TEX_NAME),
                   metavar="FILE",
                   help="manuscript to check against (read only, never "
                        f"modified); default {TEX_NAME}")
    return p


if __name__ == "__main__":
    _args = build_parser().parse_args()
    try:
        sys.exit(run_verify(_args.tex) if _args.verify else main())
    except (CheckFailure, AssertionError) as exc:
        # FAIL LOUDLY: no figure is worth shipping if a check broke.
        print("\n" + "!" * 74, file=sys.stderr)
        print("CHECK FAILED -- figures are NOT trustworthy:", file=sys.stderr)
        print(f"  {type(exc).__name__}: {exc}", file=sys.stderr)
        print("!" * 74, file=sys.stderr)
        sys.exit(1)
