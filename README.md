# Hidden symmetry at the diabolical points of a biaxial spin — verification code

Code accompanying

> S. Ghassemi Tabrizi, *Hidden symmetry at the diabolical points of a biaxial spin*,
> [arXiv:2607.26308](https://arxiv.org/abs/2607.26308) (under review at Physical Review B, 2026).

There is one program. It generates the five figures of the article and, in
`--verify` mode, recomputes the quantitative claims listed below **from the
definition of the model** and compares them with the manuscript source,
exiting with an error on any discrepancy.

Version 1.1.0 refers to the revised manuscript (v10) submitted to Physical
Review B in response to the referees. All equation, section and table numbers
below and in the program's output follow that version. Version 1.0.0 refers to
the originally submitted manuscript (arXiv:2607.26308v1).

## The model

$$H(\mathbf h)=k_1S_x^2+k_2S_y^2-\mathbf h\cdot\mathbf S,\qquad k_1>k_2>0$$

for spin $J$, $N=2J+1$. With $c=\sqrt{k_2/k_1}$, $s=\sqrt{1-k_2/k_1}$ and
$Z_\pm=\pm sS_x+cS_z$, the diabolical points are the lattice

$$h_x=k_1s\\,(m-n),\qquad h_z=-k_1c\\,(m+n+1),\qquad h_y=0,$$

with $m,n\in\\{-J,\dots,J-1\\}$, and the multiplicity of a point is
$f(m,n)=\min\\{p,q,N-p,N-q\\}$ with $p=J+m+1$, $q=J+n+1$.

## Install

```
python3 -m pip install -r requirements.txt
```

Python 3.9 or newer. `numpy` and `matplotlib` are needed for the figures;
`scipy` and `mpmath` are needed in addition for `--verify`. The program does not
install anything itself; if `mpmath` is missing, `--verify` stops with a message.

## Run

**Figures.**

```
python3 make_figures.py
```

Writes `fig_chains.pdf`, `fig_lattice.pdf`, `fig_stars.pdf`, `fig_detM.pdf`
and `fig_cutline.pdf` into the directory of the program, prints a SHA-256 for
each and runs its own internal consistency checks. Nothing random is used, so
a second run with the same software versions reproduces the same bytes. The
figures in the article were produced with matplotlib 3.10.9; other versions
give the same data but may place axis ticks slightly differently. About 10 s.

**Verification.**

```
python3 make_figures.py --verify --tex /path/to/Biaxial_DP_PRB_v10.tex
```

Point `--tex` at the LaTeX source of the article; the source of the
Supplemental Material must lie in the same directory under the name
`<stem>_supp.tex` (here `Biaxial_DP_PRB_v10_supp.tex`). The LaTeX sources are
not part of this repository; they are available from the author on request.
The program extracts
the numbers listed below from the source, recomputes them and compares. Exit
status 0 and `no mismatches` only if everything agrees; otherwise the
manuscript value, the computed value and the location are printed and the
exit status is 1. About 20 s.

What is compared with the manuscript source:

- the sweep of Sec. VI ($J=1,\tfrac32,\dots,4$, $k_2/k_1=0.05,0.3,0.9$): the
  largest commutator norm $\lVert[H_{mn},P_m]\rVert$, $\lVert[H_{mn},Q_n]\rVert$,
  the relative error of the factorized formula Eq. (27) against the directly
  computed determinant, Eq. (24), and the 60-digit check of Eq. (27);
- every entry of Table II (global checks) and of Table S1 of the Supplemental
  Material (complete lattice data for $J=2$ at $k_2/k_1=0.3$), and the claims
  of the Table II caption;
- all thresholds and margins of the numerical method (Supplemental Material,
  Sec. S5): rank cutoff, degeneracy thresholds, required and worst margins,
  smallest retained singular value and its location, and the failure of the
  older absolute cutoff of $10^{-8}$ just outside the grid;
- the four surviving hard-axis fields and the three off-axis mirror pairs of
  ground-pair diabolical points of Fe₈ with the fourth-order term (Sec. VII);
- how the other six leave the hard axis as $C$ grows from 0 to 29 µK
  (Sec. VII): the three values of $C$ (2.43, 4.99 and 11.91 µK) at which one
  crossing of a merging pair reverses its charge and emits a mirror pair, and
  the bound (0.01 µK) on the delay until the pair annihilates on the axis.

For the last item the program follows the crossings of the two lowest levels
on the positive $h_x$ axis, where the π rotation about $x$ separates them into
two sectors. The charge of a crossing is the sign of
$\partial(E_+-E_-)/\partial h_x\cdot\mathrm{Im}\big(\langle+|S_y|-\rangle\langle-|S_z|+\rangle\big)$,
a gauge-invariant product. The program checks that all crossings carry one
charge at $C=0$ (Result 4), that a merging pair has equal charges well before
it merges and opposite charges just before, and that just after the
annihilation a degeneracy lies off the axis in the plane $h_z=0$. It locates
the charge reversal and the annihilation by bisection in $C$.

Checked at every lattice point of the sweep, without a number to compare:
$[H_{mn},P_m]=[H_{mn},Q_n]=0$, $\mathrm{rank}\\,B_{mn}=f(m,n)$,
$\mathrm{rank}\\,[P_m,Q_n]=2f(m,n)$, the observed multiplicities against
$f(m,n)$, no eigenvalue of multiplicity above two, and $\det M<0$ for every
doublet.

Compared with reference values stored in the program: the on-axis census for
the Fe₈ parameters with the fourth-order term (55 / 55 / 31 degeneracies),
computed from sign changes between the two sectors of the π rotation about the
field axis rather than from a gap threshold.

Not covered by the program, and checked separately: the largest discarded
singular value and the minima of $|\det M|$ quoted in the text of Sec. VI (the
latter are also entries of Table II); the separation of the
branches that leave multiple lattice points under the fourth-order term,
including the spread at $J=3$ (Sec. VII); the comparison of the Fe₈ fields with
the table of Li and Garg in their units; the limit $J=1$, $k_2\to k_1$
(Sec. V); and the ratio range of the singular-value bound and the values at
$J=4$ quoted in Secs. S3 and S5 of the Supplemental Material.

## Checking that the checker is not vacuous

The verifier is meant to fail when the manuscript is wrong. Copy the two
`.tex` files, change one of the compared numbers (for example an entry of
Table II, or the cutoff in Sec. VI) in the copy, and run `--verify` against it;
it must report a mismatch and exit 1.

```
cp Biaxial_DP_PRB_v10.tex doctored.tex
cp Biaxial_DP_PRB_v10_supp.tex doctored_supp.tex
# change one compared number in doctored.tex, then
python3 make_figures.py --verify --tex doctored.tex   # must exit 1
```

## Implementation notes

Spin operators are built in the $S_z$ eigenbasis with ascending $r$, and the
construction of $S_+$ is asserted at run time ($[S_z,S_+]=S_+$ and
$(S_++S_-)/2=S_x$).

The fourth-order term is written $-C(S_+^4+S_-^4)$ with $C>0$, the sign
convention of Li and Garg, PRB **83**, 132401 (2011). The census depends on it.

Ranks are taken with an explicit singular-value cutoff and the run aborts if
any classification margin falls below ten, so a rank is never assigned on a
close call.

## Contents

```
make_figures.py     the program
requirements.txt    dependencies
CITATION.cff        citation metadata
LICENSE             MIT
```

## Citing

Please cite the article. If you use the code itself, see `CITATION.cff` or the
"Cite this repository" button above.

## License

MIT, see `LICENSE`.
