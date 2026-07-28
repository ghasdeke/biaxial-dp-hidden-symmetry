# Hidden symmetry at the diabolical points of a biaxial spin — verification code

Code accompanying

> S. Ghassemi Tabrizi, *Hidden symmetry at the diabolical points of a biaxial spin*
> (submitted to Physical Review B, 2026).

There is one program. It generates the five figures of the article and, in
`--verify` mode, recomputes the numerical claims of the article **from the
definition of the model** and compares each one against the manuscript source,
exiting with an error on any discrepancy.

## The model

$$H(\mathbf h)=k_1S_x^2+k_2S_y^2-\mathbf h\cdot\mathbf S,\qquad k_1>k_2>0$$

for spin $J$, $N=2J+1$. With $c=\sqrt{k_2/k_1}$, $s=\sqrt{1-k_2/k_1}$ and
$Z_\pm=\pm sS_x+cS_z$, the diabolical points are the lattice

$$h_x=k_1s\,(m-n),\qquad h_z=-k_1c\,(m+n+1),\qquad h_y=0,$$

with $m,n\in\{-J,\dots,J-1\}$, and the multiplicity of a point is
$f(m,n)=\min\\{p,q,N-p,N-q\\}$ with $p=J+m+1$, $q=J+n+1$.

## Install

```
python3 -m pip install -r requirements.txt
```

Python 3.9 or newer. `numpy`, `scipy` and `matplotlib` are required; `sympy`
and `mpmath` are used for the exact symbolic and high-precision checks and are
skipped if absent.

## Run

**Figures.**

```
python3 make_figures.py
```

Writes `fig_chains.pdf`, `fig_lattice.pdf`, `fig_stars.pdf`, `fig_detM.pdf`
and `fig_cutline.pdf`, prints a SHA-256 for each and runs its own internal
consistency checks. Nothing random is used anywhere, so a second run
reproduces the same bytes. About 10 s.

**Verification.**

```
python3 make_figures.py --verify --tex /path/to/Biaxial_DP_PRB_v9.tex
```

Point `--tex` at the LaTeX source of the article, with the Supplemental Material
source in the same directory. Both are distributed as ancillary files with the
arXiv version of the paper. The script extracts every number the text asserts,
recomputes it, and compares. Exit status 0 and
`no mismatches` only if everything agrees; otherwise the manuscript value, the
computed value and the location are printed and the exit status is 1. About
20 s.

What is compared:

- $[H_{mn},P_m]=[H_{mn},Q_n]=0$ at every lattice point, for
  $J=1,\tfrac32,\dots,4$ and $k_2/k_1=0.05,0.3,0.9$;
- $\operatorname{rank}B_{mn}=f(m,n)$ and
  $\operatorname{rank}[P_m,Q_n]=2f(m,n)$ at every point;
- the observed multiplicities against $f(m,n)$, and that no eigenvalue has
  multiplicity above two;
- $\det M<0$ for every doublet, and the closed form of Eq. (26) against the
  directly computed determinant;
- every entry of Table I and of the complete $J=2$ table in the Supplemental
  Material;
- all singular-value thresholds and classification margins quoted in the paper;
- the on-axis census for the Fe₈ parameters with the fourth-order term,
  55 / 55 / 31, computed from sign changes between the two sectors of the π
  rotation about the field axis rather than from a gap threshold;
- the four surviving hard-axis fields and the three off-axis mirror pairs of
  ground-pair diabolical points.

Two numbers in the article are **not** covered and were checked separately: the
field spread of the split threefold point at $J=3$, and $\chi=-2$ in the merged
uniaxial limit.

## Checking that the checker is not vacuous

The verifier is meant to fail when the manuscript is wrong. Copy the `.tex`,
change any single number, and run `--verify` against the copy; it must report a
mismatch and exit 1.

```
cp Biaxial_DP_PRB_v9.tex doctored.tex
# change any single number in doctored.tex, then
python3 make_figures.py --verify --tex doctored.tex   # must exit 1
```

## Implementation notes

Spin operators are built in the $S_z$ eigenbasis with ascending $r$, and the
construction of $S_+$ is asserted at run time ($[S_z,S_+]=S_+$ and
$(S_++S_-)/2=S_x$): an earlier version had this wrong in a way that was silent.

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
