"""
Convert docs/paper/manuscript_v9.md into an IEEEtran journal manuscript (main.tex).

Handles the manuscript's own Markdown subset: headings, paragraphs, lists, tables with
a preceding "**Table N.** caption" paragraph, images, one code block, and author-year
citations, which become numbered IEEE citations in order of first appearance.
Body text goes through pandoc; tables, figures, headings and the bibliography are
emitted as raw LaTeX.

Usage (from docs/paper/ieee): python md2ieee.py ../manuscript_v9.md main.tex
"""
import re
import subprocess
import sys

SRC, OUT = sys.argv[1], sys.argv[2]

# ---------------------------------------------------------------- references
REFS = {
    "bischoff1995": "E. E. Bischoff and M. S. W. Ratcliff, ``Issues in the development of approaches to container loading,'' \\emph{Omega}, vol.~23, no.~4, pp.~377--390, 1995.",
    "blank2020": "J. Blank and K. Deb, ``pymoo: Multi-objective optimization in Python,'' \\emph{IEEE Access}, vol.~8, 2020, doi: 10.1109/ACCESS.2020.2990567.",
    "bortfeldt2013": "A. Bortfeldt and G. W\\\"ascher, ``Constraints in container loading---A state-of-the-art review,'' \\emph{Eur. J. Oper. Res.}, vol.~229, no.~1, pp.~1--20, 2013.",
    "ceschia2013": "S. Ceschia and A. Schaerf, ``Local search for a multi-drop multi-container loading problem,'' \\emph{J. Heuristics}, vol.~19, no.~2, pp.~275--294, 2013.",
    "cliff1993": "N. Cliff, ``Dominance statistics: Ordinal analyses to answer ordinal questions,'' \\emph{Psychol. Bull.}, vol.~114, no.~3, pp.~494--509, 1993.",
    "crainic2008": "T. G. Crainic, G. Perboli, and R. Tadei, ``Extreme point-based heuristics for three-dimensional bin packing,'' \\emph{INFORMS J. Comput.}, vol.~20, no.~3, pp.~368--384, 2008.",
    "davies1999": "A. P. Davies and E. E. Bischoff, ``Weight distribution considerations in container loading,'' \\emph{Eur. J. Oper. Res.}, vol.~114, no.~3, pp.~509--527, 1999.",
    "deb2000": "K. Deb, ``An efficient constraint handling method for genetic algorithms,'' \\emph{Comput. Methods Appl. Mech. Eng.}, vol.~186, no.~2--4, pp.~311--338, 2000.",
    "deb2002": "K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, ``A fast and elitist multiobjective genetic algorithm: NSGA-II,'' \\emph{IEEE Trans. Evol. Comput.}, vol.~6, no.~2, pp.~182--197, 2002.",
    "derrac2011": "J. Derrac, S. Garc\\'{\\i}a, D. Molina, and F. Herrera, ``A practical tutorial on the use of nonparametric statistical tests as a methodology for comparing evolutionary and swarm intelligence algorithms,'' \\emph{Swarm Evol. Comput.}, vol.~1, no.~1, pp.~3--18, 2011.",
    "fasano2014": "G. Fasano, \\emph{Solving Non-standard Packing Problems by Global Optimization and Heuristics} (SpringerBriefs in Optimization). Cham, Switzerland: Springer, 2014.",
    "fasano2015": "G. Fasano, C. Lavopa, D. Negri, and M. C. Vola, ``CAST: A successful project in support of the International Space Station logistics,'' in \\emph{Optimized Packings with Applications} (Springer Optimization and Its Applications). Cham, Switzerland: Springer, 2015, pp.~87--117.",
    "friedrich2015": "T. Friedrich and M. Wagner, ``Seeding the initial population of multi-objective evolutionary algorithms: A computational study,'' \\emph{Appl. Soft Comput.}, vol.~33, pp.~223--230, 2015.",
    "goncalves2013": "J. F. Gon\\c{c}alves and M. G. C. Resende, ``A biased random key genetic algorithm for 2D and 3D bin packing problems,'' \\emph{Int. J. Prod. Econ.}, vol.~145, no.~2, pp.~500--510, 2013.",
    "ho2023": "K. Ho, ``Space logistics modeling and optimization: Review of the state of the art,'' arXiv:2306.01107, 2023, doi: 10.2514/1.A35982.",
    "holm1979": "S. Holm, ``A simple sequentially rejective multiple test procedure,'' \\emph{Scand. J. Statist.}, vol.~6, no.~2, pp.~65--70, 1979.",
    "martello2000": "S. Martello, D. Pisinger, and D. Vigo, ``The three-dimensional bin packing problem,'' \\emph{Oper. Res.}, vol.~48, no.~2, pp.~256--267, 2000.",
    "ozcan2016": "S. O. \\\"Ozcan, T. Dokero\\u{g}lu, A. Co\\c{s}ar, and A. Yaz\\i c\\i, ``A novel grouping genetic algorithm for the one-dimensional bin packing problem on GPU,'' in \\emph{Computer and Information Sciences (ISCIS 2016)}. Cham, Switzerland: Springer, 2016.",
    "parreno2008": "F. Parre\\~no, R. Alvarez-Valdes, J. M. Tamarit, and J. F. Oliveira, ``A maximal-space algorithm for the container loading problem,'' \\emph{INFORMS J. Comput.}, vol.~20, no.~3, pp.~412--422, 2008.",
    "ramos2016": "A. G. Ramos, J. F. Oliveira, J. F. Gon\\c{c}alves, and M. P. Lopes, ``A container loading algorithm with static mechanical equilibrium stability constraints,'' \\emph{Transp. Res. B, Methodol.}, vol.~91, pp.~565--581, 2016.",
    "salcedo2009": "S. Salcedo-Sanz, ``A survey of repair methods used as constraint handling techniques in evolutionary algorithms,'' \\emph{Comput. Sci. Rev.}, 2009, doi: 10.1016/j.cosrev.2009.07.001.",
    "whitley1994": "D. Whitley, V. S. Gordon, and K. Mathias, ``Lamarckian evolution, the Baldwin effect and function optimization,'' in \\emph{Parallel Problem Solving from Nature---PPSN III} (LNCS 866). Berlin, Germany: Springer, 1994.",
    "zhao2016": "X. Zhao, J. A. Bennell, T. Bekta\\c{s}, and K. Dowsland, ``A comparative review of 3D container loading algorithms,'' \\emph{Int. Trans. Oper. Res.}, vol.~23, no.~1--2, pp.~287--320, 2016.",
    "zitzler1999": "E. Zitzler and L. Thiele, ``Multiobjective evolutionary algorithms: A comparative case study and the strength Pareto approach,'' \\emph{IEEE Trans. Evol. Comput.}, vol.~3, no.~4, pp.~257--271, 1999.",
    "zitzler2003": "E. Zitzler, L. Thiele, M. Laumanns, C. M. Fonseca, and V. G. da Fonseca, ``Performance assessment of multiobjective optimizers: An analysis and review,'' \\emph{IEEE Trans. Evol. Comput.}, vol.~7, no.~2, pp.~117--132, 2003.",
}

# Exact citation strings in the manuscript -> replacement (narrative keeps the name).
CITES = [
    ("(Deb, 2000; Deb et al., 2002)", "[@deb2000,deb2002]"),
    ("NSGA-II (Deb et al., 2002)", "NSGA-II [@deb2002]"),
    ("(Deb et al., 2002)", "[@deb2002]"),
    ("(Gonçalves and Resende, 2013)", "[@goncalves2013]"),
    ("(Salcedo-Sanz, 2009)", "[@salcedo2009]"),
    ("(Friedrich and Wagner, 2015)", "[@friedrich2015]"),
    ("(Ramos et al., 2016)", "[@ramos2016]"),
    ("(Fasano, Lavopa, Negri and Vola, 2015)", "[@fasano2015]"),
    ("(Whitley, Gordon and Mathias, 1994)", "[@whitley1994]"),
    ("(Blank and Deb, 2020)", "[@blank2020]"),
    ("(Zitzler and Thiele, 1999)", "[@zitzler1999]"),
    ("(Zitzler, Thiele, Laumanns, Fonseca and da Fonseca, 2003)", "[@zitzler2003]"),
    ("Bischoff and Ratcliff (1995)", "Bischoff and Ratcliff [@bischoff1995]"),
    ("Bortfeldt and Wäscher (2013)", "Bortfeldt and Wäscher [@bortfeldt2013]"),
    ("Davies and Bischoff (1999)", "Davies and Bischoff [@davies1999]"),
    ("Ramos, Oliveira, Gonçalves and Lopes (2016)", "Ramos et al. [@ramos2016]"),
    ("Ceschia and Schaerf (2013)", "Ceschia and Schaerf [@ceschia2013]"),
    ("Zhao, Bennell, Bektaş and Dowsland (2016)", "Zhao et al. [@zhao2016]"),
    ("Martello, Pisinger and Vigo (2000)", "Martello et al. [@martello2000]"),
    ("Crainic, Perboli and Tadei (2008)", "Crainic et al. [@crainic2008]"),
    ("Parreño, Alvarez-Valdes, Tamarit and Oliveira (2008)", "Parreño et al. [@parreno2008]"),
    ("Gonçalves and Resende (2013)", "Gonçalves and Resende [@goncalves2013]"),
    ("Fasano (2014)", "Fasano [@fasano2014]"),
    ("Ho (2023)", "Ho [@ho2023]"),
    ("Deb's (2000)", "Deb's [@deb2000]"),
    ("Salcedo-Sanz (2009)", "Salcedo-Sanz [@salcedo2009]"),
    ("Friedrich and Wagner (2015)", "Friedrich and Wagner [@friedrich2015]"),
    ("Derrac, García, Molina and Herrera (2011)", "Derrac et al. [@derrac2011]"),
    ("Holm's (1979)", "Holm's [@holm1979]"),
    ("Cliff's (1993)", "Cliff's [@cliff1993]"),
    ("Özcan, Dokeroğlu, Coşar and Yazıcı (2016)", "Özcan et al. [@ozcan2016]"),
]

# Characters the Times-like font lacks or that read better in math.
SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
SYMBOLS = {"≤": "$\\le$", "≥": "$\\ge$", "σ": "$\\sigma$", "λ": "$\\lambda$", "κ": "$\\kappa$",
           "ρ": "$\\rho$", "δ": "$\\delta$", "χ²": "$\\chi^2$", "Σ": "$\\Sigma$", "±": "$\\pm$",
           "≈": "$\\approx$", "→": "$\\rightarrow$", "∪": "$\\cup$", "⌈": "$\\lceil$", "⌉": "$\\rceil$",
           "₀": "$_0$", "kg⁻¹": "kg$^{-1}$", "ⁿ": "$^n$", "∈": "$\\in$", "α": "$\\alpha$", "…": "\\ldots{}", "₁": "$_1$", "₂": "$_2$", "ᵢ": "$_i$", "ᴺ": "$^N$", "−": "$-$"}


def math_numbers(s):
    s = s.replace("kg⁻¹", "kg$^{-1}$")
    # "× 10⁻⁸" -> $\times 10^{-8}$
    s = re.sub(r"× 10([⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+)", lambda m: "$\\times 10^{" + m.group(1).translate(SUP) + "}$", s)
    s = s.replace("×", "$\\times$")
    for k, v in SYMBOLS.items():
        s = s.replace(k, v)
    return s


def inline(s):
    """Minimal inline Markdown -> LaTeX for table cells and captions."""
    out, parts = [], re.split(r"(`[^`]*`)", s)
    for p in parts:
        if p.startswith("`"):
            out.append("\\texttt{" + p[1:-1].replace("_", "\\_") + "}")
            continue
        p = p.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")
        p = re.sub(r"(?<![\\$])_", "\\_", p)
        p = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", p)
        p = re.sub(r"\*(.+?)\*", r"\\emph{\1}", p)
        p = re.sub(r"\[@([\w,]+)\]", lambda m: "\\cite{" + m.group(1) + "}", p)
        out.append(math_numbers(p))
    return "".join(out)


STORE = []


def raw(tex):
    """Raw LaTeX block: replaced by a placeholder paragraph, restored after pandoc."""
    STORE.append(tex)
    return f"\n\nRAWBLOCK{len(STORE) - 1}END\n\n"


def main():
    md = open(SRC, encoding="utf-8").read()
    for a, b in CITES:
        pass
        md = md.replace(a, b)
    left = re.findall(r"[A-Z][^\s()]* \((?:19|20)\d\d\)|\([^()]*?(?:19|20)\d\d\)", md.split("## References")[0])
    left = [x for x in left if "ISCIS" not in x and "2,100" not in x]
    assert not left, f"unconverted citations: {left}"

    title = re.search(r"^# (.+)$", md, re.M).group(1)
    abstract = md.split("## Abstract")[1].split("**Keywords:**")[0].strip()
    keywords = md.split("**Keywords:**")[1].split("\n")[0].strip()
    body = md.split("## 1. Introduction")[1].split("## References")[0]
    body = "## 1. Introduction" + body

    # the FAPS pseudocode block becomes a float; the reproducibility commands stay verbatim
    def code(m):
        txt = m.group(1)
        if txt.lstrip().startswith("FAPS("):
            return raw("\\begin{figure*}[!t]\n\\small\n\\begin{verbatim}\n" + txt.translate(str.maketrans({"σ": "s", "≥": ">=", "⌈": "ceil(", "⌉": ")", "∪": "U", "·": "*", "₀": "0", "←": "<-"})) +
                       "\\end{verbatim}\n\\caption{Feasibility-adaptive portfolio seeding (FAPS).}\\label{alg:faps}\n\\end{figure*}")
        return raw("{\\tiny\n\\begin{verbatim}\n" + txt + "\\end{verbatim}\n}")
    body = re.sub(r"^```\n(.*?)^```\s*$", code, body, flags=re.M | re.S)


    # cited keys in order of first appearance
    order = []
    for m in re.finditer(r"\[@([\w,]+)\]", abstract + body):
        for k in m.group(1).split(","):
            if k not in order:
                order.append(k)
    assert set(order) == set(REFS), set(REFS) ^ set(order)

    # headings -> raw latex with labels keyed by the original numbers
    def heading(m):
        level, text = len(m.group(1)), m.group(2).strip()
        num = re.match(r"^([0-9.]+|Appendix [A-Z]\.)\s+(.*)$", text)
        label = ""
        if num:
            key = num.group(1).rstrip(".")
            text = num.group(2)
            label = "\\label{sec:" + key.replace("Appendix ", "app") + "}"
        if text.startswith("Reproducibility Statement"):
            return raw("\\section*{Reproducibility Statement}")
        if num and num.group(1).startswith("Appendix"):
            return raw("\\appendix\n\\section{" + inline(text) + "}" + label)
        cmd = {2: "section", 3: "subsection", 4: "subsubsection"}[level]
        return raw(f"\\{cmd}{{{inline(text)}}}{label}")
    body = re.sub(r"^(#{2,4}) (.+)$", heading, body, flags=re.M)

    # tables: caption paragraph followed by a pipe table
    tab_no = [0]

    def table(m):
        cap, lines = m.group(1), [l for l in m.group(2).strip().split("\n")]
        n = re.match(r"\*\*Table (\d+)\.\*\*\s*", cap).group(1)
        cap = re.sub(r"\*\*Table \d+\.\*\*\s*", "", cap)
        rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines if not re.match(r"^\|[-| ]+\|$", l)]
        ncol = len(rows[0])
        spec = "l" + "c" * (ncol - 1)
        head = " & ".join("\\textbf{" + inline(c) + "}" for c in rows[0]) + " \\\\"
        body_rows = "\n".join(" & ".join(inline(c) for c in r) + " \\\\" for r in rows[1:])
        wide = ncol > 4 or sum(len(c) for c in rows[0]) > 60
        env = "table*" if wide else "table"
        width = "\\textwidth" if wide else "\\columnwidth"
        tex = (f"\\begin{{{env}}}[!t]\n\\caption{{{inline(cap)}}}\\label{{tab:{n}}}\n\\centering\\footnotesize\n"
               f"\\resizebox{{{width}}}{{!}}{{%\n\\begin{{tabular}}{{{spec}}}\n\\toprule\n{head}\n\\midrule\n"
               f"{body_rows}\n\\bottomrule\n\\end{{tabular}}}}\n\\end{{{env}}}")
        return raw(tex)
    body = re.sub(r"^(\*\*Table \d+\.\*\*[^\n]*)\n\n((?:\|[^\n]*\|\n?)+)", table, body, flags=re.M)

    # figures
    fig_no = [0]

    def figure(m):
        cap, path = m.group(1), m.group(2).replace(".png", ".pdf")
        fig_no[0] += 1
        return raw(f"\\begin{{figure*}}[!t]\n\\centering\n\\includegraphics[width=0.92\\textwidth]{{{path}}}\n"
                   f"\\caption{{{inline(cap)}}}\\label{{fig:{fig_no[0]}}}\n\\end{{figure*}}")
    body = re.sub(r"^!\[(.*?)\]\((.*?)\)\s*$", figure, body, flags=re.M)

    # cross-references
    def secref(m):
        words, nums = m.group(1), m.group(2)
        nums = re.sub(r"\d+(?:\.\d+)?", lambda k: "\\ref{sec:" + k.group(0) + "}", nums)
        return f"{words}~{nums}"
    body = re.sub(r"(Sections?) ((?:\d+(?:\.\d+)?)(?:(?:, | and |–)\d+(?:\.\d+)?)*)", lambda m: raw_inline(secref(m)), body)
    body = re.sub(r"Table (\d+)\b", lambda m: raw_inline(f"Table~\\ref{{tab:{m.group(1)}}}"), body)
    body = body.replace("Appendix A", raw_inline("Appendix~\\ref{sec:appA}"))
    body = re.sub(r"\[@([\w,]+)\]", lambda m: raw_inline("\\cite{" + m.group(1) + "}"), body)
    body = math_numbers_outside_code(body)

    latex_body = subprocess.run(["pandoc", "-f", "markdown+raw_tex+raw_attribute", "-t", "latex", "--wrap=preserve"],
                                input=body, capture_output=True, text=True, check=True).stdout
    abs_tex = subprocess.run(["pandoc", "-f", "markdown", "-t", "latex", "--wrap=preserve"],
                             input=math_numbers(re.sub(r"\[@([\w,]+)\]", r"\\cite{\1}", abstract)),
                             capture_output=True, text=True, check=True).stdout

    for i in range(len(STORE) - 1, -1, -1):
        latex_body = latex_body.replace(f"RAWBLOCK{i}END", STORE[i]).replace(f"RAWINL{i}END", STORE[i])
    assert "RAWBLOCK" not in latex_body and "RAWINL" not in latex_body
    bib = "\n".join(f"\\bibitem{{{k}}} {REFS[k]}" for k in order)
    tex = TEMPLATE.replace("@TITLE@", inline(title)).replace("@ABSTRACT@", abs_tex.strip()) \
        .replace("@KEYWORDS@", inline(keywords.rstrip("."))).replace("@BODY@", latex_body) \
        .replace("@BIB@", bib)
    open(OUT, "w", encoding="utf-8").write(tex)
    print(f"wrote {OUT}: {len(order)} references, {fig_no[0]} figures")


def raw_inline(tex):
    STORE.append(tex)
    return f"RAWINL{len(STORE) - 1}END"


def math_numbers_outside_code(s):
    return math_numbers(s)


TEMPLATE = r"""\documentclass[journal]{IEEEtran}
\usepackage{fontspec}
\setmainfont{TeX Gyre Termes}
\setmonofont{DejaVu Sans Mono}[Scale=0.85]
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{url}
\usepackage[hidelinks]{hyperref}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}

\begin{document}

\title{@TITLE@}

\author{Ronit and [Second Author]%
\thanks{Ronit is with the Department of Computer Science and Engineering, Amity University, Noida, Uttar Pradesh, India (e-mail: ronitm0108@gmail.com).}%
\thanks{[Second Author] is with [Department, Institution, City, Country] (e-mail: [email]).}%
\thanks{Code, instances and derived data are available in the project repository.}}

\markboth{Manuscript submitted for review}{Ronit and [Second Author]: Feasibility-Adaptive Portfolio Seeding for Spacecraft Payload Packing}

\maketitle

\begin{abstract}
@ABSTRACT@
\end{abstract}

\begin{IEEEkeywords}
@KEYWORDS@
\end{IEEEkeywords}

@BODY@

\begin{thebibliography}{99}
@BIB@
\end{thebibliography}

\end{document}
"""

if __name__ == "__main__":
    main()
