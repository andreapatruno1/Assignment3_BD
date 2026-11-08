"""Summary table and figures from results/raw.csv (and results/limits.csv).

Central value: median of the 5 measured repetitions; variability: IQR = Q3 - Q1 (numpy linear quantiles).
Speedup  S(p) = median Tseq(n) / median Tp(n), efficiency E(p) = S(p) / p   (medians, same n, same inputs).
Speedup band [Q1seq / Q3p, Q3seq / Q1p]: rough range from the quartiles (not a confidence interval).
cpu_over_wall = median process CPU time / median wall time = average number of busy threads.
Memory: peak_rss_MiB = VmHWM of the process; storage_MiB = 3 * 8 n^2 bytes (A, B, C), analytical.
Outputs: results/summary.csv, results/fig/time.png, results/fig/speedup.png,
         results/summary_check.csv (control experiment, results/raw_check.csv, all rounds pooled and per round)
"""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "fig")
HW = 2                                   # logical processors available (nproc), see results/environment.txt
MB = 1024 ** 2


def main():
    t, cpu, rss, chk = defaultdict(list), defaultdict(list), {}, defaultdict(set)
    with open(os.path.join(RES, "raw.csv")) as f:
        for r in csv.DictReader(f):
            assert r["method"] == "seq" or int(r["team"]) == int(r["p"]), r   # thread count really used
            k = (r["method"], int(r["n"]), int(r["p"]))
            t[k].append(float(r["time_s"]))
            cpu[k].append(float(r["cpu_s"]))
            rss[k] = max(rss.get(k, 0), int(r["peak_rss_kb"]) / 1024)
            chk[int(r["n"])].add(r["checksum"])
    for n, s in chk.items():
        assert len(s) == 1, f"different checksums at n={n}: {s}"                # same result for every p

    rows = []
    for (m, n, p), v in sorted(t.items(), key=lambda x: (x[0][1], x[0][0] != "seq", x[0][2])):
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        rows.append(dict(method=m, n=n, p=p, reps=len(v), median_s=med, q1_s=q1, q3_s=q3, iqr_s=q3 - q1,
                         min_s=min(v), max_s=max(v), cpu_median_s=np.median(cpu[(m, n, p)]),
                         cpu_over_wall=np.median(cpu[(m, n, p)]) / med,
                         peak_rss_MiB=rss[(m, n, p)], storage_MiB=3 * 8 * n * n / MB,
                         oversubscribed=int(m == "par" and p > HW)))
    seq = {r["n"]: r for r in rows if r["method"] == "seq"}
    for r in rows:
        b = seq.get(r["n"])
        if b:
            r["speedup"] = b["median_s"] / r["median_s"]
            r["speedup_lo"] = b["q1_s"] / r["q3_s"]
            r["speedup_hi"] = b["q3_s"] / r["q1_s"]
            r["efficiency"] = r["speedup"] / r["p"]
        else:
            r["speedup"] = r["speedup_lo"] = r["speedup_hi"] = r["efficiency"] = float("nan")
    with open(os.path.join(RES, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})

    os.makedirs(FIG, exist_ok=True)
    ps = sorted({r["p"] for r in rows if r["method"] == "par"})
    cmap = plt.get_cmap("viridis")
    col = {p: cmap(i / max(1, len(ps) - 1) * 0.9) for i, p in enumerate(ps)}

    # 1. time vs n
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for m, p, c, lab in [("seq", 1, "black", "seq (baseline)")] + [("par", p, col[p], f"par p={p}") for p in ps]:
        sel = [r for r in rows if r["method"] == m and r["p"] == p]
        x = [r["n"] for r in sel]
        y = [r["median_s"] for r in sel]
        e = [[r["median_s"] - r["q1_s"] for r in sel], [r["q3_s"] - r["median_s"] for r in sel]]
        ax.errorbar(x, y, yerr=e, marker="o", ms=3.5, capsize=2, color=c, lw=1.2, ls="--" if m == "seq" else "-", label=lab)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("matrix size n (n x n)")
    ax.set_ylabel("time per multiplication [s] (median, IQR bars)")
    ax.set_title("Runtime vs n, sequential and OpenMP (5 reps)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "time.png"), dpi=160)

    # 2. speedup vs p, one line per n
    ns = sorted(seq)
    ncol = {n: plt.get_cmap("plasma")(i / max(1, len(ns) - 1) * 0.85) for i, n in enumerate(ns)}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.axvspan(HW + 0.0, max(ps) + 0.5, color="0.9", label="oversubscribed (p > 2)")
    ax.plot([1, HW, max(ps)], [1, HW, HW], color="0.4", ls=":", label="ideal (min(p, 2))")
    for n in ns:
        sel = [r for r in rows if r["method"] == "par" and r["n"] == n]
        y = [r["speedup"] for r in sel]
        e = [[r["speedup"] - r["speedup_lo"] for r in sel], [r["speedup_hi"] - r["speedup"] for r in sel]]
        ax.errorbar([r["p"] for r in sel], y, yerr=e, marker="o", ms=3.5, capsize=2, color=ncol[n], label=f"n={n}")
    ax.set_xticks(ps)
    ax.set_xlim(0.7, max(ps) + 0.3)
    ax.set_xlabel("OpenMP threads p")
    ax.set_ylabel("speedup S(p) = Tseq / Tp (medians)")
    ax.set_title("Speedup vs threads (bars: quartile band)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "speedup.png"), dpi=160)

    for r in rows:
        print(f"{r['method']:4} n={r['n']:5} p={r['p']} med={r['median_s']:.5g} iqr={r['iqr_s']:.3g} "
              f"S={r['speedup']:.3g} [{r['speedup_lo']:.3g},{r['speedup_hi']:.3g}] E={r['efficiency']:.3g} "
              f"cpu/wall={r['cpu_over_wall']:.2f} rss={r['peak_rss_MiB']:.1f}")


def check():
    path = os.path.join(RES, "raw_check.csv")
    if not os.path.exists(path):
        return
    g = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            for rnd in (r["round"], "all"):
                g[(int(r["n"]), r["condition"], rnd)].append(float(r["time_s"]))
    with open(os.path.join(RES, "summary_check.csv"), "w") as f:
        f.write("n,condition,round,reps,median_s,q1_s,q3_s,iqr_s\n")
        for (n, c, rnd), v in sorted(g.items(), key=lambda x: (x[0][0], x[0][2] != "all", x[0][2], x[0][1])):
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            f.write(f"{n},{c},{rnd},{len(v)},{med:.6g},{q1:.6g},{q3:.6g},{q3 - q1:.6g}\n")
            if rnd == "all":
                print(f"check n={n} {c:12} reps={len(v)} median={med:.4g} IQR={q3 - q1:.3g}")


if __name__ == "__main__":
    main()
    check()
