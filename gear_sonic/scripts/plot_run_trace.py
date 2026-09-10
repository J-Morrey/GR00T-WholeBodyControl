import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

d = json.load(open("/tmp/run_trace.json"))
S, A = d["scratch"], d["any2any"]
g = lambda r, k: np.array([x[k] for x in r], float)
LR_MIN, LR_CFG, LR_MAX = 1e-5, 2e-5, 2e-4
C_S, C_A = "#c44e52", "#3465a4"
XMAX = 2700
BOX = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=.85)

def band(a):
    a.axvspan(0, 200, color=C_A, alpha=.07, lw=0)

fig, ax = plt.subplots(2, 2, figsize=(13.5, 9))
fig.suptitle("R1 from-scratch (success)  vs  G1$\\rightarrow$R1 Any2Any (collapse)          "
             "shaded band = the 200 iterations the Any2Any run lasted\n"
             "values sampled at checkpoint saves; LR is its value after the last minibatch of that iteration",
             fontsize=11)

# ---- A: learning rate ------------------------------------------------------
a = ax[0, 0]; band(a)
for lv, lab, st in [(LR_MAX, "adaptive_lr_max = 2e-4", ":"),
                    (LR_CFG, "actor_learning_rate = 2e-5", "--"),
                    (LR_MIN, "adaptive_lr_min = 1e-5  (floor)", "-")]:
    a.axhline(lv, color="0.55", ls=st, lw=1)
    a.text(XMAX*0.99, lv*1.06, lab, color="0.35", fontsize=8, ha="right", va="bottom", bbox=BOX)
a.plot(g(S, "step"), g(S, "lr"), "o-", color=C_S, label="from-scratch", ms=4)
a.plot(g(A, "step"), g(A, "lr"), "s-", color=C_A, label="Any2Any (LoRA)", ms=6)
a.set_yscale("log"); a.set_ylim(7e-6, 3.2e-4); a.set_xlim(0, XMAX)
a.set_xlabel("iteration"); a.set_ylabel("learning rate")
a.set_title("A.  KL-adaptive learning rate", loc="left", fontweight="bold")
a.legend(loc="center right", fontsize=9); a.grid(alpha=.25)
a.annotate("pinned at the FLOOR, 4/4 checkpoints\n$\\Rightarrow$ KL > 2$\\times$desired_kl on every update;\n"
           "the controller wants to brake and can't",
           xy=(210, 1e-5), xytext=(620, 1.15e-5), color=C_A, fontsize=8.5, bbox=BOX,
           arrowprops=dict(arrowstyle="->", color=C_A, lw=1.3))

# ---- B: exploration std ----------------------------------------------------
b = ax[0, 1]; band(b)
b.plot(g(S, "step"), g(S, "std_mean"), "o-", color=C_S, label="from-scratch  (trainable)", ms=4)
b.plot(g(A, "step"), g(A, "std_max"), "s-", color=C_A, label="Any2Any  (frozen at init_noise_std)", ms=6)
for lv, lab, st in [(0.5, "std_clamp_max", ":"), (0.05, "init_noise_std", "--")]:
    b.axhline(lv, color="0.55", ls=st, lw=1)
    b.text(XMAX*0.99, lv+0.012, lab, color="0.35", fontsize=8, ha="right", bbox=BOX)
b.set_xlabel("iteration"); b.set_ylabel(r"exploration $\sigma$")
b.set_xlim(0, XMAX); b.set_ylim(0, 0.57)
b.set_title("B.  Exploration std  —  the root cause", loc="left", fontweight="bold")
b.legend(loc="lower right", fontsize=9); b.grid(alpha=.25)
b.annotate("scratch self-regulates to $\\sigma\\approx0.46$:\nthe task WANTS 9$\\times$ more noise",
           xy=(1500, .451), xytext=(880, .245), color=C_S, fontsize=8.5, bbox=BOX,
           arrowprops=dict(arrowstyle="->", color=C_S, lw=1.3))
b.annotate("$\\sigma$ frozen at 0.05 $\\Rightarrow$ KL $\\propto 1/\\sigma^2$\ninflated $\\sim$80$\\times$ for the same $\\Delta\\mu$",
           xy=(205, .05), xytext=(430, .085), color=C_A, fontsize=8.5, bbox=BOX,
           arrowprops=dict(arrowstyle="->", color=C_A, lw=1.3))

# ---- C: reward -------------------------------------------------------------
c = ax[1, 0]; band(c)
c.axhline(S[-1]["reward"], color=C_S, ls=":", lw=1)
c.plot(g(S, "step"), g(S, "reward"), "o-", color=C_S, label="from-scratch", ms=4)
c.plot(g(A, "step"), g(A, "reward"), "s-", color=C_A, label="Any2Any", ms=6)
c.set_xlabel("iteration"); c.set_ylabel("mean episode reward")
c.set_xlim(0, XMAX); c.set_ylim(0.6, 5.9)
c.set_title("C.  Mean episode reward (last 100 episodes)", loc="left", fontweight="bold")
c.grid(alpha=.25); c.legend(loc="center right", fontsize=9)
c.annotate("Any2Any at iter 50 already beats\nfrom-scratch at iter 2600 ($\\sim$50$\\times$ fewer iters)\n"
           "$\\rightarrow$ the transferred prior WORKS",
           xy=(55, 4.61), xytext=(430, 5.05), color=C_A, fontsize=8.5, bbox=BOX,
           arrowprops=dict(arrowstyle="->", color=C_A, lw=1.3))
ci = inset_axes(c, width="38%", height="34%", loc="lower right", borderpad=1.6)
ci.plot(g(A, "step"), g(A, "reward"), "s-", color=C_A, ms=4)
ci.set_xlim(30, 220); ci.set_title("zoom: 0-200", fontsize=7.5, color=C_A, pad=2)
ci.tick_params(labelsize=6.5); ci.grid(alpha=.25)
for s in ci.spines.values(): s.set_color(C_A)

# ---- D: LoRA drift ---------------------------------------------------------
dd = ax[1, 1]
keys = sorted(A[0]["lora_rel"].keys(), key=lambda k: int(k.split(".")[-2]))
cm = plt.cm.viridis(np.linspace(0, .88, len(keys)))
for k, col in zip(keys, cm):
    y = np.array([r["lora_rel"][k] for r in A]) * 100
    dd.plot(g(A, "step"), y, "s-", ms=4, lw=1.4, color=col,
            label="g1_dyn Linear[%s]" % k.split(".")[-2])
dd.set_xlabel("iteration"); dd.set_ylabel(r"$\|(\alpha/r)\,BA\|_F\ /\ \|W\|_F$   [%]")
dd.set_title("D.  LoRA drift off the pretrained weights (Any2Any only)", loc="left", fontweight="bold")
dd.set_xlim(35, 215); dd.grid(alpha=.25); dd.legend(fontsize=7, ncol=2, loc="upper left")
dd.annotate("monotone, no sign of converging\n(doubles over 150 iterations)",
            xy=(200, max(A[-1]["lora_rel"].values())*100), xytext=(95, 0.30),
            fontsize=8.5, color="0.25", bbox=BOX,
            arrowprops=dict(arrowstyle="->", color="0.35", lw=1.3))

fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("any2any_lr_diagnosis.png", dpi=155)
print("ok")
