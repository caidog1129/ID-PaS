"""
Create N independent SMip instances from one .mps/.lp file by giving each
x[p] and y[a] objective coefficient, and each weight coefficient of x[p],
its own random perturbation.

The variable and constraint pattern stays *identical* across instances,
but the optimal solution almost certainly changes.
"""
import gurobipy as gp
import numpy as np, os, re, itertools

# ─────────────────────────── user parameters ────────────────────────────
BASE_MPS   = "h/3reg_8fc_80vnd_50s0.55sp_20m0.6sp_10l0.65sp_55lmd_30s_15m_10l_0.5minP_1.2oor_4arcPaths_wgtRed1.25_loc139_it1.lp"       # ← your original file
# BASE_MPS = "MMCN-hard-BI/val/3reg_8fc_22vnd_15s0.55sp_5m0.6sp_2l0.65sp_17lmd_10s_5m_2l_0.5minP_1.2oor_4arcPaths_wgtRed1.0_loc105_it1.lp"
OUT_DIR    = "variants_h_test"    # where to write the new models
N_VARIANTS = 100
# SEED       = 42                     # global RNG seed
SEED = 0

os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- load template
tmpl = gp.read(BASE_MPS)
tmpl.update()

# variable buckets (order is fixed inside Gurobi)
x_vars = [v for v in tmpl.getVars() if v.VarName.startswith("x[")]
y_vars = [v for v in tmpl.getVars() if v.VarName.startswith("y[")]

# capacity rows look like  cap_(arcID)
cap_rows = [c for c in tmpl.getConstrs()
            if c.RHS == 0]

print(f"template: {len(x_vars):,} x-vars, {len(y_vars):,} y-vars, "
      f"{len(cap_rows):,} capacity rows")

# ---------------------------------------------------------------- variants
for k in range(1, N_VARIANTS + 1):
    m = tmpl.copy()

    cap_rows = [c for c in m.getConstrs()
            if c.RHS == 0]

    for row in cap_rows:
        # Each x[p] appears at most once per row; scan the row sparsely
        lexpr = m.getRow(row)

        for i in range(lexpr.size()):
            var = lexpr.getVar(i)
            coeff = lexpr.getCoeff(i)
            if coeff > 0:
                m.chgCoeff(row, var, coeff * rng.uniform(0.50, 1.50))

    m.update()
    out = os.path.join(OUT_DIR, f"smip_rand_{k:03d}.lp")
    m.write(out)
    print(f"✓ {out}")

print("All variants written.")
