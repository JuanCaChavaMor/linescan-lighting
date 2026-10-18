"""LaTeX tables of the report from results/.

    python report/make_tables.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", Path(__file__).resolve().parent / "tables"
OUT.mkdir(exist_ok=True)


def write(name, body):
    (OUT / f"{name}.tex").write_text(body)
    print("wrote", name)


def tab(spec, header, rows):
    return f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{header} \\\\\n\\midrule\n" + "\n".join(rows) + \
        "\n\\bottomrule\n\\end{tabular}\n"


lab = pd.read_csv(RES / "design_lab.csv")
p = pd.read_csv(RES / "design_pareto.csv")
rows = []
l0 = lab[lab.case == "lab set-up"].iloc[0]
rows.append(f"lab set-up & {l0.W:.0f} & {l0.z0:.0f} & {l0.theta:.0f} & {l0.U0:.2f} & {l0.CV:.3f} & {l0.eff_rel:.2f} & "
            f"{'yes' if l0.blocks else 'no'} \\\\")
t25 = lab[(lab.case == "lab, tilt sweep") & (lab.theta == 25)].iloc[0]
rows.append(f"lab, tilt 25$^\\circ$ & {t25.W:.0f} & {t25.z0:.0f} & {t25.theta:.0f} & {t25.U0:.2f} & {t25.CV:.3f} & "
            f"{t25.eff_rel:.2f} & {'yes' if t25.blocks else 'no'} \\\\ \\midrule")
for r in p[(p.m == 1) & p.feasible].itertuples():
    rows.append(f"optimum, efficiency $\\ge$ {r.min_eff_rel:.1f} & {r.W:.0f} & {r.z0:.0f} & {r.theta:.0f} & {r.U0:.3f} & "
                f"{r.CV:.3f} & {r.eff_rel:.2f} & no \\\\")
write("design", tab("lrrrrrrc", r"design & $W$ [mm] & $z_0$ [mm] & $\theta$ [$^\circ$] & $U_0$ & CV & efficiency & "
                                r"blocks view", rows))

c = pd.read_csv(RES / "calibration_per_image.csv", dtype={"image": str})
rows = []
for _, r in c.iterrows():
    j = r["rms_joint (without 00000037)"]
    rows.append(f"{r.image} & {100 * r.saturated:.0f} & {r.first_edge:.0f}--{r.last_edge:.0f} & {r.scale_mean:.3f} & "
                f"{r.rms_linear:.2f} & {r.rms_projective:.2f} & " + (f"{j:.3f}" if pd.notna(j) else "--") + " \\\\")
write("calibration", tab("lrcrrrr", r"image & saturated [\%] & edges [px] & scale [px/mm] & \multicolumn{3}{c}{RMS residual [px]} \\ "
                         r"\cmidrule(lr){5-7}" "\n& & & & linear & projective & + target", rows))
