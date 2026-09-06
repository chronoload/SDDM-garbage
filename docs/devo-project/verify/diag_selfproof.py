"""检验「先自证，再连接」：能否让跨模态成边从有害变为有益"""
import numpy as np
from devo_control import agg

RHO = 0.025
print("=" * 94)
print(f"先自证再连接：self_proof 扫描（ρ={RHO}，跨模态成边 ON）")
print("=" * 94)
print(f"{'self_proof':>12}{'代价':>11}{'改进%':>9}{'通路':>8}{'有效':>7}"
      f"{'跨模态':>8}{'对角真':>8}{'对角噪':>8}")
print("-" * 94)
rows = []
for sp in (0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30):
    r = agg(rho=RHO, self_proof=sp)
    rows.append((sp, r))
    print(f"{sp:>12.2f}{r['cost_high'][0]:>11.4f}{r['improve'][0]*100:>8.1f}%"
          f"{r['n_paths'][0]:>8.1f}{r['n_eff'][0]:>7.1f}{r['cross'][0]:>8.1f}"
          f"{r['diag_real'][0]:>8.1f}{r['diag_noise'][0]:>8.1f}")

print()
print("参照基线：")
for name, kw in [
    ("仅对角（跨模态 OFF）", dict(rho=RHO, allow_cross=False, wire=True)),
    ("无成边", dict(rho=RHO, wire=False)),
    ("爬虫脑反射", dict(rho=RHO, wire=False)),
]:
    if name == "爬虫脑反射":
        r = agg(**kw)
        print(f"  {name:<22} 代价 {r['cost_reflex'][0]:.4f}（反射基线）")
    else:
        r = agg(**kw)
        print(f"  {name:<22} 代价 {r['cost_high'][0]:.4f}  "
              f"改进 {r['improve'][0]*100:.1f}%  通路 {r['n_paths'][0]:.1f}")

best = min(rows, key=lambda x: x[1]["cost_high"][0])
print()
print(f"最优 self_proof = {best[0]}  代价 {best[1]['cost_high'][0]:.4f}  "
      f"改进 {best[1]['improve'][0]*100:.1f}%  "
      f"有效通路 {best[1]['n_eff'][0]:.1f}")
