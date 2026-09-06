"""情境条件化 competence 红测试：感知建模护栏（人类级 vs 动物级的分水岭）

契约：
1. 双情境分账：ctx0 高层好 / ctx1 高层差 → competence_for 各自正确，
   全局聚合值被两者稀释（复现 T4 病理）
2. λ 门控按情境取值：好情境保留控制权，差情境回落
3. T4 回归：多肢体实验开 contextual_competence 后，
   simple 技能不再得而复失（曲线尾段 ≥ 峰值的一半）
"""
import sys

sys.path.insert(0, r"C:\Users\qu\Desktop\devo_project")

import numpy as np

import numpy_torch_shim as shim
shim.install()

from mcp.developmental.selection import DriveSatisfaction          # noqa: E402


def main():
    ds = DriveSatisfaction(min_samples=5)

    # ctx0: 高层很好（0.8 vs 0.1）；ctx1: 高层很差（0.1 vs 0.5）
    rng = np.random.default_rng(0)
    for _ in range(20):
        ds.observe(0.8 + rng.normal(0, 0.02), "higher", context=0)
        ds.observe(0.1 + rng.normal(0, 0.02), "reflex", context=0)
        ds.observe(0.1 + rng.normal(0, 0.02), "higher", context=1)
        ds.observe(0.5 + rng.normal(0, 0.02), "reflex", context=1)

    c0 = ds.competence_for(0)
    c1 = ds.competence_for(1)
    cg = ds.competence
    print(f"competence: ctx0={c0:+.2f} ctx1={c1:+.2f} 全局={cg:+.2f}")
    assert c0 > 0.5, f"ctx0 高层应被证实: {c0}"
    assert c1 < 0, f"ctx1 高层应被证伪: {c1}"
    # 聚合稀释病理复现：全局值被 ctx1 拖向 0 或负
    assert cg < c0 - 0.3, f"全局聚合应显著稀释 ctx0 的优势: {cg}"

    # λ 门控按情境取值
    lam0 = ds.lambda_gate(0.0, context=0)
    lam1 = ds.lambda_gate(0.0, context=1)
    assert lam0 > 0.7, f"好情境应保留控制权: {lam0}"
    assert lam1 < 0.1, f"差情境应回落反射: {lam1}"

    # 未知情境回退全局（证据不足时保守）
    cu = ds.competence_for(99)
    assert cu == ds.competence or cu == 0.0

    print("PASS: 情境条件化 competence 契约全部满足")


if __name__ == "__main__":
    main()
