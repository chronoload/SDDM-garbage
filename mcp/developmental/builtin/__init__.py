"""⚠ 本地测试桩 —— 不是交付物

原始项目中的 ``builtin`` 子包未包含在本次沙盒内。
它只被 ``__init__`` / ``cli`` / ``gui`` 引用，不进核心链路
（system / higher_brain / myelin / neuron 均不依赖它）。

这里给出最小实现，目的是让包能 import，从而跑通核心实验。
真实项目应使用原始实现。
"""
