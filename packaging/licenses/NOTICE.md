# 第三方许可证摘要

本仓库自有代码见根目录 `LICENSE`（MIT）。

便携包在构建时会把下列上游许可证复制到 `licenses/`：

| 组件 | 版本（见 versions.json） | 许可证 |
| --- | --- | --- |
| Python | 3.10.11 | PSF |
| COLMAP | 4.2.0 | BSD-3-Clause |
| GLOMAP | 1.2.0 | BSD-3-Clause |
| gsplat | 1.5.3+pt24cu124 官方 Windows wheel | Apache-2.0 |
| PyTorch / torchvision | 2.4.1+cu124 / 0.19.1+cu124 | BSD-style |
| pycolmap | 3.11.1 | BSD-3-Clause |
| SciPy | 1.14.1 | BSD-3-Clause |
| FastAPI / Starlette / Uvicorn / Pydantic | 见清单 | MIT |
| Pillow | 10.4.0 | HPND / MIT-CMU |
| NumPy | 1.26.4 | BSD-3-Clause |

NVIDIA 显卡驱动不属于本包，由其最终用户许可协议约束。

GLOMAP / COLMAP `global_mapper` 的全局求解主要在 CPU 上运行；不要将其宣传为全 GPU 重建。
