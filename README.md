# 航拍高斯工坊（AerialGS）

Windows 本地网页工具：用 HTML 界面控制照片整理、**COLMAP 4.2 CUDA** 特征处理、**COLMAP `global_mapper`（内置 GLOMAP）** 全局重建、**gsplat** GPU 训练，以及 Gaussian PLY **轻量预览**。地面绕拍、天空航拍、室内外序列都可以，不限无人机。

项目从一次摄影专业采风作业长出来：12GB 显存上必须分块、顺序匹配、空地分开训。来龙去脉、拍摄参数和踩坑见 [docs/PROJECT_HISTORY.md](docs/PROJECT_HISTORY.md)。

本仓库是**源码与可重复打包脚本**。COLMAP / PyTorch / gsplat 等数 GB 二进制**不会**进 git；便携包首次启动会按 `packaging/versions.json` 自动下载，失败可手动导入。

内置网页查看器是 **point sprite 轻量预览，不等同于精确椭球 Gaussian splat 渲染**。本工具**没有**内置多块 PLY 自动对齐合并。

## 界面里现在能做什么

- **低 / 中 / 高**三档预设，同时套训练和 COLMAP 相关参数；分类折叠；标题旁问号悬停说明。
- 一次加入多个源图文件夹（系统选文件夹窗口，可多选），支持拖入；训练中途还能继续往队列里加。失败自动跳过下一个。失败或中断的任务可续跑，队列里可删除。
- 默认每个文件夹**成功跑完**后清中间缓存，PLY 留在 `output/`；失败会留缓存以便续跑。
- 只绑定 `127.0.0.1`。源图可以含中文路径，不改原图。工作目录建议 ASCII 短路径。

## 目标电脑要求（便携包）

- Windows 10/11 **x64**
- **兼容的 NVIDIA 驱动**（唯一外部前置条件）
- **不需要** Python、Node、CUDA Toolkit，也**不需要联网**
- 不支持 AMD / Intel / 纯 CPU 训练
- 建议工作目录使用 ASCII 短路径，例如 `C:\AerialGS-Cache`（源图路径可以含中文；原图不会被修改）

解压后预计还需要大约 **8–20+ GB** 磁盘（包体本身数 GB，加上项目缓存与训练产物）。

## 重要：GLOMAP 不是全 GPU

| 阶段 | 设备 |
| --- | --- |
| COLMAP 特征提取 / 匹配 | **GPU** |
| `view_graph_calibrator`、GLOMAP / `global_mapper` 位姿与全局优化 | **主要 CPU** |
| 增量 `mapper`（仅回退） | **主要 CPU**，通常更慢 |
| gsplat 训练 | **GPU** |

界面会标注每个阶段的 CPU/GPU。不要把 GLOMAP 说成“全 GPU 重建”。

## 源码开发启动

需要本机已有 Python 3.10+（开发用）。这**不会**安装完整训练栈：

```powershell
cd F:\全域\AerialGS
powershell -ExecutionPolicy Bypass -File .\bootstrap.ps1 -InstallOnly
.\启动高斯工坊.bat
```

或直接：

```powershell
powershell -ExecutionPolicy Bypass -File .\bootstrap.ps1
```

浏览器打开 `http://127.0.0.1:8765/`（只绑定 loopback）。

开发环境没有 COLMAP/gsplat 时，诊断页会列出阻止项；可用假进程联调：

```powershell
.venv\Scripts\python.exe backend\run.py --fake --host 127.0.0.1
```

## 测试

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m compileall backend trainer
```

测试使用假进程，**不会**跑真实 COLMAP / gsplat 训练。

## 流水线行为

1. 扫描源图（文件名自然排序或 EXIF 时间），硬链接优先、失败则复制到工作目录的 ASCII 数字文件名。重新整理前会清掉工作目录里的旧帧。
2. COLMAP CUDA `feature_extractor`：`--ImageReader.single_camera 1`，GPU 开关为 `--FeatureExtraction.use_gpu`。
3. 匹配：自动 / 顺序 / 穷举；GPU 开关为 `--FeatureMatching.use_gpu`。自动模式下小图集走穷举。
4. 复制 `database.db` → `database_global.db` 后运行 `view_graph_calibrator`。
5. 默认 `colmap global_mapper`（COLMAP 4.2 内置 GLOMAP）。独立 **GLOMAP 1.2** 仅作为兼容适配入口。重跑 mapper 前会清空对应 sparse 输出。
6. 在多个稀疏模型中选择注册图像最多的一个；注册率过低则阻止训练。增量 mapper 只作回退。
7. 使用 COLMAP `image_undistorter` 一次性生成无畸变训练图和 PINHOLE 模型，再整理 `images/`、`images_2/`、`images_4/` 与 `sparse/0/`。
8. gsplat 训练预设：低 / 中 / 高。12GB 默认「中」为 **factor=4、15000 步、约 450 万高斯**。写出检查点与标准 Gaussian PLY。CUDA OOM 会给出可执行建议，而不是假装训完。

训练适配层使用 `pycolmap.Reconstruction`（失败时回退到本仓库解析器），**不会伪造训练**。场景会做相似变换归一化；DefaultStrategy 按 gsplat 1.5.3 传入 `scene_scale` 与 `absgrad`。

## 构建 NVIDIA 离线便携包

在一台已装 NVIDIA 驱动的开发机上运行（用于校验预编译 gsplat；**不需要** CUDA Toolkit / MSVC 来编译扩展）：

```powershell
powershell -ExecutionPolicy Bypass -File .\packaging\build_portable.ps1 -UpdateLock
```

脚本会：

- 按 `packaging/versions.json` 下载**固定版本**并校验 SHA256
- 组装可重定位目录 `dist/AerialGS-Portable/`（启动脚本只用包内相对路径；**不**把 `packaging/cache` 打进 app）
- 安装 Python 3.10.11 embed + **PyTorch 2.4.1+cu124** + **官方预编译** `gsplat-1.5.3+pt24cu124-cp310-cp310-win_amd64.whl` + COLMAP 4.2 CUDA + GLOMAP 1.2
- 包内写入第三方 `ARTIFACT_SHA256.txt`（不含 zip 自哈希）；外层 `dist/SHA256SUMS.txt` 另含 zip SHA256
- 生成 `dist/AerialGS-Portable-win64.zip`（仅在本脚本成功结束后才存在）

目标机解压后运行 `启动高斯工坊.bat`。首次启动只做本地自检，不联网。

固定版本见 `packaging/versions.json`，例如：

- COLMAP `4.2.0` CUDA zip `sha256:991e0bae403a496fcc4de0c1f1f428619bf12f8000978f77bc6799d9bfeac23e`
- GLOMAP `1.2.0` CUDA zip `sha256:7b32a8b0ecfaec28b82d5e0b7d40e38198259849ad1af8ba3e34dab301a3a773`
- Python embed `3.10.11` `sha256:608619f8619075629c9c69f361352a0da6ed7e62f83a0e19c63e0ea32eb7629d`
- gsplat 官方 Windows wheel `gsplat-1.5.3+pt24cu124-cp310-cp310-win_amd64.whl` `sha256:62fae62e2cf233233527ba890fd322825476118edd4bd27a4e6cb36b1723003e`
- PyTorch `2.4.1+cu124`、torchvision `0.19.1+cu124`

## 许可证

- 本仓库源码：MIT（`LICENSE`）
- 第三方：见 `packaging/licenses/NOTICE.md`。便携包构建时复制上游许可证。
- NVIDIA 驱动不包含在包内。
