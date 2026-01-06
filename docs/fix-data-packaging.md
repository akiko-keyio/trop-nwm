# 修复：数据文件打包问题

## 问题

在别的项目中使用 `trop-nwm` 库时，会报错：

```text
Error in trop_nwm_month: File not found:
Y:\trop-system\.venv\lib\reference\egm96-5.pgm
```

## 根本原因

1. **路径计算错误**：代码使用 `Path(__file__).parent.parent.parent / "reference"` 计算数据文件路径
   - 在开发环境：`__file__` → `y:\trop-nwm\src\trop_nwm\geoid.py`，路径正确
   - 在安装环境：`__file__` → `site-packages\trop_nwm\geoid.py`，路径错误指向外部

2. **数据文件未打包**：`reference/` 目录默认不会被打包到 wheel 中

## 修复内容

### 1. 修改了 3 个文件

#### `pyproject.toml`
添加了打包配置：
```toml
[tool.hatchling.build.targets.wheel.force-include]
"src/trop_nwm/data" = "trop_nwm/data"
```

#### `src/trop_nwm/geoid.py`
修改了路径查找逻辑，支持双路径：
- 优先查找：`site-packages/trop_nwm/data/egm96-5.pgm`（安装位置）
- 回退查找：`项目根目录/reference/egm96-5.pgm`（开发位置）

#### `src/trop_nwm/height_correction.py`
同样修改了默认模型文件路径查找逻辑

### 2. 复制了数据文件

将 `reference/` 的内容复制到 `src/trop_nwm/data/`，包括：
- `egm96-5.pgm` (18.7 MB)
- `ztdht_model_01.dat` (1.8 MB)
- `h.nc` (8.3 MB)

## 测试验证

```bash
uv sync
uv run python -c "from trop_nwm.geoid import GeoidHeight; g = GeoidHeight('egm96-5'); print('Success!')"
# 输出：Success! Geoid file loaded.
```

## 使用说明

现在在其他项目中安装 `trop-nwm` 时，数据文件会被正确打包和安装：

```bash
# 在任何项目中
pip install trop-nwm

# 或使用 editable 模式
pip install -e path/to/trop-nwm
```

不会再出现找不到 `egm96-5.pgm` 的错误。

## 维护注意

如果更新了 `reference/` 目录中的数据文件，记得同步到 `src/trop_nwm/data/`：

```powershell
Copy-Item -Path "reference\*" -Destination "src\trop_nwm\data\" -Recurse -Force
```
