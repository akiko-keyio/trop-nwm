# 数据文件打包说明

## 问题描述

当 `trop-nwm` 作为库安装到其他项目时（例如通过 `pip install` 或 `uv add`），会出现找不到数据文件的错误：

```
File not found: Y:\trop-system\.venv\lib\reference\egm96-5.pgm
```

## 原因分析

这是因为：

1. **路径计算依赖 `__file__` 位置**：代码中使用 `Path(__file__).parent.parent.parent / "reference"` 来查找数据文件
2. **开发环境 vs 安装环境的差异**：
   - 开发环境：`__file__` = `y:\trop-nwm\src\trop_nwm\geoid.py`，正确指向项目根目录
   - 安装环境：`__file__` = `site-packages\trop_nwm\geoid.py`，错误地向上3层查找
3. **数据文件未被打包**：默认情况下，`reference/` 目录不会被包含在发布的 wheel 包中

## 解决方案

已实施以下修复：

### 1. 配置打包规则 (`pyproject.toml`)

```toml
[tool.hatchling.build.targets.wheel]
packages = ["src/trop_nwm"]

[tool.hatchling.build.targets.wheel.shared-data]
"reference" = "trop_nwm/data"
```

这会将 `reference/` 目录打包到 `site-packages/trop_nwm/data/` 中。

### 2. 双路径查找逻辑

修改了 `geoid.py` 和 `height_correction.py`，使其：
- 首先尝试从安装位置查找：`site-packages/trop_nwm/data/`
- 如果找不到，回退到开发位置：`项目根目录/reference/`

这样在开发和生产环境都能正常工作。

### 3. 数据文件复制

将 `reference/` 目录的内容复制到 `src/trop_nwm/data/`，确保在本地开发时也能找到文件。

## 验证方法

在其他项目中测试：

```bash
# 在 trop-system 或其他项目中
pip install -e path/to/trop-nwm

# 或使用 uv
uv add --editable path/to/trop-nwm
```

然后运行代码，应该不再出现文件找不到的错误。

## 维护注意事项

- 如果修改了 `reference/` 目录中的数据文件，需要同时更新 `src/trop_nwm/data/` 目录
- 可以使用以下命令同步：
  ```powershell
  Copy-Item -Path "reference\*" -Destination "src\trop_nwm\data\" -Recurse -Force
  ```
