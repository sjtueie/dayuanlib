# 发布指南

## 版本管理

本项目使用 **动态版本管理**，版本号直接从 git tag 读取，无需手动修改 `pyproject.toml`。

## 发布新版本

### 方法 1：使用发布脚本（推荐）

```bash
./scripts/release.sh 0.3.2
```

脚本会：
- ✅ 检查 tag 是否已存在
- ✅ 提示未提交的更改
- ✅ 创建带注释的 git tag
- ✅ 显示下一步操作指引

### 方法 2：手动创建 tag

```bash
# 创建 tag
git tag -a v0.3.2 -m "Release 0.3.2"

# 推送 tag
git push origin v0.3.2
```

## 自动发布流程

1. **推送 tag** → 触发 GitHub Actions
2. **构建包** → 从 git tag 自动读取版本号
3. **发布到 PyPI** → 使用 `PYPI_API_TOKEN`

## 版本号格式

- **标准版本**: `v0.3.2` → PyPI 显示为 `0.3.2`
- **开发版本**: 无 tag 时 → `0.3.2.dev5+g1234567`

## 查看当前版本

```bash
# 在本地查看（需要安装 hatch-vcs）
python -c "from importlib.metadata import version; print(version('dayuanlib'))"

# 从 git 推断版本
git describe --tags --abbrev=0
```

## 注意事项

- ⚠️ **不要**手动修改 `pyproject.toml` 中的版本号
- ✅ 版本号统一由 git tag 管理
- ✅ 确保 tag 格式为 `v*`（如 `v0.3.2`）
- ✅ 推送 tag 前确保代码已提交
