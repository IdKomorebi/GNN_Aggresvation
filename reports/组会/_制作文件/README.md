# 制作与检查

面向使用者的入口在上一级 `README.md`。本目录只保存制作文件，六份成品在各次汇报的 `V1`。

- `ppt_01_02.py`、`ppt_03_04.py`、`ppt_05_06.py`：逐页内容、原图路径、显示裁剪、备注。
- `build_pptx.py`：使用 python-pptx 生成真正的可编辑 PowerPoint。显式白底；图片按原字节嵌入。
- `replot_123_panel.py`：唯一的 Matplotlib 重绘，读取 123 号原 CSV，将原合并图 B 单独画出。
- `check_pptx.py`：核对页数、白底、备注、图片 SHA256 和 V2 空目录，并用 LibreOffice 转 PDF、生成逐页检查图。
- `ppt_manifest.json`、`source_manifest.json`、`VALIDATION.json`：成品索引、实际引用源文件及哈希、最终检查结果。
- `旧版HTML归档`：原 `reports/组会汇报`，仅作历史保留。

制作依赖：Python、python-pptx、Pillow、pandas、Matplotlib。检查另需 LibreOffice、Poppler 和本机 Noto CJK 字体。

本次使用 `/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python`。python-pptx 的临时依赖安装在 `/tmp/group-meeting-pptx-deps`，构建脚本会在它存在时加载；将来可在独立 Python 环境中安装这些依赖。

```bash
python replot_123_panel.py
python build_pptx.py
python check_pptx.py
```

构建会覆盖本次 V1 的生成文件。**下一轮 V2 请另设输出目录，不覆盖 V1。** 脚本不会训练模型，也不会更改编号实验目录。
