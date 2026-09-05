# BUILD.md — 编译说明

## 编译方式

使用 **XeLaTeX** 编译（中文支持）。

```bash
cd presentation
xelatex -interaction=nonstopmode -halt-on-error project_overview.tex
xelatex -interaction=nonstopmode -halt-on-error project_overview.tex
```

需要编译**两遍**（第二遍用于稳定导航与页数信息；本演示文稿无参考文献，无需 BibTeX）。

若安装了 `latexmk`，也可用：

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error project_overview.tex
```

## 本机编译环境（实测成功）

- 编译器：MiKTeX 26.5，XeTeX 3.141592653-2.6（`xelatex`）
- 操作系统：Windows 11
- 关键宏包：`ctex`（Windows 字体集）、`beamer`、`tikz`、`booktabs`、`graphicx`
- 中文字体（ctex Windows 字体集，均已安装）：宋体（SimSun）、黑体（SimHei）、楷体（KaiTi）、仿宋（FangSong）

## 输出

- 成功生成 `project_overview.pdf`（11 页），无 Overfull 溢出、无缺图、无未定义引用。

## 注意事项

1. **Logo**：模板原引用 `pic/Westlake_University_Logo.png` 在项目中不存在，已删除封面 Logo 引用，未伪造学校 Logo。
2. **目录页**：Westlake 主题默认在每个 section 前插入目录页，已在主文件加载主题后通过 `\AtBeginSection[]{}` 取消，保持 PPT 简洁（共 11 页）。
3. **元信息**：作者/学校/指导教师为占位命令（`\ProjectAuthor` 等，默认 `待填写`），位于 `project_overview.tex` 开头，需手动填写。
4. **图片**：`assets/` 内 3 张预测示例图来自项目现有产物 `fire_detection/runs/detect/predict/`（已缩放，保持宽高比），使用相对路径引用。
5. **参考文献**：正文无 `\cite`，`ref.bib` 已清空（仅注释）。如后续需引用 YOLO/SigLIP/DINO 原论文，书目须从项目资料确认后补充。
6. **字体大小**：正文 `\small` 及以上，仅封面指导教师一行用 `\footnotesize`。
