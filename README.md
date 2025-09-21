# MinerU Lite CLI

English | [简体中文](#简体中文)

A lightweight CLI tool for document processing using MinerU's VLM capabilities. This project provides a simplified command-line interface to convert PDFs and images to Markdown format.

## Features

- Convert PDF documents to Markdown
- Convert images (PNG/JPG) to Markdown
- Process specific pages with page range selection
- Generate debug PDFs with bounding box visualizations
- Interactive configuration wizard
- Local text-based PDF summaries (without backend calls)

## Prerequisites

Before using this tool, you need to set up a VLLM server as described in the [MinerU Extension Modules documentation](https://opendatalab.github.io/MinerU/quick_start/extension_modules/).

## Installation

```bash
pip install mineru-lite-cli
```

## Usage

### Basic Commands

#### Convert PDF to Markdown
```bash
mineru-lite run input.pdf
```

#### Convert Image to Markdown
```bash
mineru-lite run image.png
```

#### Process Specific Pages
```bash
# Pages 1-3
mineru-lite run input.pdf --pages 1-3

# Pages 1,2,5-6
mineru-lite run input.pdf --pages 1,2,5-6

# Pages 3 to end
mineru-lite run input.pdf --pages 3-

# Up to page 3
mineru-lite run input.pdf --pages -3
```

#### Generate Debug PDF
```bash
mineru-lite run input.pdf --debug-pdf
```

### Configuration

#### Interactive Setup
```bash
mineru-lite config wizard
```

#### Show Current Configuration
```bash
mineru-lite config show
```

#### Set Server URL
```bash
mineru-lite config set --server http://localhost:8000
```

#### Reset Configuration
```bash
mineru-lite config reset
```

### Summary Generation

Generate text-based PDF summary without backend calls:
```bash
mineru-lite summary input.pdf
```

### Command Line Options

#### Run Command
- `input`: Input file path (PDF or PNG/JPG)
- `--dpi`: Set DPI for PDF processing (default: 220)
- `--debug-pdf`: Generate debug PDF with visualization boxes
- `--server`: Override server URL
- `--configure`: Run interactive configuration wizard
- `--pages`: Page selection (1-based)

#### Summary Command
- `input`: Input PDF file path
- `--pages`: Page selection (1-based)

## Configuration

The tool uses a configuration file to store server settings. You can configure it using the interactive wizard or manually set the server URL.

## Acknowledgments

This project is built upon the excellent work of the [MinerU](https://github.com/opendatalab/MinerU) project by OpenDataLab. Special thanks to the MinerU team for their powerful document processing capabilities.

---

# 简体中文

一个基于 MinerU VLM 能力的轻量级命令行工具。此项目提供简化的命令行界面，用于将 PDF 和图像转换为 Markdown 格式。

## 功能特性

- 将 PDF 文档转换为 Markdown
- 将图像（PNG/JPG）转换为 Markdown
- 支持页面范围选择，处理特定页面
- 生成带有边界框可视化的调试 PDF
- 交互式配置向导
- 本地基于文本的 PDF 摘要（无需调用后端）

## 使用前提

在使用此工具之前，您需要根据 [MinerU 扩展模块文档](https://opendatalab.github.io/MinerU/quick_start/extension_modules/) 的说明设置 VLLM 服务器。

## 安装

```bash
pip install mineru-lite-cli
```

## 使用方法

### 基本命令

#### PDF 转 Markdown
```bash
mineru-lite run input.pdf
```

#### 图像转 Markdown
```bash
mineru-lite run image.png
```

#### 处理指定页面
```bash
# 页面 1-3
mineru-lite run input.pdf --pages 1-3

# 页面 1,2,5-6
mineru-lite run input.pdf --pages 1,2,5-6

# 从页面 3 到结尾
mineru-lite run input.pdf --pages 3-

# 到页面 3
mineru-lite run input.pdf --pages -3
```

#### 生成调试 PDF
```bash
mineru-lite run input.pdf --debug-pdf
```

### 配置管理

#### 交互式设置
```bash
mineru-lite config wizard
```

#### 显示当前配置
```bash
mineru-lite config show
```

#### 设置服务器地址
```bash
mineru-lite config set --server http://localhost:8000
```

#### 重置配置
```bash
mineru-lite config reset
```

### 摘要生成

无需调用后端，生成基于文本的 PDF 摘要：
```bash
mineru-lite summary input.pdf
```

### 命令行选项

#### Run 命令
- `input`: 输入文件路径（PDF 或 PNG/JPG）
- `--dpi`: 设置 PDF 处理的 DPI（默认：220）
- `--debug-pdf`: 生成带可视化框的调试 PDF
- `--server`: 覆盖服务器 URL
- `--configure`: 运行交互式配置向导
- `--pages`: 页面选择（1-based）

#### Summary 命令
- `input`: 输入 PDF 文件路径
- `--pages`: 页面选择（1-based）

## 配置

工具使用配置文件存储服务器设置。您可以使用交互式向导进行配置，或手动设置服务器地址。

## 致谢

本项目基于 OpenDataLab 的 [MinerU](https://github.com/opendatalab/MinerU) 项目的优秀工作构建。特别感谢 MinerU 团队提供的强大文档处理能力。