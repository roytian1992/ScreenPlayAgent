# ScreenPlayAgent: 智能剧本一致性分析智能体

## 🌍 项目简介

本项目是一个基于大语言模型（LLM）的剧本叙事一致性分析智能体，旨在帮助影视创作者、编剧与文学编辑从结构层面深入理解剧本内容，自动识别剧本中的各类**叙事一致性问题**，并输出结构化的分析报告与上下文相关的修改建议。其核心目标是提升剧本的**逻辑连贯性、人物设定一致性、时空结构完整性**，并增强叙事因果链的合理性，助力剧本从创意草稿向高质量成片打磨的全过程。

与传统的文本润色或语法纠错工具不同，本系统聚焦于“叙事层级”的高阶内容分析，具备以下关键能力：

- 自动检测**时序错误、场景逻辑矛盾、事件前后因果断裂**；
- 识别**人物设定不一致、行为动机缺失、关系链混乱**等问题；
- 追踪和校验**时间线与空间跳转**，发现潜在“穿帮”与设定崩坏；
- 在剧本长度超出模型上下文窗口的情况下，仍能保持全局语义分析能力；
- 基于模块化提示和工具链机制，提出上下文感知的改写建议。

为应对剧本作为长文本、结构复杂、多角色、多线并行的特殊挑战，本系统引入了如下技术手段：

- 构建基于 LLM 的**语义工具链**（Toolchain），明确模块边界，提升多问题类型的可拆解性与可调试性；
- 使用**长文本处理与摘要策略**压缩跨场景信息，同时维持语义连贯；
- 结合 LangChain 框架，实现可编排、可回溯的多轮推理流程；
- 引入 MCP（Model Context Protocol）理念，构建智能体的“认知边界”与推理记忆结构，支持任务规划、反思与策略优化；
- 兼容 GPT-4o 与本地 Qwen3 等主流大语言模型，可根据场景需求灵活切换。

## 🤖 系统功能概览

- ✅ 自动识别叙事时序错误、场景跳跃矛盾
- ✅ 分析角色设定是否前后一致，行为是否合逻辑
- ✅ 构建事件因果链，识别断链、动机缺失等问题
- ✅ 结合时间线、空间信息，检测潜在穿帮
- ✅ 提出上下文相关的修改建议与重写建议
- ✅ 支持多模型后端（GPT-4o / Qwen3-14B / LLM + Embedding）
- ✅ 命令行与配置化运行，适配离线、本地部署场景

## 📚 系统架构与模块划分

ScreenPlayAgent 遵循四层模块架构设计：

1. **用户接口层**（CLI / API）
2. **Agent 控制层**（TaskPlanner / Executor / Memory / Reflection）
3. **工具链层**（结构提取、角色图谱、因果链、时空逻辑、一致性分析器）
4. **模型与数据层**（LLM 模型、Embedding 检索、Prompt 管理）

系统支持通过配置文件切换使用远程 OpenAI GPT-4o 接口或本地 Qwen3 模型与 bge 向量模型，满足不同部署需求。

## ⚙️ 安装与环境配置

### 1. 创建环境

```bash
conda env create -f environment.yml
conda activate screenplay
```

### 2. 本地模型与 Embedding 准备

确保你已准备以下模型文件并指定路径：

- 本地大模型：如 Qwen3-14B
- 向量模型：如 bge-large-zh-v1.5

---

## 🔧 使用说明

### 1. 命令行调用

```bash
python main.py ./test_data/sample_screenplay.json --config config_local.json
```

或者使用 GPT-4o：

```bash
python main.py ./test_data/sample_screenplay.json --config config.json
```

### 2. 配置文件结构

- `config.json`：使用 GPT-4o 和 OpenAI Embedding
- `config_local.json`：使用本地 Qwen3 和 bge 向量模型

配置项包括：

- LLM 模型路径、token、设备
- 文本分块参数（最大 chunk 长度、摘要策略）
- agent 记忆与 RAG 检索模块
- 输出目录与格式设置

### 3. 输出内容

分析结果默认保存在：

```
output/reports/{filename}.json
```

其中包含：

- 各模块返回的结构化 JSON
- 一致性分析结论摘要
- 修改建议（按场景编号标注）

## 📅 示例文件

- `test_data/sample_screenplay.json`：测试用样例剧本
- `prompts/`：提示词模板集合，支持多轮调用链（CoT / ReAct）
- `output/`：系统生成报告与中间知识图谱

## 🌟 开发者支持

如需自定义分析流程、扩展提示模板或集成前端系统，可参考以下模块：

- `src/models/llm_interface.py`：统一模型接口封装
- `src/tools/`：LangChain 工具定义
- `src/agent/`：调度器、任务追踪器、智能体主逻辑
- `src/utils/`：提示模板管理器、长文本切分工具