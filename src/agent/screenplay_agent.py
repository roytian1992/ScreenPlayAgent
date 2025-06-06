"""
主Agent模块
负责协调各个功能模块，实现剧本叙事一致性校对功能
"""

import os
import json
import uuid
from typing import Dict, List, Any, Optional, Union

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

class ScreenplayConsistencyAgent:
    """剧本叙事一致性校对智能体"""
    
    def __init__(self, config_path: str = "config.json"):
        """
        初始化剧本叙事一致性校对智能体
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化组件
        self._init_components()
        
        # 初始化Agent
        self._init_agent()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置数据
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载配置失败: {e}")
            return {}
    
    def _init_components(self) -> None:
        """初始化组件"""
        # 导入组件
        from src.models.llm_interface import LLMFactory
        from src.utils.prompt_template_manager import PromptTemplateManager
        from src.utils.long_text_processor import LongTextProcessor
        from src.agent.agent_memory import AgentMemory, TaskTracker, ErrorCorrector
        from src.agent.agent_rag import AgentRAG, TaskPlanner
        from src.input_processor.input_processor import InputProcessor
        from src.knowledge_builder.knowledge_builder import KnowledgeBuilder
        from src.consistency_analyzer.consistency_analyzer import ConsistencyAnalyzer
        from src.output_generator.output_generator import OutputGenerator
        
        # 初始化LLM接口
        self.llm_factory = LLMFactory()
        self.llm_interface = self.llm_factory.create_llm(self.config.get("llm", {}))
        
        # 初始化提示模板管理器
        self.template_manager = PromptTemplateManager(self.config.get("templates_dir", "prompts"))
        
        # 初始化长文本处理器
        self.long_text_processor = LongTextProcessor(self.config.get("long_text_processor", {}), self.llm_interface)
        
        # 初始化Agent记忆
        self.agent_memory = AgentMemory(self.llm_interface, self.template_manager, self.config.get("agent_memory", {}))
        
        # 初始化任务追踪器
        self.task_tracker = TaskTracker(self.config.get("task_tracker", {}))
        
        # 初始化错误纠正器
        self.error_corrector = ErrorCorrector(self.llm_interface, self.template_manager, self.config.get("error_corrector", {}))
        
        # 初始化Agent RAG
        self.agent_rag = AgentRAG(self.llm_interface, self.config.get("agent_rag", {}))
        
        # 初始化任务规划器
        self.task_planner = TaskPlanner(self.llm_interface, self.template_manager, self.config.get("task_planner", {}))
        
        # 初始化输入处理器
        self.input_processor = InputProcessor(self.config.get("input_processor", {}))
        
        # 初始化知识构建器
        self.knowledge_builder = KnowledgeBuilder(self.llm_interface, self.template_manager, self.config.get("knowledge_builder", {}))
        
        # 初始化一致性分析器
        self.consistency_analyzer = ConsistencyAnalyzer(self.llm_interface, self.template_manager, self.long_text_processor, self.config.get("consistency_analyzer", {}))
        
        # 初始化输出生成器
        self.output_generator = OutputGenerator(self.llm_interface, self.template_manager, self.config.get("output_generator", {}))
    
    def _init_agent(self) -> None:
        """初始化Agent"""
        # 定义工具
        tools = [
            Tool(
                name="InputProcessor",
                func=self._process_input,
                description="处理剧本输入，解析剧本结构和内容"
            ),
            Tool(
                name="KnowledgeBuilder",
                func=self._build_knowledge,
                description="构建剧本知识，包括角色图谱、时空关系和事件因果网络"
            ),
            Tool(
                name="ConsistencyAnalyzer",
                func=self._analyze_consistency,
                description="分析剧本一致性，检测叙事问题"
            ),
            Tool(
                name="OutputGenerator",
                func=self._generate_output,
                description="生成分析报告和修改建议"
            ),
            Tool(
                name="TaskPlanner",
                func=self._plan_task,
                description="规划任务，拆分子任务"
            ),
            Tool(
                name="AgentRAG",
                func=self._query_knowledge,
                description="查询知识库，获取相关信息"
            ),
            Tool(
                name="ErrorCorrector",
                func=self._correct_errors,
                description="纠正错误，检查重复和冲突"
            )
        ]
        
        # 创建Agent记忆
        agent_memory = ConversationBufferMemory(memory_key="chat_history")
        
        # 获取Agent提示模板
        agent_prompt = self.template_manager.get_langchain_prompt("agent_prompt")
        if not agent_prompt:
            # 使用默认提示模板
            agent_prompt = PromptTemplate.from_template(
                """你是一个专业的剧本叙事一致性校对智能体，负责检测剧本中的各类叙事问题。
                
                当前任务: {task}
                
                可用工具:
                {tools}
                
                使用工具时，请使用以下格式:
                ```
                思考: 我需要思考如何解决这个问题
                行动: 工具名称
                行动输入: {{
                    "参数1": "值1",
                    "参数2": "值2"
                }}
                观察: 工具返回的结果
                ```
                
                历史对话:
                {chat_history}
                
                人类: {input}
                AI: """
            )
        
        # 创建Agent
        agent = create_react_agent(
            llm=self.llm_interface.get_llm(),
            tools=tools,
            prompt=agent_prompt
        )
        
        # 创建Agent执行器
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=agent_memory,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def _process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理输入
        
        Args:
            input_data: 输入数据
            
        Returns:
            处理结果
        """
        # 记录操作
        task_id = input_data.get("task_id", str(uuid.uuid4()))
        self.task_tracker.add_operation(task_id, {
            "type": "process_input",
            "input_data": input_data
        })
        
        # 处理输入
        try:
            result = self.input_processor.process(input_data)
            
            # 添加到RAG知识库
            if "screenplay_data" in result:
                self.agent_rag.add_screenplay(result["screenplay_data"])
            
            return result
        except Exception as e:
            error_msg = f"处理输入失败: {e}"
            print(error_msg)
            return {"error": error_msg}
    
    def _build_knowledge(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建知识
        
        Args:
            input_data: 输入数据
            
        Returns:
            构建结果
        """
        # 记录操作
        task_id = input_data.get("task_id", str(uuid.uuid4()))
        self.task_tracker.add_operation(task_id, {
            "type": "build_knowledge",
            "input_data": input_data
        })
        
        # 构建知识
        try:
            # 使用长文本处理器处理大型剧本
            if "screenplay_data" in input_data:
                result = self.long_text_processor.process_screenplay(
                    input_data["screenplay_data"],
                    self.knowledge_builder.build,
                    **{k: v for k, v in input_data.items() if k != "screenplay_data"}
                )
            else:
                result = self.knowledge_builder.build(input_data)
            
            return result
        except Exception as e:
            error_msg = f"构建知识失败: {e}"
            print(error_msg)
            return {"error": error_msg}
    
    def _analyze_consistency(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析一致性
        
        Args:
            input_data: 输入数据
            
        Returns:
            分析结果
        """
        # 记录操作
        task_id = input_data.get("task_id", str(uuid.uuid4()))
        self.task_tracker.add_operation(task_id, {
            "type": "analyze_consistency",
            "input_data": input_data
        })
        
        # 分析一致性
        try:
            # 使用长文本处理器处理大型剧本
            if "screenplay_data" in input_data and "knowledge" in input_data:
                result = self.long_text_processor.process_screenplay(
                    input_data["screenplay_data"],
                    self.consistency_analyzer.analyze,
                    knowledge=input_data["knowledge"],
                    **{k: v for k, v in input_data.items() if k not in ["screenplay_data", "knowledge"]}
                )
            else:
                result = self.consistency_analyzer.analyze(input_data)
            
            # 纠正错误
            if "issues" in result:
                # 检查重复
                duplicates = self.error_corrector.check_for_duplicates(result["issues"])
                
                # 检查冲突
                conflicts = self.error_corrector.check_for_conflicts(result["issues"])
                
                # 纠正问题
                result["issues"] = self.error_corrector.correct_issues(result["issues"], duplicates, conflicts)
                
                # 记录纠正结果
                result["corrections"] = {
                    "duplicates": duplicates,
                    "conflicts": conflicts
                }
            
            return result
        except Exception as e:
            error_msg = f"分析一致性失败: {e}"
            print(error_msg)
            return {"error": error_msg}
    
    def _generate_output(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成输出
        
        Args:
            input_data: 输入数据
            
        Returns:
            生成结果
        """
        # 记录操作
        task_id = input_data.get("task_id", str(uuid.uuid4()))
        self.task_tracker.add_operation(task_id, {
            "type": "generate_output",
            "input_data": input_data
        })
        
        # 生成输出
        try:
            result = self.output_generator.generate(input_data)
            return result
        except Exception as e:
            error_msg = f"生成输出失败: {e}"
            print(error_msg)
            return {"error": error_msg}
    
    def _plan_task(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        规划任务
        
        Args:
            input_data: 输入数据
            
        Returns:
            规划结果
        """
        # 记录操作
        task_id = input_data.get("task_id", str(uuid.uuid4()))
        self.task_tracker.add_operation(task_id, {
            "type": "plan_task",
            "input_data": input_data
        })
        
        # 规划任务
        try:
            # 创建大纲
            if input_data.get("create_outline", False):
                result = {
                    "outline": self.task_planner.create_outline(input_data)
                }
            else:
                # 规划子任务
                result = {
                    "subtasks": self.task_planner.plan_task(input_data)
                }
            
            return result
        except Exception as e:
            error_msg = f"规划任务失败: {e}"
            print(error_msg)
            return {"error": error_msg}
    
    def _query_knowledge(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        查询知识
        
        Args:
            input_data: 输入数据
            
        Returns:
            查询结果
        """
        # 记录操作
        task_id = input_data.get("task_id", str(uuid.uuid4()))
        self.task_tracker.add_operation(task_id, {
            "type": "query_knowledge",
            "input_data": input_data
        })
        
        # 记录检索操作
        query = input_data.get("query", "")
        self.agent_memory.record_retrieval(query, None)
        
        # 查询知识
        try:
            # 检查是否有相关经验
            relevant_experience = self.agent_memory.get_relevant_experience(query)
            
            # 如果有相关经验，优先使用经验
            if relevant_experience:
                result = {
                    "source": "experience",
                    "result": relevant_experience
                }
            else:
                # 否则查询知识库
                if "search" in input_data and input_data["search"]:
                    # 搜索相关文档
                    docs = self.agent_rag.search(query, top_k=input_data.get("top_k", 5))
                    result = {
                        "source": "search",
                        "result": docs
                    }
                else:
                    # 查询知识库
                    answer = self.agent_rag.query(query)
                    result = {
                        "source": "query",
                        "result": answer
                    }
            
            # 更新检索记录
            self.agent_memory.record_retrieval(query, result)
            
            return result
        except Exception as e:
            error_msg = f"查询知识失败: {e}"
            print(error_msg)
            return {"error": error_msg}
    
    def _correct_errors(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        纠正错误
        
        Args:
            input_data: 输入数据
            
        Returns:
            纠正结果
        """
        # 记录操作
        task_id = input_data.get("task_id", str(uuid.uuid4()))
        self.task_tracker.add_operation(task_id, {
            "type": "correct_errors",
            "input_data": input_data
        })
        
        # 纠正错误
        try:
            issues = input_data.get("issues", [])
            
            # 检查重复
            duplicates = self.error_corrector.check_for_duplicates(issues)
            
            # 检查冲突
            conflicts = self.error_corrector.check_for_conflicts(issues)
            
            # 纠正问题
            corrected_issues = self.error_corrector.correct_issues(issues, duplicates, conflicts)
            
            return {
                "original_count": len(issues),
                "corrected_count": len(corrected_issues),
                "duplicates": duplicates,
                "conflicts": conflicts,
                "corrected_issues": corrected_issues
            }
        except Exception as e:
            error_msg = f"纠正错误失败: {e}"
            print(error_msg)
            return {"error": error_msg}
    
    def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行智能体
        
        Args:
            task: 任务数据
            
        Returns:
            运行结果
        """
        # 生成任务ID
        task_id = str(uuid.uuid4())
        
        # 添加任务
        self.task_tracker.add_task(task_id, task)
        
        # 开始任务
        self.task_tracker.start_task(task_id)
        
        # 记录任务
        self.agent_memory.record_task(task)
        
        try:
            # 运行Agent
            result = self.agent_executor.run(
                task=json.dumps(task),
                input=f"请分析以下剧本的叙事一致性问题：{task.get('screenplay_path', '')}"
            )
            
            # 完成任务
            self.task_tracker.complete_task(task_id, result)
            
            # 总结经验
            self.agent_memory.summarize_experience()
            
            return {
                "task_id": task_id,
                "status": "completed",
                "result": result
            }
        except Exception as e:
            error_msg = f"运行智能体失败: {e}"
            print(error_msg)
            
            # 任务失败
            self.task_tracker.fail_task(task_id, error_msg)
            
            return {
                "task_id": task_id,
                "status": "failed",
                "error": error_msg
            }
