"""
主程序模块
剧本叙事一致性校对智能体的入口点
"""

import os
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"

# 禁用 LangChain tracing 和 analytics（包括 PostHog）
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_PROJECT"] = ""



from openai import OpenAI

import sys
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# 导入各模块
from src.models.llm_interface import LLMFactory
from src.input_processor.input_processor import InputProcessor
from src.knowledge_builder.knowledge_builder import KnowledgeBuilder
from src.consistency_analyzer.consistency_analyzer import ConsistencyAnalyzer
from src.output_generator.output_generator import OutputGenerator
from src.utils.prompt_template_manager import PromptTemplateManager
from src.utils.long_text_processor import LongTextProcessor
from src.agent.agent_memory import AgentMemory
from src.agent.agent_rag import AgentRAG
from src.agent.screenplay_agent import ScreenplayConsistencyAgent

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("screenplay_agent.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("screenplay_agent")

class ScreenplayConsistencyAgent:
    """剧本叙事一致性校对智能体"""
    
    def __init__(self, config_path: str = None):
        """
        初始化智能体
        
        Args:
            config_path (str, optional): 配置文件路径
        """
        # 加载配置
        self.config = self._load_config(config_path)
        api_key = self.config.get("llm", {}).get("api_key")
        os.environ["OPENAI_API_KEY"] = api_key
        
        # 初始化LLM接口
        self.llm_interface = LLMFactory.create_llm(self.config.get("llm", {}))
        
        # 初始化提示模板管理器
        self.template_manager = PromptTemplateManager(
            templates_dir=self.config.get("templates_dir", "prompts")
        )
        
        # 初始化长文本处理器
        self.text_processor = LongTextProcessor(
            config=self.config.get("text_processor", {}),
            llm_interface=self.llm_interface
        )
        
        # 初始化Agent记忆
        self.agent_memory = AgentMemory(
            llm_interface=self.llm_interface,
            template_manager=self.template_manager,
            config=self.config.get("agent_memory", {})
        )
        
        # 初始化Agent RAG
        self.agent_rag = AgentRAG(
            llm_interface=self.llm_interface,
            config=self.config.get("agent_rag", {})
        )
        
        # 初始化各模块
        self.input_processor = InputProcessor(
            config=self.config.get("input_processor", {})
        )
        
        self.knowledge_builder = KnowledgeBuilder(
            llm_interface=self.llm_interface,
            template_manager=self.template_manager,
            config=self.config.get("knowledge_builder", {})
        )
        
        self.consistency_analyzer = ConsistencyAnalyzer(
            llm_interface=self.llm_interface,
            template_manager=self.template_manager,
            long_text_processor=self.text_processor,
            config=self.config.get("consistency_analyzer", {})
        )
        
        self.output_generator = OutputGenerator(
            llm_interface=self.llm_interface,
            template_manager=self.template_manager,
            config=self.config.get("output_generator", {})
        )
        
        logger.info("剧本叙事一致性校对智能体初始化完成")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_path (str): 配置文件路径
            
        Returns:
            dict: 配置参数
        """
        default_config = {
            "llm": {
                "model_type": "openai",
                "model_name": "gpt-4o",
                "api_key": os.environ.get("OPENAI_API_KEY", ""),
                "temperature": 0.2
            },
            "templates_dir": "prompts",
            "output_dir": "output",
            "output_format": "json",
            "text_processor": {
                "max_chunk_size": 4000,
                "chunk_overlap": 500,
                "use_global_summary": True,
                "max_summary_size": 1000
            },
            "agent_memory": {
                "memory_file": "memory.json",
                "max_task_history": 100,
                "max_retrieval_history": 500
            },
            "agent_rag": {
                "index_file": "index.json",
                "chunk_size": 1000,
                "chunk_overlap": 200
            },
            "input_processor": {},
            "knowledge_builder": {},
            "consistency_analyzer": {},
            "output_generator": {}
        }
        
        # 如果没有提供配置文件，使用默认配置
        if not config_path:
            return default_config
        
        # 加载配置文件
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            # 合并默认配置和用户配置
            merged_config = default_config.copy()
            for key, value in config.items():
                if isinstance(value, dict) and key in default_config and isinstance(default_config[key], dict):
                    merged_config[key].update(value)
                else:
                    merged_config[key] = value
            
            return merged_config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return default_config
    
    def analyze_screenplay(self, input_path: str, output_path: str = None) -> Dict[str, Any]:
        """
        分析剧本
        
        Args:
            input_path (str): 输入文件路径
            output_path (str, optional): 输出文件路径
            
        Returns:
            dict: 分析报告
        """
        try:
            # 记录任务
            task = {
                "type": "analyze_screenplay",
                "input_path": input_path,
                "output_path": output_path,
                "timestamp": datetime.now().isoformat()
            }
            self.agent_memory.record_task(task)
            
            # 1. 加载和预处理剧本
            logger.info(f"开始加载和预处理剧本: {input_path}")
            input_data = {"screenplay_path": input_path}
            processed_data = self.input_processor.process(input_data)
            screenplay_data = processed_data["screenplay_data"]
            
            # 2. 构建知识模型
            logger.info("开始构建知识模型")
            knowledge = self.knowledge_builder.build({"screenplay_data": screenplay_data})
            
            # 3. 分析一致性问题
            logger.info("开始分析一致性问题")
            analysis_input = {
                "screenplay_data": screenplay_data,
                "knowledge": knowledge
            }
            analysis_result = self.consistency_analyzer.analyze(analysis_input)
            
            # 4. 生成报告
            logger.info("开始生成分析报告")
            output_input = {
                "issues": analysis_result["issues"],
                "screenplay_data": screenplay_data
            }
            report = self.output_generator.generate(output_input)
            
            # 5. 保存报告
            if not output_path:
                output_path = os.path.join(self.config.get("output_dir"), f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            logger.info(f"保存分析报告: {output_path}")
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            # 记录任务完成
            self.agent_memory.update_task_status(task, "completed", {
                "issue_count": analysis_result["issue_count"],
                "report_generated": True
            })
            
            logger.info("剧本分析完成")
            return report
        
        except Exception as e:
            logger.error(f"分析剧本失败: {e}")
            # 记录任务失败
            if 'task' in locals():
                self.agent_memory.update_task_status(task, "failed", {
                    "error": str(e)
                })
            raise
    
    def analyze_screenplay_with_long_text_processing(self, input_path: str, output_path: str = None) -> Dict[str, Any]:
        """
        使用长文本处理器分析剧本
        
        Args:
            input_path (str): 输入文件路径
            output_path (str, optional): 输出文件路径
            
        Returns:
            dict: 分析报告
        """
        try:
            # 记录任务
            task = {
                "type": "analyze_screenplay_with_long_text_processing",
                "input_path": input_path,
                "output_path": output_path,
                "timestamp": datetime.now().isoformat()
            }
            self.agent_memory.record_task(task)
            
            # 1. 加载和预处理剧本
            logger.info(f"开始加载和预处理剧本: {input_path}")
            input_data = {"screenplay_path": input_path}
            processed_data = self.input_processor.process(input_data)
            screenplay_data = processed_data["screenplay_data"]
            
            # 2. 使用长文本处理器构建知识模型
            logger.info("开始构建知识模型（使用长文本处理）")
            
            def build_knowledge_models_for_chunk(screenplay_chunk, **kwargs):
                return self.knowledge_builder.build({"screenplay_data": screenplay_chunk})
            
            knowledge = self.text_processor.process_screenplay(
                screenplay_data, build_knowledge_models_for_chunk
            )
            
            # 3. 使用长文本处理器分析一致性问题
            logger.info("开始分析一致性问题（使用长文本处理）")
            
            def analyze_consistency_for_chunk(screenplay_chunk, **kwargs):
                analysis_input = {
                    "screenplay_data": screenplay_chunk,
                    "knowledge": kwargs.get("knowledge", {})
                }
                return self.consistency_analyzer.analyze(analysis_input)
            
            analysis_result = self.text_processor.process_screenplay(
                screenplay_data, analyze_consistency_for_chunk, knowledge=knowledge
            )
            
            # 4. 生成报告
            logger.info("开始生成分析报告")
            output_input = {
                "issues": analysis_result["issues"],
                "screenplay_data": screenplay_data
            }
            report = self.output_generator.generate(output_input)
            
            # 5. 保存报告
            if output_path:
                output_path = os.path.join(self.config.get("output_dir"), f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            logger.info(f"保存分析报告: {output_path}")
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            # 记录任务完成
            self.agent_memory.update_task_status(task, "completed", {
                "issue_count": analysis_result["issue_count"],
                "report_generated": True
            })
            
            logger.info("剧本分析完成")
            return report
        
        except Exception as e:
            logger.error(f"分析剧本失败: {e}")
            # 记录任务失败
            if 'task' in locals():
                self.agent_memory.update_task_status(task, "failed", {
                    "error": str(e)
                })
            raise

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="剧本叙事一致性校对智能体")
    parser.add_argument("input", help="输入文件路径")
    parser.add_argument("-o", "--output", help="输出文件路径", default=None)
    parser.add_argument("-c", "--config", help="配置文件路径")
    parser.add_argument("-m", "--model", choices=["openai", "local"], help="模型类型")
    parser.add_argument("--long-text", action="store_true", help="使用长文本处理")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出详细日志")
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 创建输出目录
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # 初始化智能体
    
    agent = ScreenplayConsistencyAgent(args.config)
    
    # 如果指定了模型类型，覆盖配置
    if args.model:
        agent.config["llm"]["model_type"] = args.model
        agent.llm_interface = LLMFactory.create_llm(agent.config.get("llm", {}))
        
    
    # 分析剧本
    if args.long_text:
        report = agent.analyze_screenplay_with_long_text_processing(args.input, args.output)
    else:
        report = agent.analyze_screenplay(args.input, args.output)

    experience_summary = agent.agent_memory.summarize_experience() 
    agent.agent_memory.experience.update(experience_summary)
    agent.agent_memory._save_memory()
    
    # 输出简要结果
    print(f"\n剧本分析完成，共发现 {report['issue_count']} 个问题")
    if "issue_types" in report:
        for category, count in report["issue_types"].items():
            print(f"- {category}: {count} 个问题")
    
    if args.output:
        print(f"\n详细报告已保存至: {args.output}")

if __name__ == "__main__":
    main()
