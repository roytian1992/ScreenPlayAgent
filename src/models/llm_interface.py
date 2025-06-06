"""
统一LLM接口模块
负责提供统一的语言模型接口，支持OpenAI API和本地模型
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
from qwen3 import Qwen3LLM

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMInterface:
    """统一LLM接口基类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化LLM接口
        
        Args:
            config (dict): 配置参数
        """
        self.config = config
    
    def get_llm(self):
        """
        获取LangChain LLM对象
        
        Returns:
            LLM: LangChain LLM对象
        """
        raise NotImplementedError
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        生成文本
        
        Args:
            prompt (str): 提示文本
            **kwargs: 其他参数
            
        Returns:
            str: 生成的文本
        """
        raise NotImplementedError
    
    def generate_with_json_output(self, prompt: str, json_schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        生成JSON格式输出
        
        Args:
            prompt (str): 提示文本
            json_schema (dict): JSON模式
            **kwargs: 其他参数
            
        Returns:
            dict: 生成的JSON对象
        """
        raise NotImplementedError

class OpenAILLM(LLMInterface):
    """OpenAI LLM接口"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化OpenAI LLM接口
        
        Args:
            config (dict): 配置参数
        """
        super().__init__(config)
        self.api_key = config.get("api_key", os.environ.get("OPENAI_API_KEY", ""))
        self.model_name = config.get("model_name", "gpt-4o")
        self.temperature = config.get("temperature", 0.2)
        self.max_tokens = config.get("max_tokens", 4000)
        
        # 初始化LangChain ChatOpenAI
        self._init_llm()
    
    def _init_llm(self):
        """初始化LangChain ChatOpenAI"""
        try:
            from langchain_openai import ChatOpenAI
            
            self.llm = ChatOpenAI(
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                openai_api_key=self.api_key
            )
            logger.info(f"初始化OpenAI LLM成功: {self.model_name}")
        except ImportError:
            logger.error("导入langchain_openai失败，请安装: pip install langchain-openai")
            raise
        except Exception as e:
            logger.error(f"初始化OpenAI LLM失败: {e}")
            raise
    
    def get_llm(self):
        """
        获取LangChain LLM对象
        
        Returns:
            ChatOpenAI: LangChain ChatOpenAI对象
        """
        return self.llm
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        生成文本
        
        Args:
            prompt (str): 提示文本
            **kwargs: 其他参数
            
        Returns:
            str: 生成的文本
        """
        try:
            from langchain_core.messages import HumanMessage
            
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"生成文本失败: {e}")
            return f"生成失败: {e}"
    
    def generate_with_json_output(self, prompt: str, json_schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        生成JSON格式输出
        
        Args:
            prompt (str): 提示文本
            json_schema (dict): JSON模式
            **kwargs: 其他参数
            
        Returns:
            dict: 生成的JSON对象
        """
        try:
            from langchain_core.output_parsers import JsonOutputParser
            from langchain_core.prompts import ChatPromptTemplate
            
            # 创建输出解析器
            parser = JsonOutputParser(pydantic_object=json_schema)
            
            # 创建提示模板
            prompt_template = ChatPromptTemplate.from_template(
                "{prompt}\n\n请以JSON格式输出，符合以下模式:\n{format_instructions}"
            )
            
            # 创建链
            chain = prompt_template | self.llm | parser
            
            # 运行链
            result = chain.invoke({
                "prompt": prompt,
                "format_instructions": parser.get_format_instructions()
            })
            
            return result
        except Exception as e:
            logger.error(f"生成JSON输出失败: {e}")
            return {"error": f"生成失败: {e}"}

class LocalLLM(LLMInterface):
    """本地LLM接口，支持Qwen3等本地模型"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化本地LLM接口
        
        Args:
            config (dict): 配置参数
        """
        super().__init__(config)

        self.model_path = config.get("model_path", "")
        self.model_name = config.get("model_name", "qwen3")
        self.device = config.get("device", "cuda:0")
        self.max_new_tokens = config.get("max_new_tokens", 4096)
        self.temperature = config.get("temperature", 0.2)
        
        # 初始化模型和分词器
        self._init_model()
        
        # 创建LangChain兼容的LLM对象
        self.llm = self._create_langchain_llm()
    
    def _init_model(self):
        """初始化模型和分词器"""
        try:
            # import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch  
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # 加载模型
            if self.device == "auto":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype="auto",
                    device_map="auto"
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype="auto"
                ).to(self.device)
            
            logger.info(f"初始化本地模型成功: {self.model_name} from {self.model_path}")

        except Exception as e:
            logger.error(f"初始化本地模型失败: {e}")
            raise
    
    def _create_langchain_llm(self):
        """创建LangChain兼容的LLM对象"""

        # 返回实例化的LLM对象
        if self.model_name == "qwen3":
            model = Qwen3LLM(
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature
            )
        else:
            raise ValueError(f"不支持的本地模型类型: {self.model_name}")

        return model
    
    def get_llm(self):
        """
        获取LangChain LLM对象
        
        Returns:
            LLM: LangChain兼容的LLM对象
        """
        return self.llm
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        生成文本
        
        Args:
            prompt (str): 提示文本
            **kwargs: 其他参数
            
        Returns:
            str: 生成的文本
        """
        try:
            return self.llm(prompt)
        except Exception as e:
            logger.error(f"生成文本失败: {e}")
            return f"生成失败: {e}"
    
    def generate_with_json_output(self, prompt: str, json_schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        生成JSON格式输出
        
        Args:
            prompt (str): 提示文本
            json_schema (dict): JSON模式
            **kwargs: 其他参数
            
        Returns:
            dict: 生成的JSON对象
        """
        try:
            from langchain.output_parsers import StructuredOutputParser
            from langchain.prompts import PromptTemplate
            
            # 创建输出解析器
            parser = StructuredOutputParser.from_json_schema(json_schema)
            format_instructions = parser.get_format_instructions()
            
            # 创建提示模板
            template = "{prompt}\n\n{format_instructions}\n"
            prompt_template = PromptTemplate(
                template=template,
                input_variables=["prompt"],
                partial_variables={"format_instructions": format_instructions}
            )
            
            # 生成文本
            formatted_prompt = prompt_template.format(prompt=prompt)
            output = self.generate(formatted_prompt, **kwargs)
            
            # 解析JSON输出
            try:
                return parser.parse(output)
            except Exception as e:
                logger.error(f"JSON解析错误: {e}")
                return {"error": "输出格式错误", "raw_output": output}
        except Exception as e:
            logger.error(f"生成JSON输出失败: {e}")
            return {"error": f"生成失败: {e}"}

class LLMFactory:
    """LLM工厂类"""
    
    @staticmethod
    def create_llm(config: Dict[str, Any]) -> LLMInterface:
        """
        创建LLM实例
        
        Args:
            config (dict): 配置参数
            
        Returns:
            LLMInterface: LLM接口实例
            
        Raises:
            ValueError: 不支持的模型类型
        """
        model_type = config.get("model_type", "openai")
        
        if model_type == "openai":
            return OpenAILLM(config)
        elif model_type == "local":
            return LocalLLM(config)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
