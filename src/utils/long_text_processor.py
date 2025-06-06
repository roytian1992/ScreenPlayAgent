"""
长文本处理模块
负责处理长文本的分块、聚合和上下文管理
"""

import os
import json
from typing import Dict, List, Any, Optional, Callable, Union
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate

class LongTextProcessor:
    """长文本处理器，负责处理长文本的分块、聚合和上下文管理"""
    
    def __init__(self, config: Dict[str, Any], llm_interface):
        """
        初始化长文本处理器
        
        Args:
            config (dict): 配置参数
            llm_interface: 统一LLM接口
        """
        self.config = config
        self.llm_interface = llm_interface
        self.llm = llm_interface.get_llm()
        
        # 配置参数
        self.max_chunk_size = config.get("max_chunk_size", 4000)
        self.chunk_overlap = config.get("chunk_overlap", 500)
        self.use_global_summary = config.get("use_global_summary", True)
        self.max_summary_size = config.get("max_summary_size", 1000)
        
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def split_text(self, text: str) -> List[str]:
        """
        分割文本
        
        Args:
            text (str): 要分割的文本
            
        Returns:
            list: 文本块列表
        """
        return self.text_splitter.split_text(text)
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        分割文档
        
        Args:
            documents (list): 要分割的文档列表
            
        Returns:
            list: 分割后的文档列表
        """
        return self.text_splitter.split_documents(documents)
    
    def create_document_from_text(self, text: str, metadata: Dict[str, Any] = None) -> Document:
        """
        从文本创建文档
        
        Args:
            text (str): 文本内容
            metadata (dict, optional): 元数据
            
        Returns:
            Document: 文档对象
        """
        return Document(page_content=text, metadata=metadata or {})
    
    def create_documents_from_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> List[Document]:
        """
        从文本列表创建文档列表
        
        Args:
            texts (list): 文本列表
            metadatas (list, optional): 元数据列表
            
        Returns:
            list: 文档列表
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]
        return [Document(page_content=text, metadata=metadata) for text, metadata in zip(texts, metadatas)]
    
    def summarize_text(self, text: str) -> str:
        """
        生成文本摘要
        
        Args:
            text (str): 要摘要的文本
            
        Returns:
            str: 摘要文本
        """
        # 如果文本长度小于最大摘要大小，直接返回
        if len(text) <= self.max_summary_size:
            return text
        
        # 创建文档
        doc = self.create_document_from_text(text)
        
        # 创建摘要提示模板
        map_prompt_template = """请为以下文本生成简洁的摘要：
        
        {text}
        
        简洁摘要："""
        
        map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
        
        combine_prompt_template = """请将以下摘要合并为一个连贯的摘要：
        
        {text}
        
        连贯摘要："""
        
        combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])
        
        # 创建摘要链
        summary_chain = load_summarize_chain(
            llm=self.llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=False
        )
        
        # 生成摘要
        summary = summary_chain.run([doc])
        
        return summary
    
    def process_screenplay(self, screenplay_data: Dict[str, Any], process_func: Callable, **kwargs) -> Any:
        """
        处理剧本数据
        
        Args:
            screenplay_data (dict): 剧本数据
            process_func (callable): 处理函数，接受剧本块和其他参数，返回处理结果
            **kwargs: 传递给处理函数的其他参数
            
        Returns:
            处理结果
        """
        # 检查剧本是否需要分块处理
        scenes = screenplay_data.get("scenes", [])
        if len(scenes) <= 5:  # 如果场景数量较少，直接处理整个剧本
            return process_func(screenplay_data, **kwargs)
        
        # 分块处理
        results = []
        scene_chunks = self._split_scenes_into_chunks(scenes)
        
        # 处理每个块
        for i, chunk in enumerate(scene_chunks):
            # 创建块剧本数据
            chunk_screenplay = {
                "metadata": screenplay_data.get("metadata", {}),
                "scenes": chunk
            }
            
            # 如果使用全局摘要，添加剧本摘要
            if self.use_global_summary:
                chunk_screenplay["summary"] = self._generate_screenplay_summary(screenplay_data)
            
            # 处理块
            chunk_result = process_func(chunk_screenplay, **kwargs)
            results.append(chunk_result)
        
        # 合并结果
        return self._merge_results(results)
    
    def _split_scenes_into_chunks(self, scenes: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        将场景列表分割成多个块
        
        Args:
            scenes (list): 场景列表
            
        Returns:
            list: 场景块列表
        """
        # 计算每个场景的大致token数
        scene_tokens = []
        for scene in scenes:
            # 简单估算：每个单词约1.3个token
            content_tokens = len(scene.get("content", "").split()) * 1.3
            scene_tokens.append(content_tokens)
        
        # 分块
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for i, scene in enumerate(scenes):
            # 如果当前块为空或添加当前场景后不超过最大块大小，添加到当前块
            if not current_chunk or current_tokens + scene_tokens[i] <= self.max_chunk_size:
                current_chunk.append(scene)
                current_tokens += scene_tokens[i]
            else:
                # 当前块已满，创建新块
                chunks.append(current_chunk)
                current_chunk = [scene]
                current_tokens = scene_tokens[i]
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _generate_screenplay_summary(self, screenplay_data: Dict[str, Any]) -> str:
        """
        生成剧本摘要
        
        Args:
            screenplay_data (dict): 剧本数据
            
        Returns:
            str: 剧本摘要
        """
        # 提取剧本内容
        scenes = screenplay_data.get("scenes", [])
        scene_summaries = []
        
        for scene in scenes:
            scene_id = scene.get("scene_id", "")
            title = scene.get("title", "")
            content = scene.get("content", "")
            
            # 生成场景摘要
            scene_summary = f"场景{scene_id} - {title}: {content[:100]}..."
            scene_summaries.append(scene_summary)
        
        # 合并场景摘要
        screenplay_text = "\n\n".join(scene_summaries)
        
        # 生成剧本摘要
        return self.summarize_text(screenplay_text)
    
    def _merge_results(self, results: List[Any]) -> Any:
        """
        合并处理结果
        
        Args:
            results (list): 处理结果列表
            
        Returns:
            合并后的结果
        """
        # 如果结果是列表，直接合并
        if all(isinstance(result, list) for result in results):
            merged = []
            for result in results:
                merged.extend(result)
            return merged
        
        # 如果结果是字典，合并键值
        if all(isinstance(result, dict) for result in results):
            merged = {}
            for result in results:
                for key, value in result.items():
                    if key in merged:
                        # 如果值是列表，合并列表
                        if isinstance(merged[key], list) and isinstance(value, list):
                            merged[key].extend(value)
                        # 如果值是字典，递归合并
                        elif isinstance(merged[key], dict) and isinstance(value, dict):
                            merged[key].update(value)
                        # 否则，保留最后一个值
                        else:
                            merged[key] = value
                    else:
                        merged[key] = value
            return merged
        
        # 其他情况，返回最后一个结果
        return results[-1] if results else None
