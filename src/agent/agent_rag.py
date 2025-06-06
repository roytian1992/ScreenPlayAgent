"""
Agent RAG模块
负责实现检索增强生成功能
"""

import os
import json
from typing import Dict, List, Any, Optional, Union
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceBgeEmbeddings
import torch

class AgentRAG:
    """Agent RAG模块，实现检索增强生成功能"""
    
    def __init__(self, llm_interface, config: Dict[str, Any] = None):
        """
        初始化Agent RAG
        
        Args:
            llm_interface: 统一LLM接口
            config: 配置参数
        """
        self.llm_interface = llm_interface
        self.config = config or {}
        
        # 配置参数
        self.persist_directory = self.config.get("persist_directory", "chroma_db")
        self.collection_name = self.config.get("collection_name", "screenplay_knowledge")
        
        # 初始化嵌入模型
        self.embedding_model = self._init_embedding_model()
        
        # 初始化向量存储
        self.vector_store = self._init_vector_store()
        
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get("chunk_size", 1000),
            chunk_overlap=self.config.get("chunk_overlap", 200),
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def _init_embedding_model(self):
        """初始化嵌入模型"""
        embedding_type = self.config.get("embedding_type", "openai")
        
        if embedding_type == "openai":
            return OpenAIEmbeddings(
                openai_api_key=self.config.get("openai_api_key", os.environ.get("OPENAI_API_KEY", ""))
            )
        elif embedding_type == "bge":
            model_name = self.config.get("embedding_model", "BAAI/bge-large-zh-v1.5")
            device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            return HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
                query_instruction="为这个句子生成表示以用于检索相关文章："
            )
        else:
            # 可以扩展支持其他嵌入模型
            raise ValueError(f"不支持的嵌入模型类型: {embedding_type}")
    
    def _init_vector_store(self):
        """初始化向量存储"""
        # 确保目录存在
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # 检查是否已有向量存储
        try:
            return Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_model,
                persist_directory=self.persist_directory
            )
        except Exception as e:
            print(f"加载向量存储失败，创建新的向量存储: {e}")
            return Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_model,
                persist_directory=self.persist_directory
            )
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        添加文档到向量存储
        
        Args:
            documents: 文档列表
        """
        # 分割文档
        chunks = self.text_splitter.split_documents(documents)
        
        # 添加到向量存储
        self.vector_store.add_documents(chunks)
        
        # 持久化向量存储
        if hasattr(self.vector_store, "persist"):
            self.vector_store.persist()
    
    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> None:
        """
        添加文本到向量存储
        
        Args:
            texts: 文本列表
            metadatas: 元数据列表
        """
        # 创建文档
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        documents = [Document(page_content=text, metadata=metadata) for text, metadata in zip(texts, metadatas)]
        
        # 添加文档
        self.add_documents(documents)
    
    def add_screenplay(self, screenplay_data: Dict[str, Any]) -> None:
        """
        添加剧本数据到向量存储
        
        Args:
            screenplay_data: 剧本数据
        """
        # 提取剧本内容
        scenes = screenplay_data.get("scenes", [])
        
        # 创建文档
        documents = []
        for scene in scenes:
            scene_id = scene.get("scene_id", "")
            title = scene.get("title", "")
            content = scene.get("content", "")
            
            # 创建文档
            document = Document(
                page_content=content,
                metadata={
                    "scene_id": scene_id,
                    "title": title,
                    "type": "scene"
                }
            )
            
            documents.append(document)
        
        # 添加文档
        self.add_documents(documents)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        搜索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            相关文档列表
        """
        # 搜索相关文档
        docs = self.vector_store.similarity_search(query, k=top_k)
        
        # 转换为字典列表
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": getattr(doc, "score", None)
            })
        
        return results
    
    def query(self, query: str) -> str:
        """
        查询知识库
        
        Args:
            query: 查询文本
            
        Returns:
            查询结果
        """
        # 创建检索QA链
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm_interface.get_llm(),
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(),
            return_source_documents=True
        )
        
        # 运行查询
        result = qa_chain({"query": query})
        
        return result["result"]
    
    def clear(self) -> None:
        """清空向量存储"""
        if hasattr(self.vector_store, "_collection"):
            self.vector_store._collection.delete(where={})
        
        # 重新初始化向量存储
        self.vector_store = self._init_vector_store()


class TaskPlanner:
    """任务规划器，负责拆分子任务"""
    
    def __init__(self, llm_interface, template_manager, config: Dict[str, Any] = None):
        """
        初始化任务规划器
        
        Args:
            llm_interface: 统一LLM接口
            template_manager: 提示模板管理器
            config: 配置参数
        """
        self.llm_interface = llm_interface
        self.template_manager = template_manager
        self.config = config or {}
    
    def plan_task(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        规划任务
        
        Args:
            task: 任务数据
            
        Returns:
            子任务列表
        """
        # 获取任务规划模板
        template = self.template_manager.get_langchain_prompt("task_planning")
        if not template:
            return [{"error": "任务规划模板不存在"}]
        
        # 准备输入数据
        input_data = {
            "task": json.dumps(task)
        }
        
        # 生成任务规划
        from langchain.chains import LLMChain
        
        chain = LLMChain(llm=self.llm_interface.get_llm(), prompt=template)
        result = chain.run(**input_data)
        
        try:
            subtasks = json.loads(result)
            return subtasks
        except json.JSONDecodeError:
            return [{"error": "无法解析任务规划", "raw_result": result}]
    
    def create_outline(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建大纲
        
        Args:
            task: 任务数据
            
        Returns:
            大纲数据
        """
        # 获取大纲创建模板
        template = self.template_manager.get_langchain_prompt("outline_creation")
        if not template:
            return {"error": "大纲创建模板不存在"}
        
        # 准备输入数据
        input_data = {
            "task": json.dumps(task)
        }
        
        # 生成大纲
        from langchain.chains import LLMChain
        
        chain = LLMChain(llm=self.llm_interface.get_llm(), prompt=template)
        result = chain.run(**input_data)
        
        try:
            outline = json.loads(result)
            return outline
        except json.JSONDecodeError:
            return {"error": "无法解析大纲", "raw_result": result}
