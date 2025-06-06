"""
知识构建模块
负责构建角色知识图谱、时空关系模型和事件因果网络
"""

import os
import json
from typing import Dict, List, Any, Optional, Union
import networkx as nx
from ..utils.format_utils import correct_json_format

class KnowledgeBuilder:
    """知识构建器，负责构建角色知识图谱、时空关系模型和事件因果网络"""
    
    def __init__(self, llm_interface, template_manager, config: Dict[str, Any] = None):
        """
        初始化知识构建器
        
        Args:
            llm_interface: 统一LLM接口
            template_manager: 提示模板管理器
            config: 配置参数
        """
        self.llm_interface = llm_interface
        self.template_manager = template_manager
        self.config = config or {}
        self.output_dir = self.config.get("location", "./")

    
    def build(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建知识
        
        Args:
            input_data: 输入数据，包含剧本数据
            
        Returns:
            构建的知识
        """
        # 检查输入
        if "screenplay_data" not in input_data:
            raise ValueError("输入数据必须包含screenplay_data")
        
        screenplay_data = input_data["screenplay_data"]
        
        # 构建角色知识图谱
        character_graph = self._build_character_graph(screenplay_data)
        
        # 构建时空关系模型
        spacetime_model = self._build_spacetime_model(screenplay_data)
        
        # 构建事件因果网络
        event_causal_network = self._build_event_causal_network(screenplay_data)

        with open(os.path.join(self.output_dir, "character_graph.json"), 'w') as f:
            json.dump(character_graph, f, ensure_ascii=False, indent=2)

        with open(os.path.join(self.output_dir, "spacetime_model.json"), 'w') as f:
            json.dump(spacetime_model, f, ensure_ascii=False, indent=2)

        with open(os.path.join(self.output_dir, "event_causal_network.json.json"), 'w') as f:
            json.dump(event_causal_network, f, ensure_ascii=False, indent=2)

        
        # 返回构建的知识
        return {
            "character_graph": character_graph,
            "spacetime_model": spacetime_model,
            "event_causal_network": event_causal_network
        }
    
    def _build_character_graph(self, screenplay_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建角色知识图谱
        
        Args:
            screenplay_data: 剧本数据
            
        Returns:
            角色知识图谱
        """
        # 获取角色图谱构建模板
        template = self.template_manager.get_langchain_prompt("character_graph_building")
        if not template:
            return {"error": "角色图谱构建模板不存在"}
        
        # 提取所有角色
        characters = set()
        for scene in screenplay_data.get("scenes", []):
            if "characters" in scene and isinstance(scene["characters"], list):
                characters.update(scene["characters"])
        
        # 创建角色图谱
        character_graph = nx.Graph()
        
        # 添加角色节点
        for character in characters:
            character_graph.add_node(character)
        
        # 分析角色关系
        from langchain.chains import LLMChain
        
        # 准备输入数据
        input_data = {
            "screenplay": json.dumps(screenplay_data, ensure_ascii=False)
        }
        
        # 运行角色图谱构建
        chain = LLMChain(llm=self.llm_interface.get_llm(), prompt=template)
        
        result = chain.run(**input_data)
        # result = chain.invoke(input_data)

        result = correct_json_format(result)
        
        try:
            # 解析结果
            character_relations = json.loads(result)
            
            # 添加角色关系
            for relation in character_relations:
                if "source" in relation and "target" in relation and "relation" in relation:
                    character_graph.add_edge(
                        relation["source"],
                        relation["target"],
                        relation=relation["relation"]
                    )
            
            # 转换为字典表示
            character_graph_dict = {
                "nodes": [{"id": node, "name": node} for node in character_graph.nodes()],
                "edges": [{"source": u, "target": v, "relation": character_graph[u][v]["relation"]} for u, v in character_graph.edges()]
            }
            
            return character_graph_dict
        except json.JSONDecodeError:
            # 如果解析失败，返回原始结果
            return {"error": "无法解析角色图谱构建结果", "raw_result": result}
    
    def _build_spacetime_model(self, screenplay_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建时空关系模型
        
        Args:
            screenplay_data: 剧本数据
            
        Returns:
            时空关系模型
        """
        # 获取时空关系构建模板
        template = self.template_manager.get_langchain_prompt("spacetime_model_building")
        if not template:
            return {"error": "时空关系构建模板不存在"}
        
        # 分析时空关系
        from langchain.chains import LLMChain
        
        # 准备输入数据
        input_data = {
            "screenplay": json.dumps(screenplay_data, ensure_ascii=False)
        }
        
        # 运行时空关系构建
        chain = LLMChain(llm=self.llm_interface.get_llm(), prompt=template)

        result = chain.run(**input_data)

        result = correct_json_format(result)
        
        try:
            # 解析结果
            spacetime_model = json.loads(result)
            return spacetime_model
        except json.JSONDecodeError:
            # 如果解析失败，返回原始结果
            return {"error": "无法解析时空关系构建结果", "raw_result": result}
    
    def _build_event_causal_network(self, screenplay_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建事件因果网络
        
        Args:
            screenplay_data: 剧本数据
            
        Returns:
            事件因果网络
        """
        # 获取事件因果网络构建模板
        template = self.template_manager.get_langchain_prompt("event_causal_network_building")
        if not template:
            return {"error": "事件因果网络构建模板不存在"}
        
        # 分析事件因果关系
        from langchain.chains import LLMChain
        
        # 准备输入数据
        input_data = {
            "screenplay": json.dumps(screenplay_data, ensure_ascii=False)
        }
        
        # 运行事件因果网络构建
        chain = LLMChain(llm=self.llm_interface.get_llm(), prompt=template)

        result = chain.run(**input_data)

        result = correct_json_format(result)
        
        try:
            # 解析结果
            event_causal_network = json.loads(result)
            return event_causal_network
        except json.JSONDecodeError:
            # 如果解析失败，返回原始结果
            return {"error": "无法解析事件因果网络构建结果", "raw_result": result}
