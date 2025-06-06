"""
一致性分析模块
负责检测剧本中的叙事一致性问题
"""

import os
import json
from typing import Dict, List, Any, Optional, Union

class ConsistencyAnalyzer:
    """一致性分析器，负责检测剧本中的叙事一致性问题"""
    
    def __init__(self, llm_interface, template_manager, long_text_processor, config: Dict[str, Any] = None):
        """
        初始化一致性分析器
        
        Args:
            llm_interface: 统一LLM接口
            template_manager: 提示模板管理器
            long_text_processor: 长文本处理器
            config: 配置参数
        """
        self.llm_interface = llm_interface
        self.template_manager = template_manager
        self.long_text_processor = long_text_processor
        self.config = config or {}
    
    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析剧本一致性
        
        Args:
            input_data: 输入数据，包含剧本数据和知识
            
        Returns:
            分析结果
        """
        # 检查输入
        if "screenplay_data" not in input_data:
            raise ValueError("输入数据必须包含screenplay_data")
        
        screenplay_data = input_data["screenplay_data"]
        knowledge = input_data.get("knowledge", {})
        
        # 分析多线叙事结构问题
        narrative_structure_issues = self._analyze_narrative_structure(screenplay_data, knowledge)
        
        # 分析时空逻辑一致性问题
        spacetime_issues = self._analyze_spacetime_consistency(screenplay_data, knowledge)
        
        # 分析角色一致性问题
        character_issues = self._analyze_character_consistency(screenplay_data, knowledge)
        
        # 分析潜在穿帮点问题
        continuity_issues = self._analyze_continuity_errors(screenplay_data, knowledge)
        
        # 分析因果链与剧情推进逻辑问题
        causal_issues = self._analyze_causal_logic(screenplay_data, knowledge)
        
        # 合并所有问题
        all_issues = []
        all_issues.extend(narrative_structure_issues)
        all_issues.extend(spacetime_issues)
        all_issues.extend(character_issues)
        all_issues.extend(continuity_issues)
        all_issues.extend(causal_issues)
        
        # 为每个问题添加唯一ID
        for i, issue in enumerate(all_issues):
            issue["id"] = f"issue_{i+1}"
        
        # 返回分析结果
        return {
            "issues": all_issues,
            "issue_count": len(all_issues),
            "issue_types": {
                "narrative_structure": len(narrative_structure_issues),
                "spacetime": len(spacetime_issues),
                "character": len(character_issues),
                "continuity": len(continuity_issues),
                "causal": len(causal_issues)
            }
        }
    
    def _analyze_narrative_structure(self, screenplay_data: Dict[str, Any], knowledge: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        分析多线叙事结构问题
        
        Args:
            screenplay_data: 剧本数据
            knowledge: 知识数据
            
        Returns:
            问题列表
        """
        # 获取多线叙事结构分析模板
        template = self.template_manager.get_langchain_prompt("narrative_structure_analysis")
        if not template:
            return [{"error": "多线叙事结构分析模板不存在"}]
        
        # 分析多线叙事结构
        from langchain.chains import LLMChain
        
        # 准备输入数据
        input_data = {
            "screenplay": json.dumps(screenplay_data, ensure_ascii=False),
            "knowledge": json.dumps(knowledge, ensure_ascii=False)
        }
        
        # 运行多线叙事结构分析
        chain = LLMChain(llm=self.llm_interface.get_llm(), prompt=template)
        result = chain.run(**input_data)
        
        try:
            # 解析结果
            issues = json.loads(result)
            
            # 添加问题类型
            for issue in issues:
                issue["type"] = "narrative_structure"
            
            return issues
        except json.JSONDecodeError:
            # 如果解析失败，返回错误
            return [{"type": "narrative_structure", "error": "无法解析多线叙事结构分析结果", "raw_result": result}]
    
    def _analyze_spacetime_consistency(self, screenplay_data: Dict[str, Any], knowledge: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        分析时空逻辑一致性问题
        
        Args:
            screenplay_data: 剧本数据
            knowledge: 知识数据
            
        Returns:
            问题列表
        """
        # 获取时空逻辑一致性分析模板
        template = self.template_manager.get_langchain_prompt("spacetime_consistency_analysis")
        if not template:
            return [{"error": "时空逻辑一致性分析模板不存在"}]
        
        # 分析时空逻辑一致性
        from langchain.chains import LLMChain
        
        # 准备输入数据
        input_data = {
            "screenplay": json.dumps(screenplay_data, ensure_ascii=False),
            "knowledge": json.dumps(knowledge, ensure_ascii=False)
        }
        
        # 运行时空逻辑一致性分析
        chain = LLMChain(llm=self.llm_interface.get_llm(), prompt=template)
        result = chain.run(**input_data)
        
        try:
            # 解析结果
            issues = json.loads(result)
            
            # 添加问题类型
            for issue in issues:
                issue["type"] = "spacetime"
            
            return issues
        except json.JSONDecodeError:
            # 如果解析失败，返回错误
            return [{"type": "spacetime", "error": "无法解析时空逻辑一致性分析结果", "raw_result": result}]
    
    def _analyze_character_consistency(self, screenplay_data: Dict[str, Any], knowledge: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        分析角色一致性问题
        
        Args:
            screenplay_data: 剧本数据
            knowledge: 知识数据
            
        Returns:
            问题列表
        """
        # 获取角色一致性分析模板
        template = self.template_manager.get_langchain_prompt("character_consistency_analysis")
        if not template:
            return [{"error": "角色一致性分析模板不存在"}]
        
        # 分析角色一致性
        from langchain.chains import LLMChain
        
        # 准备输入数据
        input_data = {
            "screenplay": json.dumps(screenplay_data, ensure_ascii=False),
            "knowledge": json.dumps(knowledge, ensure_ascii=False)
        }
        
        # 运行角色一致性分析
        chain = LLMChain(llm=self.llm_interface.get_llm(), prompt=template)
        result = chain.run(**input_data)
        
        try:
            # 解析结果
            issues = json.loads(result)
            
            # 添加问题类型
            for issue in issues:
                issue["type"] = "character"
            
            return issues
        except json.JSONDecodeError:
            # 如果解析失败，返回错误
            return [{"type": "character", "error": "无法解析角色一致性分析结果", "raw_result": result}]
    
    def _analyze_continuity_errors(self, screenplay_data: Dict[str, Any], knowledge: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        分析潜在穿帮点问题
        
        Args:
            screenplay_data: 剧本数据
            knowledge: 知识数据
            
        Returns:
            问题列表
        """
        # 获取潜在穿帮点分析模板
        template = self.template_manager.get_langchain_prompt("continuity_error_analysis")
        if not template:
            return [{"error": "潜在穿帮点分析模板不存在"}]
        
        # 分析潜在穿帮点
        from langchain.chains import LLMChain
        
        # 准备输入数据
        input_data = {
            "screenplay": json.dumps(screenplay_data, ensure_ascii=False),
            "knowledge": json.dumps(knowledge, ensure_ascii=False)
        }
        
        # 运行潜在穿帮点分析
        chain = LLMChain(llm=self.llm_interface.get_llm(), prompt=template)
        result = chain.run(**input_data)
        
        try:
            # 解析结果
            issues = json.loads(result)
            
            # 添加问题类型
            for issue in issues:
                issue["type"] = "continuity"
            
            return issues
        except json.JSONDecodeError:
            # 如果解析失败，返回错误
            return [{"type": "continuity", "error": "无法解析潜在穿帮点分析结果", "raw_result": result}]
    
    def _analyze_causal_logic(self, screenplay_data: Dict[str, Any], knowledge: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        分析因果链与剧情推进逻辑问题
        
        Args:
            screenplay_data: 剧本数据
            knowledge: 知识数据
            
        Returns:
            问题列表
        """
        # 获取因果链与剧情推进逻辑分析模板
        template = self.template_manager.get_langchain_prompt("causal_logic_analysis")
        if not template:
            return [{"error": "因果链与剧情推进逻辑分析模板不存在"}]
        
        # 分析因果链与剧情推进逻辑
        from langchain.chains import LLMChain
        
        # 准备输入数据
        input_data = {
            "screenplay": json.dumps(screenplay_data, ensure_ascii=False),
            "knowledge": json.dumps(knowledge, ensure_ascii=False)
        }
        
        # 运行因果链与剧情推进逻辑分析
        chain = LLMChain(llm=self.llm_interface.get_llm(), prompt=template)
        result = chain.run(**input_data)
        
        try:
            # 解析结果
            issues = json.loads(result)
            
            # 添加问题类型
            for issue in issues:
                issue["type"] = "causal"
            
            return issues
        except json.JSONDecodeError:
            # 如果解析失败，返回错误
            return [{"type": "causal", "error": "无法解析因果链与剧情推进逻辑分析结果", "raw_result": result}]
