"""
输出生成模块
负责整合分析结果，生成问题报告与修改建议
"""

import os
import json
from typing import Dict, List, Any, Optional, Union
from ..utils.format_utils import correct_json_format


class OutputGenerator:
    """输出生成器，负责整合分析结果，生成问题报告与修改建议"""
    
    def __init__(self, llm_interface, template_manager, config: Dict[str, Any] = None):
        """
        初始化输出生成器
        
        Args:
            llm_interface: 统一LLM接口
            template_manager: 提示模板管理器
            config: 配置参数
        """
        self.llm_interface = llm_interface
        self.template_manager = template_manager
        self.config = config or {}
    
    def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成输出
        
        Args:
            input_data: 输入数据，包含分析结果
            
        Returns:
            生成结果
        """
        # 检查输入
        if "issues" not in input_data:
            raise ValueError("输入数据必须包含issues")
        
        issues = input_data["issues"]
        screenplay_data = input_data.get("screenplay_data", {})
        
        # 生成问题报告
        report = self._generate_report(issues, screenplay_data)
        
        # 生成修改建议
        suggestions = self._generate_suggestions(issues, screenplay_data)
        
        # 返回生成结果
        return {
            "report": report,
            "suggestions": suggestions,
            "issue_count": len(issues)
        }
    
    def _generate_report(self, issues: List[Dict[str, Any]], screenplay_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成问题报告
        
        Args:
            issues: 问题列表
            screenplay_data: 剧本数据
            
        Returns:
            问题报告
        """
        # 获取问题报告生成模板
        template = self.template_manager.get_langchain_prompt("report_generation")
        if not template:
            return {"error": "问题报告生成模板不存在"}
        
        # 生成问题报告
        from langchain.chains import LLMChain
        
        # 准备输入数据
        input_data = {
            "issues": json.dumps(issues, ensure_ascii=False),
            "screenplay": json.dumps(screenplay_data, ensure_ascii=False)
        }
        
        # 运行问题报告生成
        chain = LLMChain(llm=self.llm_interface.get_llm(), prompt=template)
        result = chain.run(**input_data)
        result = correct_json_format(result)
        
        try:
            # 解析结果
            report = json.loads(result)
            return report
        except json.JSONDecodeError:
            # 如果解析失败，尝试作为Markdown格式处理
            return {
                "format": "markdown",
                "content": result
            }
    
    def _generate_suggestions(self, issues: List[Dict[str, Any]], screenplay_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        生成修改建议
        
        Args:
            issues: 问题列表
            screenplay_data: 剧本数据
            
        Returns:
            修改建议列表
        """
        # 获取修改建议生成模板
        template = self.template_manager.get_langchain_prompt("suggestion_generation")
        if not template:
            return [{"error": "修改建议生成模板不存在"}]
        
        # 按问题类型分组
        issue_types = {}
        for issue in issues:
            issue_type = issue.get("type", "unknown")
            if issue_type not in issue_types:
                issue_types[issue_type] = []
            issue_types[issue_type].append(issue)
        
        # 为每种问题类型生成修改建议
        suggestions = []
        for issue_type, type_issues in issue_types.items():
            # 生成修改建议
            from langchain.chains import LLMChain
            
            # 准备输入数据
            input_data = {
                "issues": json.dumps(type_issues, ensure_ascii=False),
                "screenplay": json.dumps(screenplay_data, ensure_ascii=False),
                "issue_type": issue_type
            }
            
            # 运行修改建议生成
            chain = LLMChain(llm=self.llm_interface.get_llm(), prompt=template)
            result = chain.run(**input_data)
            result = correct_json_format(result)
            
            try:
                # 解析结果
                type_suggestions = json.loads(result)
                
                # 添加问题类型
                for suggestion in type_suggestions:
                    suggestion["issue_type"] = issue_type
                
                suggestions.extend(type_suggestions)
            except json.JSONDecodeError:
                # 如果解析失败，添加错误
                suggestions.append({
                    "issue_type": issue_type,
                    "error": "无法解析修改建议生成结果",
                    "raw_result": result
                })
        
        return suggestions
