"""
提示模板管理模块
负责加载和管理JSON格式的提示模板
"""

import os
import json
from typing import Dict, List, Any, Optional

from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate


class PromptTemplateManager:
    """提示模板管理器，负责加载和管理JSON格式的提示模板"""
    
    def __init__(self, templates_dir: str = "prompts"):
        """
        初始化提示模板管理器
        
        Args:
            templates_dir (str): 提示模板目录
        """
        self.templates_dir = templates_dir
        self.templates = {}
        
        # 加载所有模板
        self._load_templates()
    
    def _load_templates(self) -> None:
        """递归加载所有JSON模板文件"""
        if not os.path.exists(self.templates_dir):
            os.makedirs(self.templates_dir, exist_ok=True)
            print(f"创建模板目录: {self.templates_dir}")
            return
            
        for root, _, files in os.walk(self.templates_dir):
            for file in files:
                if file.endswith(".json"):
                    template_path = os.path.join(root, file)
                    try:
                        with open(template_path, 'r', encoding='utf-8') as f:
                            template = json.load(f)
                            if "id" in template:
                                self.templates[template["id"]] = template
                                print(f"加载模板: {template['id']}")
                    except Exception as e:
                        print(f"加载模板 {template_path} 失败: {e}")
    
    def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """
        获取提示模板
        
        Args:
            template_id (str): 模板ID
            
        Returns:
            dict: 提示模板，如果不存在则返回None
        """
        return self.templates.get(template_id)
    
    def get_template_content(self, template_id: str, variables: Dict[str, Any] = None) -> Optional[str]:
        """
        获取填充变量后的模板内容
        
        Args:
            template_id (str): 模板ID
            variables (dict): 变量值
            
        Returns:
            str: 填充变量后的模板内容，如果模板不存在则返回None
        """
        template = self.get_template(template_id)
        if not template:
            return None
        
        content = template.get("template", "")
        if variables:
            # 替换模板中的变量
            for var_name, var_value in variables.items():
                content = content.replace(f"{{{var_name}}}", str(var_value))
        
        return content
    
    def list_templates(self, category: str = None) -> List[Dict[str, Any]]:
        """
        列出所有模板或指定类别的模板
        
        Args:
            category (str, optional): 模板类别
            
        Returns:
            list: 模板列表
        """
        if category:
            return [t for t in self.templates.values() if t.get("category") == category]
        return list(self.templates.values())
    
    def save_template(self, template: Dict[str, Any]) -> bool:
        """
        保存模板
        
        Args:
            template (dict): 模板数据
            
        Returns:
            bool: 是否保存成功
        """
        if "id" not in template or "category" not in template:
            return False
        
        # 确定保存路径
        category_dir = os.path.join(self.templates_dir, template["category"])
        os.makedirs(category_dir, exist_ok=True)
        
        template_path = os.path.join(category_dir, f"{template['id']}.json")
        
        try:
            with open(template_path, 'w', encoding='utf-8') as f:
                json.dump(template, f, ensure_ascii=False, indent=2)
            
            # 更新内存中的模板
            self.templates[template["id"]] = template
            return True
        except Exception as e:
            print(f"保存模板失败: {e}")
            return False
    
    def get_langchain_prompt(self, template_id: str) -> Optional[PromptTemplate]:
        """
        获取LangChain格式的提示模板
        
        Args:
            template_id (str): 模板ID
            
        Returns:
            PromptTemplate: LangChain提示模板，如果模板不存在则返回None
        """
        template = self.get_template(template_id)
        if not template:
            return None
        
        # 提取变量名
        variables = [var["name"] for var in template.get("variables", [])]
        
        # 创建LangChain PromptTemplate
        return PromptTemplate(
            template=template.get("template", ""),
            input_variables=variables
        )
    
    def get_langchain_chat_prompt(self, template_id: str, system_message: str = None) -> Optional[ChatPromptTemplate]:
        """
        获取LangChain格式的聊天提示模板
        
        Args:
            template_id (str): 模板ID
            system_message (str, optional): 系统消息
            
        Returns:
            ChatPromptTemplate: LangChain聊天提示模板，如果模板不存在则返回None
        """
        template = self.get_template(template_id)
        if not template:
            return None
        
        # 提取变量名
        variables = [var["name"] for var in template.get("variables", [])]
        
        # 创建消息模板列表
        messages = []
        
        # 添加系统消息（如果有）
        if system_message:
            messages.append(SystemMessagePromptTemplate.from_template(system_message))
        
        # 添加人类消息
        messages.append(HumanMessagePromptTemplate.from_template(template.get("template", "")))
        
        # 创建LangChain ChatPromptTemplate
        return ChatPromptTemplate.from_messages(messages)
