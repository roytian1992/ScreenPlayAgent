"""
输入处理模块
负责剧本文件的加载、解析和预处理
"""

import os
import json
import pandas as pd
from typing import Dict, List, Any, Optional, Union

class InputProcessor:
    """输入处理器，负责剧本文件的加载、解析和预处理"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化输入处理器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理输入数据
        
        Args:
            input_data: 输入数据，可以是文件路径或直接的剧本数据
            
        Returns:
            处理后的剧本数据
        """
        # 检查输入类型
        if "screenplay_path" in input_data:
            # 从文件加载剧本
            screenplay_data = self._load_screenplay_from_file(input_data["screenplay_path"])
        elif "screenplay_data" in input_data:
            # 直接使用提供的剧本数据
            screenplay_data = input_data["screenplay_data"]
        else:
            raise ValueError("输入数据必须包含screenplay_path或screenplay_data")
        
        # 预处理剧本
        processed_data = self._preprocess_screenplay(screenplay_data)
        
        return {
            "screenplay_data": processed_data,
            "metadata": {
                "title": processed_data.get("metadata", {}).get("title", "未命名剧本"),
                "scene_count": len(processed_data.get("scenes", [])),
                "character_count": len(self._extract_characters(processed_data))
            }
        }
    
    def _load_screenplay_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        从文件加载剧本
        
        Args:
            file_path: 剧本文件路径
            
        Returns:
            剧本数据
        """
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"剧本文件不存在: {file_path}")
        
        # 根据文件扩展名选择加载方法
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == ".json":
            # 加载JSON格式剧本
            return self._load_json_screenplay(file_path)
        elif file_ext in [".xlsx", ".xls"]:
            # 加载Excel格式剧本
            return self._load_excel_screenplay(file_path)
        elif file_ext == ".csv":
            # 加载CSV格式剧本
            return self._load_csv_screenplay(file_path)
        elif file_ext in [".txt", ".md"]:
            # 加载文本格式剧本
            return self._load_text_screenplay(file_path)
        else:
            raise ValueError(f"不支持的剧本文件格式: {file_ext}")
    
    def _load_json_screenplay(self, file_path: str) -> Dict[str, Any]:
        """
        加载JSON格式剧本
        
        Args:
            file_path: 剧本文件路径
            
        Returns:
            剧本数据
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            screenplay_data = json.load(f)
        
        # 验证剧本格式
        if "scenes" not in screenplay_data:
            # 尝试转换格式
            if isinstance(screenplay_data, list):
                # 假设是场景列表
                screenplay_data = {
                    "metadata": {
                        "title": os.path.basename(file_path),
                        "source": file_path
                    },
                    "scenes": screenplay_data
                }
        
        return screenplay_data
    
    def _load_excel_screenplay(self, file_path: str) -> Dict[str, Any]:
        """
        加载Excel格式剧本
        
        Args:
            file_path: 剧本文件路径
            
        Returns:
            剧本数据
        """
        # 读取Excel文件
        df = pd.read_excel(file_path)
        
        # 转换为剧本数据
        return self._convert_dataframe_to_screenplay(df, file_path)
    
    def _load_csv_screenplay(self, file_path: str) -> Dict[str, Any]:
        """
        加载CSV格式剧本
        
        Args:
            file_path: 剧本文件路径
            
        Returns:
            剧本数据
        """
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 转换为剧本数据
        return self._convert_dataframe_to_screenplay(df, file_path)
    
    def _convert_dataframe_to_screenplay(self, df: pd.DataFrame, file_path: str) -> Dict[str, Any]:
        """
        将DataFrame转换为剧本数据
        
        Args:
            df: DataFrame
            file_path: 剧本文件路径
            
        Returns:
            剧本数据
        """
        # 检查必要的列
        required_columns = ["scene_id"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"缺少必要的列: {col}")
        
        # 映射列名
        column_mapping = {
            "scene_id": "scene_id",
            "title": "title",
            "content": "content",
            "location": "location",
            "time": "time",
            "characters": "characters"
        }
        
        # 创建场景列表
        scenes = []
        for _, row in df.iterrows():
            scene = {}
            
            # 添加场景属性
            for src_col, dst_col in column_mapping.items():
                if src_col in df.columns and not pd.isna(row[src_col]):
                    scene[dst_col] = row[src_col]
            
            # 确保场景ID存在
            if "scene_id" not in scene:
                scene["scene_id"] = str(len(scenes) + 1)
            
            # 处理角色列表
            if "characters" in scene and isinstance(scene["characters"], str):
                scene["characters"] = [c.strip() for c in scene["characters"].split(",")]
            
            scenes.append(scene)
        
        # 创建剧本数据
        screenplay_data = {
            "metadata": {
                "title": os.path.basename(file_path),
                "source": file_path
            },
            "scenes": scenes
        }
        
        return screenplay_data
    
    def _load_text_screenplay(self, file_path: str) -> Dict[str, Any]:
        """
        加载文本格式剧本
        
        Args:
            file_path: 剧本文件路径
            
        Returns:
            剧本数据
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析文本剧本
        scenes = self._parse_text_screenplay(content)
        
        # 创建剧本数据
        screenplay_data = {
            "metadata": {
                "title": os.path.basename(file_path),
                "source": file_path
            },
            "scenes": scenes
        }
        
        return screenplay_data
    
    def _parse_text_screenplay(self, content: str) -> List[Dict[str, Any]]:
        """
        解析文本剧本
        
        Args:
            content: 剧本内容
            
        Returns:
            场景列表
        """
        # 分割场景
        scene_markers = ["场景", "SCENE", "Scene", "INT.", "EXT.", "内景", "外景"]
        
        # 尝试按场景分割
        scenes = []
        current_scene = None
        
        # 按行分割
        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检查是否是场景标记
            is_scene_marker = False
            for marker in scene_markers:
                if line.startswith(marker):
                    is_scene_marker = True
                    break
            
            if is_scene_marker or line.startswith("第") and ("场" in line or "幕" in line):
                # 保存当前场景
                if current_scene:
                    scenes.append(current_scene)
                
                # 创建新场景
                current_scene = {
                    "scene_id": str(len(scenes) + 1),
                    "title": line,
                    "content": ""
                }
            elif current_scene:
                # 添加内容到当前场景
                if current_scene["content"]:
                    current_scene["content"] += "\n"
                current_scene["content"] += line
            else:
                # 创建第一个场景
                current_scene = {
                    "scene_id": "1",
                    "title": "开场",
                    "content": line
                }
        
        # 保存最后一个场景
        if current_scene:
            scenes.append(current_scene)
        
        # 如果没有找到场景，将整个内容作为一个场景
        if not scenes:
            scenes = [{
                "scene_id": "1",
                "title": "整体内容",
                "content": content
            }]
        
        return scenes
    
    def _preprocess_screenplay(self, screenplay_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        预处理剧本
        
        Args:
            screenplay_data: 剧本数据
            
        Returns:
            预处理后的剧本数据
        """
        # 确保metadata存在
        if "metadata" not in screenplay_data:
            screenplay_data["metadata"] = {}
        
        # 确保scenes存在
        if "scenes" not in screenplay_data:
            screenplay_data["scenes"] = []
        
        # 处理场景
        for i, scene in enumerate(screenplay_data["scenes"]):
            # 确保scene_id存在
            if "scene_id" not in scene:
                scene["scene_id"] = str(i + 1)
            
            # 确保title存在
            if "title" not in scene:
                scene["title"] = f"场景{scene['scene_id']}"
            
            # 确保content存在
            if "content" not in scene:
                scene["content"] = ""
            
            # 提取角色
            if "characters" not in scene:
                scene["characters"] = self._extract_scene_characters(scene["content"])
        
        return screenplay_data
    
    def _extract_scene_characters(self, content: str) -> List[str]:
        """
        从场景内容中提取角色
        
        Args:
            content: 场景内容
            
        Returns:
            角色列表
        """
        # 简单实现：提取大写单词作为角色名
        import re
        
        # 匹配可能的角色名（中文或大写英文单词）
        character_pattern = r'([\u4e00-\u9fa5]{1,4}|[A-Z][A-Z]+)(?=[:：])'
        characters = set(re.findall(character_pattern, content))
        
        return list(characters)
    
    def _extract_characters(self, screenplay_data: Dict[str, Any]) -> List[str]:
        """
        从剧本中提取所有角色
        
        Args:
            screenplay_data: 剧本数据
            
        Returns:
            角色列表
        """
        characters = set()
        
        for scene in screenplay_data.get("scenes", []):
            if "characters" in scene and isinstance(scene["characters"], list):
                characters.update(scene["characters"])
        
        return list(characters)
