"""
Agent记忆与反思模块
负责实现任务记忆、经验总结与反思机制
"""

import os
import json
import datetime
from typing import Dict, List, Any, Optional, Union
from ..utils.format_utils import correct_json_format

class AgentMemory:
    """Agent记忆与反思机制"""
    
    def __init__(self, llm_interface, template_manager, config: Dict[str, Any] = None):
        """
        初始化Agent记忆
        
        Args:
            llm_interface: 统一LLM接口
            template_manager: 提示模板管理器
            config: 配置参数
        """
        self.llm_interface = llm_interface
        self.template_manager = template_manager
        self.config = config or {}
        
        # 初始化记忆存储
        self.task_history = []
        self.retrieval_history = []
        self.experience = {}
        
        # 加载持久化记忆（如果有）
        self.location = self.config.get("location", "./")
        self.memory_file = self.config.get("memory_file", "memory.json")
        self.memory_file = os.path.join(self.location, self.memory_file)
        self._load_memory()
    
    def record_task(self, task: Dict[str, Any]) -> None:
        """
        记录任务
        
        Args:
            task: 任务数据
        """
        # 添加时间戳
        task_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "task": task
        }
        
        self.task_history.append(task_record)
        
        # 保存记忆
        self._save_memory()
    
    def record_retrieval(self, query: str, result: Any) -> None:
        """
        记录检索操作
        
        Args:
            query: 检索查询
            result: 检索结果
        """
        # 添加时间戳
        retrieval_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "query": query,
            "result": result
        }
        
        self.retrieval_history.append(retrieval_record)
        
        # 保存记忆
        self._save_memory()
    
    def summarize_experience(self) -> Dict[str, Any]:
        """
        总结经验
        
        Returns:
            经验总结
        """
        # 获取经验总结模板
        template = self.template_manager.get_langchain_prompt("experience_summary")
        if not template:
            return {"error": "经验总结模板不存在"}
        
        # 准备输入数据
        input_data = {
            "task_history": json.dumps(self.task_history[-10:] if len(self.task_history) > 10 else self.task_history, ensure_ascii=False),
            "retrieval_history": json.dumps(self.retrieval_history[-10:] if len(self.retrieval_history) > 10 else self.retrieval_history, ensure_ascii=False),
            "current_experience": json.dumps(self.experience, ensure_ascii=False)
        }

        
        # 生成经验总结
        from langchain.chains import LLMChain
        
        chain = LLMChain(llm=self.llm_interface.get_llm(), prompt=template)
        result = chain.run(**input_data)
        result = correct_json_format(result)
        
        try:
            experience = json.loads(result)
            
            # 更新经验
            self.experience.update(experience)
            
            # 保存记忆
            self._save_memory()
            
            return experience
        except json.JSONDecodeError:
            return {"error": "无法解析经验总结", "raw_result": result}
    
    def reflect_on_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        反思任务
        
        Args:
            task: 任务数据
            
        Returns:
            反思结果
        """
        # 获取任务反思模板
        template = self.template_manager.get_langchain_prompt("task_reflection")
        if not template:
            return {"error": "任务反思模板不存在"}
        
        # 准备输入数据
        input_data = {
            "task": json.dumps(task),
            "experience": json.dumps(self.experience)
        }
        
        # 生成任务反思
        from langchain.chains import LLMChain
        
        chain = LLMChain(llm=self.llm_interface.get_llm(), prompt=template)
        result = chain.run(**input_data)
        
        try:
            reflection = json.loads(result)
            return reflection
        except json.JSONDecodeError:
            return {"error": "无法解析任务反思", "raw_result": result}
    
    def optimize_strategy(self, task_type: str) -> Dict[str, Any]:
        """
        优化策略
        
        Args:
            task_type: 任务类型
            
        Returns:
            优化策略
        """
        # 获取策略优化模板
        template = self.template_manager.get_langchain_prompt("strategy_optimization")
        if not template:
            return {"error": "策略优化模板不存在"}
        
        # 筛选相关任务历史
        relevant_tasks = [t for t in self.task_history if t.get("task", {}).get("type") == task_type]
        
        # 准备输入数据
        input_data = {
            "task_type": task_type,
            "task_history": json.dumps(relevant_tasks[-5:] if len(relevant_tasks) > 5 else relevant_tasks),
            "experience": json.dumps(self.experience)
        }
        
        # 生成策略优化
        from langchain.chains import LLMChain
        
        chain = LLMChain(llm=self.llm_interface.get_llm(), prompt=template)
        result = chain.run(**input_data)
        result = correct_json_format(result)
        
        try:
            strategy = json.loads(result)
            return strategy
        except json.JSONDecodeError:
            return {"error": "无法解析策略优化", "raw_result": result}
    
    def get_relevant_experience(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        获取相关经验
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            相关经验列表
        """
        # 如果经验为空，返回空列表
        if not self.experience:
            return []
        
        # 使用简单的关键词匹配
        relevant_experience = []
        for key, value in self.experience.items():
            # 计算简单的相关性分数
            score = sum(1 for word in query.lower().split() if word in key.lower())
            if score > 0:
                relevant_experience.append({"key": key, "value": value, "score": score})
        
        # 按相关性排序
        relevant_experience.sort(key=lambda x: x["score"], reverse=True)
        
        # 返回前top_k个结果
        return relevant_experience[:top_k]
    
    def _load_memory(self) -> None:
        """加载持久化记忆"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    memory_data = json.load(f)
                    
                    self.task_history = memory_data.get("task_history", [])
                    self.retrieval_history = memory_data.get("retrieval_history", [])
                    self.experience = memory_data.get("experience", {})
                    
                    print(f"从 {self.memory_file} 加载记忆成功")
            except Exception as e:
                print(f"加载记忆失败: {e}")
    
    def _save_memory(self) -> None:
        """保存持久化记忆"""
        try:
            memory_data = {
                "task_history": self.task_history,
                "retrieval_history": self.retrieval_history,
                "experience": self.experience
            }
            
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(self.memory_file)), exist_ok=True)
            
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存记忆失败: {e}")


    def update_task_status(self, task: Dict[str, Any], status: str, result: Optional[Dict[str, Any]] = None) -> None:
        """
        更新任务状态记录

        Args:
            task (dict): 原始任务记录（包含 type, input_path 等）
            status (str): 任务状态，如 "completed", "failed"
            result (dict, optional): 可选的结果数据或错误信息
        """
        for record in self.task_history:
            if record["task"] == task:
                record["task"]["status"] = status
                record["task"]["result"] = result
                record["task"]["updated_at"] = datetime.datetime.now().isoformat()
                break
        else:
            # 如果没有找到，就添加一条新的
            self.task_history.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "task": {
                    **task,
                    "status": status,
                    "result": result,
                    "updated_at": datetime.datetime.now().isoformat()
                }
            })
        self._save_memory()




class TaskTracker:
    """任务状态追踪器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化任务追踪器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.tasks = {}
        self.current_task_id = None
        self.task_history = []
        
        # 加载持久化任务状态（如果有）
        self.tasks_file = self.config.get("tasks_file", "task_tracker.json")
        self._load_tasks()
    
    def add_task(self, task_id: str, task_data: Dict[str, Any]) -> None:
        """
        添加任务
        
        Args:
            task_id: 任务ID
            task_data: 任务数据
        """
        self.tasks[task_id] = {
            "data": task_data,
            "status": "pending",
            "start_time": None,
            "end_time": None,
            "result": None,
            "operations": []
        }
        
        # 保存任务状态
        self._save_tasks()
    
    def start_task(self, task_id: str) -> None:
        """
        开始任务
        
        Args:
            task_id: 任务ID
        """
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = "in_progress"
            self.tasks[task_id]["start_time"] = datetime.datetime.now().isoformat()
            self.current_task_id = task_id
            
            # 保存任务状态
            self._save_tasks()
    
    def complete_task(self, task_id: str, result: Any = None) -> None:
        """
        完成任务
        
        Args:
            task_id: 任务ID
            result: 任务结果
        """
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = "completed"
            self.tasks[task_id]["end_time"] = datetime.datetime.now().isoformat()
            self.tasks[task_id]["result"] = result
            
            # 记录任务历史
            self.task_history.append({
                "task_id": task_id,
                "data": self.tasks[task_id]["data"],
                "start_time": self.tasks[task_id]["start_time"],
                "end_time": self.tasks[task_id]["end_time"],
                "operations": self.tasks[task_id]["operations"]
            })
            
            if self.current_task_id == task_id:
                self.current_task_id = None
            
            # 保存任务状态
            self._save_tasks()
    
    def fail_task(self, task_id: str, error: str) -> None:
        """
        任务失败
        
        Args:
            task_id: 任务ID
            error: 错误信息
        """
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = "failed"
            self.tasks[task_id]["end_time"] = datetime.datetime.now().isoformat()
            self.tasks[task_id]["result"] = {"error": error}
            
            # 记录任务历史
            self.task_history.append({
                "task_id": task_id,
                "data": self.tasks[task_id]["data"],
                "start_time": self.tasks[task_id]["start_time"],
                "end_time": self.tasks[task_id]["end_time"],
                "operations": self.tasks[task_id]["operations"],
                "error": error
            })
            
            if self.current_task_id == task_id:
                self.current_task_id = None
            
            # 保存任务状态
            self._save_tasks()
    
    def add_operation(self, task_id: str, operation: Dict[str, Any]) -> None:
        """
        添加操作记录
        
        Args:
            task_id: 任务ID
            operation: 操作记录
        """
        if task_id in self.tasks:
            self.tasks[task_id]["operations"].append({
                "timestamp": datetime.datetime.now().isoformat(),
                **operation
            })
            
            # 保存任务状态
            self._save_tasks()
    
    def get_task_status(self, task_id: str) -> Optional[str]:
        """
        获取任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务状态
        """
        if task_id in self.tasks:
            return self.tasks[task_id]["status"]
        return None
    
    def get_current_task(self) -> Optional[Dict[str, Any]]:
        """
        获取当前任务
        
        Returns:
            当前任务
        """
        if self.current_task_id:
            return {
                "task_id": self.current_task_id,
                **self.tasks[self.current_task_id]
            }
        return None
    
    def get_task_history(self) -> List[Dict[str, Any]]:
        """
        获取任务历史
        
        Returns:
            任务历史
        """
        return self.task_history
    
    def _load_tasks(self) -> None:
        """加载持久化任务状态"""
        if os.path.exists(self.tasks_file):
            try:
                with open(self.tasks_file, 'r', encoding='utf-8') as f:
                    tasks_data = json.load(f)
                    
                    self.tasks = tasks_data.get("tasks", {})
                    self.current_task_id = tasks_data.get("current_task_id")
                    self.task_history = tasks_data.get("task_history", [])
                    
                    print(f"从 {self.tasks_file} 加载任务状态成功")
            except Exception as e:
                print(f"加载任务状态失败: {e}")
    
    def _save_tasks(self) -> None:
        """保存持久化任务状态"""
        try:
            tasks_data = {
                "tasks": self.tasks,
                "current_task_id": self.current_task_id,
                "task_history": self.task_history
            }
            
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(self.tasks_file)), exist_ok=True)
            
            with open(self.tasks_file, 'w', encoding='utf-8') as f:
                json.dump(tasks_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存任务状态失败: {e}")


class ErrorCorrector:
    """错误纠正器"""
    
    def __init__(self, llm_interface, template_manager, config: Dict[str, Any] = None):
        """
        初始化错误纠正器
        
        Args:
            llm_interface: 统一LLM接口
            template_manager: 提示模板管理器
            config: 配置参数
        """
        self.llm_interface = llm_interface
        self.template_manager = template_manager
        self.config = config or {}
    
    def check_for_duplicates(self, issues: List[Dict[str, Any]]) -> List[str]:
        """
        检查重复问题
        
        Args:
            issues: 问题列表
            
        Returns:
            重复问题ID列表
        """
        # 获取重复检测模板
        template = self.template_manager.get_langchain_prompt("duplicate_detection")
        if not template:
            return []
        
        # 准备输入数据
        input_data = {
            "issues": json.dumps(issues)
        }
        
        # 运行重复检测
        from langchain.chains import LLMChain
        
        chain = LLMChain(llm=self.llm_interface.get_llm(), prompt=template)
        result = chain.run(**input_data)
        
        try:
            duplicates = json.loads(result)
            return duplicates
        except json.JSONDecodeError:
            print(f"无法解析重复检测结果: {result}")
            return []
    
    def check_for_conflicts(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        检查冲突问题
        
        Args:
            issues: 问题列表
            
        Returns:
            冲突问题列表
        """
        # 获取冲突检测模板
        template = self.template_manager.get_langchain_prompt("conflict_detection")
        if not template:
            return []
        
        # 准备输入数据
        input_data = {
            "issues": json.dumps(issues)
        }
        
        # 运行冲突检测
        from langchain.chains import LLMChain
        
        chain = LLMChain(llm=self.llm_interface.get_llm(), prompt=template)
        result = chain.run(**input_data)
        
        try:
            conflicts = json.loads(result)
            return conflicts
        except json.JSONDecodeError:
            print(f"无法解析冲突检测结果: {result}")
            return []
    
    def correct_issues(self, issues: List[Dict[str, Any]], duplicates: List[str], conflicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        纠正问题
        
        Args:
            issues: 问题列表
            duplicates: 重复问题ID列表
            conflicts: 冲突问题列表
            
        Returns:
            纠正后的问题列表
        """
        # 移除重复问题
        unique_issues = [issue for issue in issues if issue.get("id") not in duplicates]
        
        # 解决冲突问题
        for conflict in conflicts:
            # 获取冲突解决模板
            template = self.template_manager.get_langchain_prompt("conflict_resolution")
            if not template:
                continue
            
            # 准备输入数据
            conflict_issues = [issue for issue in unique_issues if issue.get("id") in conflict.get("issue_ids", [])]
            input_data = {
                "issues": json.dumps(conflict_issues)
            }
            
            # 运行冲突解决
            from langchain.chains import LLMChain
            
            chain = LLMChain(llm=self.llm_interface.get_llm(), prompt=template)
            result = chain.run(**input_data)
            
            try:
                resolved_issue = json.loads(result)
                
                # 移除冲突问题
                unique_issues = [issue for issue in unique_issues if issue.get("id") not in conflict.get("issue_ids", [])]
                
                # 添加解决后的问题
                unique_issues.append(resolved_issue)
            except json.JSONDecodeError:
                print(f"无法解析冲突解决结果: {result}")
        
        return unique_issues
