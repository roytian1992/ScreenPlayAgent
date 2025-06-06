"""
GPT-4O API客户端模块
负责处理与OpenAI API的通信、缓存和错误处理
"""

import os
import json
import hashlib
from datetime import datetime
import time
import openai

class GPT4OClient:
    """GPT-4O API客户端，处理API调用、缓存和错误处理"""
    
    def __init__(self, config=None):
        """
        初始化API客户端
        
        Args:
            config (dict, optional): 配置参数，包含API密钥、模型名称等
        """
        self.config = config or {}
        self.api_key = self.config.get("api_key", os.environ.get("OPENAI_API_KEY"))
        self.model = self.config.get("model", "gpt-4o")
        self.max_tokens = self.config.get("max_tokens", 4096)
        self.temperature = self.config.get("temperature", 0.2)
        
        # 初始化缓存
        self.cache_enabled = self.config.get("cache_enabled", True)
        self.cache_path = self.config.get("cache_path", "cache/")
        if self.cache_enabled and not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        
        # 重试设置
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 2)
        
        # 初始化API客户端
        if not self.api_key:
            raise ValueError("API密钥未提供，请在配置中设置api_key或设置环境变量OPENAI_API_KEY")
        
        openai.api_key = self.api_key
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def call(self, prompt, system_message=None, temperature=None, max_tokens=None):
        """
        调用GPT-4O API
        
        Args:
            prompt (str): 用户提示
            system_message (str, optional): 系统消息
            temperature (float, optional): 温度参数，控制随机性
            max_tokens (int, optional): 最大生成令牌数
            
        Returns:
            str: API响应文本，失败时返回None
        """
        # 使用传入的参数，如果没有则使用默认值
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        # 检查缓存
        if self.cache_enabled:
            cache_key = self._generate_cache_key(prompt, system_message, temperature, max_tokens)
            cached_response = self._get_from_cache(cache_key)
            if cached_response:
                return cached_response
        
        # 准备消息
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        # 重试机制
        for attempt in range(self.max_retries):
            try:
                # 调用API
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                # 提取响应文本
                response_text = response.choices[0].message.content
                
                # 缓存响应
                if self.cache_enabled:
                    self._save_to_cache(cache_key, response_text)
                
                return response_text
            
            except Exception as e:
                print(f"API调用失败 (尝试 {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    # 指数退避策略
                    sleep_time = self.retry_delay * (2 ** attempt)
                    print(f"等待 {sleep_time} 秒后重试...")
                    time.sleep(sleep_time)
                else:
                    print("达到最大重试次数，放弃请求")
                    return None
    
    def _generate_cache_key(self, prompt, system_message=None, temperature=None, max_tokens=None):
        """
        生成缓存键
        
        Args:
            prompt (str): 用户提示
            system_message (str, optional): 系统消息
            temperature (float, optional): 温度参数
            max_tokens (int, optional): 最大生成令牌数
            
        Returns:
            str: 缓存键（MD5哈希）
        """
        # 组合所有参数
        combined = f"{prompt}|{system_message or ''}|{temperature}|{max_tokens}"
        
        # 生成MD5哈希
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key):
        """
        从缓存获取响应
        
        Args:
            cache_key (str): 缓存键
            
        Returns:
            str: 缓存的响应，不存在时返回None
        """
        cache_file = os.path.join(self.cache_path, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                return cache_data.get("response")
            except Exception as e:
                print(f"读取缓存失败: {e}")
                return None
        
        return None
    
    def _save_to_cache(self, cache_key, response):
        """
        保存响应到缓存
        
        Args:
            cache_key (str): 缓存键
            response (str): API响应文本
        """
        cache_file = os.path.join(self.cache_path, f"{cache_key}.json")
        
        try:
            cache_data = {
                "response": response,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"缓存保存失败: {e}")
    
    def batch_call(self, prompts, system_message=None, temperature=None, max_tokens=None):
        """
        批量调用API
        
        Args:
            prompts (list): 提示列表
            system_message (str, optional): 系统消息
            temperature (float, optional): 温度参数
            max_tokens (int, optional): 最大生成令牌数
            
        Returns:
            list: 响应列表，与提示列表一一对应
        """
        results = []
        for prompt in prompts:
            result = self.call(prompt, system_message, temperature, max_tokens)
            results.append(result)
        return results
