from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any

class Qwen3LLM(LLM):
    model: Any
    tokenizer: Any
    max_new_tokens: int
    temperature: float
    
    @property
    def _llm_type(self) -> str:
        return "qwen3"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # 生成文本
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature
        )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        # 解析思考内容和回答
        try:
            # 查找</think>标记 (151668)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        
        # 只返回回答部分
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        return content