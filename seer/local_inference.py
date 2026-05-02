import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
import threading

class LocalPeftQueryGenerator:
    """Generate search queries using a local LLM with PEFT adapters."""
    
    def __init__(
        self,
        base_model_path: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        adapter_path: str = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype = torch.bfloat16,
        temperature: float = 0.0, # Greedy by default for eval
        max_new_tokens: int = 128,
    ):
        print(f"Loading base model: {base_model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        # Llama 3 requires a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map=device,
        )
        
        if adapter_path:
            print(f"Loading adapter: {adapter_path}...")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            
        self.model.eval()
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self._lock = threading.Lock()

    def generate_query(
        self, 
        question: str, 
        context_str: str,
    ) -> str:
        """
        Match the prompt format used during SFT:
        ### Instruction:
        Generate a search query...
        ### Input:
        Context: ... Question: ...
        ### Response:
        """
        instruction = "Generate a search query to find information needed to answer the following question. Use the provided context if available."
        
        input_text = f"Context: {context_str}\n\nQuestion: {question}"
        
        # Alpaca-style format matching Axolotl 'alpaca' type
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract response after the prompt
        if "### Response:" in full_output:
            response = full_output.split("### Response:")[1].strip()
        else:
            response = full_output[len(prompt):].strip()
            
        return response
