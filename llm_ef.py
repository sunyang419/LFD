import torch
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForCausalLM, 
    BitsAndBytesConfig, StoppingCriteriaList, StoppingCriteria
)
from typing import List, Tuple, Optional
import time

class LLM:
    def __init__(
        self, 
        model_id: str, 
        device: str = 'cuda', 
        quantization_bits: Optional[int] = None, 
        stop_list: Optional[List[str]] = None, 
        model_max_length: int = 4096
    ):
        self.device = device
        self.model_max_length = model_max_length

        self.stop_list = stop_list
        if stop_list is None:
            self.stop_list = ['\n']
        
        self.bnb_config = self._set_quantization(quantization_bits)
        self.model, self.tokenizer = self._initialize_model_tokenizer(model_id)
        self.stopping_criteria = self._define_stopping_criteria()
        


    def _set_quantization(self, quantization_bits: Optional[int]) -> Optional[BitsAndBytesConfig]:
        if quantization_bits in [4, 8]:
            bnb_config = BitsAndBytesConfig()
            if quantization_bits == 4:
                bnb_config.load_in_4bit = True
                bnb_config.bnb_4bit_quant_type = 'nf4'
                bnb_config.bnb_4bit_use_double_quant = True
                bnb_config.bnb_4bit_compute_dtype = torch.bfloat16
            elif quantization_bits == 8:
                bnb_config.load_in_8bit = True
            return bnb_config
        return None


    def _initialize_model_tokenizer(self, model_id: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        model_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        model_config.max_seq_len = self.model_max_length

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            config=model_config,
            # quantization_config=self.bnb_config,
            torch_dtype=torch.bfloat16,
            device_map='auto',
        )
        model.eval() # Set the model to evaluation mode

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, padding_side="left", truncation_side="left", trust_remote_code=True,
            model_max_length=self.model_max_length
        )
        # Most LLMs don't have a pad token by default
        tokenizer.pad_token = tokenizer.eos_token  

        return model, tokenizer


    def _define_stopping_criteria(self) -> StoppingCriteriaList:
        stop_token_ids = []
        for x in self.stop_list:
            encoded = self.tokenizer.encode(x, add_special_tokens=False)[-1]
            if encoded: 
                stop_token_ids.append([encoded])

        stop_token_tensors = []
        for ids in stop_token_ids:
            tensor = torch.tensor(ids, dtype=torch.long, device=self.device)
            stop_token_tensors.append(tensor)

        class StopOnTokens(StoppingCriteria):
            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                current_seq = input_ids[0] 
                for stop_ids in stop_token_tensors:
                    if len(current_seq) >= len(stop_ids):
                        if torch.equal(current_seq[-len(stop_ids):], stop_ids):
                            return True
                return False

        return StoppingCriteriaList([StopOnTokens()])


    def generate(self, prompt: str, max_new_tokens: int = 15):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        memory_before = torch.cuda.memory_allocated() 
        
        if self.tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_max_length
        ).to(self.device)
        
        input_token_len = inputs.input_ids.shape[-1]  
        
        start_time = time.perf_counter()
        
        sequences = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.1,
            stopping_criteria=self.stopping_criteria,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            lfd_layers=[16, 18, 20, 22, 24, 26, 28, 30]
        )
        
        total_time = time.perf_counter() - start_time
        
        output_token_len = sequences.shape[-1] - input_token_len
        # gen_sequences = sequences[:, input_token_len:][0, :]
        # generated_text = self.tokenizer.decode(gen_sequences, skip_special_tokens=True)
        
        metrics = {
            "input_token_len": input_token_len,
            "output_token_len": output_token_len,
            "total_time": total_time,
            "memory_before_mb": memory_before / (1024 ** 2), # MB
            "peak_memory_mb": torch.cuda.max_memory_allocated() / (1024 ** 2), # MB
        }
        
        return metrics