from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import datasets as _hf_datasets  # Force HF datasets into sys.modules before peft/awq imports.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import uvicorn
import json
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Global variables for model and tokenizer
model = None
tokenizer = None
prompt_format = "alpaca"  # "alpaca" or "chat"

class QueryRequest(BaseModel):
    instruction: str
    input_text: str
    max_new_tokens: int = 128
    temperature: float = 0.0

class BatchQueryRequest(BaseModel):
    requests: list[QueryRequest]

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]
    max_new_tokens: int = 512
    temperature: float = 0.0

class BatchChatRequest(BaseModel):
    requests: list[ChatRequest]

def build_prompt(instruction: str, input_text: str) -> str:
    """Build prompt using the configured format (alpaca or chat)."""
    if prompt_format == "chat":
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"

def extract_response(full_output: str, prompt: str) -> str:
    """Extract generated response from full model output."""
    if prompt_format == "chat":
        return full_output[len(prompt):].strip() if len(full_output) > len(prompt) else full_output
    else:
        if "### Response:" in full_output:
            return full_output.split("### Response:")[-1].strip()
        return full_output[len(prompt):].strip()

@app.get("/")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "model_loaded": model is not None, "prompt_format": prompt_format}

@app.post("/v1/generate")
async def generate(request: QueryRequest):
    try:
        logger.info(f"Received single request (format={prompt_format})")

        prompt = build_prompt(request.instruction, request.input_text)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature if request.temperature > 0 else None,
                do_sample=request.temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = extract_response(full_output, prompt)

        logger.info(f"Successfully generated response")
        return {"response": response}
        
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/v1/chat")
async def chat(request: ChatRequest):
    """Chat endpoint using messages format (matches training data)."""
    try:
        logger.info(f"Received chat request with {len(request.messages)} messages")
        
        # Convert messages to chat format using tokenizer's chat template
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Use tokenizer's chat template if available
        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback: simple concatenation
            system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
            user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
            prompt = f"{system_msg}\n\n{user_msg}"
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature if request.temperature > 0 else None,
                do_sample=request.temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the generated part (after the prompt)
        response = full_output[len(prompt):].strip() if len(full_output) > len(prompt) else full_output
        
        logger.info(f"Successfully generated chat response ({len(response)} chars)")
        return {"response": response}
        
    except Exception as e:
        logger.error(f"Error during chat generation: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Chat generation failed: {str(e)}")

@app.post("/v1/chat_batch")
async def chat_batch(batch_request: BatchChatRequest):
    """Process multiple chat requests in a single GPU batch for efficiency."""
    try:
        requests = batch_request.requests
        logger.info(f"Received batch chat request with {len(requests)} items")
        
        # Build all prompts using chat template
        prompts = []
        for req in requests:
            messages = [{"role": msg.role, "content": msg.content} for msg in req.messages]
            
            if tokenizer.chat_template:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
                user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
                prompt = f"{system_msg}\n\n{user_msg}"
            prompts.append(prompt)
        
        # Tokenize all prompts with padding
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to("cuda")
        
        logger.debug(f"Batch input shape: {inputs['input_ids'].shape}")
        
        # Generate for all prompts in batch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=requests[0].max_new_tokens,
                temperature=requests[0].temperature if requests[0].temperature > 0 else None,
                do_sample=requests[0].temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode all outputs
        responses = []
        for i, output_ids in enumerate(outputs):
            # Skip the input tokens
            input_length = inputs['input_ids'][i].shape[0]
            generated_ids = output_ids[input_length:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            responses.append(response)
        
        logger.info(f"Successfully generated {len(responses)} chat responses")
        return {"responses": responses}
        
    except Exception as e:
        logger.error(f"Error during batch chat generation: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Batch chat generation failed: {str(e)}")

@app.post("/v1/generate_batch")
async def generate_batch(batch_request: BatchQueryRequest):
    """Process multiple requests in a single GPU batch for efficiency."""
    try:
        requests = batch_request.requests
        logger.info(f"Received batch request with {len(requests)} items (format={prompt_format})")

        # Build all prompts using configured format
        prompts = [build_prompt(req.instruction, req.input_text) for req in requests]

        # Tokenize all prompts with padding
        # Use 2048 to match training sequence length and avoid truncating hop 2 context
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to("cuda")

        logger.debug(f"Batch input shape: {inputs['input_ids'].shape}")

        # Generate for all prompts in batch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=requests[0].max_new_tokens,
                temperature=requests[0].temperature if requests[0].temperature > 0 else None,
                do_sample=requests[0].temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode all outputs — skip input tokens for clean extraction
        responses = []
        for i, output_ids in enumerate(outputs):
            input_length = inputs['input_ids'][i].shape[0]
            generated_ids = output_ids[input_length:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            responses.append(response)

        logger.info(f"Successfully generated {len(responses)} responses")
        return {"responses": responses}
        
    except Exception as e:
        logger.error(f"Error during batch generation: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Batch generation failed: {str(e)}")

def load_model(base_model_path, adapter_path, merge_adapter_path=None, load_in_4bit=False):
    global model, tokenizer
    print(f"Loading base model: {base_model_path}...")

    # For local paths, convert to absolute path to avoid HF Hub validation
    import os
    if os.path.exists(base_model_path):
        base_model_path = os.path.abspath(base_model_path)
        print(f"Using local model at: {base_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # CRITICAL: Use left padding for decoder-only models in batch inference
    tokenizer.padding_side = "left"

    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        **model_kwargs,
    )

    import sys
    sys.modules["datasets"] = _hf_datasets

    if merge_adapter_path:
        print(f"Loading and merging adapter: {merge_adapter_path}...")
        model = PeftModel.from_pretrained(model, merge_adapter_path)
        model = model.merge_and_unload()
        print("  Adapter merged into base model")

    if adapter_path:
        print(f"Loading adapter: {adapter_path}...")
        sys.modules["datasets"] = _hf_datasets
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    print("Model loaded and ready!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--adapter", default=None, help="LoRA adapter path (omit for full FT models)")
    parser.add_argument("--merge-adapter", default=None, help="LoRA adapter to merge into base before loading --adapter (for DPO on top of SFT)")
    parser.add_argument("--prompt-format", choices=["alpaca", "chat"], default="alpaca",
                        help="Prompt format: 'alpaca' for SFT models, 'chat' for base Llama-Instruct")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load the base model in 4-bit to reduce inference memory.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    prompt_format = args.prompt_format
    print(f"Prompt format: {prompt_format}")

    load_model(args.base_model, args.adapter, args.merge_adapter, load_in_4bit=args.load_in_4bit)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
