# run sentiment analysis for llama (AutoModelForCausalLM) with vllm and lora

from utils import *
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm.lora.request import LoRARequest
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("----- Load Dataset -----")
dataset="FPB"
# dataset="FIQASA"
test_dataset = TestDataset(dataset=dataset)
batch_data = [test_dataset[i]['sentence'] for i in range(len(test_dataset))]

print("----- Load Model -----")
# model_name_or_path = "meta-llama/Llama-2-7b-hf"
# lora_name_or_path = "FinGPT/fingpt-mt_llama2-7b_lora"
model_name_or_path = "NousResearch/Llama-2-13b-hf"
lora_name_or_path = "FinGPT/fingpt-sentiment_llama2-13b_lora"
model_name = lora_name_or_path.split("/")[-1]

model = LLM(model=model_name_or_path, 
            tensor_parallel_size=1, 
            dtype="bfloat16", 
            gpu_memory_utilization=0.9, 
            enable_lora=True)
lora=LoRARequest("sql_adapter", 1, lora_name_or_path)
sampling_params = SamplingParams(temperature=0, max_tokens=4)

print("----- Start Inference -----")
outputs = model.generate(batch_data, 
                         sampling_params, 
                         lora_request=lora)
labels = [output.outputs[0].text for output in outputs]
save_to_csv(dataset_map[dataset]['path'], model_name, labels)
