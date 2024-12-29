# run sentiment analysis for llama (AutoModelForCausalLM) with vllm

from utils import *
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

print("----- Load Dataset -----")
# data_path = "train_easy.csv"
data_path = "train_hard.csv"
# data_path = "FPB.csv"
test_dataset = TestDataset(data_path, use_sys_prompt=True, use_few_shot=True, use_template=True)
batch_data = [test_dataset[i]['sentence'] for i in range(len(test_dataset))]

print("----- Load Model -----")
model_name_or_path = "TheFinAI/finma-7b-nlp"
model_name = model_name_or_path.split("/")[-1]

model = LLM(model=model_name_or_path, 
            tensor_parallel_size=1, 
            dtype="bfloat16", 
            gpu_memory_utilization=0.9)
sampling_params = SamplingParams(temperature=0, max_tokens=4)

print("----- Start Inference -----")
outputs = model.generate(batch_data, sampling_params)
print(batch_data[0])
labels = [output.outputs[0].text for output in outputs]
save_to_csv(data_path, model_name, labels)
