import pandas as pd
from torch.utils.data import Dataset

dataset_map = {
    "FPB": {"path": "data/sentimental_analysis/FPB/test.csv", "type": "FPB"},
    "FIQASA": {"path": "data/sentimental_analysis/FIQASA/test.csv", "type": "FIQASA"},
    "OXBRIDGE_EASY": {"path": "data/sentimental_analysis/OXBRIDGE/test_easy.csv", "type": "OXBRIDGE"},
    "OXBRIDGE_HARD": {"path": "data/sentimental_analysis/OXBRIDGE/test_hard.csv", "type": "OXBRIDGE"},
    "OXBRIDGE": {"path": "data/sentimental_analysis/OXBRIDGE/test.csv", "type": "OXBRIDGE"},
}

task_prompt_map = {
    "FPB": '''Analyze the sentiment of this statement extracted from a financial news article. 
Provide your answer as either negative, positive or neutral without anything else.
For instance, The company's stocks plummeted following the scandal. would be classified as negative.''',
    "FIQASA": '''What is the sentiment of this news? Please choose an answer from {negative/neutral/positive} without anything else.''', 
    "OXBRIDGE": '''Analyze the sentiment of this statement extracted from a financial news article. 
Provide your answer as either negative, positive or neutral without anything else.
For instance, The company's stocks plummeted following the scandal. would be classified as negative.''',
}

template_prompt = '''Instruction: [Task Prompt]
Input: [Input Text]
Answer: '''

class TestDataset(Dataset):
    def __init__(self, dataset, use_template=True, keys=['sentence']):
        dataset_path = dataset_map[dataset]['path']
        self.dataset_type = dataset_map[dataset]['type']
        self.data = pd.read_csv(dataset_path)
        self.keys = keys
        self.use_template = use_template

    def apply_template(self, sentence):
        if self.use_template: 
            prompt = template_prompt.replace("[Task Prompt]", task_prompt_map[self.dataset_type])
            prompt = prompt.replace("[Input Text]", sentence)
            return prompt
        return sentence
    
    def __len__(self):        
        return len(self.data)
    
    def __getitem__(self, index):
        return{key: self.apply_template(self.data[key][index]) for key in self.keys}
    
    def collate_fn(self, batch):
        return {key: [self.apply_template(x[key]) for x in batch] for key in self.keys}
    
def save_to_csv(output_dir, key, values):
    df = pd.read_csv(output_dir)
    df[key] = values
    df.to_csv(output_dir, index=False)
    print(f"Save to {output_dir}")