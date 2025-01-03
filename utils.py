import pandas as pd
from torch.utils.data import Dataset

task_prompt = '''Analyze the sentiment of this statement extracted from a financial news article. 
Provide your answer as either negative, positive or neutral without anything else.
For instance, The company's stocks plummeted following the scandal. would be classified as negative.'''

template_prompt = '''Instruction: [Task Prompt]
Text: [Input Text]
Response: '''

class TestDataset(Dataset):
    def __init__(self, dataset_path, use_sys_prompt=False, use_few_shot=False, use_template=False):
        self.data = pd.read_csv(dataset_path)
        self.keys = ['sentence']
        # self.keys = list(self.data.keys())
        self.use_sys_prompt = use_sys_prompt
        self.use_few_shot = use_few_shot
        self.use_template = use_template

    def apply_template(self, sentence):
        prompt = template_prompt.replace("[Task Prompt]", task_prompt)
        prompt = prompt.replace("[Input Text]", sentence)
        return prompt
    
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