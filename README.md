# Financial Sentiment Analysis
## Dataset
* Oxbrige Sentimental Analysis Dataset: The whole unprocessed dataset is saved in `data/sentences.csv` with the processed data in `data/train_easy.csv` and `data/train_hard.csv`. The `train_easy` dataset contains 310 easier data points (that Yingxian and Zhang Kai agree with) and the `train_hard` dataset contains 100 harder data points (that Yingxian and Zhang Kai disagree with and Qingyi dives deeper into).
* [Financial PhraseBank (FPB) Dataset](https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis): The `FPB-all` contains 5843 labels sentences. The `FPB` is a small subject for simple evaluation.

## Models
* [NLTK](https://www.nltk.org/api/nltk.sentiment.sentiment_analyzer.html)
* [FinBERT](https://github.com/ProsusAI/finBERT)
* [FinMA](https://github.com/The-FinAI/PIXIU): we use [vLLM](https://github.com/vllm-project/vllm) for fast inference for LLMs as shown in `run_vllm.py`. Note that you need CUDA of around 20GB to inference a 7B model efficiently.
* GPT-4O mini: we use the Azure API for inference in `openai.ipynb`.
* DeepSeek: we use the OpenAI API for inference in `openai.ipynb`.

## Results
After you get the inferece result from the models, you may use the `data/eval.ipynb` to generate the accuracy table. By default, the result will be saved in `data/results.md`.

| Model | train_easy.csv | train_hard.csv | FPB.csv |
|-------|-----|-----|-----|
| fingpt-sentiment_llama2-13b_lora | 0.7573 | 0.5200 | 0.8696 | 
| fingpt-mt_llama2-7b_lora | 0.7832 | 0.5500 | 0.8495 | 
| openai | 0.8641 | 0.6200 | 0.7659 | 
| finma-7b-nlp | 0.7152 | 0.5400 | 0.8127 | 
| finbert | 0.6958 | 0.4100 | 0.7358 | 
| NLTK | 0.3786 | 0.4400 | 0.2642 | 
| deepseek | 0.8317 | 0.5500 | 0.8027 | 