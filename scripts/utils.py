from datasets import load_dataset
from transformers import pipeline
import json


def get_dataset(dataset_name):
    return load_dataset(dataset_name)

def save_json(data, save_path):
    with open(save_path, 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False)

def load_json(data_path):
    with open(data_path, 'r', encoding='utf8') as f:
        return json.load(f)

def get_model_pipeline(model_name):
    return pipeline('text-generation', model=model_name, device='cuda')

def make_prompt(data, example_indices_full, example_index_to_summarize):
    prompt = ''
    for index in example_indices_full:
        dialogue = data['dialog'][index]
        summary = data['summary'][index]

        prompt += f"""
Диалог:

{dialogue}

Что в данном диалоге произошло?
{summary}


"""
    
    dialogue = data['dialog'][example_index_to_summarize]
    
    prompt += f"""
Диалог:

{dialogue}

Что в данном диалоге произошло?
"""
        
    return prompt

def make_prompt_final(data, example_indices_full, example_index_to_summarize):
    prompt = ''
    for index in example_indices_full:
        # dialogue = data['dialog'][index]
        summary = data['summary'][index]

        prompt += f"""
Пример сокращенного диалога:
{summary}
"""
    
    dialogue = data['dialog'][example_index_to_summarize]
    
    prompt += f"""
Диалог для сокращения:

{dialogue}

Перепишите диалог, сократив его до короткого списка с ключевыми задачами и сроками, без добавления дополнительных комментариевю.
В твоем сокращенном тексте обязательно должны быть слова "Имя работника", "задача" и "срок", как в примере.
"""
        
    return prompt