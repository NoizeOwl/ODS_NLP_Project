from ollama import chat
from ollama import ChatResponse

def get_predicts_hf(pipe, message, max_new_tokens=512):
    try:
        res = pipe(message, max_new_tokens=max_new_tokens)
        return res[0]['generated_text'][-1]['content']
    except:
        print('Error')
        return None

def get_predicts_ollama(model_name, message, options=None):
    # try:
        if options is not None:
            res: ChatResponse = chat(model_name, message, options)
        else:
            res: ChatResponse = chat(model_name, message)
        return res['message']['content']
    # except:
    #     print('Error')
        return None