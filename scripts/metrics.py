import evaluate
import numpy as np

def _get_bertscore(preds, targets):
    bert_score = evaluate.load("bertscore")
    metrics_bertscore = bert_score.compute(predictions=preds, references=targets, lang="ru")
    bert_array = np.array([val for val in metrics_bertscore['precision']]).mean()
    return bert_array.mean()

def get_metrics(preds, targets):
    rouge_score = evaluate.load("rouge")
    bleu_score = evaluate.load("bleu")
    meteor_score = evaluate.load("meteor")

    print('Computing metrics...')
    metrics_rouge = rouge_score.compute(predictions=preds, references=targets)
    print('Rouge computed')
    metrics_bleu = bleu_score.compute(predictions=preds, references=targets)
    print('Bleu computed')
    metrics_meteor = meteor_score.compute(predictions=preds, references=targets)
    print('Meteor computed')
    metrics_bertscore = _get_bertscore(preds, targets)
    print('Bertscore computed')
    print('Metrics computed')
    return {
        'rouge1': metrics_rouge['rouge1'],
        'rouge2': metrics_rouge['rouge2'],
        'rougeL': metrics_rouge['rougeL'],
        'rougeLsum': metrics_rouge['rougeLsum'],
        'bleu': metrics_bleu['bleu'],
        'meteor': metrics_meteor['meteor'],
        'bertscore': metrics_bertscore
    }