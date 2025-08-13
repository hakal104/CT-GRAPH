import os
import re
from types import SimpleNamespace
import argparse
import json
import numpy as np

from report_gen.evaluation.get_f1_metrics import get_abn_labels
from report_gen.evaluation.nlg_metrics import calculate_bleu, calculate_meteor, calculate_rouge_l
from report_gen.data.get_data import get_data
from sklearn.metrics import f1_score, precision_score, recall_score
from report_gen.model.CTRepgen import CTConfig

def clean_report_mimic_cxr(report):
    """
    Preprocessing the report.
    Args:
        report (str): input report.
    Returns:
        str: Cleaned report.
    """
    report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
        .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
        .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
        .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
        .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
        .strip().lower().split('. ')
    sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                    .replace('\\', '').replace("'", '').strip().lower())
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
    report = ' . '.join(tokens) + ' .'
    return report

def evaluate_model(gt_report, gen_report):
    """
    Calculates basic NLG metrics based on given ground-truth report
    and generated report.
    
    Args:
        gt_report (str): ground-truth report.
        gen_report (str): generated report.
    Returns:
        dict: Contains relevant NLG metrics.
    """
    
    gt_report = clean_report_mimic_cxr(gt_report)
    gen_report = clean_report_mimic_cxr(gen_report)
    bleu_1,bleu_2,bleu_3,bleu_4 = calculate_bleu(gt_report, gen_report)
    print(bleu_1)
    meteor = calculate_meteor(gt_report, gen_report)
    rouge_l = calculate_rouge_l(gt_report, gen_report)
    
    metrics ={'BLEU-1': bleu_1,
              'BLEU-2': bleu_2,
              'BLEU-3': bleu_3,
              'BLEU-4': bleu_4,
              'METEOR': meteor,
              'ROUGE-L': rouge_l}
    
    return metrics

def get_ce_metrics(gt_abn,gen_abn):
    """
    Calculates cross-entropy (CE) metrics using different averaging methods 
    (e.g., macro) and per-label values based on a ground-truth report 
    and a generated report.
    
    Args:
        gt_report (str): ground-truth report.
        gen_report (str): generated report.
    Returns:
        ce_avg (dict): Contains averaged CE metrics.
        per_labe (dict): Contains CE metrics per label.
    """
    
    ce_avg = {}
    per_label = {}
    
    ce_avg['f1_micro'] = f1_score(gt_abn, gen_abn, average="micro")
    ce_avg['f1_macro'] = f1_score(gt_abn, gen_abn, average="macro")
    ce_avg['f1_weighted'] = f1_score(gt_abn, gen_abn, average="weighted")

    ce_avg['precision_micro'] = precision_score(gt_abn, gen_abn, average="micro")
    ce_avg['precision_macro'] = precision_score(gt_abn, gen_abn, average="macro")
    ce_avg['precision_weighted']  = precision_score(gt_abn, gen_abn, average="weighted")

    ce_avg['recall_micro'] = recall_score(gt_abn, gen_abn, average="micro")
    ce_avg['recall_macro'] = recall_score(gt_abn, gen_abn, average="macro")
    ce_avg['recall_weighted'] = recall_score(gt_abn, gen_abn, average="weighted")

    per_label['f1'] = f1_score(gt_abn, gen_abn, average=None).tolist()
    per_label['precision'] = precision_score(gt_abn, gen_abn, average=None).tolist()
    per_label['recall'] = recall_score(gt_abn, gen_abn, average=None).tolist()
    
    return ce_avg, per_label

def get_gt_abn_labels(model_dir):
    """
    Provides the ground-truth validation results and abnormality labels.
    
    Args:
        model_dir (str): Path to model directory.
    Returns:
        val_reports (list): Contains averaged CE metrics.
        val_abn_labels (np.ndarray): Contains CE metrics per label.
    """
    
    config_path = os.path.join(model_dir,'model_config.json')
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    config = CTConfig.from_dict(config_dict) 
    
    data_config_path = os.path.join(model_dir,'data_config.json')
    with open(data_config_path) as f:
        data_args = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    
    _, _, val_reports, val_abn_labels = get_data(mode='valid', model_config=config, data_args=data_args)
    
    return val_reports, val_abn_labels

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Evaluate generated reports.")
    parser.add_argument("--model_dir", type=str, default=None, help="Path to the weights of the trained report generation model.")
    parser.add_argument("--ce_model_path", type=str, default=None, help="Path to weights of the trained abnormality classification model.")
    
    args = parser.parse_args()

    input_path = os.path.join(args.model_dir,'reports.json')
    
    save_nlg = os.path.join(args.model_dir,'nlg_metrics.json')
    save_ce = os.path.join(args.model_dir,'ce_metrics.json')
    save_nlg_per_report = os.path.join(args.model_dir,'nlg_metrics_per_report.json')
    save_ce_per_label = os.path.join(args.model_dir,'ce_metrics_per_label.json')
    
    with open(input_path) as f:
        pred_reports = json.load(f)
    gt_reports, gt_abn_labels = get_gt_abn_labels(args.model_dir)

    nlg_metrics_per_report = {
        'BLEU-1': [],
        'BLEU-2': [],
        'BLEU-3': [],
        'BLEU-4': [],
        'METEOR': [],
        'ROUGE-L': []
    }

    for pred_rep, gt_rep in zip(pred_reports, gt_reports):

        report_metrics = evaluate_model(gt_rep, pred_rep)
        for key, value in report_metrics.items():
            print(key,value)
            nlg_metrics_per_report[key].append(value)

    mean_nlg_metrics = {k: float(np.mean(v)) for k, v in nlg_metrics_per_report.items()}

    gen_abn_labels = get_abn_labels(pred_reports, args.ce_model_path)
    gen_abn_labels.to_csv(os.path.join(args.model_dir,'gen_ce_labels.csv'))
    gen_abn_labels = gen_abn_labels.to_numpy()
    ce_metrics, ce_metrics_per_label = get_ce_metrics(gt_abn_labels, gen_abn_labels)
    
    with open(save_nlg, "w") as f:
        json.dump(mean_nlg_metrics, f, indent=4)

    with open(save_ce, "w") as f:
        json.dump(ce_metrics, f, indent=4)
        
    with open(save_nlg_per_report, "w") as f:
        json.dump(nlg_metrics_per_report, f, indent=4)
        
    with open(save_ce_per_label, "w") as f:
        json.dump(ce_metrics_per_label, f, indent=4)