import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

#nltk.download('wordnet')
#nltk.download('omw-1.4')  # Optional: Ensures compatibility with wordnet for multilingual support

def tokenize(text):
    """
    Tokenizes input text into a list of words.
    Args:
        text (str): Input string to tokenize.
    Returns:
        list: Tokenized text.
    """
    return text.split()

def calculate_bleu(ground_truth, gen_report):
    """
    Calculates BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores.
    Args:
        ground_truth (str): Ground truth text.
        gen_report (str): Generated text.
    Returns:
        dict: BLEU-1 to BLEU-4 scores.
    """
    gt = [ground_truth]
    gen = gen_report
    gt = [ground_truth.split()]
    gen = gen_report.split()
    smoothing_function = SmoothingFunction().method1  # To avoid zero scores with small texts

    bleu1 = sentence_bleu(gt, gen, weights=(1, 0, 0, 0), smoothing_function=smoothing_function)
    bleu2 = sentence_bleu(gt, gen, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function)
    bleu3 = sentence_bleu(gt, gen, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing_function)
    bleu4 = sentence_bleu(gt, gen, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)

    return  bleu1, bleu2, bleu3, bleu4

def calculate_meteor(ground_truth, gen_report):
    """
    Calculates METEOR score.
    Args:
        ground_truth (str): Ground truth text.
        gen_report (str): Generated text.
    Returns:
        float: METEOR score.
    """
    gt = tokenize(ground_truth)
    gen = tokenize(gen_report)
    return nltk.translate.meteor_score.single_meteor_score(gt, gen)

def calculate_rouge_l(ground_truth, gen_report):
    """
    Calculates ROUGE-L score.
    Args:
        ground_truth (str): Ground truth text.
        gen_report (str): Generated text.
    Returns:
        float: ROUGE-L score.
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = scorer.score(ground_truth, gen_report)
    rouge_l = scores["rougeL"].fmeasure
    return rouge_l