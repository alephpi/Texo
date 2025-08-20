import evaluate
from rapidfuzz.distance import Levenshtein

bleu = evaluate.load("bleu", keep_in_memory=False)

def compute_bleu(pred_str: list[str], ref_str: list[str]):
    """compute bleu score of two lists of strings, notice that the max order is 4, so any ref_str less than 4 words would have 0 bleu score."""
    results = bleu.compute(
        predictions=pred_str, references=ref_str, tokenizer=lambda x: x.split(" ")
    )
    return results["bleu"]

def compute_edit_distance(pred_str: list[str], ref_str: list[str]):
    results = [Levenshtein.normalized_distance(p, r, processor=str.split) for p, r in zip(pred_str, ref_str)]
    return sum(results)/len(results)

if __name__ == '__main__':
    pred_str = "hello world"
    ref_str = "hello world"
    pred_str2 = "[ C ] h t g k b S b"
    ref_str2 = "[ C ] h t a b S b"
    assert compute_bleu([pred_str], [ref_str]) == 0
    assert compute_bleu([pred_str2], [ref_str2]) == 0.5253819788848316
    assert compute_edit_distance([pred_str], [ref_str]) == 0
    assert compute_edit_distance([pred_str2], [ref_str2]) == 0.2