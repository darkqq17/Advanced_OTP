import json
from collections import defaultdict

def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_aspect_sentiments(data):
    aspect_sentiments = []
    for item in data:
        outputs = item.get('output', [])
        for output in outputs:
            # Unpack the output list
            if len(output) == 4:
                aspect, category, opinion, sentiment = output
                aspect_sentiments.append((item['input'], aspect.strip().lower(), sentiment.strip()))
            else:
                print(f"Warning: Unexpected output format in item: {item}")
    return aspect_sentiments

def calculate_metrics(true_data, pred_data):
    true_aspects = extract_aspect_sentiments(true_data)
    pred_aspects = extract_aspect_sentiments(pred_data)

    # Create sets for comparison
    true_set = set(true_aspects)
    pred_set = set(pred_aspects)

    # True Positives (TP): Correctly predicted sentiments for aspects present in both sets
    tp = len(true_set & pred_set)

    # False Positives (FP): Aspects in pred_set not in true_set
    fp = len(pred_set - true_set)

    # False Negatives (FN): Aspects in true_set not in pred_set
    fn = len(true_set - pred_set)

    # Precision, Recall, F1-Score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1_score, tp, fp, fn

def main():
    # Load data from JSON files
    true_data = load_data('answer_part.json')
    pred_data = load_data('output.json')

    # Calculate metrics
    precision, recall, f1_score, tp, fp, fn = calculate_metrics(true_data, pred_data)

    # Print the results
    print("Sentiment Analysis Metrics:")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1_score:.4f}")

if __name__ == "__main__":
    main()
