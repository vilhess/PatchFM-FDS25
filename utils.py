import json

def add_result(human_loss, ai_loss):
    try:
        with open('results.json', 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print("File not found, creating a new one.")
        results = {"human_loss": 0.0, "ai_loss": 0.0, "counts": 0}
    
    results['human_loss'] += human_loss
    results['ai_loss'] += ai_loss
    results['counts'] += 1
    
    with open('results.json', 'w') as f:
        json.dump(results, f)

def get_results():
    try:
        with open('results.json', 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print("File not found, returning default values.")
        return 0.0, 0.0, 0
    
    mean_human_loss = results['human_loss'] / results['counts'] if results['counts'] > 0 else 0.0
    mean_ai_loss = results['ai_loss'] / results['counts'] if results['counts'] > 0 else 0.0
    return mean_human_loss, mean_ai_loss, results['counts']
