import os
import time
import logging
import json
import numpy as np

from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Optional


class Timer:
    def __init__(self, name: Optional[str] = None, verbose: bool = True):
        logging.basicConfig(
            level=logging.INFO,
            format="[%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler()],
        )

        self.name = name or "Timer"
        self.verbose = verbose
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        
        if self.verbose:
            logging.info(f"Elapsed time: {self.elapsed_time:.3f}s")
    
    def get_elapsed_time(self) -> float:
        """Get the elapsed time in seconds."""
        if self.elapsed_time is None:
            raise RuntimeError("Timer has not been used yet or is still running")
        return self.elapsed_time
    
    def get_elapsed_time_ms(self) -> float:
        """Get the elapsed time in milliseconds."""
        return self.get_elapsed_time() * 1000


def calculate_metrics(
    predictions: list[str], true_labels: list[str]
) -> dict[str, float]:

    try:
        error_count = sum(1 for pred in predictions if pred == "ERROR")
        error_rate = error_count / len(predictions) if len(predictions) > 0 else 0.0
        
        valid_indices = [i for i, pred in enumerate(predictions) if pred != "ERROR"]
        valid_predictions = [predictions[i] for i in valid_indices]
        valid_true_labels = [true_labels[i] for i in valid_indices]
        
        if len(valid_predictions) > 0:
            accuracy = accuracy_score(valid_true_labels, valid_predictions)
            precision = precision_score(
                valid_true_labels, valid_predictions, average="weighted", zero_division=0
            )
            recall = recall_score(
                valid_true_labels, valid_predictions, average="weighted", zero_division=0
            )
            f1 = f1_score(valid_true_labels, valid_predictions, average="weighted", zero_division=0)
        else:
            accuracy = precision = recall = f1 = 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "error_rate": error_rate,
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "error_rate": 1.0}


def evaluate_multiple_runs(
    evaluation_results: list[tuple[list[str], list[str]]], 
    name: str = "Evaluation",
    save_json: bool = False,
    output_dir: str = "results"
):
    all_metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "error_rate": []
    }
    
    # Store detailed results for JSON export
    detailed_results = {
        "evaluation_name": name,
        "timestamp": datetime.now().isoformat(),
        # "num_runs": len(evaluation_results),
        # "runs": [],
        "summary": {}
    }
    
    for i, (predictions, true_labels) in enumerate(evaluation_results):
        metrics = calculate_metrics(predictions, true_labels)
        # run_data = {
        #     "run_id": i + 1,
        #     "num_predictions": len(predictions),
        #     "num_labels": len(true_labels),
        #     "metrics": metrics,
        #     "predictions": predictions,
        #     "true_labels": true_labels
        # }
        # detailed_results["runs"].append(run_data)
        
        for metric_name, value in metrics.items():
            all_metrics[metric_name].append(value)
    
    print(f"\n{name} - Summary Statistics across {len(evaluation_results)} runs:")
    print("=" * 60)
    
    summary_stats = {}
    for metric_name, values in all_metrics.items():
        metric_summary = summarize(values, metric_name.capitalize().replace("_", " "), save_to_dict=True)
        summary_stats[metric_name] = metric_summary
    
    detailed_results["summary"] = summary_stats
    
    if save_json:
        save_results_to_json(detailed_results, name, output_dir)
    
    return detailed_results


def summarize(metric_values, name, save_to_dict=False):
    values = np.array(metric_values)
    mean = np.mean(values)
    
    summary_dict = {
        "metric_name": name,
        # "num_values": len(values),
        # "values": values.tolist(),
        "mean": float(mean)
    }
    
    if len(values) <= 1:
        summary_dict.update({
            "std": None,
            # "ci_95_lower": None,
            # "ci_95_upper": None,
            # "min": float(mean),
            # "max": float(mean),
            "latex_format": f"{mean:.2%}"
        })
        
        if not save_to_dict:
            print(f"{name} Summary:")
            print(f"  Mean           = {mean:.4f}")
            print(f"  Std Dev        = N/A (insufficient data)")
            # print(f"  95% CI         = N/A (insufficient data)")
            # print(f"  Min / Max      = {mean:.4f} / {mean:.4f}")
            print(f"  LaTeX format   = {mean:.2%}")
            print()
        
        return summary_dict if save_to_dict else None
    
    std = np.std(values, ddof=1)
    # minimum = np.min(values)
    # maximum = np.max(values)
    
    if std == 0:
        summary_dict.update({
            "std": float(std),
            # "ci_95_lower": float(mean),
            # "ci_95_upper": float(mean),
            # "min": float(minimum),
            # "max": float(maximum),
            "latex_format": f"{mean:.2%} ± {std:.2%}"
        })
        
        if not save_to_dict:
            print(f"{name} Summary:")
            print(f"  Mean           = {mean:.4f}")
            print(f"  Std Dev        = {std:.4f}")
            # print(f"  95% CI         = [{mean:.4f}, {mean:.4f}]")
            # print(f"  Min / Max      = {minimum:.4f} / {maximum:.4f}")
            print(f"  LaTeX format   = {mean:.2%} ± {std:.2%}")
            print()
        
        return summary_dict if save_to_dict else None
    
    # ci95 = stats.t.interval(0.95, len(values)-1, loc=mean, scale=std/np.sqrt(len(values)))
    
    summary_dict.update({
        "std": float(std),
        # "ci_95_lower": float(ci95[0]),
        # "ci_95_upper": float(ci95[1]),
        # "min": float(minimum),
        # "max": float(maximum),
        "latex_format": f"{mean:.2%} ± {std:.2%}"
    })
    
    if not save_to_dict:
        print(f"{name} Summary:")
        print(f"  Mean           = {mean:.4f}")
        print(f"  Std Dev        = {std:.4f}")
        # print(f"  95% CI         = [{ci95[0]:.4f}, {ci95[1]:.4f}]")
        # print(f"  Min / Max      = {minimum:.4f} / {maximum:.4f}")
        print(f"  LaTeX format   = {mean:.2%} ± {std:.2%}")
        print()
    
    return summary_dict if save_to_dict else None


def save_results_to_json(results_dict, experiment_name, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {filepath}")
        print(f"Summary statistics:")
        # print(f"   - Number of runs: {results_dict['num_runs']}")
        print(f"   - Timestamp: {results_dict['timestamp']}")
        
        # Print key metrics summary
        if 'summary' in results_dict:
            for metric, stats in results_dict['summary'].items():
                if stats and 'mean' in stats:
                    print(f"   - {metric.capitalize()}: {stats['mean']:.4f}")
        
    except Exception as e:
        print(f"Error saving results to JSON: {e}")