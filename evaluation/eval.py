from eval_utils import score_name_ori
import json
import logging

# Configure logging.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_metrics(data_subset):
    """
    Compute precision and recall for function, parameter, and variable names.
    Global totals are used to avoid per-entry averaging bias.
    """
    # Initialize global accumulators.
    global_funnames_precision_sum = 0
    global_funnames_recall_sum = 0
    global_funnames_count = 0

    global_params_precision_sum = 0
    global_params_recall_sum = 0
    global_params_count = 0

    global_vars_precision_sum = 0
    global_vars_recall_sum = 0
    global_vars_count = 0

    # Accumulators for overall precision and recall.
    global_total_precision_sum = 0
    global_total_recall_sum = 0
    global_total_count = 0

    # Evaluate each entry and accumulate precision and recall.
    for entry in data_subset:
        # Extract aligned names and predictions.
        funnames_aligned = entry.get("funname_aligned", {})  # Added field.
        params_aligned = entry.get("params_aligned", {})
        vars_aligned = entry.get("vars_aligned", {})
        predict = entry.get("predict", "")

        if not predict:
            logging.warning("Missing prediction for an entry; skipping it.")
            continue  # Skip entries without predictions.

        # Parse predict (multiple predictions may be comma-separated).
        predict_pairs = predict.split(",")
        predict_dict = {}
        for pair in predict_pairs:
            if ":" in pair:
                name, result = pair.split(":", 1)
            else:
                name, result = pair, None
            predict_dict[name] = result

        # Calculate precision and recall for each name category.
        for name, result in predict_dict.items():
            if name in funnames_aligned:  # Added field.
                precision, recall = score_name_ori(funnames_aligned[name], result)
                global_funnames_precision_sum += precision
                global_funnames_recall_sum += recall
                global_funnames_count += 1
                global_total_precision_sum += precision
                global_total_recall_sum += recall
                global_total_count += 1

            if name in params_aligned:
                precision, recall = score_name_ori(params_aligned[name], result)
                global_params_precision_sum += precision
                global_params_recall_sum += recall
                global_params_count += 1
                global_total_precision_sum += precision
                global_total_recall_sum += recall
                global_total_count += 1

            if name in vars_aligned:
                precision, recall = score_name_ori(vars_aligned[name], result)
                global_vars_precision_sum += precision
                global_vars_recall_sum += recall
                global_vars_count += 1
                global_total_precision_sum += precision
                global_total_recall_sum += recall
                global_total_count += 1

    # Calculate global average precision and recall.
    average_funnames_precision = global_funnames_precision_sum / global_funnames_count if global_funnames_count > 0 else 0
    average_funnames_recall = global_funnames_recall_sum / global_funnames_count if global_funnames_count > 0 else 0

    average_params_precision = global_params_precision_sum / global_params_count if global_params_count > 0 else 0
    average_params_recall = global_params_recall_sum / global_params_count if global_params_count > 0 else 0

    average_vars_precision = global_vars_precision_sum / global_vars_count if global_vars_count > 0 else 0
    average_vars_recall = global_vars_recall_sum / global_vars_count if global_vars_count > 0 else 0

    # Calculate combined precision and recall (parameters + variables).
    params_vars_precision = (global_params_precision_sum + global_vars_precision_sum) / (global_params_count + global_vars_count) if (global_params_count + global_vars_count) > 0 else 0
    params_vars_recall = (global_params_recall_sum + global_vars_recall_sum) / (global_params_count + global_vars_count) if (global_params_count + global_vars_count) > 0 else 0

    # Calculate combined precision and recall (functions + parameters + variables).
    funnames_params_vars_precision = (global_funnames_precision_sum + global_params_precision_sum + global_vars_precision_sum) / (global_funnames_count + global_params_count + global_vars_count) if (global_funnames_count + global_params_count + global_vars_count) > 0 else 0
    funnames_params_vars_recall = (global_funnames_recall_sum + global_params_recall_sum + global_vars_recall_sum) / (global_funnames_count + global_params_count + global_vars_count) if (global_funnames_count + global_params_count + global_vars_count) > 0 else 0

    # Overall precision and recall (functions + parameters + variables).
    average_precision = global_total_precision_sum / global_total_count if global_total_count > 0 else 0
    average_recall = global_total_recall_sum / global_total_count if global_total_count > 0 else 0

    # Calculate each F1 score.
    def calculate_f1_score(precision, recall):
        return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    # F1-scores
    funnames_f1 = calculate_f1_score(average_funnames_precision, average_funnames_recall)
    params_f1 = calculate_f1_score(average_params_precision, average_params_recall)
    vars_f1 = calculate_f1_score(average_vars_precision, average_vars_recall)
    params_vars_f1 = calculate_f1_score(params_vars_precision, params_vars_recall)
    funnames_params_vars_f1 = calculate_f1_score(funnames_params_vars_precision, funnames_params_vars_recall)
    total_f1 = calculate_f1_score(average_precision, average_recall)

    return {
        "funnames_count": global_funnames_count,
        "params_count": global_params_count,
        "vars_count": global_vars_count,
        "total_count": global_total_count,
        "funnames_precision": average_funnames_precision,
        "funnames_recall": average_funnames_recall,
        "funnames_f1": funnames_f1,  # F1-score for funnames
        "params_precision": average_params_precision,
        "params_recall": average_params_recall,
        "params_f1": params_f1,  # F1-score for params
        "vars_precision": average_vars_precision,
        "vars_recall": average_vars_recall,
        "vars_f1": vars_f1,  # F1-score for vars
        "params_vars_precision": params_vars_precision,
        "params_vars_recall": params_vars_recall,
        "params_vars_f1": params_vars_f1,  # F1-score for params & vars
        "funnames_params_vars_precision": funnames_params_vars_precision,
        "funnames_params_vars_recall": funnames_params_vars_recall,
        "funnames_params_vars_f1": funnames_params_vars_f1,  # F1-score for funnames, params & vars
        "total_precision": average_precision,
        "total_recall": average_recall,
        "total_f1_score": total_f1  # F1-score for total
    }

def evaluate_json_file(file_path):
    try:
        # Open the file and read it line by line.
        with open(file_path, 'r', encoding='utf-8') as f:
            data = []
            for line_number, line in enumerate(f, 1):
                try:
                    data.append(json.loads(line))  # Parse each JSON line.
                except json.JSONDecodeError as e:
                    logging.error(f"Error decoding JSON on line {line_number}: {e}")
                    continue  # Skip the invalid line.

        if not data:
            logging.warning("No valid data found in the file.")
            return 0, 0

        # Split data into in_train=1, in_train=0, and all records.
        data_in_train_1 = [entry for entry in data if entry.get("in_train") == 1]
        data_in_train_0 = [entry for entry in data if entry.get("in_train") == 0]

        # Calculate and print metrics for in_train=1.
        logging.info("="*50)
        logging.info("Calculating metrics for in_train = 1")
        logging.info("="*50)
        results_1 = calculate_metrics(data_in_train_1)
        logging.info(f"Sample Counts (in_train=1): Funnames={results_1['funnames_count']}, Params={results_1['params_count']}, Vars={results_1['vars_count']}, Total={results_1['total_count']}")
        logging.info(f"Funnames Precision: {results_1['funnames_precision']:.4f}")
        logging.info(f"Funnames Recall: {results_1['funnames_recall']:.4f}")
        logging.info(f"Funnames F1-score: {results_1['funnames_f1']:.4f}")
        logging.info(f"Params Precision: {results_1['params_precision']:.4f}")
        logging.info(f"Params Recall: {results_1['params_recall']:.4f}")
        logging.info(f"Params F1-score: {results_1['params_f1']:.4f}")
        logging.info(f"Vars Precision: {results_1['vars_precision']:.4f}")
        logging.info(f"Vars Recall: {results_1['vars_recall']:.4f}")
        logging.info(f"Vars F1-score: {results_1['vars_f1']:.4f}")
        logging.info(f"Params & Vars Precision: {results_1['params_vars_precision']:.4f}")
        logging.info(f"Params & Vars Recall: {results_1['params_vars_recall']:.4f}")
        logging.info(f"Params & Vars F1-score: {results_1['params_vars_f1']:.4f}")
        logging.info(f"Funnames & Params & Vars Precision: {results_1['funnames_params_vars_precision']:.4f}")
        logging.info(f"Funnames & Params & Vars Recall: {results_1['funnames_params_vars_recall']:.4f}")
        logging.info(f"Funnames & Params & Vars F1-score: {results_1['funnames_params_vars_f1']:.4f}")
        logging.info(f"Total Precision: {results_1['total_precision']:.4f}")
        logging.info(f"Total Recall: {results_1['total_recall']:.4f}")
        logging.info(f"Total F1-score: {results_1['total_f1_score']:.4f}")

        # Calculate and print metrics for in_train=0.
        logging.info("-"*50)
        logging.info("Calculating metrics for in_train = 0")
        logging.info("-"*50)
        results_0 = calculate_metrics(data_in_train_0)
        logging.info(f"Sample Counts (in_train=0): Funnames={results_0['funnames_count']}, Params={results_0['params_count']}, Vars={results_0['vars_count']}, Total={results_0['total_count']}")
        logging.info(f"Funnames Precision: {results_0['funnames_precision']:.4f}")
        logging.info(f"Funnames Recall: {results_0['funnames_recall']:.4f}")
        logging.info(f"Funnames F1-score: {results_0['funnames_f1']:.4f}")
        logging.info(f"Params Precision: {results_0['params_precision']:.4f}")
        logging.info(f"Params Recall: {results_0['params_recall']:.4f}")
        logging.info(f"Params F1-score: {results_0['params_f1']:.4f}")
        logging.info(f"Vars Precision: {results_0['vars_precision']:.4f}")
        logging.info(f"Vars Recall: {results_0['vars_recall']:.4f}")
        logging.info(f"Vars F1-score: {results_0['vars_f1']:.4f}")
        logging.info(f"Params & Vars Precision: {results_0['params_vars_precision']:.4f}")
        logging.info(f"Params & Vars Recall: {results_0['params_vars_recall']:.4f}")
        logging.info(f"Params & Vars F1-score: {results_0['params_vars_f1']:.4f}")
        logging.info(f"Funnames & Params & Vars Precision: {results_0['funnames_params_vars_precision']:.4f}")
        logging.info(f"Funnames & Params & Vars Recall: {results_0['funnames_params_vars_recall']:.4f}")
        logging.info(f"Funnames & Params & Vars F1-score: {results_0['funnames_params_vars_f1']:.4f}")
        logging.info(f"Total Precision: {results_0['total_precision']:.4f}")
        logging.info(f"Total Recall: {results_0['total_recall']:.4f}")
        logging.info(f"Total F1-score: {results_0['total_f1_score']:.4f}")

        # Calculate and print metrics for all data.
        logging.info("-"*50)
        logging.info("Calculating metrics for ALL DATA (in_train=1 + in_train=0)")
        logging.info("-"*50)
        results_all = calculate_metrics(data)  # Use all data.
        logging.info(f"Sample Counts (ALL): Funnames={results_all['funnames_count']}, Params={results_all['params_count']}, Vars={results_all['vars_count']}, Total={results_all['total_count']}")
        logging.info(f"Funnames Precision: {results_all['funnames_precision']:.4f}")
        logging.info(f"Funnames Recall: {results_all['funnames_recall']:.4f}")
        logging.info(f"Funnames F1-score: {results_all['funnames_f1']:.4f}")
        logging.info(f"Params Precision: {results_all['params_precision']:.4f}")
        logging.info(f"Params Recall: {results_all['params_recall']:.4f}")
        logging.info(f"Params F1-score: {results_all['params_f1']:.4f}")
        logging.info(f"Vars Precision: {results_all['vars_precision']:.4f}")
        logging.info(f"Vars Recall: {results_all['vars_recall']:.4f}")
        logging.info(f"Vars F1-score: {results_all['vars_f1']:.4f}")
        logging.info(f"Params & Vars Precision: {results_all['params_vars_precision']:.4f}")
        logging.info(f"Params & Vars Recall: {results_all['params_vars_recall']:.4f}")
        logging.info(f"Params & Vars F1-score: {results_all['params_vars_f1']:.4f}")
        logging.info(f"Funnames & Params & Vars Precision: {results_all['funnames_params_vars_precision']:.4f}")
        logging.info(f"Funnames & Params & Vars Recall: {results_all['funnames_params_vars_recall']:.4f}")
        logging.info(f"Funnames & Params & Vars F1-score: {results_all['funnames_params_vars_f1']:.4f}")
        logging.info(f"Total Precision: {results_all['total_precision']:.4f}")
        logging.info(f"Total Recall: {results_all['total_recall']:.4f}")
        logging.info(f"Total F1-score: {results_all['total_f1_score']:.4f}")

    except FileNotFoundError:
        logging.error("The input file was not found.")
        return 0, 0
    except json.JSONDecodeError:
        logging.error("The input file contains invalid JSON.")
        return 0, 0
    except Exception as e:
        logging.error("Error processing the input file: %s", type(e).__name__)
        return 0, 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate inference predictions.")
    parser.add_argument(
        "file_path",
        help="Path to the JSONL prediction file, for example ./test/predictions.jsonl",
    )
    args = parser.parse_args()
    evaluate_json_file(args.file_path)
