import argparse
# import gzip
import json
# import logging
import os
# import re
# import sys
# from functools import partial
# from pathlib import Path
# from typing import Union

import numpy as np

from lm_eval.evaluator import simple_evaluate
from lm_eval.tasks import TaskManager
from dataclasses import dataclass

from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
api = HfApi()

# from collect_blimp import glue_parser, devbench_parser, zeroshot_harness_parser
# from score_predictions import score_predictions
# from tabulate import tabulate

@dataclass
class args_default: 
    model:str = 'hf'
    seed:int = 1
    tasks:str = ''
    image_src:str = None
    model_args:str = ''
    num_fewshot:int = None
    image_src_split:str = None
    batch_size:int = 32
    max_batch_size:int = None
    device:str = 'cuda'
    limit = None
    use_cache = None
    check_integrity = False
    write_out = False
    log_samples = False
    gen_kwargs = None
    verbosity = 'CRITICAL' #CRITICAL|ERROR|WARNING|INFO|DEBUG
    predict_only = False
    include_path = None

def main(model_name, revision, tasks, out_file): 
    args = args_default(
        model_args=f'pretrained={model_name},revision={revision},backend=causal',
        tasks=tasks)
    task_manager = TaskManager(args.verbosity, include_path=args.include_path)

    results = simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=args.tasks.split(','),
        image_src=args.image_src,
        image_src_split=args.image_src_split,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        use_cache=args.use_cache,
        limit=args.limit,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        log_samples=args.log_samples,
        gen_kwargs=args.gen_kwargs,
        task_manager=task_manager,
        verbosity=args.verbosity,
        predict_only=args.predict_only,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
    )

    def _handle_non_serializable(o):
        if isinstance(o, np.int64) or isinstance(o, np.int32):
            return int(o)
        elif isinstance(o, set):
            return list(o)
        else:
            return str(o)
        
    with open(out_file, 'a') as f: 
        for task, res in results['results'].items(): 
            line = f'{task},{model_name.split("/")[-1]},{revision},{res["acc,none"]},{res["acc_stderr,none"]}\n'
            f.write(line)
            f.flush()

if __name__=='__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('model_names', type=str, nargs='+')
    # parser.add_argument('-r', '--revision', type=str, default='main')
    parser.add_argument('-t', '--tasks', type=str, default='blimp_supplement', metavar='blimp_supplement,blimp_filtered')
    parser.add_argument('-o', '--out_file', type=str, default='rasta_results.txt')
    args = parser.parse_args()

    for model_name_and_revision in args.model_names: 
        model_name_and_revision = model_name_and_revision.split(':')
        if len(model_name_and_revision) == 1: 
            model_name, revision = model_name_and_revision[0], 'main'
        else: 
            model_name, revision = model_name_and_revision

        if revision.isnumeric():
            try: 
                commit_ids = [commit.commit_id for commit in api.list_repo_commits(model_name)[:int(revision)]]
            except RepositoryNotFoundError: 
                print(f'could not find model {model_name}, skipping')
        else: 
            commit_ids = [revision]

        for commit_id in commit_ids: 
            main(model_name, commit_id, args.tasks, args.out_file)

# print(make_table(results))
# if "groups" in results:
#     print(make_table(results, "groups"))


# def make_task_dict(task_name, subtask_name, preds_path):
#     def _add_to_dict(index, prediction, task_dict):
#         example_id = f"{subtask_name}_{index}"
#         if type(prediction) == str:
#             prediction = prediction.replace("\\n", "\n")
#         task_dict["predictions"].append({"id": example_id, "pred": prediction})
    
#     if not os.path.exists(preds_path):
#         raise FileNotFoundError(f"Error: no predictions found for \"{subtask_name}\" (in {task_name}).")
    
#     task_dict = {"predictions": []}
#     with open(preds_path, 'r') as predictions_file:
#         if task_name in ("blimp", "blimp_supplement", "ewok", "vqa", "winoground"):
#             predictions = zeroshot_harness_parser(predictions_file)
#         elif task_name == "glue":
#             predictions = glue_parser(predictions_file)
#         else:
#             prediction_matrix = devbench_parser(preds_path)
#             task_dict["predictions"] = prediction_matrix
#             return task_dict
        
#         for idx, prediction in enumerate(predictions):
#             _add_to_dict(idx, prediction, task_dict)
    
#     return task_dict

# model_basename = model_name.split("/")[-1]
# task_dicts = {"glue": {}, "blimp": {}, "blimp_supplement": {}, "ewok": {}}

# TEXT_TASKS = {
#     # "glue": [],
#     "glue": ["cola", "sst2", "mrpc", "qqp", "mnli", "mnli-mm", "qnli", "rte",
#              "boolq", "multirc", "wsc"],
#     # Lots of BLiMP tasks – use verifier function below to see if you've included everything.
#     "blimp": [taskname.split(".jsonl")[0] for taskname in os.listdir("evaluation_data/blimp_filtered/")],
#     "blimp_supplement": ["hypernym", "qa_congruence_easy", "qa_congruence_tricky",
#                    "subject_aux_inversion", "turn_taking"],
#     "ewok": []
# }

# # Build task predictions dictionaries
# for bigtask in TEXT_TASKS: 
#     if bigtask not in tasks: 
#         continue
#     if bigtask == 'glue': 
#         for task in TEXT_TASKS["glue"]:
#             results_dir = "finetune"
#             # if args.glue_lora:
#             #     results_dir = "lora"
#             preds_path = f"results/{results_dir}/{model_basename}/{task}/predictions.txt"
#             task_dicts["glue"][task] = make_task_dict("glue", task, preds_path)
#     elif bigtask == 'blimp': 
#         for task in TEXT_TASKS["blimp"]:
#             preds_path = f"results/blimp/{model_basename}/blimp_{task}_filtered_results.jsonl"
#             task_dicts["blimp"][task] = make_task_dict("blimp", task, preds_path)
#     elif bigtask == 'blimp_supplement': 
#         for task in TEXT_TASKS["blimp_supplement"]:
#             preds_path = f"results/blimp/{model_basename}/blimp_supplement_{task}_results.jsonl"
#             task_dicts["blimp_supplement"][task] = make_task_dict("blimp_supplement", task, preds_path)
#     elif bigtask == 'ewok': 
#         for task in TEXT_TASKS["ewok"]:
#             preds_path = f"results/ewok/{model_basename}/ewok_{task}_filtered_results.jsonl"
#             task_dicts["ewok"][task] = make_task_dict("ewok", task, preds_path)

# # Save predictions
# # preds_name = "withvision" if args.include_vision_tasks else "textonly"
# # with gzip.open(f"{model_basename}_{preds_name}_predictions.json.gz", 'wt') as predictions_out:
# #     json.dump(task_dicts, predictions_out)

# predictions = task_dicts
# for task in predictions:
#     if task == "glue":
#         gold_dir = "evaluation_data/glue_filtered/"
#     elif task == "blimp":
#         gold_dir = "evaluation_data/blimp_filtered/"
#     elif task == "blimp_supplement":
#         gold_dir = "evaluation_data/supplement_filtered/"
#     elif task == "ewok":
#         gold_dir = "evaluation_data/ewok_filtered/"
#     elif task == "vqa":
#         gold_dir = "evaluation_data/vqa_filtered/"
#     elif task == "winoground":
#         gold_dir = "evaluation_data/winoground_filtered/"
#     # elif task == "devbench":
#     #     scores = score_devbench(predictions)
#     #     score_rows = []
#     #     accs, human_sims = [], []
#     #     for subtask in scores:
#     #         if type(scores[subtask]) == tuple:
#     #             acc, kl_metric = scores[subtask]
#     #             accs.append(acc)
#     #             human_sims.append(kl_metric)
#     #             score_rows.append([subtask, f"{acc:.3f}", f"{kl_metric:.3f}"])
#     #         else:
#     #             human_sims.append(scores[subtask])
#     #             accs.append(scores[subtask])
#     #             score_rows.append([subtask, f"{scores[subtask]:.3f}", f"{scores[subtask]:.3f}"])
#     #     avg_acc = np.mean(accs)
#     #     avg_human_sim = np.mean(human_sims)
#     #     score_rows.append(["*Average*", f"{avg_acc:.3f}", f"{avg_human_sim:.3f}"])
#     #     print(tabulate(score_rows, headers=[f"{task} subtask", "Acc", "Human Similarity"]))
#     #     print()
#     #     continue

#     scores = score_predictions(predictions, task, gold_dir)
#     score_rows = [[k, f"{v:.3f}"] for k, v in scores.items()]
#     scores_list = list(scores.values())
#     avg_task_score = np.mean(scores_list)
#     if len(scores_list) > 1:
#         score_rows.append(["*Average*", f"{avg_task_score:.3f}"])
#     print(tabulate(score_rows, headers=[f"{task} subtask", "Score"]))
#     print()

# # Make sure dictionary includes everything and is formatted correctly
# # verify_dict(task_dicts, args.include_vision_tasks)