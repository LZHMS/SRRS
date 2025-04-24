"""
Goal
---
1. Read test results from log.txt files
2. Compute mean and std across different folders (seeds)

Usage
---
Assume the output files are saved under output/my_experiment,
which contains results of different seeds, e.g.,

my_experiment/
    seed1/
        log.txt
    seed2/
        log.txt
    seed3/
        log.txt

Run the following command from the root directory:

$ python tools/parse_test_res.py output/my_experiment

Add --ci95 to the argument if you wanna get 95% confidence
interval instead of standard deviation:

$ python tools/parse_test_res.py output/my_experiment --ci95

If my_experiment/ has the following structure,

my_experiment/
    exp-1/
        seed1/
            log.txt
            ...
        seed2/
            log.txt
            ...
        seed3/
            log.txt
            ...
    exp-2/
        ...
    exp-3/
        ...

Run

$ python tools/parse_test_res.py output/my_experiment --multi-exp
"""
import os
import re
import numpy as np
import os.path as osp
import argparse
from collections import OrderedDict, defaultdict

from dassl.utils import check_isfile, listdir_nohidden


def compute_ci95(res):
    return 1.96 * np.std(res) / np.sqrt(len(res))

def listdir_sorted_by_fp(directory):
    subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

    def sort_key(name):
        match = re.search(r'(\d+)FP', name)
        return int(match.group(1)) if match else float('inf')
    
    sorted_subdirs = sorted(subdirs, key=sort_key)
    
    return sorted_subdirs

def parse_function(*metrics, directory="", args=None, end_train_signal=None, end_test_signal=None):
    print(f"Parsing files in {directory}")
    subdirs = listdir_nohidden(directory, sort=True)
    metrics = metrics[0]

    outputs = []

    for subdir in subdirs:
        fpath = osp.join(directory, subdir, "log.txt")
        assert check_isfile(fpath)
        train_good_to_go, test_good_to_go = False, False
        output = OrderedDict()

        with open(fpath, "r") as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()

                if line == end_train_signal:
                    train_good_to_go = True
                    test_good_to_go = False

                if line == end_test_signal:
                    test_good_to_go = True
                    train_good_to_go = False

                for metric in metrics:
                    match = metric["regex"].search(line)
                    if match and train_good_to_go:
                        if "file" not in output:
                            output["file"] = fpath
                        num_list_str = match.group(1)
                        num_list = [float(num) for num in num_list_str.split(',')]
                        name = metric["name"]
                        output[name] = num_list
                
                    if match and test_good_to_go:
                        if "file" not in output:
                            output["file"] = fpath
                        num = float(match.group(1))
                        name = metric["name"]
                        output[name] = num

        if output:
            outputs.append(output)

    assert len(outputs) > 0, f"Nothing found in {directory}"

    metrics_results = defaultdict(list)

    for output in outputs:
        msg = ""
        for key, value in output.items():
            if isinstance(value, float):
                msg += f"{key}: {value:.2f}%. "
            elif isinstance(value, list):
                pass
            else:
                msg += f"{key}: {value}. "
            if key != "file":
                metrics_results[key].append(value)
        print(msg)

    output_results = OrderedDict()
    output_std = OrderedDict()

    print("===")
    print(f"Summary of directory: {directory}")
    for key, values in metrics_results.items():
        if isinstance(values[0], list):
            values = np.array(values)
            avg = np.round(np.mean(values, axis=0), 2)
            std = np.round(np.std(values, axis=0), 2)
            print(f"* {key} avg: {list(avg)}")
            print(f"* {key} std: {list(std)}")
        else:
            avg = np.mean(values)
            std = compute_ci95(values) if args.ci95 else np.std(values)
            print(f"* {key}: {avg:.2f}% +- {std:.2f}%")
        output_results[key] = avg
        output_std[key] = std
    print("===")
    return output_results, output_std


def main(args, end_train_signal, end_test_signal=None):
    noise_rate, acc = args.keyword.split(',')[0], args.keyword.split(',')[1]
    metrics = [{
        "name": noise_rate,
        "regex": re.compile(fr"\* {noise_rate}: \[([\d\.,\sEe+-]+)\]"),
        },
        {
        "name": acc,
        "regex": re.compile(fr"\* {acc}: ([\.\deE+-]+)%"),
        }
    ]

    if args.multi_exp:

        final_results = defaultdict()
        final_std = defaultdict()
        for directory in listdir_sorted_by_fp(args.directory):
            attributes = directory.rsplit('_', 2)
            directory = osp.join(args.directory, directory)
            for sub_directory in listdir_nohidden(directory):
                gce = sub_directory.split('_')[-1]
                
                sub_directory = osp.join(directory, sub_directory)
                results, out_std = parse_function(
                    metrics, directory=sub_directory, args=args, end_train_signal=end_train_signal, end_test_signal=end_test_signal
                )
                config_key = f"{attributes[-3]}_{attributes[-1]}_{gce}"
                fp_key = attributes[-2]
                if config_key not in final_results:
                    final_results[config_key] = defaultdict()
                if fp_key not in final_results[config_key]:
                    final_results[config_key][fp_key] = defaultdict(list)
                
                if config_key not in final_std:
                    final_std[config_key] = defaultdict()
                if fp_key not in final_std[config_key]:
                    final_std[config_key][fp_key] = defaultdict(list)

                for key, value in results.items():
                    final_results[config_key][fp_key][key].append(value)
                    final_std[config_key][fp_key][key].append(out_std[key])

        for config_key, config_values in final_results.items():
            noise = ""
            msg = defaultdict(str)
            msgv = defaultdict(str)
            average = defaultdict(list)
            for fp_key in sorted(config_values.keys(), key=lambda x: int(x[:-2])):
                fp_values = config_values[fp_key]
                for key, values in fp_values.items():
                    for id, v in enumerate(values):
                        if np.ndim(v) == 0:
                            msg[key] += f"| {v:.2f} +- {final_std[config_key][fp_key][key][id]:.2f}% "
                            msgv[key] += f"{v:.2f} "
                            average[key].append(np.mean(values))
                noise += f"| {fp_key} "
            print(f"Experimental Config: {config_key}")
            print(f"* noise: {noise}| average |")
            for metric, value in msg.items():
                print(f"* {metric}: {value}| {np.mean(average[metric]):.2f}% |")
            
            for metric, value in msgv.items():
                print(f"* {metric}: {value} {np.mean(average[metric]):.2f}")
            print("===")
    else:
        parse_function(
            metric, directory=args.directory, args=args, end_signal=end_signal
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, help="path to directory")
    parser.add_argument(
        "--ci95", action="store_true", help=r"compute 95\% confidence interval"
    )
    parser.add_argument("--test-log", action="store_true", help="parse test-only logs")
    parser.add_argument(
        "--multi-exp", action="store_true", help="parse multiple experiments"
    )
    parser.add_argument(
        "--keyword", default="refined noise rate,accuracy", type=str, help="which keyword to extract"
    )
    args = parser.parse_args()

    end_train_signal = "Finish training"
    if args.test_log:
        end_test_signal = "=> result"
    else:
        end_test_signal = None

    main(args, end_train_signal, end_test_signal)
