"""
@author: Zhihao Li
@date: 2024-11-11
@homepage: https://zhihaoli.top/
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import re
import numpy as np
import os
import argparse
from collections import OrderedDict, defaultdict
from dassl.utils import check_isfile, listdir_nohidden
import warnings
warnings.filterwarnings("ignore")

class ANALYZER:
    def __init__(self, data, epoch, save_path):
        epoch_data = data["Epoch " + str(epoch)]
        self.num_samples = len(epoch_data)
        self.epoch = epoch
        self.save_path = save_path
        # split the data source
        self.split_dataset_epoch(epoch_data)
        
        # get samples ID
        self.clean_sample_id = self.label == self.gt_label
        self.noisy_sample_id = self.label != self.gt_label
        self.clean_num = np.sum(self.clean_sample_id)
        self.noisy_num = np.sum(self.noisy_sample_id)
        self.noise_rate = self.noisy_num / self.num_samples
        
        # analysis generator
        self.analysis_generator()
        # analysis refurbish
        self.analysis_refurbish()
        # analysis discriminator
        self.analysis_discriminator()
        
        # print the analysed table
        # self.print_analysis_results()
        
        self.clean_sample_id_D = self.refurbished_label == self.gt_label
        self.noisy_sample_id_D = self.refurbished_label != self.gt_label
        # visualize lossG, probsG
        self.visualize_samples('LossG', self.lossG[self.clean_sample_id], self.lossG[self.noisy_sample_id])
        self.visualize_samples('ProbsG', self.probsG[self.clean_sample_id], self.probsG[self.noisy_sample_id])
        self.visualize_samples('ProbsD', self.probsD[self.clean_sample_id_D], self.probsD[self.noisy_sample_id_D])
        
    def split_dataset_epoch(self, data):
        self.label = np.zeros((len(data)), dtype=int)
        self.gt_label = np.zeros((len(data)), dtype=int)
        self.lossG = np.zeros((len(data)), dtype=np.float32)     # generator
        self.probsG = np.zeros((len(data)), dtype=np.float32)    # GMM
        self.pred_labelG = np.zeros((len(data)), dtype=int)   # generator
        self.refurbished_label = np.zeros((len(data)), dtype=int)
        self.probsD = np.zeros((len(data)), dtype=np.float32)

        for id, item in enumerate(data):
            self.label[id] = item['label']
            self.gt_label[id] = item['gt_label']
            self.lossG[id] = item['lossG']
            self.probsG[id] = item['probsG']
            self.pred_labelG[id] = item['pred_labelG']
            self.refurbished_label[id] = item['refurbished_label']
            self.probsD[id] = item['probsD']
    
    def analysis_generator(self):
        # prepare
        TP = np.sum(self.pred_labelG[self.clean_sample_id] == self.gt_label[self.clean_sample_id])
        TN = np.sum(self.pred_labelG[self.noisy_sample_id] != self.gt_label[self.noisy_sample_id])
        FN = np.sum(self.pred_labelG[self.clean_sample_id] != self.gt_label[self.clean_sample_id])
        FP = np.sum(self.pred_labelG[self.noisy_sample_id] == self.gt_label[self.noisy_sample_id])
        
        # indicators
        self.Keep_G = TP / (TP + FN)
        self.Adjust_G = FP / (TN + FP)
    
    def analysis_refurbish(self):
        # prepare
        self.true_G = self.pred_labelG == self.gt_label
        self.true_R = self.refurbished_label == self.gt_label
        TP = np.sum(self.true_G & self.true_R)
        TN = np.sum(~self.true_G & ~self.true_R)
        FN = np.sum(self.true_G & ~self.true_R)
        FP = np.sum(~self.true_G & self.true_R)
        
        # indicators
        self.Keep_R = TP / (TP + FN)
        self.Adjust_R = FP / (TN + FP)
        self.Accuracy_R = (TP+FP) / self.num_samples
        
    def analysis_discriminator(self):
        # prepare
        self.higher_ID = self.probsD >= 0.4
        self.lower_ID = self.probsD < 0.4
        TP = self.refurbished_label[self.higher_ID] == self.gt_label[self.higher_ID]
        
        # indicators
        self.higher_true_num = np.sum(TP)
        self.higher_num = np.sum(self.higher_ID)
        self.Accuracy_D = self.higher_true_num / self.higher_num
        
    def print_analysis_results(self):
        # Prepare data for tabulation
        print(f"{35 * '-'}")
        print(f"Analysis results for the epoch {self.epoch}")
        print(f"{35 * '-'}\nBasic information:")
        print(f"\tTotal samples: {self.num_samples}\n\tClean samples: {self.clean_num}")
        print(f"\tNoisy samples: {self.noisy_num}\n\tNoise rate: {self.noise_rate * 100:.2f}%")
        
        print(f"{35 * '-'}\nGenerator metric:")
        print(f"\tPotential: {self.Keep_G * 100:.2f}%\t↑")
        print(f"\tFlexibility: {self.Adjust_G * 100:.2f}%\t↑")
        
        print(f"{35 * '-'}\nRefurbish metric:")
        print(f"\tPotential: {self.Keep_R * 100:.2f}%\t↑")
        print(f"\tFlexibility: {self.Adjust_R * 100:.2f}%\t↑")
        print(f"\tAccuracy: {self.Accuracy_R * 100:.2f}%\t↑")
        
        print(f"{35 * '-'}\nDiscriminator metric:")
        print(f"\tHigher true samples: {self.higher_true_num}")
        print(f"\tHigher samples: {self.higher_num}")
        print(f"\tAccuracy: {self.Accuracy_D * 100:.2f}%\t↑")
        print(f"{35 * '-'}")
              
    def visualize_samples(self, data_name, data_clean, data_noisy):
        # settings
        # Fonts style
        font_prop1 = FontProperties(fname="./fonts/TIMES.TTF")
        font_prop2 = FontProperties(fname="./fonts/ARLRDBD.TTF")
        font_size = 10
        sns.set(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=500)

        # histplot for sub-figure1
        sns.histplot(data_clean, color="blue", label="Clean", kde=False, stat="probability", bins=20, alpha=0.6, ax=axes[0])
        sns.histplot(data_noisy, color="red", label="Noisy", kde=False, stat="probability", bins=20, alpha=0.6, ax=axes[0])
        axes[0].set_title(f'{data_name} Density (Histogram)', fontproperties=font_prop2, fontsize=font_size + 3)
        axes[0].set_xlabel(r'$L_{Divide}$', fontproperties=font_prop2, fontsize=font_size + 1)
        axes[0].set_ylabel('Probability Density', fontproperties=font_prop2, fontsize=font_size + 1)
        axes[0].legend(prop=font_prop2)
        
        # kdeplot for sub-figure2
        sns.kdeplot(data_clean, color="blue", label="Clean", fill=True, ax=axes[1], alpha=0.3)
        sns.kdeplot(data_noisy, color="red", label="Noisy", fill=True, ax=axes[1], alpha=0.3)
        axes[1].set_title(f'{data_name} Density (KDE)', fontproperties=font_prop2, fontsize=font_size + 3)
        axes[1].set_xlabel(r'$L_{Divide}$', fontproperties=font_prop2, fontsize=font_size + 1)
        axes[1].set_ylabel('Probability Density', fontproperties=font_prop2, fontsize=font_size + 1)
        axes[1].legend(prop=font_prop2)
        
        plt.tight_layout()

        folder_path = os.path.join(self.save_path, 'images')

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        plt.savefig(os.path.join(folder_path, f"{data_name}_Probability_Density_Epoch{self.epoch}.png"), format="png")
        plt.show()

def compute_ci95(res):
    return 1.96 * np.std(res) / np.sqrt(len(res))

def listdir_sorted_by_fp(directory):
    subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

    def sort_key(name):
        match = re.search(r'fp(\d+)', name)
        return int(match.group(1)) if match else float('inf')
    
    sorted_subdirs = sorted(subdirs, key=sort_key)
    
    return sorted_subdirs

def analysis_function(directory="", args=None):
    print(f"Analyzing files in {directory}")
    subdirs = listdir_nohidden(directory, sort=True)
    
    outputs = []
    epoch_id = [1, 10, 20, 30, 40, 50]
    for subdir in subdirs:
        fpath = os.path.join(directory, subdir, "analysis.json")
        assert check_isfile(fpath)

        output = OrderedDict()

        # load the raw materials
        with open(fpath, 'r', encoding='utf-8') as file:
            analysis_results = json.load(file)

        for epoch in epoch_id:
            analyzer_epoch = ANALYZER(analysis_results, epoch, os.path.join(directory, subdir))

            if str(epoch) not in output:
                output[str(epoch)] = OrderedDict()
            
            output[str(epoch)]['clean_num'] = analyzer_epoch.clean_num
            output[str(epoch)]['noisy_num'] = analyzer_epoch.noisy_num
            output[str(epoch)]['noise_rate'] = analyzer_epoch.noise_rate
            output[str(epoch)]['Keep_G'] = analyzer_epoch.Keep_G
            output[str(epoch)]['Adjust_G'] = analyzer_epoch.Adjust_G
            output[str(epoch)]['Keep_R'] = analyzer_epoch.Keep_R
            output[str(epoch)]['Adjust_R'] = analyzer_epoch.Adjust_R
            output[str(epoch)]['Accuracy_R'] = analyzer_epoch.Accuracy_R
            output[str(epoch)]['higher_true_num'] = analyzer_epoch.higher_true_num
            output[str(epoch)]['higher_num'] = analyzer_epoch.higher_num
            output[str(epoch)]['Accuracy_D'] = analyzer_epoch.Accuracy_D

        if output:
            outputs.append(output)

    assert len(outputs) > 0, f"Nothing found in {directory}"

    for epoch in epoch_id:
        metrics_results = defaultdict(list)
        for output in outputs:
            for key, value in output[str(epoch)].items():
                metrics_results[key].append(value)

        output_results = OrderedDict()
        output_std = OrderedDict()

        print("===")
        print(f"Summary of epoch: {epoch}")
        for key, values in metrics_results.items():
            avg = np.mean(values)
            std = compute_ci95(values) if args.ci95 else np.std(values)
            print(f"* {key}: {avg:.2f} +- {std:.2f}%")
            output_results[key] = avg
            output_std[key] = std
        print("===")


def main(args):

    if args.multi_exp:
        for directory in listdir_sorted_by_fp(args.directory):
            directory = os.path.join(args.directory, directory)
            analysis_function(directory=directory, args=args)
    else:
        analysis_function(directory=args.directory, args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, help="path to directory")
    parser.add_argument(
        "--multi-exp", action="store_true", help="parse multiple experiments"
    )
    parser.add_argument(
        "--ci95", action="store_true", help=r"compute 95\% confidence interval"
    )
    args = parser.parse_args()

    main(args)
