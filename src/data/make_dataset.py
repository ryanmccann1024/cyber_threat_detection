import os
import re

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import skew

BASE_FP = os.path.join('..', '..', 'data', 'external')
PCAP_YEAR_LIST = ['2017']


class MakeDataset:
    def __init__(self):
        self.input_df = None

    @staticmethod
    def _to_title_case(text):
        return text.title().replace('_', ' ')

    @staticmethod
    def freedman_diaconis_rule(data):
        IQR = np.subtract(*np.percentile(data, [75, 25]))
        bin_width = 2 * IQR * (len(data) ** (-1 / 3))
        data_range = np.max(data) - np.min(data)
        num_bins = int(np.ceil(data_range / bin_width))
        return num_bins

    @staticmethod
    def scotts_rule(data):
        std_dev = np.std(data, ddof=1)
        n = len(data)
        bin_width = (3.5 * std_dev) / (n ** (1 / 3))
        data_range = np.max(data) - np.min(data)
        num_bins = int(np.ceil(data_range / bin_width))
        return num_bins

    @staticmethod
    def rice_rule(data):
        num_bins = int(2 * (len(data) ** (1 / 3)))
        return num_bins

    def doanes_formula(self, data):
        n = len(data)
        if n <= 1:
            return 1

        g1 = skew(data)
        if np.isnan(g1) or np.isinf(g1):
            return self.sturges_rule(data=data)

        sigma_g1 = ((6 * (n - 2)) / ((n + 1) * (n + 3))) ** 0.5
        if sigma_g1 == 0:
            return self.sturges_rule(data=data)

        num_bins = int(1 + np.log2(n) + np.log2(1 + abs(g1) / sigma_g1))
        return max(num_bins, 1)

    @staticmethod
    def sturges_rule(data):
        num_bins = int(np.ceil(1 + np.log2(len(data))))
        return num_bins

    @staticmethod
    def square_root_choice(data):
        num_bins = int(np.sqrt(len(data)))
        return num_bins

    @staticmethod
    def _plot_pie(data, title):
        value_counts = data.value_counts()
        plt.figure(figsize=(8, 8))
        patches, texts, autotexts = plt.pie(value_counts, labels=value_counts.index, autopct='%1.2f%%', startangle=140)

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_size('x-large')

        plt.legend(patches, [f'{label}: {count}' for label, count in zip(value_counts.index, value_counts)],
                   loc='center left', bbox_to_anchor=(1, 0.5))

        plt.title(f'Pie Chart of {title}', fontweight='bold')
        plt.ylabel('')
        plt.tight_layout()

        save_fp = os.path.join('plots', PCAP_YEAR_LIST[0], 'pie_charts', f'{title}_{PCAP_YEAR_LIST[0]}.png')
        plt.savefig(save_fp)
        plt.close()

    @staticmethod
    def _plot_box(data, title):
        plt.figure(figsize=(10, 6))
        sns.set(style="whitegrid")
        sns.boxplot(y=data, width=0.3, palette="Blues")

        plt.title(f'Box Plot of {title} - {PCAP_YEAR_LIST[0]}', fontweight='bold', fontsize=14)
        plt.yscale('log')
        plt.ylabel('Values', fontweight='bold', fontsize=12)

        # median = data.median()
        # upper_quartile = data.quantile(0.75)
        # lower_quartile = data.quantile(0.25)
        # iqr = upper_quartile - lower_quartile
        # upper_whisker = data[data <= upper_quartile + 1.5 * iqr].max()
        # lower_whisker = data[data >= lower_quartile - 1.5 * iqr].min()

        # plt.text(x=0.2, y=median, s=" Median", color='black', va='center', fontweight='bold')
        # plt.text(x=0.2, y=upper_quartile, s=" 75th percentile", color='black', va='center', fontweight='bold')
        # plt.text(x=0.2, y=lower_quartile, s=" 25th percentile", color='black', va='center', fontweight='bold')
        # plt.text(x=0.2, y=upper_whisker, s=" Max", color='black', va='center', fontweight='bold')
        # plt.text(x=0.2, y=lower_whisker, s=" Min", color='black', va='center', fontweight='bold')

        # plt.tick_params(axis='both', which='major', labelsize=10)
        # plt.tight_layout()

        save_fp = os.path.join('plots', PCAP_YEAR_LIST[0], 'box_plots', f'{title}_{PCAP_YEAR_LIST[0]}.png')
        plt.savefig(save_fp)
        plt.close()

    def _plot_hist(self, data, title):
        num_bins = self.doanes_formula(data=data)
        plt.figure(figsize=(12, 6))
        sns.histplot(data, bins=num_bins, kde=False, edgecolor='black')

        plt.yscale('log')
        plt.title(f'Histogram of {title} - {PCAP_YEAR_LIST[0]}', fontweight='bold')
        plt.xlabel('Values', fontweight='bold')
        plt.ylabel('Frequency (Log Scale)', fontweight='bold')
        plt.grid(True)

        save_fp = os.path.join('plots', PCAP_YEAR_LIST[0], 'histograms', f'{title}_{PCAP_YEAR_LIST[0]}.png')
        plt.savefig(save_fp)
        plt.close()

    def _plot_column_data(self):
        for column in self.input_df.columns:
            if pd.api.types.is_numeric_dtype(self.input_df[column]):
                continue
                non_null_data = self.input_df[column].dropna()
                column_title = self._to_title_case(column)

                data_range = non_null_data.max() - non_null_data.min()

                if data_range < 5:
                    self._plot_pie(data=non_null_data, title=column_title)
                else:
                    self._plot_box(data=non_null_data, title=column_title)

            else:
                if column == 'label':
                    print(self.input_df[column])

    def analyze_data(self):
        self._plot_column_data()

    def clean_columns(self):
        self.input_df.columns = [
            re.sub(r'\.1$', '',
                   re.sub(r'ss\b', 's',
                          re.sub(r'/', '',
                                 re.sub(r'/bulk', '_bulk',
                                        column.strip().lower().replace(' ', '_')))))
            for column in self.input_df.columns
        ]

    def read_data(self):
        self.input_df = pd.DataFrame()

        for year in PCAP_YEAR_LIST:
            curr_fp = os.path.join(BASE_FP, year)
            pcap_files = os.listdir(curr_fp)
            for curr_file in pcap_files:
                print(f'Completed reading file:\t\t{curr_file} {year}')
                curr_df = pd.read_csv(os.path.join(curr_fp, curr_file), encoding='latin1')
                self.input_df = pd.concat([self.input_df, curr_df], ignore_index=True)

    def make_dataset(self):
        self.read_data()
        self.clean_columns()
        self.analyze_data()


if __name__ == '__main__':
    make_data_obj = MakeDataset()
    make_data_obj.make_dataset()
