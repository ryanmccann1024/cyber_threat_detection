import os
import re

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE_FP = os.path.join('..', '..', 'data', 'external')
PCAP_YEAR_LIST = ['2017']


class MakeDataset:
    def __init__(self):
        self.input_df = None

    @staticmethod
    def _plot_box(data, title):
        plt.figure(figsize=(10, 6), dpi=300)
        sns.set(style="whitegrid")
        sns.boxplot(y=data, width=0.3, palette="Blues")

        plt.title(f'Box Plot of {title}', fontweight='bold', fontsize=14)
        plt.ylabel('Values', fontweight='bold', fontsize=12)

        median = data.median()
        upper_quartile = data.quantile(0.75)
        lower_quartile = data.quantile(0.25)
        iqr = upper_quartile - lower_quartile
        upper_whisker = data[data <= upper_quartile + 1.5 * iqr].max()
        lower_whisker = data[data >= lower_quartile - 1.5 * iqr].min()

        plt.text(x=0.2, y=median, s=" Median", color='black', va='center', fontweight='bold')
        plt.text(x=0.2, y=upper_quartile, s=" 75th percentile", color='black', va='center', fontweight='bold')
        plt.text(x=0.2, y=lower_quartile, s=" 25th percentile", color='black', va='center', fontweight='bold')
        plt.text(x=0.2, y=upper_whisker, s=" Max", color='black', va='center', fontweight='bold')
        plt.text(x=0.2, y=lower_whisker, s=" Min", color='black', va='center', fontweight='bold')

        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.tight_layout()
        # plt.subplots_adjust(right=0.75)

        plt.show()

    @staticmethod
    def _plot_hist(data, title):
        num_bins = int(np.ceil(1 + np.log2(len(data))))
        plt.figure(figsize=(12, 6), dpi=300)
        sns.histplot(data, bins=num_bins, kde=True, edgecolor='black')

        plt.title(f'Histogram of {title} Feature', fontweight='bold')
        plt.xlabel('Values', fontweight='bold')
        plt.ylabel('Frequency', fontweight='bold')
        plt.grid(True)
        plt.show()

    @staticmethod
    def _to_title_case(text):
        return text.title().replace('_', ' ')

    def _plot_column_data(self):
        for column in self.input_df.columns:
            if pd.api.types.is_numeric_dtype(self.input_df[column]):
                non_null_data = self.input_df[column].dropna()
                column_title = self._to_title_case(column)
                self._plot_hist(data=non_null_data, title=column_title)
                self._plot_box(data=self.input_df[column], title=column_title)
                break

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
