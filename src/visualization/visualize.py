import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
from scipy.stats import skew

PCAP_YEARS_LIST = ['2017', '2018']
VERSION = 'v0'
SAVE_FP = os.path.join('..', '..', 'reports', 'figures')
COLOR_MAP = {
    '2017': 'darkred',
    '2018': 'navy',
}


def _to_title_case(text):
    return text.title().replace('_', ' ')


def freedman_diaconis_rule(data):
    IQR = np.subtract(*np.percentile(data, [75, 25]))
    bin_width = 2 * IQR * (len(data) ** (-1 / 3))
    data_range = np.max(data) - np.min(data)
    num_bins = int(np.ceil(data_range / bin_width))
    return num_bins


def scotts_rule(data):
    std_dev = np.std(data, ddof=1)
    n = len(data)
    bin_width = (3.5 * std_dev) / (n ** (1 / 3))
    data_range = np.max(data) - np.min(data)
    num_bins = int(np.ceil(data_range / bin_width))
    return num_bins


def rice_rule(data):
    num_bins = int(2 * (len(data) ** (1 / 3)))
    return num_bins


def doanes_formula(data):
    n = len(data)
    if n <= 1:
        return 1

    g1 = skew(data)
    if np.isnan(g1) or np.isinf(g1):
        return sturges_rule(data=data)

    sigma_g1 = ((6 * (n - 2)) / ((n + 1) * (n + 3))) ** 0.5
    if sigma_g1 == 0:
        return sturges_rule(data=data)

    num_bins = int(1 + np.log2(n) + np.log2(1 + abs(g1) / sigma_g1))
    return max(num_bins, 1)


def sturges_rule(data):
    num_bins = int(np.ceil(1 + np.log2(len(data))))
    return num_bins


def square_root_choice(data):
    num_bins = int(np.sqrt(len(data)))
    return num_bins


def _plot_pie(data, title, pcap_year, plot_labels=True):
    value_counts = data.value_counts()
    plt.figure(figsize=(8, 8))

    if plot_labels:
        patches, texts, autotexts = plt.pie(value_counts, startangle=140, autopct='%1.1f%%')
    else:
        patches, texts = plt.pie(value_counts, startangle=140)
        autotexts = []

    if not plot_labels:
        for autotext in autotexts:
            autotext.set_visible(False)

    percentages = [f'{(count / value_counts.sum()) * 100:.1f}%' for count in value_counts]
    legend_labels = [f'{label}: {count} ({perc})' for label, count, perc in
                     zip(value_counts.index, value_counts, percentages)]
    plt.legend(patches, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.title(f'Pie Chart of {title}', fontweight='bold')
    plt.ylabel('')
    plt.tight_layout()

    save_fp = os.path.join(SAVE_FP, f"{pcap_year}_{VERSION}", 'pie_charts', f'{title}_{pcap_year}.png')
    plt.savefig(save_fp)
    plt.close()


def plot_histograms(data_dict, title, plot_density=False):
    plt.figure(figsize=(12, 6))
    palette = sns.color_palette("deep", len(data_dict))

    for idx, (year, data) in enumerate(data_dict.items()):
        num_bins = 20
        if plot_density:
            sns.histplot(data, bins=num_bins, kde=False, edgecolor='black', color=palette[idx],
                         label=f'Year {year}', stat='density')
        else:
            sns.histplot(data, bins=num_bins, kde=False, edgecolor='black', color=palette[idx],
                         label=f'Year {year}')
            plt.title(f'Density Histogram of {title}', fontweight='bold')

            plt.yscale('log')
            y_ticks = [10 ** x for x in range(-1, 8)]
            plt.yticks(y_ticks)

            def log_format(x, pos):
                return r'$10^{%d}$' % (np.log10(x))

            plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(log_format))
            plt.title(f'Histogram of {title}', fontweight='bold')

    plt.xlabel('Values', fontweight='bold')
    plt.ylabel('Frequency (Log Scale)', fontweight='bold')
    plt.grid(True)
    plt.legend()

    save_fp = os.path.join(SAVE_FP, f"{VERSION}", 'histograms', f'{title}.png')
    plt.savefig(save_fp)
    plt.close()


def collect_and_plot_data(PCAP_YEARS_LIST):
    all_data = {year: pd.read_csv(os.path.join('..', '..', 'data', 'processed', f'pcap_data_{year}.csv'))
                for year in PCAP_YEARS_LIST}

    # num_bins_list, bin_widths_list = manual_num_bins(data=all_data)
    for column in all_data[PCAP_YEARS_LIST[0]].columns:
        # Does not exist in 2018 data
        if column == 'source_port':
            continue
        is_numeric = all_data[PCAP_YEARS_LIST[0]][column].dtype.kind in 'biufc'
        if is_numeric:
            data_dict = {}
            for year, df in all_data.items():
                # fixme
                if column == 'fwd_header_length.1':
                    column = 'fwd_header_length'

                non_null_data = df[column].dropna()
                data_dict[year] = non_null_data

            column_title = _to_title_case(column)
            plot_histograms(data_dict, column_title)
        elif column == 'label':
            for year, df in all_data.items():
                _plot_pie(data=df[column], title='Output Features', pcap_year=year, plot_labels=False)


collect_and_plot_data(PCAP_YEARS_LIST)
