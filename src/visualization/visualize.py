import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind, ks_2samp, anderson_ksamp, mannwhitneyu

PCAP_YEARS_LIST = ['2017', '2018']
VERSION = 'v1'
SAVE_FP = os.path.join('..', '..', 'reports', 'figures')


def _to_title_case(text):
    return text.title().replace('_', ' ')


def _plot_pie(data, title, pcap_year=None, plot_labels=True):
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

    if pcap_year is None:
        save_fp = os.path.join(SAVE_FP, f'{title}.png')
    else:
        save_fp = os.path.join(SAVE_FP, f"{pcap_year}_{VERSION}", 'pie_charts', f'{title}_{pcap_year}.png')
    plt.savefig(save_fp)
    plt.close()


def _get_dists(input_df, column):
    dist_one = input_df[input_df['Year'] == '2017'][column].dropna().values
    dist_two = input_df[input_df['Year'] == '2018'][column].dropna().values

    return dist_one, dist_two


def welch_t_test(input_df, column):
    dist_one, dist_two = _get_dists(input_df=input_df, column=column)
    result = ttest_ind(dist_one, dist_two, equal_var=False)
    return result


def mann_test(input_df, column):
    dist_one, dist_two = _get_dists(input_df=input_df, column=column)
    result = mannwhitneyu(dist_one, dist_two)
    return result


def kolmogorov_test(input_df, column):
    dist_one, dist_two = _get_dists(input_df=input_df, column=column)
    result = ks_2samp(dist_one, dist_two)
    return result


def anderson_test(input_df, column):
    dist_one, dist_two = _get_dists(input_df=input_df, column=column)
    try:
        result = anderson_ksamp([dist_one, dist_two])
    except ValueError:
        return None

    return result


def plot_histograms(input_df, column, log_scale=False, probability=True):
    input_df[column] = input_df[column].dropna()
    title = _to_title_case(column)
    plot_df = input_df[[column, 'Year']]
    plt.figure(figsize=(12, 6))
    color_palette = sns.color_palette("bright", n_colors=len(plot_df['Year'].unique()))

    num_bins = 30
    if not log_scale and not probability:
        sns.histplot(plot_df, bins=num_bins, x=column, kde=False, edgecolor='black', hue='Year',
                     palette=color_palette)
        title = f'{title}'

        plt.ylabel('Frequency', fontweight='bold')
    elif probability:
        title = f'{title} Percentages'
        sns.histplot(plot_df, bins=num_bins, x=column, kde=False, edgecolor='black', hue='Year',
                     palette=color_palette, stat='percent')

        plt.ylabel('Percent', fontweight='bold')
    else:
        sns.histplot(plot_df, bins=num_bins, x=column, kde=False, edgecolor='black', hue='Year',
                     palette=color_palette)
        title = f'{title} Log'

        plt.yscale('log')
        y_ticks = [10 ** x for x in range(-1, 8)]
        plt.yticks(y_ticks)

        def log_format(x, pos):
            return r'$10^{%d}$' % (np.log10(x))

        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(log_format))

        plt.ylabel('Frequency (Log scale)', fontweight='bold')

    plt.title(f'Histogram of {title}', fontweight='bold')
    plt.xlabel('Values', fontweight='bold')
    plt.grid(True)
    years = input_df['Year'].unique()
    plt.legend(years.astype(str))

    save_fp = os.path.join(SAVE_FP, f"{VERSION}", 'histograms', f'{title}.png')
    plt.savefig(save_fp)
    plt.close()


def plot_p_values(feature_names, p_values_dict, n_features=10):
    bar_width = 0.3
    index = np.arange(n_features)

    selected_features = np.random.choice(feature_names, size=n_features, replace=False)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)

    for i, test in enumerate(p_values_dict.keys()):
        selected_p_values = [p_values_dict[test][feature_names.index(feature)] for feature in selected_features]
        ax.bar(index + i * bar_width, selected_p_values, bar_width, label=test)

    ax.set_xlabel('Features', fontweight='bold')
    ax.set_ylabel('P-value', fontweight='bold')
    ax.set_title('Features vs. P-values', fontweight='bold')
    ax.set_xticks(index + bar_width * len(p_values_dict.keys()) / 2)
    ax.set_xticklabels(selected_features, rotation=45, ha='right')
    ax.set_yscale('log')
    ax.legend()

    plt.tight_layout()
    save_fp = os.path.join(SAVE_FP, f"{VERSION}", 'p_values.png')
    plt.savefig(save_fp)


def collect_and_plot_data():
    all_data = {year: pd.read_csv(os.path.join('..', '..', 'data', 'processed', f'pcap_data_{year}.csv'))
                for year in PCAP_YEARS_LIST}
    for year, df in all_data.items():
        df['Year'] = year

    combined_df = pd.concat(all_data.values(), ignore_index=True)

    p_values_dict = {test: [] for test in ['Mann', 'Kolmogorov']}
    feature_names = []

    for column in all_data[PCAP_YEARS_LIST[0]].columns:
        # Does not exist in 2018 data
        if column == 'source_port':
            continue
        is_numeric = all_data[PCAP_YEARS_LIST[0]][column].dtype.kind in 'biufc'
        if is_numeric:
            if column == 'fwd_header_length.1':
                column = 'fwd_header_length'

            mann_result = mann_test(input_df=combined_df, column=column)
            kolmogorov_result = kolmogorov_test(input_df=combined_df, column=column)

            plot_histograms(combined_df, column)

            p_values_dict['Mann'].append(mann_result.pvalue)
            p_values_dict['Kolmogorov'].append(kolmogorov_result.pvalue)
            feature_names.append(column)
        elif column == 'label':
            for year, df in all_data.items():
                _plot_pie(data=df[column], title='Output Features', pcap_year=year, plot_labels=False)

    plot_p_values(feature_names=feature_names, p_values_dict=p_values_dict)


def plot_data():
    for file_name in ('train', 'test'):
        fp = os.path.join('..', '..', 'data', 'processed', f'{file_name}_{VERSION}.csv')
        train_csv = pd.read_csv(fp)
        _plot_pie(data=train_csv['label'], title=f'{file_name.title()} Output Features', plot_labels=False)


collect_and_plot_data()
