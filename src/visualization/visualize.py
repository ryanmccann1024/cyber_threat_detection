import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
from scipy.stats import skew
from scipy.stats import ks_2samp, anderson_ksamp, ttest_ind, chi2_contingency

PCAP_YEARS_LIST = ['2017', '2018']
VERSION = 'v0'
SAVE_FP = os.path.join('..', '..', 'reports', 'figures')


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


def kolmogorov_smirnov_test(data1, data2):
    D, p_value = ks_2samp(data1, data2)
    return D, p_value


def anderson_darling_test(data1, data2):
    try:
        result = anderson_ksamp([data1, data2])
    except ValueError:
        return None

    return result.statistic, result.critical_values, result.significance_level


def welchs_t_test(data1, data2):
    t_statistic, p_value = ttest_ind(data1, data2, equal_var=False)
    return t_statistic, p_value


def plot_histograms(input_df, column, plot_density=False):
    input_df[column] = input_df[column].dropna()
    title = _to_title_case(column)
    plot_df = input_df[[column, 'Year']]
    plt.figure(figsize=(12, 6))
    color_palette = sns.color_palette("bright", n_colors=len(plot_df['Year'].unique()))

    num_bins = 30
    if plot_density:
        sns.histplot(plot_df, bins=num_bins, x=column, kde=False, edgecolor='black', hue='Year',
                     stat='density', palette=color_palette)
        plt.title(f'Density Histogram of {title}', fontweight='bold')
    else:
        sns.histplot(plot_df, bins=num_bins, x=column, kde=False, edgecolor='black', hue='Year',
                     palette=color_palette)

        plt.yscale('log')
        y_ticks = [10 ** x for x in range(-1, 8)]
        plt.yticks(y_ticks)

        def log_format(x, pos):
            return r'$10^{%d}$' % (np.log10(x))

        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(log_format))
        plt.title(f'Histogram of {title}', fontweight='bold')

    data_2017 = plot_df[plot_df['Year'] == '2017'][column].values
    data_2018 = plot_df[plot_df['Year'] == '2018'][column].values

    kolmogorov_smirnov_result = kolmogorov_smirnov_test(data_2017, data_2018)
    anderson_darling_result = anderson_darling_test(data_2017, data_2018)
    welchs_t_test_result = welchs_t_test(data_2017, data_2018)

    if anderson_darling_result is None:
        ad_test_str = "AD: None"
        anderson_darling_result = [np.nan, np.nan, np.nan]
    else:
        ad_test_str = f"AD test: stat={anderson_darling_result[0]:.2f}, p={anderson_darling_result[2]:.2f}"

    ks_test_str = f"KS test: D={kolmogorov_smirnov_result[0]:.2f}, p={kolmogorov_smirnov_result[1]:.2e}"
    t_test_str = f"Welch's t-test: t={welchs_t_test_result[0]:.2f}, p={welchs_t_test_result[1]:.2e}"
    test_result_str = f"{ks_test_str} | {ad_test_str} | {t_test_str}"

    plt.xlabel('Values', fontweight='bold')
    plt.ylabel('Frequency (Log scale)', fontweight='bold')
    plt.grid(True)
    plt.legend([f'2017 {test_result_str}',
                f'2018'])

    save_fp = os.path.join(SAVE_FP, f"{VERSION}", 'histograms', f'{title}.png')
    plt.savefig(save_fp)
    plt.close()

    return kolmogorov_smirnov_result[1], anderson_darling_result[2], welchs_t_test_result[1]


def collect_and_plot_data(PCAP_YEARS_LIST):
    all_data = {year: pd.read_csv(os.path.join('..', '..', 'data', 'processed', f'pcap_data_{year}.csv'))
                for year in PCAP_YEARS_LIST}
    for year, df in all_data.items():
        df['Year'] = year

    combined_df = pd.concat(all_data.values(), ignore_index=True)

    test_results = {}
    for column in all_data[PCAP_YEARS_LIST[0]].columns:
        # Does not exist in 2018 data
        if column == 'source_port':
            continue
        is_numeric = all_data[PCAP_YEARS_LIST[0]][column].dtype.kind in 'biufc'
        if is_numeric:
            if column == 'fwd_header_length.1':
                column = 'fwd_header_length'

            ks_p_value, ad_p_value, welch_p_value = plot_histograms(combined_df, column)
            test_results[column] = {'KS p-value': ks_p_value, 'AD p-value': ad_p_value, 'Welch p-value': welch_p_value}
        elif column == 'label':
            for year, df in all_data.items():
                _plot_pie(data=df[column], title='Output Features', pcap_year=year, plot_labels=False)

    test_results_df = pd.DataFrame.from_dict(test_results, orient='index')

    plt.figure(figsize=(14, 10))
    heatmap = sns.heatmap(test_results_df, cmap='coolwarm', linewidths=0.2, linecolor='black')
    heatmap.set_title('P-values Heatmap', fontsize=18, fontweight='bold')
    heatmap.set_xlabel('P-values', fontsize=14, fontweight='bold')
    heatmap.set_ylabel('Features', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    save_fp = os.path.join(SAVE_FP, f"{VERSION}", f'p_value_heat.png')
    plt.savefig(save_fp)


collect_and_plot_data(PCAP_YEARS_LIST)
