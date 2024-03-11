import os
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from make_dataset_args import STANDARD_COLS_DICT

BASE_FP = os.path.join('..', '..', 'data', 'external')
PCAP_YEARS_LIST = ['2017', '2018']


class MakeDataset:
    def __init__(self):
        self.input_df = None
        self.dtypes_dict = dict()
        self.parse_dates = list()

        self.pcap_year = None

    def clean_columns(self):
        cleaned_columns = [
            re.sub(r'\.1$', '',
                   re.sub(r'ss\b', 's',
                          re.sub(r'/', '',
                                 re.sub(r'/bulk', '_bulk',
                                        column.strip().lower().replace(' ', '_')))))
            for column in self.input_df.columns
        ]

        final_columns = [STANDARD_COLS_DICT.get(column, column) for column in cleaned_columns]
        self.input_df.columns = final_columns

    def make_dtypes_dict(self, sample_fp: str):
        sample_df = pd.read_csv(sample_fp, nrows=1000)
        sample_df.columns = sample_df.columns.str.strip()
        sample_df = sample_df.infer_objects()
        for col in sample_df.columns:
            if sample_df[col].dtype == 'object':
                sample_df[col] = pd.to_datetime(sample_df[col], errors='ignore')
                if sample_df[col].dtype == 'object':
                    sample_df[col] = pd.to_numeric(sample_df[col], errors='ignore')

        for col, dtype in sample_df.dtypes.items():
            if pd.api.types.is_datetime64_any_dtype(dtype):
                self.parse_dates.append(col)
            elif pd.api.types.is_integer_dtype(dtype):
                self.dtypes_dict[col] = pd.Int64Dtype()
            else:
                self.dtypes_dict[col] = dtype.name

    def convert_dtypes(self, curr_df: pd.DataFrame):
        for col, dtype in self.dtypes_dict.items():
            if col in curr_df.columns:
                curr_df[col] = curr_df[col].astype(dtype)

        if self.parse_dates:
            for col in self.parse_dates:
                if col in curr_df.columns:
                    curr_df[col] = pd.to_datetime(curr_df[col], errors='coerce')

        return curr_df

    @staticmethod
    def remove_repeat_headers(df: pd.DataFrame, header_list: list):
        headers_joined = ''.join(header_list).lower().replace(' ', '_')
        mask = df.apply(lambda x: ''.join(x.astype(str)).lower().replace(' ', '_').replace('.1', ''),
                        axis=1) == headers_joined

        df = df[~mask]
        return df

    def read_data(self):
        self.input_df = pd.DataFrame()
        curr_fp = os.path.join(BASE_FP, self.pcap_year)
        pcap_files = os.listdir(curr_fp)
        for curr_file in pcap_files:
            print(f'Began reading:\t\t\t{curr_file} from {self.pcap_year}')

            pcap_fp = os.path.join(curr_fp, curr_file)
            if self.dtypes_dict == {}:
                self.make_dtypes_dict(sample_fp=pcap_fp)

            curr_df = pd.read_csv(pcap_fp, encoding='latin1', skipinitialspace=True, low_memory=False)
            curr_df.columns = curr_df.columns.str.strip()
            curr_df = self.remove_repeat_headers(curr_df, self.input_df.columns.tolist())
            curr_df = self.convert_dtypes(curr_df=curr_df)

            self.input_df = pd.concat([self.input_df, curr_df], ignore_index=True)
            print(f'Completed reading:\t\t\t{curr_file} from {self.pcap_year}')

    def make_dataset(self):
        for pcap_year in PCAP_YEARS_LIST:
            self.pcap_year = pcap_year

            self.read_data()
            self.clean_columns()

            save_fp = os.path.join('..', '..', 'data', 'interim', f'pcap_data_{pcap_year}.csv')
            self.input_df.to_csv(save_fp, index=False)

    @staticmethod
    def get_sampled_df(df_2017: pd.DataFrame, df_2018: pd.DataFrame, ):
        df_combined = pd.concat([df_2017, df_2018])
        df_combined['label'] = df_combined['label'].str.lower()
        benign_count = df_combined[df_combined['label'] == 'benign'].shape[0]
        not_benign_count = df_combined[df_combined['label'] != 'benign'].shape[0]
        benign_to_sample = not_benign_count

        df_not_benign_sampled = df_combined[df_combined['label'] != 'benign']
        df_benign_sampled = df_combined[df_combined['label'] == 'benign'].sample(n=benign_to_sample, replace=False)
        df_sampled = pd.concat([df_benign_sampled, df_not_benign_sampled])
        df_sampled = df_sampled.sample(frac=1).reset_index(drop=True)

        return df_sampled

    def split_datasets(self, test_size: float):
        load_fp = os.path.join('..', '..', 'data', 'interim')
        df_2017 = pd.read_csv(os.path.join(load_fp, 'pcap_data_2017.csv'))
        df_2018 = pd.read_csv(os.path.join(load_fp, 'pcap_data_2018.csv'))

        df_sampled = self.get_sampled_df(df_2017=df_2017, df_2018=df_2018)
        df_train, df_test = train_test_split(df_sampled, test_size=test_size, random_state=42)
        save_fp = os.path.join('..', '..', 'data', 'processed')
        train_fp = os.path.join(save_fp, 'train_v1.csv')
        test_fp = os.path.join(save_fp, 'test_v1.csv')

        df_train.to_csv(train_fp, index=False)
        df_test.to_csv(test_fp, index=False)


if __name__ == '__main__':
    make_data_obj = MakeDataset()
    # make_data_obj.make_dataset()

    make_data_obj.split_datasets(test_size=0.2)
