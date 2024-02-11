import os
import re

import pandas as pd

BASE_FP = os.path.join('..', '..', 'data', 'external')
PCAP_YEAR = '2017'


class MakeDataset:
    def __init__(self):
        self.input_df = None
        self.dtypes_dict = {}
        self.parse_dates = []

    def clean_columns(self):
        self.input_df.columns = [
            re.sub(r'\.1$', '',
                   re.sub(r'ss\b', 's',
                          re.sub(r'/', '',
                                 re.sub(r'/bulk', '_bulk',
                                        column.strip().lower().replace(' ', '_')))))
            for column in self.input_df.columns
        ]

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
                    curr_df[col] = pd.to_datetime(curr_df[col])

        return curr_df

    def read_data(self):
        self.input_df = pd.DataFrame()
        curr_fp = os.path.join(BASE_FP, PCAP_YEAR)
        pcap_files = os.listdir(curr_fp)
        for curr_file in pcap_files:
            pcap_fp = os.path.join(curr_fp, curr_file)
            if self.dtypes_dict == {}:
                self.make_dtypes_dict(sample_fp=pcap_fp)

            curr_df = pd.read_csv(pcap_fp, encoding='latin1', skipinitialspace=True)
            curr_df.columns = curr_df.columns.str.strip()
            curr_df = self.convert_dtypes(curr_df=curr_df)

            self.input_df = pd.concat([self.input_df, curr_df], ignore_index=True)
            print(f'Completed reading:\t\t{curr_file} from {PCAP_YEAR}')

    def make_dataset(self):
        self.read_data()
        self.clean_columns()

        save_fp = os.path.join('..', '..', 'data', 'processed', f'pcap_data_{PCAP_YEAR}.csv')
        self.input_df.to_csv(save_fp, index=False)


if __name__ == '__main__':
    make_data_obj = MakeDataset()
    make_data_obj.make_dataset()
