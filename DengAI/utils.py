from pandas import read_csv
import numpy as np

class DataReader:
    feature_headers = ['city', 'year', 'weekofyear', 'week_start_date', 'ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw', 'precipitation_amt_mm', 'reanalysis_air_temp_k', 'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k', 'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k', 'reanalysis_precip_amt_kg_per_m2', 'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm', 'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k', 'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c', 'station_min_temp_c', 'station_precip_mm']
    label_headers = ['city', 'year', 'weekofyear', 'total_cases']

    @classmethod
    def read_features(cls, file_path):
        data_frame = read_csv(file_path)
        # handle NaN
        data_frame = data_frame.fillna(0.0)
        return data_frame.values

    @classmethod
    def read_labels(cls, file_path):
        data_frame = read_csv(file_path)
        return data_frame.values


class DataWriter:
    def __init__(self, test_data_path):
        data = DataReader.read_features(test_data_path)
        self.cols = self.get_output_cols_name(data)

    def get_output_cols_name(self, data):
        cols = {'city': np.copy(data[:, 0]),
                'year': np.copy(data[:, 1]),
                'weekofyear': np.copy(data[:, 2])}
        return cols
