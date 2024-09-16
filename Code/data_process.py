# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


'''
    Data preprocessing Extract by row
    Output
    label_all: all labels,
    min_values: normalized minimum value array,
    max_values: normalized maximum value, array,
    label_encoders: normalized encoder,
    dataNormalization: normalized data
'''
def data_pre_process(file_path):
    with open(file_path, 'r') as file:
        label_all = []
        data_all = []
        for line in file:
            data = [''] * 32
            line_1 = line.split(';', 4)
            # Label, Activity, Physics, Net = s_1[0], s_1[1], s_1[2], s_1[3]
            Label = line_1[1]
            data[0] = Label[0]
            Activity = line_1[2].split(',')[0].split('|')
            for i in range(len(Activity)):
                data[i + 1] = Activity[i]

            Physics = line_1[3].split('{')[1].split('}')[0].split(',')
            for i in range(len(Physics)):
                data[i + 11] = Physics[i].split(':')[1]
            # print(Physics)
            Net = line_1[4].split('{')[1].split('}')[0].split(',')
            for i in range(4):
                data[i + 17] = Net[i].split(':')[1]
            # Net[4] is a special type 'frame_protocols': 'eth:ethertype:ip:tcp'
            data[21] = Net[4].split("'frame_protocols':")[-1]
            for i in range(5, len(Net)):
                data[i + 17] = Net[i].split(':')[1]

            #label 
            # data_process_1 31-dimensional feature data
            label, data_process_1 = data_process_line(data)
            label_all.append(label)
            data_all.append(data_process_1)
        # print(label_all)
        # print(data_all[2])
            # print(data_process_1)
        min_values, max_values, label_encoders, dataNormalization = data_normalization(data_all)
        # print(dataNormalization)
        return label_all, min_values, max_values, label_encoders, dataNormalization

        # print(data[21])

            # return data
'''
    Preprocess each row
    Input is row data
    Output label label
    Data 31-dimensional feature data
'''
def data_process_line(data):
    # Define a list for storing processed data
    label = int(data[0]) - 1  # Labels are integers
    # Handling Empty
    for i in range(1, len(data)):
        if data[i].strip() == '':
            data[i] = 0
    # Handles access time as an integer
    h1, m1, s1 = map(int, data[5].split(':'))
    data[5] = h1 * 3600 + m1 * 60 + s1
    # Processing frequency
    if data[9] == '1000':
        data[9] = 0
    else:
        h2, m2, s2 = map(int, data[9].split(':'))
        data[9] = h2 * 3600 + m2 * 60 + s2
    # Treat as floating point number
    for i in range(10, 17):
        if data[i].strip() == 'True':
            data[i] = 1
        elif data[i].strip() == 'False':
            data[i] = 0
        else:
            data[i] = float(data[i].strip())
    for i in range(17, len(data)):
        data[i] = data[i].split('\'')[1]
    for i in range(22, 25):
        if data[i].strip() == '':
            data[i] = 0
        else:
            data[i] = float(data[i].strip())
    for i in range(25, 31):
        if data[i].strip() == '':
            data[i] = 0
        else:
            data[i] = int(data[i].strip())
    data[31] = float(data[i])
    # if data[31].strip() == '':
    #     data[31] = 0
    # else:
    #     data[31] = float(data[i].strip())
    # print(data)
    return label, data[1:]

'''
    Normalization function
    Input: Features 31 dimensions
    Output: dataNormalization: Normalized array 31 dimensions
    max_values: 31-dimensional array maximum value
    min_values: 31-dimensional array minimum value
    label_encoders: Encoders for categorical data in each column
'''
def data_normalization(data):

    # 1. Column index for categorical data
    category_columns = [0, 1, 2, 3, 5, 6, 7, 16, 17, 18, 19, 20]

    # 2. Column index for numeric data
    numeric_columns = [4, 8, 9, 10, 11, 12, 13, 14, 15, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

    # Extracting categorical data
    category_data = np.array(data)[:, category_columns]

    # Extracting Numeric Data
    numeric_data = np.array(data)[:, numeric_columns].astype(float)

    # 3. Label Encoding for Categorical Data
    label_encoders = []
    encoded_category_data = np.empty_like(category_data, dtype=int)

    for i, col in enumerate(category_columns):
        le = LabelEncoder()
        encoded_category_data[:, i] = le.fit_transform(np.array(category_data[:, i]))
        label_encoders.append(le)

    # 4. Min-Max normalization of numerical data
    scaler = MinMaxScaler()
    normalized_numeric_data = scaler.fit_transform(numeric_data)

    # Save the maximum and minimum values ​​of each column
    min_values = scaler.data_min_
    max_values = scaler.data_max_

    # 5. Merge the encoded categorical data and normalized numerical data
    dataNormalization = np.hstack((encoded_category_data, normalized_numeric_data))

    return min_values, max_values, label_encoders, dataNormalization

'''
    Subsequent calls process each row of data as normalized
'''
def normalize_new_data(new_data, min_values, max_values, label_encoders):

    new_data_array = np.array(new_data)  # Convert the input data into an array
    # 1. Column index for categorical data
    category_columns = [0, 1, 2, 3, 5, 6, 7, 16, 17, 18, 19, 20]

    # 2. Column index for numeric data
    numeric_columns = [4, 8, 9, 10, 11, 12, 13, 14, 15, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

    # Encode the categorical data using the previous LabelEncoder
    encoded_category_data = np.empty(len(category_columns), dtype=int)

    for i, col in enumerate(category_columns):
        encoded_category_data[i] = label_encoders[i].transform([new_data_array[col]])[0]

    # 2. Extracting Numeric Data
    numeric_data = new_data_array[numeric_columns].astype(float)

    # 3. Normalize numerical data
    range_ = max_values - min_values
    range_[range_ == 0] = 1  # Avoid division by 0

    normalized_numeric_data = (numeric_data - min_values) / range_

    # 4. Merge categorically encoded data and normalized numeric data
    normalized_new_data = np.hstack((encoded_category_data, normalized_numeric_data))

    return normalized_new_data


if __name__ == '__main__':
    # s = "Label;3,Activity;user26|Role3|DB10.52|WorkPlace|01:28:49|new|new|GoodPlace|1000|23.935556411743164,Physics;{'gas_flow': 800.0, 'temperature_high': 24.30555534362793, 'pressure_high': 470.77545166015625, 'valve_high': True, 'pressure_mid': 230.03472900390625, 'valve_mid': True},Net;{'ip_src': '192.168.1.10', 'tcp_srcport': '53078', 'ip_dst': '192.168.1.3', 'tcp_dstport': '102', 'frame_protocols': 'eth:ethertype:ip:tcp', 'frame_time_delta': '0.000095000', 'frame_time_relative': '0.325782000', 'frame_time_delta_displayed': '0.000000000', 'frame_len': '54', 'eth_src_oui': '3782', 'eth_dst_oui': '6939', 'tcp_len': '0', 'tcp_ack': '2', 'tcp_analysis_bytes_in_flight': '', 'tcp_analysis_ack_rtt': ''}"
    file_path = '/home/wanghangyu/pythonProjects/testProject/data/data_test.log'
    abel_all, min_values, max_values, label_encoders, dataNormalization = data_pre_process(file_path)
    print(dataNormalization[2])
    print('_______________________________________')
    #Test single message
    new_data = ['abnormal_user5', 'Temp_role', 'DB10.52', 'StrangePlace2', 17, 'old', 'old', 'BadPlace', 0,
                25.176048278808594, 800.0, 24.467592239379883, 497.97454833984375, 1, 234.6643524169922, 1,
                '192.168.1.10', '56251', '192.168.1.3', '102', 'eth:ethertype:ip:tcp:tpkt:cotp:s7comm', 0.00018,
                9.881918, 0.023853, 85, 3782, 6939, 31, 534, 31, 31.0]
    normalized_data = normalize_new_data(new_data, min_values, max_values, label_encoders)
    print(normalized_data)
    print(len(normalized_data))


