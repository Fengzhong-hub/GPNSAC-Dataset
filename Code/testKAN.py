# Testing the Kan model
from sklearn.model_selection import train_test_split
import data_process
import numpy as np
import pickle
import torch
import numpy as np
from Fast_KAN_model import FastKAN
import time
import Is_nomal_behavior
'''
    Select test set
'''



def select_data(file_path):
    # Divide the training set and test set
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
            # Net[4]是特殊类型 'frame_protocols': 'eth:ethertype:ip:tcp'
            data[21] = Net[4].split("'frame_protocols':")[-1]
            for i in range(5, len(Net)):
                data[i + 17] = Net[i].split(':')[1]

            #label 
            # data_process_1 31-dimensional feature data
            label, data_process_1 = data_process.data_process_line(data)
            label_all.append(label)
            data_all.append(data_process_1)
            # print(label)
            # print(data_all)
    X_train, X_test, y_train, y_test = split_data(data_all, label_all, test_size=0.2)
    return X_test, y_test
    # print(X_train)
    # print(y_train)
    # print('-----')
    # print(X_test)
    # print(y_test)

# 3. 划分数据集
def split_data(data, labels, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

def testKAN_1(X_test,y_test, min_values, max_values, label_encoders):
    model = load_model('kanModel_1')
    start_time = time.time()  # Start recording time
    num_X_test = len(X_test)
    accuracy_num = 0
    normalization_time = [] # Record normalized time
    for data_line, label_line in zip(X_test, y_test):
        # Normalization
        normalization_time_1 = time.time()  #
        new_data = data_process.normalize_new_data(data_line, min_values, max_values, label_encoders)
        # print(X_test)
        new_data_tensor = torch.tensor(new_data, dtype=torch.float32).unsqueeze(0)  # unsqueeze(0) Increase the batch dimension
        normalization_time_2 = time.time()
        normalization_time.append((normalization_time_2 - normalization_time_1) * 1000) # Unit: milliseconds
        # 6. Perform inference (classification prediction)
        with torch.no_grad():  # Disable gradient calculation
            outputs = model(new_data_tensor)

            # Get the category with the highest score (predicted category)
            _, predicted_class = torch.max(outputs, dim=1)  # The output is (batch_size, num_classes), select the class index of the maximum value
        # 7. Output prediction results
        # Note: If the category starts at 1 during training, add 1 here to make the index consistent with the label
        # print(f"Predicted class: {predicted_class.item() + 1}")  # The addition of 1 is to align the predictions with the original label range (1-5)
        if predicted_class.item() == label_line:
            accuracy_num += 1
    # print(accuracy_num)
    end_time = time.time()  # Start recording time
    print(f'测试数量为：{num_X_test}')
    print(f'准确率：{accuracy_num / num_X_test * 100} %')
    print(f'花费平均时间{(end_time - start_time) * 1000 / num_X_test} ms')
    normalization_time_all = 0
    for i in normalization_time:
        normalization_time_all = normalization_time_all + i
    print(f'归一化平均时间：{normalization_time_all / len(normalization_time)} ms')
    # print(normalization_time)

def testKAN_2(X_test,y_test, min_values, max_values, label_encoders):
    model = load_model('kanModel_2', output_dim=4)
    start_time = time.time()  # Start recording time
    num_X_test = len(X_test)
    accuracy_num = 0

    for data_line, label_line in zip(X_test, y_test):
        if not Is_nomal_behavior.role_access(data_line[1], data_line[2]):
            accuracy_num += 1
            # print(label_line)
            continue
        else:
            if label_line > 0:
                label_line -= 1
            # Normalization
            new_data = data_process.normalize_new_data(data_line, min_values, max_values, label_encoders)
            # print(X_test)
            new_data_tensor = torch.tensor(new_data, dtype=torch.float32).unsqueeze(0)  # unsqueeze(0) Increase the batch dimension
            # 6. Perform inference (classification prediction)
            with torch.no_grad():  # Disable gradient calculation
                outputs = model(new_data_tensor)

                # Get the category with the highest score (predicted category)
                _, predicted_class = torch.max(outputs, dim=1)  # The output is (batch_size, num_classes), select the class index of the maximum value
            # 7. Output prediction results
            # Note: If the category starts at 1 during training, add 1 here to make the index consistent with the label
            if predicted_class.item() == label_line:
                accuracy_num += 1
    # print(accuracy_num)
    end_time = time.time()  # Start recording time
    print(f'The number of tests is：{num_X_test}')
    print(f'Accuracy：{accuracy_num / num_X_test * 100} %')
    print(f'Average time spent{(end_time - start_time) * 1000 / num_X_test} ms')

'''
Importing the Model
'''
def load_model(model_path, output_dim = 5):
    # 1. Define the same model structure as during training
    input_dim = 31  # The dimension of the input data (assuming your input dimension is 31)
    # output_dim = 5  # Number of categories (1-5)
    hidden_layers = [input_dim, 64, 32, output_dim]  # Hidden layer structure

    # 2. Instantiate Model
    model = FastKAN(hidden_layers)

    # 3. Loading saved model weights
    # model_path = 'kanModel_1'  # This is the path where you saved the model.
    model.load_state_dict(torch.load(model_path))  # Loading model weights
    model.eval()  # Switch to evaluation mode and disable the effects of Dropout etc. during inference
    return model
'''
Import normalization parameters
'''
def load_normalization_parameters(min_path, max_path, encoder_path):
    min_values = np.load(min_path)
    max_values = np.load(max_path)
    with open(encoder_path, 'rb') as f:
        loaded_label_encoders = pickle.load(f)
    return min_values, max_values, loaded_label_encoders

if __name__ == '__main__':
    file_path = '/home/wanghangyu/pythonProjects/testProject/data/output.log'
    X_test, y_test = select_data(file_path)
    # Normalization parameters
    min_values, max_values, label_encoders = load_normalization_parameters(min_path='min_values.npy', max_path='max_values.npy', encoder_path='label_encoders.pkl')

    testKAN_1(X_test, y_test, min_values, max_values, label_encoders)

    print('---------------------')
    # Normalization parameters
    min_values, max_values, label_encoders = load_normalization_parameters(min_path='min_values_2.npy', max_path='max_values_2.npy', encoder_path='label_encoders_2.pkl')

    testKAN_2(X_test,y_test, min_values, max_values, label_encoders)
