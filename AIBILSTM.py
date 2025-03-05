import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import Bidirectional, Dropout

# 读取数据
file_path = 'C:\\Users\\11318\\Desktop\\NGARI\\lstm\\SQ07\\merged_soil_moisture_data.csv'  # 修改为您的文件路径
data = pd.read_csv(file_path)
data['time'] = pd.to_datetime(data['time'])



# 定义24节气日期范围
solar_terms = {
    '立春': ('2018-02-04', '2018-02-18'), '雨水': ('2018-02-19', '2018-03-05'),
    '惊蛰': ('2018-03-06', '2018-03-20'), '春分': ('2018-03-21', '2018-04-03'),
    '清明': ('2018-04-04', '2018-04-19'), '谷雨': ('2018-04-20', '2018-05-05'),
    '立夏': ('2018-05-06', '2018-05-20'), '小满': ('2018-05-21', '2018-06-05'),
    '芒种': ('2018-06-06', '2018-06-20'), '夏至': ('2018-06-21', '2018-07-06'),
    '小暑': ('2018-07-07', '2018-07-22'), '大暑': ('2018-07-23', '2018-08-06'),
    '立秋': ('2018-08-07', '2018-08-22'), '处暑': ('2018-08-23', '2018-09-07'),
    '白露': ('2018-09-08', '2018-09-22'), '秋分': ('2018-09-23', '2018-10-07'),
    '寒露': ('2018-10-08', '2018-10-22'), '霜降': ('2018-10-23', '2018-11-06'),
    '立冬': ('2018-11-07', '2018-11-21'), '小雪': ('2018-11-22', '2018-12-06'),
    '大雪': ('2018-12-07', '2018-12-20'), '冬至': ('2018-12-21', '2019-01-05'),
    '小寒': ('2019-01-05', '2019-01-19'),'大寒': ('2019-01-19', '2019-02-03')

}

# 添加节气列
def assign_solar_term(date, terms):
    for term, (start, end) in terms.items():
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            return term
    return None

data['solar_term'] = data['time'].apply(lambda x: assign_solar_term(x, solar_terms))

# 按节气分组
data_by_solar_term = {term: group for term, group in data.groupby('solar_term')}

# 定义LSTM模型训练和预测函数
def train_and_predict_bilstm(data, look_back=24):
    """
    使用双向 LSTM 和 Dropout 模型训练并预测土壤水分随时间的变化。
    :param data: 包含 measured_soil_moisture 的 DataFrame
    :param look_back: 用于时间序列预测的时间窗口大小
    :return: 预测值数组
    """
    # 数据缩放
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['measured_soil_moisture']])

    # 创建训练数据
    def create_dataset(dataset, look_back=look_back):
        X, Y = [], []
        for i in range(len(dataset) - look_back):
            X.append(dataset[i:(i + look_back), 0])
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    X, Y = create_dataset(scaled_data)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # 构建双向 LSTM 模型
    model = Sequential([
        Bidirectional(LSTM(64, activation='relu', return_sequences=True, input_shape=(X.shape[1], 1))),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 训练模型
    model.fit(X, Y, epochs=20, batch_size=32, verbose=0)

    # 使用模型预测
    predictions = model.predict(X)
    predictions = np.concatenate((np.zeros((look_back, 1)), predictions), axis=0)  # 填充前 look_back 个值
    return scaler.inverse_transform(predictions)

# 按节气处理并预测
predicted_data = {}
for term, term_data in data_by_solar_term.items():
    term_pred = train_and_predict_bilstm(term_data[['measured_soil_moisture']])
    predicted_data[term] = term_pred

# 将预测结果汇总到原始数据中
for term, term_data in data_by_solar_term.items():
    term_length = len(term_data)
    data.loc[term_data.index, 'predicted_soil_moisture'] = predicted_data[term][:term_length]

# 保存结果到文件
data.to_csv('C:\\Users\\11318\\Desktop\\NGARI\\lstm\\SQ07\\jieguo\\predicted_smap_soil_moisture1.csv', index=False,encoding='utf_8_sig')


# 可视化部分（如果需要）

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'

plt.figure(figsize=(10, 6))
plt.plot(data['time'], data['measured_soil_moisture'], label='Measured Soil Moisture', alpha=0.6)
plt.plot(data['time'], data['predicted_soil_moisture'], label='Predicted Soil Moisture', alpha=0.6)
plt.legend()
plt.xlabel('Time')
plt.ylabel('Soil Moisture')
plt.title('Measured vs Predicted Soil Moisture')
plt.show()
