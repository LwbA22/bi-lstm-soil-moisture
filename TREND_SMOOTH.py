import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 文件路径
file_path = 'C:\\Users\\11318\\Desktop\\MAQU\\lstm\\NST08\\jieguo\\fused_smap_soil_moisture.csv'
data = pd.read_csv(file_path)

# 时间格式转换
data['time'] = pd.to_datetime(data['time'], format='%Y/%m/%d %H:%M')

# 修正函数
def trend_based_correction_v2(data):
    """
    基于X1和X2时间点SMAP原始值与实测值的比值修正缺失的SMAP数据。
    
    参数：
    - data: DataFrame，包含列 'smap_soil_moisture' 和 'predicted_soil_moisture'
    
    返回：
    - 修正后的土壤水分值列表
    """
    corrected_values = data['predicted_soil_moisture'].copy()
    smap_values = data['smap_soil_moisture']
    measured_values = data['measured_soil_moisture']
    
    # 遍历所有时刻
    for i in range(1, len(data) - 1):
        # 如果当前时刻有SMAP原始值
        if not np.isnan(smap_values.iloc[i]):
            corrected_values.iloc[i] = smap_values.iloc[i]  # 使用原始SMAP值
        else:
            # 查找前后两个有SMAP原始值的时间点
            prev_idx = i - 1
            next_idx = i + 1
            
            # 直到找到前后都有SMAP数据的点
            while prev_idx >= 0 and np.isnan(smap_values.iloc[prev_idx]):
                prev_idx -= 1
            while next_idx < len(data) and np.isnan(smap_values.iloc[next_idx]):
                next_idx += 1
            
            # 如果前后有有效的SMAP数据
            if prev_idx >= 0 and next_idx < len(data):
                # 获取前后两个时间点的SMAP原始值与实测值
                smap_prev = smap_values.iloc[prev_idx]
                smap_next = smap_values.iloc[next_idx]
                measured_prev = measured_values.iloc[prev_idx]
                measured_next = measured_values.iloc[next_idx]
                
                # 计算前后比值
                ratio_prev = smap_prev / measured_prev if measured_prev != 0 else 1
                ratio_next = smap_next / measured_next if measured_next != 0 else 1
                
                # 计算前后比值的平均值
                avg_ratio = (ratio_prev + ratio_next) / 2
                
                # 使用LSTM预测值并根据平均比值进行修正
                corrected_values.iloc[i] = corrected_values.iloc[i] * avg_ratio


    smap_indices = smap_values.dropna().index            
    first_smap_idx = smap_indices[0]
    first_smap_ratio = smap_values[first_smap_idx] / measured_values[first_smap_idx]
    corrected_values[:first_smap_idx] = corrected_values[:first_smap_idx] * first_smap_ratio

    # 处理最后一个SMAP点之后的值
    last_smap_idx = smap_indices[-1]
    last_smap_ratio = smap_values[last_smap_idx] / measured_values[last_smap_idx]
    corrected_values[last_smap_idx + 1:] = corrected_values[last_smap_idx + 1:] * last_smap_ratio
    return corrected_values

# 应用修正函数
data['fused_smap_soil_moisture'] = trend_based_correction_v2(data)

# 保存结果
output_path = 'C:\\Users\\11318\\Desktop\\MAQU\\lstm\\SQ07\\jieguo\\trend_ratio_corrected_smap_soil_moisture_v21.csv'
data.to_csv(output_path, index=False,encoding='utf_8_sig')

# 绘图检查修正效果                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
plt.figure(figsize=(12, 6))
plt.plot(data['time'], data['measured_soil_moisture'], label='Measured Soil Moisture', alpha=0.6)
plt.plot(data['time'], data['smap_soil_moisture'], label='Original SMAP Soil Moisture', alpha=0.6)
plt.plot(data['time'], data['fused_smap_soil_moisture'], label='Trend Ratio Corrected Fused SMAP Soil Moisture', alpha=0.8)
plt.legend()
plt.xlabel('Time')
plt.ylabel('Soil Moisture')
plt.title('Trend Ratio Corrected Fused SMAP Soil Moisture')
plt.show()

# 输出结果路径
print(f"修正后的数据已保存到 {output_path}")
 