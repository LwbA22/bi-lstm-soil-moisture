# 加载数据
import pandas as pd

# 文件路径
file_path = 'C:\\Users\\11318\\Desktop\\MAQU\\lstm\\NST08\\jieguo\\predicted_smap_soil_moisture1.csv'

# 读取数据
data = pd.read_csv(file_path)

# Step 1: 修复预测值中的缺失值
# 使用线性插值填补缺失值
data['predicted_soil_moisture'] = data['predicted_soil_moisture'].interpolate(method='linear', limit_direction='both')

# Step 2: 整合 SMAP 数据
# 如果 SMAP 数据存在，替换预测值
data['fused_soil_moisture'] = data['predicted_soil_moisture']  # 初始化为预测值
data.loc[~data['smap_soil_moisture'].isna(), 'fused_soil_moisture'] = data['smap_soil_moisture']

# 保存修复后的数据
output_path = 'C:\\Users\\11318\\Desktop\\MAQU\\lstm\\NST08\\jieguo\\fused_smap_soil_moisture.csv'
data.to_csv(output_path, index=False)

print(f"修复后的数据已保存到 {output_path}")
