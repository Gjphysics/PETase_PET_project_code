import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
plt.rcParams['font.family'] = 'Arial'

# 固定参数
mer = 10
chain_num = int(2000/mer)
file_name = f'{mer}-{chain_num}'
interval = 50
density_data_file = f'density_data_mer{mer}.txt'
save_png_name = f'Tg-{mer}-{chain_num}-v3.png'

# 更改当前工作路径
os.chdir(f"/data/hcz/M-run-2000PET/{file_name}")
wkdir = os.getcwd()

# 检查是否存在已保存的密度数据
if os.path.exists(density_data_file):
    print(f"检测到已存在的数据文件 {density_data_file}，直接读取...")
    # 读取已保存的数据: 温度, 平均密度, 标准差
    saved_data = np.loadtxt(density_data_file, comments='#')
    i_values = saved_data[:, 0]
    mean_densities = saved_data[:, 1]
    std_densities = saved_data[:, 2]
else:
    print("未检测到数据文件，开始计算密度数据...")
    results = []
    
    for i in range(600, 99, -interval):
        input_file = f'{wkdir}/{i}/density-md.xvg'
        data = np.loadtxt(input_file, comments=['@', '#'])
        
        # 筛选最后10ns的数据
        mask = (data[:, 0] / 1000 >= 40)
        filtered_data = data[mask]
        
        # 计算y数据平均数和标准差
        mean_density = np.mean(filtered_data[:, 1] / 1000)
        std_density = np.std(filtered_data[:, 1] / 1000)
        
        # 将温度、平均密度和标准差添加到结果数组
        results.append((i, mean_density, std_density))
    
    # 分别获取温度、平均密度和标准差
    i_values, mean_densities, std_densities = zip(*results)
    i_values = np.array(i_values)
    mean_densities = np.array(mean_densities)
    std_densities = np.array(std_densities)
    
    # 保存数据到文件
    header = "Temperature(K)\tDensity(g/cm^3)\tStd_Dev(g/cm^3)"
    data_to_save = np.column_stack((i_values, mean_densities, std_densities))
    np.savetxt(density_data_file, data_to_save, header=header, 
               fmt='%.6f', delimiter='\t')
    print(f"密度数据已保存到 {density_data_file}")

# 计算比体积 (specific volume = 1/density)
specific_volumes = 1 / mean_densities
# 计算比体积的误差传播: std(1/x) = std(x) / x^2
std_specific_volumes = std_densities / (mean_densities ** 2)

# 拟合直线
def linear_fit(x, y):
    return np.polyfit(x, y, 1)

# 将数据分为两部分进行拟合
fit1_indices = [i for i, x in enumerate(i_values) if 100 <= x <= 300]
fit2_indices = [i for i, x in enumerate(i_values) if 400 <= x <= 600]

fit1_x = np.array([i_values[i] for i in fit1_indices])
fit1_y = np.array([specific_volumes[i] for i in fit1_indices])
fit2_x = np.array([i_values[i] for i in fit2_indices])
fit2_y = np.array([specific_volumes[i] for i in fit2_indices])

fit1_params = linear_fit(fit1_x, fit1_y)
fit2_params = linear_fit(fit2_x, fit2_y)

# 计算交点
def find_intersection(params1, params2):
    def func(x):
        return np.polyval(params1, x) - np.polyval(params2, x)
    return fsolve(func, 300)[0]  # 初始猜测交点在 300K 附近

intersection_x = find_intersection(fit1_params, fit2_params)
intersection_y = np.polyval(fit1_params, intersection_x)

# 确定用于绘制拟合直线的范围
extended_x_values_1 = np.linspace(100, 400, 500)
extended_x_values_2 = np.linspace(300, 600, 500)

# 计算扩展范围内的拟合直线的 y 值
fit1_y_values = np.polyval(fit1_params, extended_x_values_1)
fit2_y_values = np.polyval(fit2_params, extended_x_values_2)

# 绘制图像 - 12:6比例
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

# 绘制带error bar的散点图
ax.errorbar(i_values, specific_volumes, yerr=std_specific_volumes, 
            fmt='o', markersize=10, capsize=4, capthick=1.5, 
            elinewidth=3,color="black", ecolor='gray')

# 绘制拟合直线 - 使用更好看的颜色
ax.plot(extended_x_values_1, fit1_y_values, color="#DB8667", linewidth=4,
        )
ax.plot(extended_x_values_2, fit2_y_values, color="#63AAE0", linewidth=4,
        )

# 标记交点
ax.scatter(intersection_x, intersection_y, color='#C1121F', s=100, zorder=5)
ax.axvline(x=intersection_x, color='black', linestyle='--', linewidth=2,
           label=f'$T_g$ = {intersection_x:.1f} K')

# 设置坐标轴标签 - 26号字体
ax.set_xlabel('Temperature (K)', fontsize=26)
ax.set_ylabel(r'$v(T)\ (\mathrm{cm}^3/\mathrm{g})$', fontsize=26)
ax.set_ylim(0.6, 1.3)

# 设置刻度标签字体大小 - 26号字体
ax.tick_params(axis='both', labelsize=26, width=1.5, length=6)

# 去除右侧和上侧的刻度线
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(right=False, top=False)

# 设置坐标轴线宽度
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

# 设置图例 - 26号字体
ax.legend(fontsize=24, frameon=False)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig(save_png_name, bbox_inches='tight')
print(f"图片已保存为 {save_png_name}")
print(f"玻璃态转变温度 Tg = {intersection_x:.2f} K")

plt.show()
