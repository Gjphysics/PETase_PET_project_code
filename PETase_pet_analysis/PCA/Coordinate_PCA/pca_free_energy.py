"""
自由能面分析与构象提取脚本

基于PCA坐标数据进行自由能面计算、极小值检测和结构提取
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
import MDAnalysis as mda
from pathlib import Path
import subprocess
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import sys

class FreeEnergyAnalyzer:
    """自由能面分析主类"""
    
    def __init__(self, config):
        """
        初始化分析器
        Args:
            config: 包含所有配置参数的类或对象
        """
        self.config = config
        self.pca_df = None
        self.free_energy = None
        self.xedges = None
        self.yedges = None
        self.minima_coords = None
        self.minima_distance_matrices = {}  # 存储每个极小值的平均距离矩阵
        self.extracted_structures = {}  # 存储提取的结构信息
        
        # 创建输出目录
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """加载PCA坐标数据"""
        try:
            self.pca_df = pd.read_csv(self.config.pca_csv_path)
            print(f"成功加载数据：{len(self.pca_df)} 个数据点")
            print("数据字段示例:")
            print(self.pca_df.head())
        except FileNotFoundError:
            raise Exception(f"文件不存在：{self.config.pca_csv_path}")
        except Exception as e:
            raise Exception(f"数据加载失败：{str(e)}")

    def calculate_free_energy(self):
        """计算自由能面"""
        coords = self.pca_df[['PC1', 'PC2']].values
        
        # 物理常数设置
        kB = 0.008314462618 if self.config.energy_unit == 'kJ/mol' else 0.0019872041
        
        # 计算二维直方图
        x, y = coords[:,0], coords[:,1]
        hist, xedges, yedges = np.histogram2d(x, y, bins=self.config.n_bins)
        hist = hist.T + 1e-10  # 避免零值
        
        # 计算概率和自由能
        prob = hist / hist.sum()
        free_energy = -kB * self.config.temperature * np.log(prob)
        free_energy -= free_energy.min()  # 归一化
        
        self.free_energy = free_energy
        self.xedges = xedges
        self.yedges = yedges

    def find_minima(self):
        """检测自由能面极小值点"""
        # 高斯平滑
        smoothed = gaussian_filter(self.free_energy, sigma=1)
        
        # 检测局部极小值
        minima = peak_local_max(
            -smoothed, 
            min_distance=self.config.min_distance,
            threshold_abs=-self.config.energy_threshold
        )
        
        # 转换为PC坐标
        minima_pc = []
        fe_values = []
        for idx in minima:
            pc1 = self.xedges[idx[1]]  # 注意坐标顺序
            pc2 = self.yedges[idx[0]]
            minima_pc.append((pc1, pc2))
            fe_values.append(self.free_energy[idx[0], idx[1]])

        # 按自由能值从小到大排序
        sorted_indices = np.argsort(fe_values)
        self.minima_coords = np.array(minima_pc)[sorted_indices]
        print(f"检测到 {len(self.minima_coords)} 个极小值点")

    def visualize_energy(self, with_minima=False, with_structures=False):
        """可视化自由能面"""
        plt.figure(figsize=(10, 8), dpi=150)
        
        # 生成网格
        x_centers = (self.xedges[:-1] + self.xedges[1:]) / 2
        y_centers = (self.yedges[:-1] + self.yedges[1:]) / 2
        X, Y = np.meshgrid(x_centers, y_centers)
        
        # 绘制等高线
        levels = np.linspace(0, 15, 16)
        contour = plt.contourf(X, Y, self.free_energy, levels=levels, cmap='viridis')
        plt.colorbar(label=f'Free Energy ({self.config.energy_unit})')
        
        # 添加坐标轴标签
        plt.xlabel('PC1', fontsize=12)
        plt.ylabel('PC2', fontsize=12)
        
        # 添加标题
        title = 'Free Energy Landscape'
        if with_minima:
            title += ' with Minima'
        if with_structures:
            title += ' and Extracted Structures'
        plt.title(title, fontsize=14)
        
        # 标注极小值
        if with_minima and self.minima_coords is not None:
            # 计算坐标范围用于偏移量计算
            x_range = self.xedges[-1] - self.xedges[0]
            y_range = self.yedges[-1] - self.yedges[0]
            
            for i, (x, y) in enumerate(self.minima_coords):
                # 绘制标记
                plt.scatter(x, y, c='red', marker='x', s=100, linewidth=2)
                # 添加编号（带偏移量）
                plt.text(x + 0.05*x_range, y + 0.05*y_range, f'{i+1}', 
                         fontsize=12, color='red', ha='center', va='center',
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # 标注提取的结构
        if with_structures and hasattr(self, 'extracted_structures') and self.extracted_structures:
            for min_idx, structure_list in self.extracted_structures.items():
                min_num = int(min_idx.split('_')[1])  # 从 'min_1' 中提取数字
                if min_num <= len(self.minima_coords):
                    x, y = self.minima_coords[min_num-1]
                    
                    # 绘制结构点的圆圈标记
                    circle = plt.Circle((x, y), 0.1, fill=False, color='orange', linewidth=2)
                    plt.gca().add_patch(circle)
                    
                    # 添加结构数量标注
                    plt.text(x, y - 0.15, f'{len(structure_list)}', 
                             fontsize=10, color='orange', ha='center', va='center',
                             bbox=dict(facecolor='white', alpha=0.8, edgecolor='orange'))
        
        # 保存图片
        suffix = ""
        if with_minima:
            suffix += "_with_minima"
        if with_structures:
            suffix += "_with_structures"
        output_path = f"{self.config.output_dir}/free_energy{suffix}.png"
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"已保存自由能面图：{output_path}")

    def display_minima_info(self):
        """显示极小值信息并等待用户确认"""
        print("\n" + "="*60)
        print("检测到的极小值点信息:")
        print("-"*60)
        for i, (pc1, pc2) in enumerate(self.minima_coords):
            print(f"极小值点 {i+1}: PC1={pc1:.3f}, PC2={pc2:.3f}")
        
        print(f"\n将为每个极小值点提取 {self.config.n_nearest} 个最近邻结构")
        print(f"总共将提取 {len(self.minima_coords) * self.config.n_nearest} 个PDB结构文件")
        print("="*60)
        
        # 显示图片路径提示
        minima_plot_path = f"{self.config.output_dir}/free_energy_with_minima.png"
        print(f"\n请查看极小值点标注图: {minima_plot_path}")
        print("图中红色 'x' 标记为检测到的极小值点，数字为编号")
        
        return self.get_user_confirmation()
    
    def get_user_confirmation(self):
        """获取用户确认"""
        while True:
            try:
                user_input = input("\n是否继续提取结构? (y/n/yes/no): ").strip().lower()
                if user_input in ['y', 'yes']:
                    return True
                elif user_input in ['n', 'no']:
                    return False
                else:
                    print("请输入 y/yes 或 n/no")
            except KeyboardInterrupt:
                print("\n\n用户中断操作")
                return False
            except EOFError:
                print("\n\n检测到EOF，默认为不继续")
                return False

    def extract_structures(self):
        """提取极小值点附近结构并计算距离矩阵"""
        self.extracted_structures = {}  # 重置结构存储
        
        for i, (target_pc1, target_pc2) in enumerate(self.minima_coords):
            print(f"\n处理极小值点 {i+1}/{len(self.minima_coords)}...")
            
            # 计算距离并添加到DataFrame
            self.pca_df['distance'] = np.sqrt(
                (self.pca_df.PC1 - target_pc1)**2 + 
                (self.pca_df.PC2 - target_pc2)**2
            )
            
            # 筛选候选点
            candidates = self.pca_df[self.pca_df.distance <= self.config.search_radius]
            
            if len(candidates) == 0:
                print(f"! 未找到 ({target_pc1:.2f}, {target_pc2:.2f}) 附近的轨迹帧")
                continue
                
            # 取最近的n个帧
            nearest = candidates.nsmallest(self.config.n_nearest, 'distance')
            
            # 加载并保存结构
            structure_info_list = self._save_structures(nearest, i)
            
            # 存储结构信息用于后续可视化
            if structure_info_list:
                self.extracted_structures[f'min_{i+1}'] = structure_info_list
            
            # 计算距离矩阵
            if structure_info_list:
                self._calculate_average_distance_matrix(structure_info_list, i)
            
        # 清理临时列（可选）
        self.pca_df.drop(columns=['distance'], inplace=True, errors='ignore')
        
        # 计算极小值之间的距离矩阵差值
        self._calculate_matrix_differences()
        
        # 生成数据总结
        self._generate_summary()

    def _save_structures(self, neighbors, min_index):
        """保存结构文件并返回结构信息列表（用于后续距离矩阵计算）"""
        structure_info = []  # 保存结构信息而不是AtomGroup对象
        for j, (idx, row) in enumerate(neighbors.iterrows()):
            try:
                u = mda.Universe(row.topology, row.trajectory)
                u.trajectory[row.frame]
                
                output_path = f"{self.config.output_dir}/min_{min_index+1}_neighbor_{j+1}.pdb"
                protein = u.select_atoms("protein")
                protein.write(output_path)
                #print(f"已保存：{output_path}")
                
                # 保存结构信息用于后续距离矩阵计算
                structure_info.append({
                    'topology': row.topology,
                    'trajectory': row.trajectory,
                    'frame': row.frame,
                    'pdb_path': output_path,
                    'n_residues': len(protein.residues)  # 记录残基数量用于维度检查
                })
                
            except Exception as e:
                print(f"加载失败：{row.trajectory} frame {row.frame}")
                print(f"错误信息：{str(e)}")
        
        return structure_info
    
    def _calculate_distance_matrix(self, structure_info, mode='residue'):
        """
        计算单个结构的距离矩阵
        Args:
            structure_info: 包含结构信息的字典
            mode: 'residue', 'BB', 'SC1'
        Returns:
            numpy.ndarray: 距离矩阵
        """
        try:
            # 重新加载结构以避免内存问题
            u = mda.Universe(structure_info['topology'], structure_info['trajectory'])
            u.trajectory[structure_info['frame']]
            protein = u.select_atoms("protein")
            
            if mode == 'residue':
                # 基于残基质心的距离
                residues = protein.residues
                positions = np.array([res.atoms.center_of_mass() for res in residues])
            elif mode == 'BB':
                # 基于主链原子的距离 (CA atoms for backbone)
                try:
                    ca_atoms = protein.select_atoms("name CA")
                    if len(ca_atoms) == 0:
                        # 如果没有CA原子，尝试使用BB (粗粒化模型)
                        ca_atoms = protein.select_atoms("name BB")
                    if len(ca_atoms) == 0:
                        print(f"警告：无法找到主链原子，使用residue模式")
                        return self._calculate_distance_matrix(structure_info, 'residue')
                    positions = ca_atoms.positions
                except:
                    print(f"警告：无法找到主链原子，使用residue模式")
                    return self._calculate_distance_matrix(structure_info, 'residue')
            elif mode == 'SC1':
                # 基于侧链SC1原子的距离 (粗粒化模型)
                try:
                    sc1_atoms = protein.select_atoms("name SC1")
                    if len(sc1_atoms) == 0:
                        print(f"警告：无法找到SC1原子，使用residue模式")
                        return self._calculate_distance_matrix(structure_info, 'residue')
                    positions = sc1_atoms.positions
                except:
                    print(f"警告：无法找到SC1原子，使用residue模式")
                    return self._calculate_distance_matrix(structure_info, 'residue')
            else:
                raise ValueError(f"不支持的距离计算模式：{mode}")
            
            # 检查是否有有效的位置数据
            if len(positions) == 0:
                raise ValueError(f"无法获取有效的原子位置数据（模式：{mode}）")
            
            # 计算距离矩阵
            distances = pdist(positions)
            distance_matrix = squareform(distances)
            
            return distance_matrix
            
        except Exception as e:
            print(f"计算距离矩阵失败：{str(e)}")
            return None
    
    def _calculate_average_distance_matrix(self, structure_info_list, min_index):
        """计算多个结构的平均距离矩阵"""
        print(f"计算极小值点 {min_index+1} 的平均距离矩阵...")
        
        for mode in self.config.distance_modes:
            matrices = []
            valid_structures = []
            
            # 首先检查所有结构的维度一致性
            reference_size = None
            
            for i, structure_info in enumerate(structure_info_list):
                try:
                    matrix = self._calculate_distance_matrix(structure_info, mode)
                    if matrix is not None:
                        if reference_size is None:
                            reference_size = matrix.shape[0]
                            matrices.append(matrix)
                            valid_structures.append(i+1)
                        elif matrix.shape[0] == reference_size:
                            matrices.append(matrix)
                            valid_structures.append(i+1)
                        else:
                            print(f"警告：结构 {i+1} 的矩阵维度 {matrix.shape} 与参考维度 ({reference_size}, {reference_size}) 不匹配，跳过")
                    else:
                        print(f"警告：结构 {i+1} 的距离矩阵计算失败，跳过")
                except Exception as e:
                    print(f"计算结构 {i+1} 的距离矩阵失败（模式：{mode}）：{str(e)}")
                    continue
            
            if len(matrices) >= 2:  # 至少需要2个有效矩阵才能计算平均值
                # 计算平均距离矩阵
                avg_matrix = np.mean(matrices, axis=0)
                
                # 保存到字典中
                key = f"min_{min_index+1}_{mode}"
                self.minima_distance_matrices[key] = avg_matrix
                
                # 保存到文件
                output_path = f"{self.config.output_dir}/distance_matrix_min_{min_index+1}_{mode}.npy"
                np.save(output_path, avg_matrix)
                print(f"已保存平均距离矩阵：{output_path} (基于 {len(matrices)}/{len(structure_info_list)} 个有效结构)")
                
                # 保存结构使用信息
                info_path = f"{self.config.output_dir}/distance_matrix_min_{min_index+1}_{mode}_info.txt"
                with open(info_path, 'w') as f:
                    f.write(f"距离矩阵计算信息 (模式: {mode})\n")
                    f.write(f"极小值点: {min_index+1}\n")
                    f.write(f"矩阵维度: {avg_matrix.shape}\n")
                    f.write(f"使用的结构: {len(matrices)}/{len(structure_info_list)}\n")
                    f.write(f"有效结构编号: {', '.join(map(str, valid_structures))}\n")
                
                # 生成热图
                self._plot_distance_matrix(avg_matrix, min_index+1, mode)
            else:
                print(f"警告：极小值点 {min_index+1} 在模式 {mode} 下只有 {len(matrices)} 个有效矩阵，无法计算平均值（需要至少2个）")
    
    def _calculate_matrix_differences(self):
        """计算不同极小值点之间平均距离矩阵的差值"""
        print("\n计算极小值点之间的距离矩阵差值...")
        
        n_minima = len(self.minima_coords)
        
        for mode in self.config.distance_modes:
            print(f"\n处理模式：{mode}")
            
            # 创建差值矩阵存储目录
            diff_dir = f"{self.config.output_dir}/matrix_differences_{mode}"
            Path(diff_dir).mkdir(exist_ok=True)
            
            # 计算两两之间的差值
            for i in range(n_minima):
                for j in range(i+1, n_minima):
                    key_i = f"min_{i+1}_{mode}"
                    key_j = f"min_{j+1}_{mode}"
                    
                    if key_i in self.minima_distance_matrices and key_j in self.minima_distance_matrices:
                        matrix_i = self.minima_distance_matrices[key_i]
                        matrix_j = self.minima_distance_matrices[key_j]
                        
                        # 检查矩阵维度一致性
                        if matrix_i.shape != matrix_j.shape:
                            print(f"警告：极小值点 {i+1} 和 {j+1} 的距离矩阵维度不匹配 ({matrix_i.shape} vs {matrix_j.shape})，跳过差值计算")
                            continue
                        
                        # 计算差值矩阵
                        diff_matrix = matrix_i - matrix_j
                        
                        # 保存差值矩阵
                        output_path = f"{diff_dir}/diff_min_{i+1}_vs_min_{j+1}.npy"
                        np.save(output_path, diff_matrix)
                        
                        # 计算并保存统计信息
                        stats = {
                            'mean_diff': np.mean(np.abs(diff_matrix)),
                            'max_diff': np.max(np.abs(diff_matrix)),
                            'rmsd': np.sqrt(np.mean(diff_matrix**2))
                        }
                        
                        stats_path = f"{diff_dir}/stats_min_{i+1}_vs_min_{j+1}.txt"
                        with open(stats_path, 'w') as f:
                            f.write(f"距离矩阵差值统计 (模式: {mode})\n")
                            f.write(f"计算公式: 极小值点 {i+1} - 极小值点 {j+1}\n")
                            f.write(f"被减数: 极小值点 {i+1}, 减数: 极小值点 {j+1}\n")
                            f.write(f"平均绝对差值: {stats['mean_diff']:.4f} Å\n")
                            f.write(f"最大绝对差值: {stats['max_diff']:.4f} Å\n")
                            f.write(f"均方根差值: {stats['rmsd']:.4f} Å\n")
                            f.write(f"\n说明:\n")
                            f.write(f"- 正值表示极小值点{i+1}在该位置的距离大于极小值点{j+1}\n")
                            f.write(f"- 负值表示极小值点{i+1}在该位置的距离小于极小值点{j+1}\n")
                        
                        print(f"已保存差值矩阵：{output_path}")
                        
                        # 生成差值矩阵热图
                        self._plot_difference_matrix(diff_matrix, i+1, j+1, mode, stats)
    
    def _plot_distance_matrix(self, matrix, min_index, mode):
        """绘制距离矩阵热图"""
        try:
            plt.figure(figsize=(12, 10), dpi=150)
            
            # 检查矩阵是否有有效数据
            if np.isnan(matrix).any() or np.isinf(matrix).any():
                print(f"警告：距离矩阵包含无效数据，跳过热图绘制")
                plt.close()
                return
            
            # 生成残基序号标签 (从30开始，1-based)
            n_residues = matrix.shape[0]
            residue_numbers = [i + 30 for i in range(n_residues)]
            
            # 根据矩阵大小决定标签显示策略
            if n_residues <= 50:
                # 小矩阵：显示所有标签
                tick_labels = residue_numbers
                tick_positions = list(range(n_residues))
            elif n_residues <= 100:
                # 中等矩阵：每5个显示一个标签
                step = 5
                tick_positions = list(range(0, n_residues, step))
                tick_labels = [residue_numbers[i] for i in tick_positions]
            else:
                # 大矩阵：每10个显示一个标签
                step = 10
                tick_positions = list(range(0, n_residues, step))
                tick_labels = [residue_numbers[i] for i in tick_positions]
            
            # 使用seaborn绘制热图
            sns.heatmap(matrix, 
                       cmap='viridis', 
                       cbar_kws={'label': 'Distance (Å)'},
                       square=True,
                       xticklabels=False,
                       yticklabels=False)
            
            # 设置坐标轴标签
            plt.xticks(tick_positions, tick_labels, rotation=45, fontsize=8)
            plt.yticks(tick_positions, tick_labels, rotation=0, fontsize=8)
            
            plt.title(f'Distance Matrix - Minimum {min_index} ({mode} mode)', fontsize=14)
            plt.xlabel('Residue Number', fontsize=12)
            plt.ylabel('Residue Number', fontsize=12)
            
            # 保存图片
            output_path = f"{self.config.output_dir}/distance_matrix_min_{min_index}_{mode}.png"
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
            print(f"已保存距离矩阵热图：{output_path}")
            
        except Exception as e:
            print(f"绘制距离矩阵热图失败：{str(e)}")
            plt.close()
    
    def _plot_difference_matrix(self, diff_matrix, min_i, min_j, mode, stats):
        """绘制差值矩阵热图"""
        try:
            plt.figure(figsize=(12, 10), dpi=150)
            
            # 检查矩阵是否有有效数据
            if np.isnan(diff_matrix).any() or np.isinf(diff_matrix).any():
                print(f"警告：差值矩阵包含无效数据，跳过热图绘制")
                plt.close()
                return
            
            # 生成残基序号标签 (从30开始，1-based)
            n_residues = diff_matrix.shape[0]
            residue_numbers = [i + 30 for i in range(n_residues)]
            
            # 根据矩阵大小决定标签显示策略
            if n_residues <= 50:
                # 小矩阵：显示所有标签
                tick_labels = residue_numbers
                tick_positions = list(range(n_residues))
            elif n_residues <= 100:
                # 中等矩阵：每5个显示一个标签
                step = 5
                tick_positions = list(range(0, n_residues, step))
                tick_labels = [residue_numbers[i] for i in tick_positions]
            else:
                # 大矩阵：每10个显示一个标签
                step = 10
                tick_positions = list(range(0, n_residues, step))
                tick_labels = [residue_numbers[i] for i in tick_positions]
            
            # 计算合适的颜色范围
            vmax = np.max(np.abs(diff_matrix))
            vmin = -vmax
            
            # 如果所有差值都为0，设置一个小的范围避免绘图错误
            if vmax == 0:
                vmax = 0.1
                vmin = -0.1
            
            # 使用seaborn绘制热图
            sns.heatmap(diff_matrix, 
                       cmap='RdBu_r', 
                       center=0,
                       vmin=vmin,
                       vmax=vmax,
                       cbar_kws={'label': 'Distance Difference (Å)'},
                       square=True,
                       xticklabels=False,
                       yticklabels=False)
            
            # 设置坐标轴标签
            plt.xticks(tick_positions, tick_labels, rotation=45, fontsize=8)
            plt.yticks(tick_positions, tick_labels, rotation=0, fontsize=8)
            
            plt.title(f'Distance Matrix Difference: Min {min_i} - Min {min_j} ({mode} mode)\n(被减数: Min {min_i}, 减数: Min {min_j})', fontsize=13)
            plt.xlabel('Residue Number', fontsize=12)
            plt.ylabel('Residue Number', fontsize=12)
            
            # 添加统计信息文本
            textstr = f'Mean |Diff|: {stats["mean_diff"]:.3f} Å\nMax |Diff|: {stats["max_diff"]:.3f} Å\nRMSD: {stats["rmsd"]:.3f} Å'
            plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # 保存图片
            diff_dir = f"{self.config.output_dir}/matrix_differences_{mode}"
            output_path = f"{diff_dir}/diff_min_{min_i}_vs_min_{min_j}.png"
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
            print(f"已保存差值矩阵热图：{output_path}")
            
        except Exception as e:
            print(f"绘制差值矩阵热图失败：{str(e)}")
            plt.close()
    
    def _generate_summary(self):
        """生成分析结果总结"""
        print("\n生成分析总结...")
        
        summary_file = f"{self.config.output_dir}/analysis_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("自由能面分析结果总结\n")
            f.write("=" * 50 + "\n\n")
            
            # 基本信息
            f.write("分析参数:\n")
            f.write(f"- 数据文件: {self.config.pca_csv_path}\n")
            f.write(f"- 输出目录: {self.config.output_dir}\n")
            f.write(f"- 温度: {self.config.temperature} K\n")
            f.write(f"- 能量阈值: {self.config.energy_threshold} kBT\n")
            f.write(f"- 提取结构数: {self.config.n_nearest} 个/极小值\n")
            f.write(f"- 距离计算模式: {', '.join(self.config.distance_modes)}\n\n")
            
            # 极小值信息
            f.write(f"检测到的极小值点: {len(self.minima_coords)} 个\n")
            f.write("-" * 30 + "\n")
            for i, (pc1, pc2) in enumerate(self.minima_coords):
                f.write(f"极小值点 {i+1}: PC1={pc1:.3f}, PC2={pc2:.3f}\n")
            f.write("\n")
            
            # 差值矩阵计算说明
            f.write("距离矩阵差值计算说明:\n")
            f.write("-" * 30 + "\n")
            f.write("差值矩阵计算公式: 极小值点(i) - 极小值点(j), 其中 i < j\n")
            f.write("结果解释:\n")
            f.write("- 正值: 极小值点(i)在该残基对的距离大于极小值点(j)\n")
            f.write("- 负值: 极小值点(i)在该残基对的距离小于极小值点(j)\n")
            f.write("- 文件命名: diff_min_i_vs_min_j 表示 min_i - min_j\n\n")
            
            # 距离矩阵统计
            for mode in self.config.distance_modes:
                f.write(f"距离矩阵统计 ({mode} 模式):\n")
                f.write("-" * 30 + "\n")
                
                # 收集所有矩阵的统计信息
                all_mean_diffs = []
                all_max_diffs = []
                all_rmsds = []
                
                for i in range(len(self.minima_coords)):
                    for j in range(i+1, len(self.minima_coords)):
                        diff_dir = f"{self.config.output_dir}/matrix_differences_{mode}"
                        stats_file = f"{diff_dir}/stats_min_{i+1}_vs_min_{j+1}.txt"
                        
                        if Path(stats_file).exists():
                            try:
                                with open(stats_file, 'r', encoding='utf-8') as stats_f:
                                    lines = stats_f.readlines()
                                    for line in lines:
                                        if "平均绝对差值:" in line:
                                            value_str = line.split(':')[1].strip()
                                            all_mean_diffs.append(float(value_str))
                                        elif "最大绝对差值:" in line:
                                            value_str = line.split(':')[1].strip()
                                            all_max_diffs.append(float(value_str))
                                        elif "均方根差值:" in line:
                                            value_str = line.split(':')[1].strip()
                                            all_rmsds.append(float(value_str))
                            except Exception as e:
                                print(f"读取统计文件失败：{stats_file}, 错误：{str(e)}")
                                continue
                
                if all_mean_diffs:
                    f.write(f"平均绝对差值范围: {min(all_mean_diffs):.3f} - {max(all_mean_diffs):.3f} Å\n")
                    f.write(f"平均绝对差值均值: {np.mean(all_mean_diffs):.3f} ± {np.std(all_mean_diffs):.3f} Å\n")
                if all_max_diffs:
                    f.write(f"最大绝对差值范围: {min(all_max_diffs):.3f} - {max(all_max_diffs):.3f} Å\n")
                    f.write(f"最大绝对差值均值: {np.mean(all_max_diffs):.3f} ± {np.std(all_max_diffs):.3f} Å\n")
                if all_rmsds:
                    f.write(f"RMSD范围: {min(all_rmsds):.3f} - {max(all_rmsds):.3f} Å\n")
                    f.write(f"RMSD均值: {np.mean(all_rmsds):.3f} ± {np.std(all_rmsds):.3f} Å\n")
                f.write("\n")
            
            # 输出文件列表
            f.write("生成的文件:\n")
            f.write("-" * 30 + "\n")
            f.write("1. 自由能面图: free_energy.png, free_energy_with_minima.png\n")
            f.write("2. 结构文件: min_X_neighbor_Y.pdb\n")
            f.write("3. 距离矩阵数据: distance_matrix_min_X_MODE.npy\n")
            f.write("4. 距离矩阵热图: distance_matrix_min_X_MODE.png\n")
            f.write("5. 距离矩阵信息: distance_matrix_min_X_MODE_info.txt\n")
            f.write("6. 差值矩阵数据: matrix_differences_MODE/diff_min_X_vs_min_Y.npy (X < Y, 计算 min_X - min_Y)\n")
            f.write("7. 差值矩阵热图: matrix_differences_MODE/diff_min_X_vs_min_Y.png (显示残基序号30+)\n")
            f.write("8. 差值统计信息: matrix_differences_MODE/stats_min_X_vs_min_Y.txt (包含计算说明)\n")
        
        print(f"已保存分析总结：{summary_file}")

class Config:
    """配置参数容器"""
    def __init__(self, type_id, state, threshold=3.0, distance_modes=None, min_num=5):
        # 输入文件路径
        self.pca_csv_path = f"/M-run-complex-FAST-10/analysis_PCA_new/pca_results/pca_{type_id}_coordinates_{state}.csv"
        
        # 输出目录
        self.output_dir = f"/M-run-complex-FAST-10/analysis_PCA_new/find_minim/{type_id}_{state}_minim_thre_{threshold}"
        
        # 自由能面参数
        self.n_bins = 100        # 分箱数
        self.temperature = 298   # 温度 (K)
        self.energy_unit = 'kJ/mol'  # 能量单位
        
        # 极小值检测参数
        self.min_distance = 5    # 极小值点最小间距（像素）
        self.energy_threshold = threshold  # 能量阈值(kBT)
        
        # 结构提取参数
        self.n_nearest = min_num        # 每个极小值提取的邻近帧数 (修改为5)
        self.search_radius = 3.0 # 搜索半径(PC单位)
        
        # 距离矩阵计算参数
        self.distance_modes = distance_modes if distance_modes else ['residue', 'BB', 'SC1']  # 距离计算模式

def main():
    """主工作流程"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='自由能面分析与构象提取')
    parser.add_argument('-t', '--type', type=str, default='WT', help='分析类型ID（如el）')
    parser.add_argument('-state', '--state', type=str, required=True, help='状态编号')
    parser.add_argument('-thre','--threshold', type=float, required=True, help='能量阈值')
    parser.add_argument('--distance-modes', nargs='+', default=['residue', 'BB', 'SC1'], 
                        choices=['residue', 'BB', 'SC1'],
                        help='距离矩阵计算模式 (可选多个)')
    #parser.add_argument('--plot', action='store_true', help='是否输出pymol脚本')
    parser.add_argument('-back','--backmap', action='store_true', help='是否反向映射')
    parser.add_argument('-min_num', '--min_num', type=int, default=5, help='极小值附近提取的结构数')
    args = parser.parse_args()

    # 初始化配置和分析器
    config = Config(args.type, args.state, args.threshold, args.distance_modes, args.min_num)
    analyzer = FreeEnergyAnalyzer(config)
    
    try:
        # 执行分析流程 - 第一阶段：自由能面和极小值检测
        print("开始自由能面分析...")
        analyzer.load_data()
        analyzer.calculate_free_energy()
        analyzer.visualize_energy()
        analyzer.find_minima()
        analyzer.visualize_energy(with_minima=True)
        
        # 交互式确认
        if len(analyzer.minima_coords) == 0:
            print("未检测到任何极小值点，分析结束。")
            return
            
        continue_extraction = analyzer.display_minima_info()
        
        if not continue_extraction:
            print("\n用户选择不继续，分析结束。")
            print("已生成文件：")
            print(f"- 基础自由能面图: {config.output_dir}/free_energy.png")
            print(f"- 带极小值标注图: {config.output_dir}/free_energy_with_minima.png")
            return
        
        # 第二阶段：结构提取和距离矩阵计算
        print("\n开始结构提取和距离矩阵计算...")
        analyzer.extract_structures()
        
        # 生成带结构标注的最终图
        print("\n生成最终的结构标注图...")
        analyzer.visualize_energy(with_minima=True, with_structures=True)
        
        print("\n分析完成！输出文件包含：")
        print(f"- 自由能面图: {config.output_dir}/free_energy*.png")
        print(f"- 各极小值点邻近结构: {config.output_dir}/min_*_neighbor_*.pdb")
        print(f"- 平均距离矩阵数据: {config.output_dir}/distance_matrix_min_*_*.npy")
        print(f"- 平均距离矩阵热图: {config.output_dir}/distance_matrix_min_*_*.png")
        print(f"- 距离矩阵计算信息: {config.output_dir}/distance_matrix_min_*_*_info.txt")
        print(f"- 距离矩阵差值数据: {config.output_dir}/matrix_differences_*/*.npy")
        print(f"- 距离矩阵差值热图: {config.output_dir}/matrix_differences_*/*.png")
        print(f"- 差值统计信息: {config.output_dir}/matrix_differences_*/stats_*.txt")
        print(f"- 分析总结: {config.output_dir}/analysis_summary.txt")
        
    except Exception as e:
        print(f"分析过程中发生错误：{str(e)}")
        raise
    
    # if args.plot:
    #     # 生成PyMOL脚本
    #     ref = "/data/hcz/M-run-complex/PETase-start/PETase-newbox.gro"
    #     # 所有在config.output_dir下的min_*_neighbor_*.pdb文件
    #     for pdb in Path(config.output_dir).glob("min_*_neighbor_*.pdb"):
    #         output_script = pdb.with_suffix(".pml")
    #         #将pdb前面加上diff_作为output_pdb

    #         output_pdb = pdb.with_name("diff_" + pdb.name)
    #         cmd = f"python /data/hcz/M-run-complex/analysis/pca_free_energy_pymol.py --ref {ref} --mob {pdb} --script {output_script} --out {output_pdb}"
    #         print(f"生成PyMOL脚本：{output_script}")
    #         subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    main()