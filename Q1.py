import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# 设置图片字体，确保中文能够正确显示
if sys.platform == 'win32':
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
elif sys.platform == 'darwin':
    plt.rcParams['font.sans-serif'] = ['PingFang SC']
    plt.rcParams['axes.unicode_minus'] = False
else:
    # 其他系统使用默认设置
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

# 保存路径，确保在不同操作系统下都能正确创建目录
figures_path = os.path.join(os.getcwd(), 'Figures')
os.makedirs(figures_path, exist_ok=True)

# ======================
# 1. 参数设置
# ======================
# 相机内参数
alpha = 1000  # u轴比例因子
beta = 1000  # v轴比例因子
gamma = 5  # 倾斜参数
u0, v0 = 320, 240  # 主点坐标

# 相机外参数（世界坐标系与相机坐标系重合）
R = np.eye(3)  # 旋转矩阵（单位矩阵）
t = np.zeros(3)  # 平移向量（零向量）

# 一维物体参数
A = np.array([0, 0, 10])  # 固定点坐标 (x,y,z)
L = 5  # 线段长度

# 灵敏性分析参数
num_segments = 6  # 线段数量
num_perturbations = 1000  # 每个点的扰动次数
angle_std = 0.01  # 角度扰动标准差（弧度）


# ======================
# 2. 生成运动状态（线段）
# ======================
def generate_segment_ends(A, L, theta, phi):
    """生成线段端点B的坐标"""
    direction = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])
    B = A + L * direction
    return B


# 随机生成6组方向角 (theta, phi)
np.random.seed(42)
thetas = np.random.uniform(0.1, np.pi / 2 - 0.1, num_segments)  # 避免z=0的情况
phis = np.random.uniform(0, 2 * np.pi, num_segments)

# 生成线段端点
Bs = [generate_segment_ends(A, L, theta, phi) for theta, phi in zip(thetas, phis)]


# ======================
# 3. 像点坐标计算
# ======================
def project_to_image(M, A, R, t):
    """将3D点投影到像平面"""
    # 转换为齐次坐标
    M_hom = np.append(M, 1)

    # 计算相机坐标系坐标 (世界坐标系与相机坐标系重合)
    cam_coords = R @ M_hom[:3] + t

    # 投影变换
    x, y, z = cam_coords
    u = (alpha * x + gamma * y + u0 * z) / z
    v = (beta * y + v0 * z) / z
    return np.array([u, v])


# 计算固定点A的像点
A_image = project_to_image(A, A, R, t)

# 计算各线段端点B的像点
B_images = [project_to_image(B, A, R, t) for B in Bs]

# ======================
# 4. 灵敏性分析
# ======================
# 存储灵敏性分析结果
distances = []

for i in range(num_segments):
    segment_distances = []
    orig_B_image = B_images[i]

    for _ in range(num_perturbations):
        # 添加角度扰动
        delta_theta = np.random.normal(0, angle_std)
        delta_phi = np.random.normal(0, angle_std)

        # 计算扰动后的端点位置
        B_perturbed = generate_segment_ends(
            A, L, thetas[i] + delta_theta, phis[i] + delta_phi
        )

        # 计算扰动后的像点
        perturbed_image = project_to_image(B_perturbed, A, R, t)

        # 计算欧氏距离
        distance = np.linalg.norm(orig_B_image - perturbed_image)
        segment_distances.append(distance)

    distances.append(segment_distances)

# ======================
# 5. 结果分析与可视化
# ======================
# 计算统计量
mean_distances = [np.mean(d) for d in distances]
std_distances = [np.std(d) for d in distances]
max_distances = [np.max(d) for d in distances]

# 打印结果
print("固定点A的像点坐标:", A_image)
print("\n各线段端点B的像点坐标:")
for i, img in enumerate(B_images):
    print(f"线段 {i + 1}: ({img[0]:.2f}, {img[1]:.2f})")

print("\n灵敏性分析统计量:")
print(f"{'线段':<6}{'平均距离(像素)':<15}{'标准差':<15}{'最大距离(像素)':<15}")
for i in range(num_segments):
    print(f"{i + 1:<6}{mean_distances[i]:<15.4f}{std_distances[i]:<15.4f}{max_distances[i]:<15.4f}")

# 整体统计
all_distances = np.concatenate(distances)
print(f"\n整体统计:")
print(f"全局平均距离: {np.mean(all_distances):.4f} 像素")
print(f"全局标准差: {np.std(all_distances):.4f} 像素")
print(f"全局最大距离: {np.max(all_distances):.4f} 像素")

# 可视化灵敏性分析结果
# 设置图表样式和参数，不使用seaborn
plt.rcParams['figure.figsize'] = (15, 6)
plt.rcParams['font.size'] = 12
fig, (ax1, ax2) = plt.subplots(1, 2)

# 像点分布图
ax1.scatter(A_image[0], A_image[1], c='red', s=150, marker='s', label='固定点A', zorder=3)
# 为不同线段端点使用不同颜色
colors = plt.cm.Set1(np.linspace(0, 1, num_segments))
for i, (img, color) in enumerate(zip(B_images, colors)):
    ax1.scatter(img[0], img[1], c=[color], label=f'B{i + 1}', s=80, zorder=2)
    ax1.annotate(f'B{i + 1}', (img[0], img[1]), textcoords="offset points", 
                xytext=(5, 5), ha='left', fontsize=9)

ax1.set_title('像点空间分布图', fontsize=14, fontweight='bold')
ax1.set_xlabel('u (像素)', fontsize=12)
ax1.set_ylabel('v (像素)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=2)

# 灵敏性箱线图
# 设置离群点的样式
flierprops = dict(marker='o', markerfacecolor='black', markersize=2, linestyle='none', alpha=0.7)
box_plot = ax2.boxplot(distances, patch_artist=True, notch=True, flierprops=flierprops)
# 为箱线图添加颜色
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax2.set_title('像点位移的灵敏性分析', fontsize=14, fontweight='bold')
ax2.set_xlabel('线段编号', fontsize=12)
ax2.set_ylabel('像点位移 (像素)', fontsize=12)
ax2.set_xticklabels(range(1, num_segments + 1))
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(figures_path,'sensitivity_analysis.png'), dpi=300, bbox_inches='tight')
plt.show()

# 输出灵敏性与角度关系图
plt.figure(figsize=(12, 7))
# 设置图表样式和参数，不使用seaborn
plt.rcParams['font.size'] = 14

# 使用颜色映射来区分不同线段
colors = plt.cm.tab10(np.linspace(0, 1, num_segments))
for i, color in enumerate(colors):
    plt.scatter(np.rad2deg(thetas[i]), mean_distances[i], s=120, 
                c=[color], label=f'线段 {i + 1}', marker='o', edgecolors='black', linewidth=0.5)

plt.title('θ角对灵敏性的影响', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('θ角度 (度)', fontsize=14)
plt.ylabel('平均像点位移 (像素)', fontsize=14)
plt.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=2)
plt.grid(True, alpha=0.3)

# 添加趋势线
z = np.polyfit(np.rad2deg(thetas), mean_distances, 1)
p = np.poly1d(z)
plt.plot(np.rad2deg(thetas), p(np.rad2deg(thetas)), "--", alpha=0.8, color='red', linewidth=2, 
         label=f'趋势线 (斜率: {z[0]:.4f})')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(figures_path,'theta_sensitivity.png'), dpi=300, bbox_inches='tight')
plt.show()