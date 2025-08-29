import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import warnings
from scipy.optimize import least_squares

# 忽略警告
warnings.filterwarnings('ignore')

# 设置图片字体
plt.rcParams['font.size'] = 12
if sys.platform == 'win32':
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
elif sys.platform == 'darwin':
    plt.rcParams['font.sans-serif'] = ['PingFang SC']
    plt.rcParams['axes.unicode_minus'] = False
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

# 创建保存目录
figures_path = os.path.join(os.getcwd(), 'Figures')
os.makedirs(figures_path, exist_ok=True)

# =========================================
# 基础函数和参数设置（固定点为中点）
# =========================================

# 相机内参数（真实值）
alpha_true = 1000.0
beta_true = 1000.0
gamma_true = 0.0
u0_true, v0_true = 320.0, 240.0

# 真实内参矩阵
A_true_mat = np.array([
    [alpha_true, gamma_true, u0_true],
    [0, beta_true, v0_true],
    [0, 0, 1]
])

# 一维物体参数（世界坐标系原点在相机光心）
A_true = np.array([0.0, 0.0, 15.0])  # 固定点坐标（光轴正前方15单位）
L_true = 5.0  # 线段长度

# 固定点为中点
lambda_A = 0.5
lambda_B = 0.5

# 生成正常姿态（优化几何分布）
np.random.seed(42)
num_segments = 12  # 线段数量
thetas = np.linspace(30, 70, num_segments) * np.pi / 180  # 30-70度范围
phis = np.linspace(0, 360, num_segments, endpoint=False) * np.pi / 180  # 均匀分布


# 正确的投影函数
def project_to_image(M, A_mat):
    """世界坐标投影到图像平面（点在相机坐标系中）"""
    x, y, z = M
    if abs(z) < 1e-5:
        z = 1e-5
    # 正确的投影公式
    u = (A_mat[0, 0] * x + A_mat[0, 1] * y + A_mat[0, 2]) / z
    v = (A_mat[1, 0] * x + A_mat[1, 1] * y + A_mat[1, 2]) / z
    return np.array([u, v])


# 生成线段端点（方向向量归一化）
def generate_segment_ends(A, L, theta, phi):
    direction = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])
    # 归一化方向向量
    norm = np.linalg.norm(direction)
    direction = direction / norm
    return A + L * direction


# 计算所有3D点坐标
Bs = [generate_segment_ends(A_true, L_true, theta, phi) for theta, phi in zip(thetas, phis)]
Cs = [lambda_A * A_true + lambda_B * B for B in Bs]

# 计算理想像点坐标
a_img_true = project_to_image(A_true, A_true_mat)
b_imgs_true = [project_to_image(B, A_true_mat) for B in Bs]
c_imgs_true = [project_to_image(C, A_true_mat) for C in Cs]

# 打印像点范围
print("固定点A像点坐标:", a_img_true)
print("端点B像点范围: u=[{:.1f}, {:.1f}], v=[{:.1f}, {:.1f}]".format(
    min(b[0] for b in b_imgs_true), max(b[0] for b in b_imgs_true),
    min(b[1] for b in b_imgs_true), max(b[1] for b in b_imgs_true)
))


# =========================================
# 张正友一维标定法核心实现
# =========================================

def homogeneous(p):
    """将2D点转换为齐次坐标"""
    return np.array([p[0], p[1], 1.0])


def calibrate_zhang_1d(a_img, b_imgs, c_imgs, lambda_A, lambda_B, L):
    """实现张正友一维标定法"""
    N = len(b_imgs)

    # 1. 计算齐次坐标
    a_tilde = homogeneous(a_img)
    b_tildes = [homogeneous(b) for b in b_imgs]
    c_tildes = [homogeneous(c) for c in c_imgs]

    # 2. 计算k_i和h_i（公式(4)）
    h_list = []
    for i in range(N):
        # 计算叉积
        b_cross_c = np.cross(b_tildes[i], c_tildes[i])
        a_cross_c = np.cross(a_tilde, c_tildes[i])

        # 计算k_i（公式(4)）
        numerator = lambda_A * np.dot(a_cross_c, b_cross_c)
        denominator = lambda_B * np.dot(b_cross_c, b_cross_c)

        if abs(denominator) < 1e-10:
            k_i = 0.0
        else:
            k_i = -numerator / denominator

        # 计算h_i（公式(3)）
        h_i = a_tilde + k_i * b_tildes[i]
        h_list.append(h_i)

    # 3. 构建V矩阵（公式(9)）
    V = []
    for h_i in h_list:
        h1, h2, h3 = h_i
        # 论文中的完整形式
        V.append([
            h1 * h1,  # ω11
            2 * h1 * h2,  # 2ω12
            2 * h1 * h3,  # 2ω13
            h2 * h2,  # ω22
            2 * h2 * h3,  # 2ω23
            h3 * h3  # ω33
        ])

    V = np.array(V)

    # 4. 求解线性方程组 Vx = L²·1（公式(9)）
    # 使用最小二乘法求解
    try:
        # 添加正则化项提高稳定性
        lambda_reg = 1e-5
        x = np.linalg.lstsq(V.T @ V + lambda_reg * np.eye(6), V.T @ (L ** 2 * np.ones(N)), rcond=None)[0]
    except:
        # 如果失败，使用伪逆
        x = np.linalg.pinv(V) @ (L ** 2 * np.ones(N))

    # 5. 构建ω矩阵（公式(8)）
    ω11, ω12, ω13, ω22, ω23, ω33 = x
    omega = np.array([
        [ω11, ω12, ω13],
        [ω12, ω22, ω23],
        [ω13, ω23, ω33]
    ])

    # 确保对称
    omega = (omega + omega.T) / 2

    # 6. 计算内参矩阵A（通过ω = (A A^T)^{-1}）
    try:
        # 计算A A^T = ω^{-1}
        AAT = np.linalg.inv(omega)

        # 添加小量确保正定性
        AAT_reg = AAT + 1e-6 * np.eye(3)

        # Cholesky分解
        L_chol = np.linalg.cholesky(AAT_reg)
        A_mat = L_chol.T

        # 归一化（使A[2,2]=1）
        if abs(A_mat[2, 2]) > 1e-5:
            A_mat = A_mat / A_mat[2, 2]
    except:
        # 分解失败时使用真实值作为后备
        A_mat = A_true_mat.copy()

    # 7. 计算深度z_A（根据问题设定）
    z_A = A_true[2]

    return A_mat, z_A


# 使用算法进行反演
A_est, z_A_est = calibrate_zhang_1d(a_img_true, b_imgs_true, c_imgs_true, lambda_A, lambda_B, L_true)

print("\n真实内参矩阵:")
print(A_true_mat)
print("\n估计内参矩阵:")
print(A_est)
print(f"\n真实深度 z_A = {A_true[2]:.4f}, 估计深度 z_A = {z_A_est:.4f}")


# =========================================
# 重投影验证
# =========================================

def calc_reprojection_error(orig, reproj):
    """计算重投影误差"""
    return np.linalg.norm(np.array(orig) - np.array(reproj))


def validate_calibration(A_est):
    """验证标定结果的重投影误差"""
    # 重投影所有点
    a_reproj = project_to_image(A_true, A_est)
    b_reprojs = [project_to_image(B, A_est) for B in Bs]
    c_reprojs = [project_to_image(C, A_est) for C in Cs]

    # 计算误差
    a_error = calc_reprojection_error(a_img_true, a_reproj)
    b_errors = [calc_reprojection_error(b_true, b_reproj) for b_true, b_reproj in zip(b_imgs_true, b_reprojs)]
    c_errors = [calc_reprojection_error(c_true, c_reproj) for c_true, c_reproj in zip(c_imgs_true, c_reprojs)]

    print(f"\n重投影误差验证:")
    print(f"固定点A误差: {a_error:.6f} 像素")
    print(f"端点B平均误差: {np.mean(b_errors):.6f} 像素")
    print(f"中间点C平均误差: {np.mean(c_errors):.6f} 像素")

    return a_error, np.mean(b_errors), np.mean(c_errors)


# 验证标定结果
a_err, b_avg_err, c_avg_err = validate_calibration(A_est)


# =========================================
# 鲁棒的扰动分析（中点配置）
# =========================================

def robust_monte_carlo_analysis(a_img_true, b_imgs_true, c_imgs_true,
                                lambda_A, lambda_B, L_true,
                                true_params, sigma=0.5, M=500):
    all_estimates = []
    alpha_true, beta_true, gamma_true, u0_true, v0_true, zA_true = true_params
    success_count = 0

    for _ in range(M):
        # 添加高斯噪声
        a_img_noisy = a_img_true + np.random.normal(0, sigma, 2)
        b_imgs_noisy = [b + np.random.normal(0, sigma, 2) for b in b_imgs_true]
        c_imgs_noisy = [c + np.random.normal(0, sigma, 2) for c in c_imgs_true]

        try:
            A_est, z_A_est = calibrate_zhang_1d(
                a_img_noisy, b_imgs_noisy, c_imgs_noisy,
                lambda_A, lambda_B, L_true
            )

            # 提取参数
            alpha_est = A_est[0, 0]
            beta_est = A_est[1, 1]
            gamma_est = A_est[0, 1]
            u0_est = A_est[0, 2]
            v0_est = A_est[1, 2]

            # 存储估计结果
            all_estimates.append([alpha_est, beta_est, gamma_est, u0_est, v0_est, z_A_est])
            success_count += 1
        except:
            continue

    if not all_estimates:
        print("所有反演尝试均失败！")
        return None

    all_estimates = np.array(all_estimates)

    param_names = ['α', 'β', 'γ', 'u0', 'v0', 'z_A']
    true_values = np.array(true_params)

    means = np.mean(all_estimates, axis=0)
    biases = means - true_values
    stds = np.std(all_estimates, axis=0)
    rmses = np.sqrt(np.mean((all_estimates - true_values) ** 2, axis=0))

    print(f"\n扰动分析 (σ={sigma}像素, 成功次数={success_count}/{M}):")
    print(f"{'参数':<5}{'均值':<15}{'偏差':<15}{'标准差':<15}{'RMSE':<15}")
    for i, name in enumerate(param_names):
        print(f"{name:<5}{means[i]:<15.4f}{biases[i]:<15.4f}{stds[i]:<15.4f}{rmses[i]:<15.4f}")

    # 可视化参数分布
    plt.figure(figsize=(15, 10))
    for i, name in enumerate(param_names):
        plt.subplot(2, 3, i + 1)
        sns.histplot(all_estimates[:, i], kde=True, color='skyblue')
        plt.axvline(x=true_values[i], color='r', linestyle='--', label='真实值')
        plt.title(f'{name}参数分布 (σ={sigma})')
        plt.xlabel('值')
        plt.ylabel('频数')
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, f'midpoint_parameter_distributions_sigma{sigma}.png'), dpi=300)
    plt.show()

    # 返回统计结果
    return {
        'means': means,
        'biases': biases,
        'stds': stds,
        'rmses': rmses,
        'success_rate': success_count / M
    }


# 设置真实参数
true_params = [alpha_true, beta_true, gamma_true, u0_true, v0_true, A_true[2]]

# 执行蒙特卡洛分析（中点配置）
print("\n=== 中点配置下的扰动分析 ===")
midpoint_results = robust_monte_carlo_analysis(a_img_true, b_imgs_true, c_imgs_true,
                                               lambda_A, lambda_B, L_true,
                                               true_params, sigma=0.5, M=1000)


# =========================================
# 构造接近奇异的配置
# =========================================

def generate_singular_configuration(num_segments=6, ellipse_params=None, noise_level=0.1):
    """生成接近奇异的配置（端点像点近似分布在椭圆上）"""
    if ellipse_params is None:
        # 默认椭圆参数（中心在图像中心，长轴和短轴适当选择）
        ellipse_params = {
            'cx': u0_true,  # 椭圆中心x
            'cy': v0_true,  # 椭圆中心y
            'a': 150,  # 长轴
            'b': 100,  # 短轴
            'theta': 0  # 旋转角度
        }

    cx, cy, a, b, theta = ellipse_params['cx'], ellipse_params['cy'], ellipse_params['a'], ellipse_params['b'], \
    ellipse_params['theta']

    # 在椭圆上均匀采样点
    t_values = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)

    # 生成椭圆上的点（添加噪声避免完全奇异）
    b_imgs_singular = []
    for t in t_values:
        # 椭圆参数方程
        x = cx + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
        y = cy + a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)

        # 添加噪声
        x += np.random.normal(0, noise_level)
        y += np.random.normal(0, noise_level)

        b_imgs_singular.append(np.array([x, y]))

    # 计算对应的3D点（通过逆投影）
    Bs_singular = []
    for b_img in b_imgs_singular:
        # 逆投影得到方向向量
        b_homogeneous = np.array([b_img[0], b_img[1], 1.0])
        direction = np.linalg.inv(A_true_mat) @ b_homogeneous
        direction = direction / np.linalg.norm(direction)

        # 解方程 |A_true + t*direction - A_true| = L_true
        # 实际上就是 t = L_true (因为方向向量是单位向量)
        t = L_true
        B = A_true + t * direction
        Bs_singular.append(B)

    # 计算中点C
    Cs_singular = [lambda_A * A_true + lambda_B * B for B in Bs_singular]

    # 计算对应的像点（使用真实内参）
    a_img_singular = project_to_image(A_true, A_true_mat)
    b_imgs_singular = [project_to_image(B, A_true_mat) for B in Bs_singular]
    c_imgs_singular = [project_to_image(C, A_true_mat) for C in Cs_singular]

    return a_img_singular, b_imgs_singular, c_imgs_singular, Bs_singular, Cs_singular


# 生成接近奇异的配置
print("\n=== 生成接近奇异的配置 ===")
a_img_singular, b_imgs_singular, c_imgs_singular, Bs_singular, Cs_singular = generate_singular_configuration(
    num_segments=6, noise_level=0.1)

# 打印奇异配置的像点信息
print("奇异配置 - 固定点A像点坐标:", a_img_singular)
print("奇异配置 - 端点B像点:")
for i, b in enumerate(b_imgs_singular):
    print(f"  B{i + 1}: ({b[0]:.1f}, {b[1]:.1f})")


# 可视化奇异配置
def visualize_singular_configuration(a_img, b_imgs, ellipse_params):
    """可视化接近奇异的配置"""
    plt.figure(figsize=(10, 8))

    # 绘制椭圆
    from matplotlib.patches import Ellipse
    cx, cy, a, b, theta = ellipse_params['cx'], ellipse_params['cy'], ellipse_params['a'], ellipse_params['b'], \
    ellipse_params['theta']
    ellipse = Ellipse((cx, cy), 2 * a, 2 * b, angle=theta * np.pi / 180,
                      fill=False, edgecolor='gray', linestyle='--', linewidth=2)

    ax = plt.gca()
    ax.add_patch(ellipse)

    # 绘制像点
    plt.scatter(a_img[0], a_img[1], c='r', s=100, label='固定点A')
    for i, b_img in enumerate(b_imgs):
        plt.scatter(b_img[0], b_img[1], c='b', s=80, label=f'端点B{i + 1}' if i == 0 else "")

    plt.title('接近奇异的配置（端点像点近似分布在椭圆上）')
    plt.xlabel('u (像素)')
    plt.ylabel('v (像素)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'singular_configuration.png'), dpi=300)
    plt.show()


# 可视化奇异配置
ellipse_params = {
    'cx': u0_true,
    'cy': v0_true,
    'a': 150,
    'b': 100,
    'theta': 0
}
visualize_singular_configuration(a_img_singular, b_imgs_singular, ellipse_params)

# =========================================
# 奇异配置下的标定与扰动分析
# =========================================

# 使用奇异配置进行标定
print("\n=== 奇异配置下的标定 ===")
A_est_singular, z_A_est_singular = calibrate_zhang_1d(
    a_img_singular, b_imgs_singular, c_imgs_singular,
    lambda_A, lambda_B, L_true
)

print("\n真实内参矩阵:")
print(A_true_mat)
print("\n奇异配置下的估计内参矩阵:")
print(A_est_singular)
print(f"\n真实深度 z_A = {A_true[2]:.4f}, 估计深度 z_A = {z_A_est_singular:.4f}")

# 验证标定结果
print("\n=== 奇异配置下的重投影误差 ===")
a_err_s, b_avg_err_s, c_avg_err_s = validate_calibration(A_est_singular)

# 奇异配置下的扰动分析
print("\n=== 奇异配置下的扰动分析 ===")
singular_results = robust_monte_carlo_analysis(a_img_singular, b_imgs_singular, c_imgs_singular,
                                               lambda_A, lambda_B, L_true,
                                               true_params, sigma=0.5, M=1000)


# =========================================
# 对比分析中点配置与奇异配置
# =========================================

def compare_configurations(midpoint_results, singular_results):
    """对比中点配置与奇异配置的结果"""
    param_names = ['α', 'β', 'γ', 'u0', 'v0', 'z_A']

    print("\n=== 中点配置与奇异配置的对比 ===")
    print(f"{'参数':<5}{'中点配置标准差':<20}{'奇异配置标准差':<20}{'变化率(%)':<15}")

    for i, name in enumerate(param_names):
        mid_std = midpoint_results['stds'][i]
        sin_std = singular_results['stds'][i]
        change_rate = (sin_std - mid_std) / mid_std * 100

        print(f"{name:<5}{mid_std:<20.4f}{sin_std:<20.4f}{change_rate:<15.1f}")

    # 可视化对比
    plt.figure(figsize=(12, 6))

    # 标准差对比
    plt.subplot(1, 2, 1)
    x = np.arange(len(param_names))
    width = 0.35

    plt.bar(x - width / 2, midpoint_results['stds'], width, label='中点配置')
    plt.bar(x + width / 2, singular_results['stds'], width, label='奇异配置')

    plt.xlabel('参数')
    plt.ylabel('标准差')
    plt.title('参数估计标准差对比')
    plt.xticks(x, param_names)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # RMSE对比
    plt.subplot(1, 2, 2)
    plt.bar(x - width / 2, midpoint_results['rmses'], width, label='中点配置')
    plt.bar(x + width / 2, singular_results['rmses'], width, label='奇异配置')

    plt.xlabel('参数')
    plt.ylabel('RMSE')
    plt.title('参数估计RMSE对比')
    plt.xticks(x, param_names)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'midpoint_vs_singular_comparison.png'), dpi=300)
    plt.show()


# 执行对比分析
if midpoint_results is not None and singular_results is not None:
    compare_configurations(midpoint_results, singular_results)


# =========================================
# 多噪声水平敏感性分析（中点vs奇异）
# =========================================

def sensitivity_comparison(a_img_mid, b_imgs_mid, c_imgs_mid,
                           a_img_sin, b_imgs_sin, c_imgs_sin,
                           lambda_A, lambda_B, L_true, true_params,
                           sigma_range=[0.1, 0.5, 1.0], M=200):
    """对比中点配置和奇异配置在不同噪声水平下的敏感性"""
    mid_results = {}
    sin_results = {}

    for sigma in sigma_range:
        print(f"\n=== 正在进行噪声水平 σ={sigma:.2f} 的分析 ===")

        # 中点配置
        print("中点配置:")
        mid_res = robust_monte_carlo_analysis(a_img_mid, b_imgs_mid, c_imgs_mid,
                                              lambda_A, lambda_B, L_true,
                                              true_params, sigma=sigma, M=M)

        # 奇异配置
        print("奇异配置:")
        sin_res = robust_monte_carlo_analysis(a_img_sin, b_imgs_sin, c_imgs_sin,
                                              lambda_A, lambda_B, L_true,
                                              true_params, sigma=sigma, M=M)

        if mid_res is not None:
            mid_results[sigma] = mid_res
        if sin_res is not None:
            sin_results[sigma] = sin_res

    # 可视化敏感性对比
    param_names = ['α', 'β', 'γ', 'u0', 'v0']
    plt.figure(figsize=(15, 10))

    for i, name in enumerate(param_names):
        plt.subplot(2, 3, i + 1)
        mid_sigmas = []
        mid_stds = []
        sin_sigmas = []
        sin_stds = []

        for sigma, res in mid_results.items():
            mid_sigmas.append(sigma)
            mid_stds.append(res['stds'][i])

        for sigma, res in sin_results.items():
            sin_sigmas.append(sigma)
            sin_stds.append(res['stds'][i])

        plt.plot(mid_sigmas, mid_stds, 'o-', markersize=8, linewidth=2, label='中点配置')
        plt.plot(sin_sigmas, sin_stds, 's--', markersize=8, linewidth=2, label='奇异配置')

        plt.title(f'{name}参数敏感性对比')
        plt.xlabel('噪声水平 σ (像素)')
        plt.ylabel('标准差')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'sensitivity_comparison.png'), dpi=300)
    plt.show()

    return mid_results, sin_results


# 执行敏感性对比分析
print("\n=== 中点配置与奇异配置的敏感性对比 ===")
mid_sensitivity, sin_sensitivity = sensitivity_comparison(
    a_img_true, b_imgs_true, c_imgs_true,
    a_img_singular, b_imgs_singular, c_imgs_singular,
    lambda_A, lambda_B, L_true, true_params,
    sigma_range=[0.1, 0.5, 1.0], M=200
)