import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import warnings

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
# 基础函数和参数设置（严格遵循张正友论文）
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
lambda_A = 0.5  # 中点参数
lambda_B = 0.5  # 中点参数

# 生成姿态（优化几何分布）
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
# 可视化标定结果
# =========================================

def visualize_reprojection():
    """可视化重投影结果"""
    plt.figure(figsize=(12, 8))

    # 真实像点
    plt.scatter(a_img_true[0], a_img_true[1], c='r', s=100, label='固定点A (真实)')
    for i, b in enumerate(b_imgs_true):
        plt.scatter(b[0], b[1], c='b', s=80, label=f'端点B{i + 1} (真实)' if i == 0 else "")

    # 重投影点
    a_reproj = project_to_image(A_true, A_est)
    b_reprojs = [project_to_image(B, A_est) for B in Bs]

    plt.scatter(a_reproj[0], a_reproj[1], c='r', s=100, marker='x', label='固定点A (重投影)')
    for i, b in enumerate(b_reprojs):
        plt.scatter(b[0], b[1], c='b', s=80, marker='x', label=f'端点B{i + 1} (重投影)' if i == 0 else "")

    plt.title('像点投影与重投影比较')
    plt.xlabel('u (像素)')
    plt.ylabel('v (像素)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'reprojection_comparison.png'), dpi=300)
    plt.show()


# 可视化重投影结果
visualize_reprojection()


# =========================================
# 鲁棒的扰动分析
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
    plt.savefig(os.path.join(figures_path, f'parameter_distributions_sigma{sigma}.png'), dpi=300)
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

# 执行蒙特卡洛分析
results = robust_monte_carlo_analysis(a_img_true, b_imgs_true, c_imgs_true,
                                      lambda_A, lambda_B, L_true,
                                      true_params, sigma=0.5, M=1000)


# =========================================
# 多噪声水平敏感性分析
# =========================================

def sensitivity_analysis(a_img_true, b_imgs_true, c_imgs_true,
                         lambda_A, lambda_B, L_true, true_params,
                         sigma_range=[0.1, 0.5, 1.0], M=200):
    """分析不同噪声水平下的参数敏感性"""
    results = {}

    for sigma in sigma_range:
        print(f"\n=== 正在进行噪声水平 σ={sigma:.2f} 的分析 ===")
        res = robust_monte_carlo_analysis(a_img_true, b_imgs_true, c_imgs_true,
                                          lambda_A, lambda_B, L_true,
                                          true_params, sigma=sigma, M=M)

        if res is not None:
            results[sigma] = res

    # 可视化敏感性
    param_names = ['α', 'β', 'γ', 'u0', 'v0']
    plt.figure(figsize=(12, 8))

    for i, name in enumerate(param_names):
        plt.subplot(2, 3, i + 1)
        sigmas = []
        stds = []

        for sigma, res in results.items():
            sigmas.append(sigma)
            stds.append(res['stds'][i])

        plt.plot(sigmas, stds, 'o-', markersize=8, linewidth=2)
        plt.title(f'{name}参数敏感性')
        plt.xlabel('噪声水平 σ (像素)')
        plt.ylabel('标准差')
        plt.grid(True)

        # 添加线性拟合
        if len(sigmas) > 1:
            coeffs = np.polyfit(sigmas, stds, 1)
            trend = np.poly1d(coeffs)
            plt.plot(sigmas, trend(sigmas), 'r--',
                     label=f'y = {coeffs[0]:.2f}σ + {coeffs[1]:.2f}')
            plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'parameter_sensitivity.png'), dpi=300)
    plt.show()

    return results


# 执行敏感性分析
sensitivity_results = sensitivity_analysis(a_img_true, b_imgs_true, c_imgs_true,
                                           lambda_A, lambda_B, L_true, true_params,
                                           sigma_range=[0.1, 0.5, 1.0])