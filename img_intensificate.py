import cv2
import numpy as np
import os

def homomorphic_filter(img):
    """实现文档中提到的同态滤波：处理光照不均 """
    # 取对数
    img_log = np.log1p(np.array(img, dtype="float") / 255)
    # 傅里叶变换
    dft = np.fft.fft2(img_log)
    dft_shift = np.fft.fftshift(dft)

    # 构建高斯同态滤波器
    rows, cols = img.shape
    rh, rl, c = 2.0, 0.5, 0.1  # 参数：高频增益, 低频增益, 锐度
    mu, mv = rows // 2, cols // 2
    y, x = np.ogrid[-mu:rows-mu, -mv:cols-mv]
    d2 = x**2 + y**2
    h = (rh - rl) * (1 - np.exp(-c * d2 / (2 * (100**2)))) + rl

    # 滤波并逆变换
    fshift = dft_shift * h
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.exp(np.real(img_back)) - 1

    # 归一化到 0-255
    res = np.uint8(cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX))
    return res

def main():
    # 1. 路径配置
    input_path = './.images/original/okita_sougo.jpeg'
    output_dir = './.images/img_intensificated'

    if not os.path.exists(output_dir):
        print(f"❌ 绝对错误: 在路径 {os.path.abspath(input_path)} 找不到文件！")
        # 列出当前目录下的文件协助排查
        if os.path.exists('./.images'):
            print(f".images 目录内容: {os.listdir('./.images')}")
        return
    
    # 2. 读取图像
    img = cv2.imread(input_path)
    if img is None:
        print(f"❌ 错误: 无法加载图片 {input_path}")
        return

    # 转换为灰度图以进行增强实验 [cite: 234]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- (1) 点运算：伽马变换 (gamma=0.5 提亮暗部) [cite: 235] ---
    gamma = 0.5
    lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img_gamma = cv2.LUT(gray, lookup_table)

    # --- (2) 点运算：直方图均衡化 [cite: 237] ---
    img_equ = cv2.equalizeHist(gray)

    # --- (3) 邻域运算：高斯平滑去噪 [cite: 243] ---
    img_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- (4) 邻域运算：中值滤波 (处理椒盐噪声) [cite: 242] ---
    img_median = cv2.medianBlur(gray, 5)

    # --- (5) 邻域运算：Laplacian 锐化  ---
    # 先降噪再锐化，防止噪声放大
    laplacian = cv2.Laplacian(img_blur, cv2.CV_64F)
    img_sharp = cv2.convertScaleAbs(gray - 0.5 * laplacian)

    # --- (6) 频率域增强：同态滤波  ---
    img_homo = homomorphic_filter(gray)

    # 3. 结果汇总与展示
    results = {
        '0_Original_Gray': gray,
        '1_Gamma_Correction': img_gamma,
        '2_Histogram_Equalization': img_equ,
        '3_Gaussian_Blur': img_blur,
        '4_Median_Filter': img_median,
        '5_Laplacian_Sharpen': img_sharp,
        '6_Homomorphic_Filter': img_homo
    }

    print("🔔 正在弹出增强后的所有结果，请点击图片窗口并按【任意键】退出...")

    for title, res in results.items():
        # 保存图片
        file_path = os.path.join(output_dir, f"{title.lower()}.jpg")
        cv2.imwrite(file_path, res)
        # 显示窗口
        cv2.imshow(title, res)

    # 4. 保持窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 针对部分 Linux 环境的强制刷新
    for i in range(5):
        cv2.waitKey(1)

if __name__ == "__main__":
    main()