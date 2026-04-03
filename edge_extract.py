import cv2
import numpy as np
import os

def main():
    # 1. 路径配置
    input_path = './.images/original/okita_sougo.jpeg'
    output_dir = './.images/edge_extracted'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 路径检查调试
    if not os.path.exists(input_path):
        print(f"❌ 错误: 找不到文件 {os.path.abspath(input_path)}")
        return

    # 2. 读取图像
    img = cv2.imread(input_path)
    if img is None:
        print("❌ 错误: 图像读取失败。")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("✅ 图像读取成功，开始提取特征...")

    # --- (1) Canny 边缘提取 [cite: 268-274] ---
    # 使用 5x5 高斯滤波去噪后进行边缘检测 [cite: 271]
    # 阈值设为 50 和 150 (推荐比例 1:3)
    img_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny_edges = cv2.Canny(img_blur, 50, 150)

    # --- (2) SIFT 特征提取 [cite: 275-283] ---
    # SIFT 具有尺度不变性和旋转不变性 [cite: 276]
    sift = cv2.SIFT_create()
    kp_sift, des_sift = sift.detectAndCompute(gray, None)
    # 绘制关键点
    img_sift = cv2.drawKeypoints(img, kp_sift, None,
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # --- (3) ORB 特征提取 [cite: 284-292] ---
    # ORB 结合了 FAST 和 BRIEF，且速度极快
    orb = cv2.ORB_create(nfeatures=1000) # 提取前1000个特征点
    kp_orb, des_orb = orb.detectAndCompute(gray, None)
    # 绘制关键点
    img_orb = cv2.drawKeypoints(img, kp_orb, None, color=(0, 255, 0), flags=0)

    # 3. 结果汇总
    results = {
        '1_Canny_Edges': canny_edges,
        '2_SIFT_Features': img_sift,
        '3_ORB_Features': img_orb
    }

    print(f"🔔 提取完成。SIFT点数: {len(kp_sift)}, ORB点数: {len(kp_orb)}")
    print("👉 正在弹出窗口，请点击图片并按【任意键】切换/退出...")

    # 保存并显示
    for title, res in results.items():
        save_path = os.path.join(output_dir, f"{title.lower()}.jpg")
        cv2.imwrite(save_path, res)
        cv2.imshow(title, res)

    # 4. 保持窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Linux 环境强制刷新刷新
    for i in range(5):
        cv2.waitKey(1)

if __name__ == "__main__":
    main()