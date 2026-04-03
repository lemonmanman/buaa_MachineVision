import cv2
import numpy as np
import os

def main():
    # 1. 路径配置 [cite: 301]
    input_path = './.images/original/okita_sougo.jpeg'
    output_dir = './.images/morphological_operated'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 已创建目录: {output_dir}")

    # 2. 读取图像并预处理 [cite: 301, 303]
    img = cv2.imread(input_path)
    if img is None:
        print(f"❌ 错误: 找不到图片 {input_path}")
        return

    # 转灰度并二值化，形态学操作核心是处理二值图像 [cite: 301, 303]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. 定义结构元素 (5x5 矩形窗口) [cite: 303, 304]
    kernel = np.ones((5, 5), np.uint8)

    # --- 执行四大核心操作 ---
    # (1) 腐蚀：收缩边界，消除小噪点 [cite: 305-307]
    img_erosion = cv2.erode(binary, kernel, iterations=1)

    # (2) 膨胀：扩张边界，填充孔洞 [cite: 308-310]
    img_dilation = cv2.dilate(binary, kernel, iterations=1)

    # (3) 开运算：先腐蚀后膨胀，用于去噪 [cite: 311-313]
    img_open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # (4) 闭运算：先膨胀后腐蚀，用于补孔 [cite: 314-316]
    img_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 4. 结果汇总
    results = {
        '0_Binary': binary,
        '1_Erosion': img_erosion,
        '2_Dilation': img_dilation,
        '3_Opening': img_open,
        '4_Closing': img_close
    }

    print("\n保存状态检查：")
    # 5. 先保存所有图片，确保文件夹里有东西
    for title, res in results.items():
        file_path = os.path.join(output_dir, f"{title.lower()}.jpg")
        success = cv2.imwrite(file_path, res)
        if success:
            print(f"✅ 已成功保存: {file_path}")
        else:
            print(f"❌ 保存失败: {file_path}")

    # 6. 一次性弹出所有窗口 [cite: 317]
    print("\n🔔 正在弹出所有窗口，请点击窗口并按【任意键】退出...")
    for title, res in results.items():
        cv2.imshow(title, res)

    # 关键：等待按键
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 针对 Linux 环境的强制刷新
    for i in range(5):
        cv2.waitKey(1)

if __name__ == "__main__":
    main()