import cv2
import os
import numpy as np

def main():
    # 1. 定义路径
    input_path = './.images/original/okita_sougo.jpeg'
    output_dir = './.images/color_converted'

    # 2. 自动创建输出目录 (如果不存在的话)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 已创建目录: {output_dir}")

    # 3. 读取图像
    image = cv2.imread(input_path)
    if image is None:
        print(f"❌ Error: 无法在 {input_path} 找到图像。")
        return

    # --- 4. 颜色空间转换 ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # --- 5. 保存图像到指定目录 ---
    # 使用 os.path.join 拼接路径，比手动写斜杠更安全
    cv2.imwrite(os.path.join(output_dir, 'okita_gray.jpg'), gray)
    cv2.imwrite(os.path.join(output_dir, 'okita_hsv.jpg'), hsv)
    cv2.imwrite(os.path.join(output_dir, 'okita_lab.jpg'), lab)
    cv2.imwrite(os.path.join(output_dir, 'okita_ycrcb.jpg'), ycrcb)

    print(f"✅ 所有转换后的图片已成功保存至: {output_dir}")

    # 6. 显示结果
    cv2.imshow('Original BGR', image)
    cv2.imshow('Gray', gray)
    cv2.imshow('HSV', hsv)
    cv2.imshow('Lab', lab)

    print("提示：在图片窗口按下任意键退出程序...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()