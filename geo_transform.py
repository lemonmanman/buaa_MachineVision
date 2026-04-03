import cv2
import numpy as np
import os

def main():
    # 1. 路径配置
    input_path = './.images/original/okita_sougo.jpeg'
    output_dir = './.images/geo_transformed'

    # 自动创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 已创建目录: {output_dir}")

    # 2. 读取原始图像
    img = cv2.imread(input_path)
    if img is None:
        print(f"❌ 错误: 无法加载图片 {input_path}")
        return

    rows, cols = img.shape[:2]
    print(f"🖼️ 原始图像尺寸: {cols}x{rows}")

    # --- (1) 平移变换 [cite: 189-193] ---
    # tm=100 (垂直方向), tn=50 (水平方向)
    M_translate = np.float32([[1, 0, 50], [0, 1, 100]])
    img_translate = cv2.warpAffine(img, M_translate, (cols, rows))

    # --- (2) 缩放变换 [cite: 194-198] ---
    # sm=0.5, sn=0.5 (缩小一半)
    img_scale = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    # --- (3) 旋转变换 [cite: 199-202] ---
    # 围绕中心点逆时针旋转 45 度
    center = (cols / 2, rows / 2)
    M_rotate = cv2.getRotationMatrix2D(center, 45, 1.0)
    img_rotate = cv2.warpAffine(img, M_rotate, (cols, rows))

    # --- (4) 翻转变换 [cite: 203-204] ---
    img_flip_h = cv2.flip(img, 1) # 水平翻转

    # --- (5) 错切变换 (Shear) [cite: 205-216] ---
    # ① 垂直错切 Sn = 0.2
    M_v_shear = np.float32([[1, 0, 0], [0.2, 1, 0]])
    img_v_shear = cv2.warpAffine(img, M_v_shear, (cols + 100, rows + 100))

    # ② 水平错切 Sm = 0.2
    M_h_shear = np.float32([[1, 0.2, 0], [0, 1, 0]])
    img_h_shear = cv2.warpAffine(img, M_h_shear, (cols + 100, rows + 100))

    # 3. 保存变换后的图像
    results = {
        'translated.jpg': img_translate,
        'scaled.jpg': img_scale,
        'rotated.jpg': img_rotate,
        'flipped.jpg': img_flip_h,
        'vertical_shear.jpg': img_v_shear,
        'horizontal_shear.jpg': img_h_shear
    }

    # 4. 显示结果
    cv2.imshow

    for name, result_img in results.items():
        save_path = os.path.join(output_dir, name)
        cv2.imwrite(save_path, result_img)
        print(f"✅ 已保存: {save_path}")

    print("\n✨ 所有几何变换处理完成！")

if __name__ == "__main__":
    main()