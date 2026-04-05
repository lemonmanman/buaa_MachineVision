import cv2
import numpy as np
import os

# 全局变量用于存储人工选取的坐标
pts_left = []
pts_right = []
drawing_mode = "LEFT"  # 当前正在哪个图选点

def select_points(event, x, y, flags, param):
    global pts_left, pts_right, drawing_mode
    img_display, scale_factor = param

    if event == cv2.EVENT_LBUTTONDOWN:
        # 映射回原始分辨率的坐标
        orig_x = int(x / scale_factor)
        orig_y = int(y / scale_factor)

        half_width = img_display.shape[1] // 2

        if drawing_mode == "LEFT" and x < half_width:
            if len(pts_left) < 8:
                pts_left.append((orig_x, orig_y))
                # 在显示图上画红色十字和编号
                cv2.drawMarker(img_display, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(img_display, str(len(pts_left)), (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if len(pts_left) == 8:
                    drawing_mode = "RIGHT"
                    print("\n✅ 左图 8 个点选取完毕！请在右图(右侧画面)选取对应的 8 个点。")

        elif drawing_mode == "RIGHT" and x >= half_width:
            if len(pts_right) < 8:
                # 注意右图在拼接图像中的偏移量
                orig_x_right = int((x - half_width) / scale_factor)
                pts_right.append((orig_x_right, orig_y))
                # 在显示图上画绿色十字和编号
                cv2.drawMarker(img_display, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(img_display, str(len(pts_right)), (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                if len(pts_right) == 8:
                    print("\n✅ 右图 8 个点选取完毕！请在窗口按【任意键】继续计算...")

        cv2.imshow("Interactive Point Selection", img_display)

def draw_epilines(img1, img2, lines, pts1, pts2):
    """在图像上绘制极线和对应点"""
    r, c = img1.shape[:2]
    img1_out = img1.copy()
    img2_out = img2.copy()

    # 使用鲜艳的颜色列表，方便在报告中区分
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255), (255,165,0), (255,192,203)]

    for i, (r, pt1, pt2) in enumerate(zip(lines, pts1, pts2)):
        color = colors[i % len(colors)]
        # 计算极线与图像边界的交点
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        # 画极线
        img1_out = cv2.line(img1_out, (x0, y0), (x1, y1), color, 2)
        # 画点
        img1_out = cv2.circle(img1_out, tuple(map(int, pt1)), 6, color, -1)
        img2_out = cv2.circle(img2_out, tuple(map(int, pt2)), 6, color, -1)

        # 加上编号
        cv2.putText(img1_out, str(i+1), (int(pt1[0])+10, int(pt1[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(img2_out, str(i+1), (int(pt2[0])+10, int(pt2[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return img1_out, img2_out

def main():
    input_left = './.images/original/left.jpg'
    input_right = './.images/original/right.jpg'
    output_dir = './.images/stereo_matched'
    os.makedirs(output_dir, exist_ok=True)

    imgL = cv2.imread(input_left)
    imgR = cv2.imread(input_right)
    if imgL is None or imgR is None:
        print("❌ 无法读取图像，请检查路径。")
        return

    # --- 1. 生成图 4-1：原始并排拼接图 ---
    img_concat_orig = np.hstack((imgL, imgR))
    cv2.imwrite(os.path.join(output_dir, 'fig4_1_original_stereo.jpg'), img_concat_orig)

    # --- 2. 交互式选点 (处理高像素手机图，缩放以适应屏幕) ---
    screen_width = 1400  # 限制拼接窗口的最大宽度
    scale = screen_width / img_concat_orig.shape[1]
    if scale < 1.0:
        img_display = cv2.resize(img_concat_orig, None, fx=scale, fy=scale)
    else:
        img_display = img_concat_orig.copy()
        scale = 1.0

    print("=========================================")
    print("👉 请在弹出的窗口中，先在【左半边图】点击 8 个特征点。")
    print("👉 然后在【右半边图】点击对应的 8 个点。")
    print("=========================================")

    cv2.namedWindow("Interactive Point Selection")
    cv2.setMouseCallback("Interactive Point Selection", select_points, [img_display, scale])
    cv2.imshow("Interactive Point Selection", img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(pts_left) != 8 or len(pts_right) != 8:
        print("❌ 未选够 8 对点，程序退出。")
        return

    # --- 3. 生成图 4-2：标注了点的图像 ---
    # 在原图分辨率上重新画点，保证保存的高清图质量
    imgL_marked = imgL.copy()
    imgR_marked = imgR.copy()
    for i in range(8):
        cv2.drawMarker(imgL_marked, pts_left[i], (0, 0, 255), cv2.MARKER_CROSS, 40, 4)
        cv2.putText(imgL_marked, str(i+1), (pts_left[i][0]+15, pts_left[i][1]-15), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)

        cv2.drawMarker(imgR_marked, pts_right[i], (0, 255, 0), cv2.MARKER_CROSS, 40, 4)
        cv2.putText(imgR_marked, str(i+1), (pts_right[i][0]+15, pts_right[i][1]-15), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)

    cv2.imwrite(os.path.join(output_dir, 'fig4_2_points_marked.jpg'), np.hstack((imgL_marked, imgR_marked)))

    # 打印 表 4-1 内容
    print("\n\n======== 【请复制以下数据填入 表 4-1】 ========")
    print("点编号\t左视图坐标(u1,v1)\t右视图坐标(u2,v2)")
    for i in range(8):
        print(f"{i+1}\t({pts_left[i][0]}, {pts_left[i][1]})\t\t({pts_right[i][0]}, {pts_right[i][1]})")
    print("==============================================\n")

    # --- 4. 八点法计算基本矩阵 F ---
    pts1 = np.float32(pts_left)
    pts2 = np.float32(pts_right)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)

    print("======== 【请复制以下矩阵填入 报告 4.3.3 节】 ========")
    print("基本矩阵 F = ")
    print(F)
    print("====================================================\n")

    # --- 5. 计算极线约束残差并生成 表 4-2 ---
    residuals = []
    print("======== 【请复制以下数据填入 表 4-2】 ========")
    print("点编号\t极线约束残差")
    for i in range(8):
        # 构造齐次坐标 [u, v, 1]^T
        p1 = np.array([pts_left[i][0], pts_left[i][1], 1.0]).reshape(3, 1)
        p2 = np.array([pts_right[i][0], pts_right[i][1], 1.0]).reshape(1, 3)
        # 计算残差 |p2^T * F * p1|
        res = np.abs(np.dot(p2, np.dot(F, p1)))[0][0]
        residuals.append(res)
        print(f"{i+1}\t{res:.6f}")

    print(f"平均残差\t{np.mean(residuals):.6f}")
    print(f"最大残差\t{np.max(residuals):.6f}")
    print("==============================================\n")

    # --- 6. 生成图 4-3：极线绘制 ---
    # 计算右图点在左图对应的极线
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, F)
    lines1 = lines1.reshape(-1,3)
    imgL_epi, _ = draw_epilines(imgL, imgL, lines1, pts1, pts1) # 仅用它的画线功能

    # 计算左图点在右图对应的极线
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1, F)
    lines2 = lines2.reshape(-1,3)
    imgR_epi, _ = draw_epilines(imgR, imgR, lines2, pts2, pts2)

    img_concat_epi = np.hstack((imgL_epi, imgR_epi))
    cv2.imwrite(os.path.join(output_dir, 'fig4_3_epilines.jpg'), img_concat_epi)

    print(f"✅ 所有图像已成功生成并保存在 {output_dir} 文件夹中！")

    # 缩小显示结果图
    cv2.imshow("Final Epipolar Lines", cv2.resize(img_concat_epi, None, fx=scale, fy=scale))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()