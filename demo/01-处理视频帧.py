import cv2
import numpy as np

VIDEO_PATH = ".\\video\\people.mp4"
RESULT_PATH = ".\\video\\result.mp4"
polygonPoints = np.array([[610, 50], [910, 50], [510, 250], [210, 250]], dtype=np.int32)

if __name__ == '__main__':
    captuer = cv2.VideoCapture(VIDEO_PATH)
    if not captuer.isOpened():
        print('ERROR OPENING')
        exit()

    fps = captuer.get(cv2.CAP_PROP_FPS)  # 获取帧率
    frame_width = int(captuer.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取宽度
    frame_height = int(captuer.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取高度

    # 初始化 VideoWriter
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    videoWriter = cv2.VideoWriter(RESULT_PATH, fourcc, fps, (frame_width, frame_height))

    while True:
        success, frame = captuer.read()  # 读取视频中的一帧
        if not success:
            print("读取结束")
            break

        # 绘制线条和多边形
        cv2.line(frame, (0, int(frame_height / 2)), (int(frame_width), int(frame_height / 2)), (0, 0, 255), 3)
        cv2.polylines(frame, [polygonPoints], True, (0, 0, 255), 3)

        # 创建掩码并叠加
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, [polygonPoints], (0, 0, 255))
        frame = cv2.addWeighted(frame, 0.7, mask, 0.3, 0)

        # 写入帧
        videoWriter.write(frame)

        # 显示帧
        cv2.imshow("people", frame)
        cv2.waitKey(1)

    # 释放资源
    captuer.release()
    videoWriter.release()
    cv2.destroyAllWindows()