import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
model_path="D:\\VScode25_3\\yoloDeepsort\\model\\yolov8n.pt"
model=YOLO(model_path)

VIDEO_PATH = ".\\video\\people.mp4"
RESULT_PATH = ".\\video\\result2.mp4"

# 记录所有的id的位置信息
track_history=defaultdict(lambda:[])

if __name__ == '__main__':
    captuer = cv2.VideoCapture(VIDEO_PATH)
    if not captuer.isOpened():
        print('ERROR OPENING')
        exit()

    fps = captuer.get(cv2.CAP_PROP_FPS)  # 获取帧率
    frame_width = int(captuer.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取宽度
    frame_height = int(captuer.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取高度

    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    videoWriter = cv2.VideoWriter(RESULT_PATH, fourcc, fps, (frame_width, frame_height))


  #   VideoWriter=None
    while True:
        success, frame = captuer.read()  # 读取视频中的一帧
        if not success:
            print("读取结束")
            break
        results = model.track(frame,persist=True)
        a_frame=results[0].plot()

        #所有id的位置信息
        boxes=results[0].boxes.xywh.cpu()
        #所有id的序列号信息
        track_ids=results[0].boxes.id.int().cpu().tolist()

        for box,track_id in zip(boxes,track_ids):
            # print(box)
            # print(track_id)
            x,y,w,h=box
            track=track_history[track_id]
            track.append((float(x),float(y)))
            if len(track)>50:
                track.pop(0)
            #当前track—id所有经过的轨迹路径，不超过50个
            points=np.hstack(track).astype(np.int32).reshape(-1,1,2)
            cv2.polylines(a_frame,[points],isClosed=False,color=(255,0,255),thickness=3)


        cv2.imshow("yolo track",a_frame)
        cv2.waitKey(1)

    captuer.release()
   #  VideoWriter.release()
    cv2.destroyAllWindows()  