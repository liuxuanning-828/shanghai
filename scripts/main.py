import cv2
import time
from ultralytics import YOLO

# 1. 加载模型
model = YOLO('models/yolov8n.pt')  

# 2. 初始化摄像头
cap = cv2.VideoCapture(0)

# 初始化计算 FPS 的变量
prev_time = 0

print("【核心优化】显卡强制加速版启动。按下 ESC 退出...")

while True:
    start_time = time.time()
    
    ret, frame = cap.read()
    if not ret:
        print("【错误】无法采集画面。")
        break

    # 3. 【核心修改】强制让 YOLO 在 GPU (device=0) 上进行推理
    results = model(frame, device=0, verbose=False) # verbose=False 可以让终端不再疯狂刷无用日志，提升效率

    # 4. 绘制 AI 检测框
    annotated_frame = results[0].plot()

    # 5. 计算并显示实时 FPS
    current_time = time.time()
    fps = 1 / (current_time - start_time)
    
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 6. 显示画面
    cv2.imshow("YOLO CUDA Vision Pipeline", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()
print("视觉核心已安全关闭。")