import cv2

from cobot2.yolo2 import YoloModel


CONFIDENCE_THRESHOLD = 0.50
IOU_THRESHOLD = 0.50
EDGE_MARGIN = 2
NORMAL_BOX_COLOR = (0, 255, 0)
CLIPPED_BOX_COLOR = (0, 0, 255)

model = YoloModel()


def is_box_on_image_edge(box, width, height, edge_margin=EDGE_MARGIN):
    x1, y1, x2, y2 = box
    return (
        x1 <= edge_margin
        or y1 <= edge_margin
        or x2 >= width - 1 - edge_margin
        or y2 >= height - 1 - edge_margin
    )


def get_split_detections(frame):
    height, width = frame.shape[:2]
    detections = model.get_detections(
        frame,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
    )

    edge_detections = []
    inner_detections = []
    for detection in detections:
        if is_box_on_image_edge(detection["box"], width, height):
            edge_detections.append(detection)
        else:
            inner_detections.append(detection)

    return edge_detections, inner_detections


def draw_detection(image, detection, color):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = map(int, detection["box"])
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))

    label = f"{detection['class']} {detection['score'] * 100:.1f}%"
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


# 카메라 열기 (0: 기본 내장 카메라)
cap = cv2.VideoCapture(8)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

print("실시간 예측 시작 (종료: Q 키 누르기)")

# 윈도우 크기 설정
cv2.namedWindow("YOLO Predict", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Predict", 960, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    edge_detections, inner_detections = get_split_detections(frame)
    for detection in inner_detections:
        draw_detection(frame, detection, NORMAL_BOX_COLOR)
    for detection in edge_detections:
        draw_detection(frame, detection, CLIPPED_BOX_COLOR)

    # 화면에 출력
    cv2.imshow("YOLO Predict", frame)

    # Q 키 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료 처리
cap.release()
cv2.destroyAllWindows()
print("예측 종료")
