import os
import queue
import threading
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_FILENAME = "yoloe-26n-seg.pt"
DEFAULT_CLASSES = ["box", "can", "plastic cup"]
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.50
WINDOW_NAME = "YOLOE 26 Segmentation"


def find_model_path():
    candidates = [
        PROJECT_ROOT / MODEL_FILENAME,
        Path.cwd() / MODEL_FILENAME,
    ]

    try:
        from ament_index_python.packages import get_package_share_directory

        share_dir = Path(get_package_share_directory("cobot2"))
        candidates.extend(
            [
                share_dir / MODEL_FILENAME,
                share_dir / "resource" / MODEL_FILENAME,
            ]
        )
    except Exception:
        pass

    for candidate in candidates:
        if candidate.exists():
            return candidate

    paths = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다:\n{paths}")


def parse_classes(text):
    classes = [item.strip() for item in text.split(",")]
    return [item for item in classes if item]


def input_worker(class_queue, stop_event):
    print("\n찾을 객체 이름을 쉼표로 입력하세요.")
    print("예: box, can, plastic cup")
    print("종료: q 입력 또는 카메라 창에서 q 키\n")

    while not stop_event.is_set():
        try:
            text = input("classes> ").strip()
        except EOFError:
            stop_event.set()
            break

        if text.lower() in {"q", "quit", "exit"}:
            stop_event.set()
            break

        classes = parse_classes(text)
        if not classes:
            print("빈 입력은 무시합니다. 예: bottle, cup")
            continue

        class_queue.put(classes)


def apply_latest_classes(model, class_queue, current_classes):
    latest_classes = None
    while True:
        try:
            latest_classes = class_queue.get_nowait()
        except queue.Empty:
            break

    if latest_classes is None or latest_classes == current_classes:
        return current_classes

    print(f"클래스 변경 중: {', '.join(latest_classes)}")
    start_time = time.time()
    model.set_classes(latest_classes)
    elapsed = time.time() - start_time
    print(f"클래스 변경 완료 ({elapsed:.2f}s)")
    return latest_classes


def draw_status(image, classes, fps):
    text = f"classes: {', '.join(classes)} | FPS: {fps:.1f}"
    cv2.rectangle(image, (0, 0), (image.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        image,
        text,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main():
    model_path = find_model_path()
    os.chdir(model_path.parent)
    model = YOLO(str(model_path))
    model.set_classes(DEFAULT_CLASSES)

    class_queue = queue.Queue()
    stop_event = threading.Event()
    thread = threading.Thread(
        target=input_worker,
        args=(class_queue, stop_event),
        daemon=True,
    )
    thread.start()

    cap = cv2.VideoCapture(8)
    if not cap.isOpened():
        stop_event.set()
        print("카메라를 열 수 없습니다.")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 720)

    current_classes = list(DEFAULT_CLASSES)
    last_time = time.time()
    fps = 0.0

    print("YOLOE 세그멘테이션 시작")
    try:
        while not stop_event.is_set():
            current_classes = apply_latest_classes(
                model,
                class_queue,
                current_classes,
            )

            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                agnostic_nms=True,
                verbose=False,
            )
            annotated = results[0].plot()

            now = time.time()
            elapsed = now - last_time
            last_time = now
            if elapsed > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / elapsed) if fps else 1.0 / elapsed

            draw_status(annotated, current_classes, fps)
            cv2.imshow(WINDOW_NAME, annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()
                break
    finally:
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()
        print("예측 종료")


if __name__ == "__main__":
    main()
