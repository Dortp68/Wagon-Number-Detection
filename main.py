from ultralytics import YOLO
import cv2
import easyocr
import time
import argparse
from traking import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, help='i/v: single image or video mode')
parser.add_argument('--path', type=str, help='path to file')
args = parser.parse_args()

reader = easyocr.Reader(['en', 'ru'])
num_model = YOLO('best.pt')
obj_model = YOLO('yolov8m.pt')
mot_tracker = Sort()

def detect_train(frame):
    results = obj_model(frame)[0]

    for detection in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) == 6:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, 'train', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255, 255), 3)



def detect(frame):
    results = num_model(frame)[0]
    detections_=[]
    for detection in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection

        crop = frame[int(y1):int(y2), int(x1):int(x2), :]
        crop_grayscale = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(crop_grayscale, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        ocr = reader.readtext(thresh, allowlist='0123456789')
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 1)

        ocr_detection = []
        for numbers in ocr:
            bbox, text, score = numbers
            text = f'{text}'
            cv2.putText(frame, text, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            ocr_detection.append([text, score])
        detections_.append([x1, y1, x2, y2, score, ocr_detection])

    return detections_


if __name__ == '__main__':
    if args.mode == 'i':
        frame = cv2.imread(args.path)
        detect_train(frame)
        detect(frame)
        cv2.imshow("frame", frame)
        cv2.imwrite('result.jpg', frame)
        cv2.waitKey(0)
    elif args.mode == 'v':
        results = []
        cap = cv2.VideoCapture(args.path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            detections_ = detect(frame)
            if detections_ == []:
                mot_tracker.update()
            else:
                track_ids = mot_tracker.update(np.asarray(detections_))
                if len(track_ids) == 2:
                    track_ids = preprocess(track_ids)
                    if track_ids!= None:
                        results.append(track_ids)
            detect_train(frame)
            # frame = cv2.resize(frame, (540, 960))
            cv2.imshow("frame", frame)
            # time.sleep(0.1)
            if cv2.waitKey(1) == ord('q'):
                break
        save_csv(results, 'result.csv')
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("this mode does not exist, please insert 'i' for image or 'v' for video")







