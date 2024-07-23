import cv2
import numpy as np
from torchvision.ops import box_iou
from scipy.optimize import linear_sum_assignment as linear_assignment
import HungarianAlgorithm as HA
class Box:
    Green_Clr = (100,255,0) # Палитра зелёного цвета в BGR
    Red_Clr = (0,0,255) # Палитра красного цвета в BGR
    lim_detArea = 2000 # Граничное значение при котором происходит отрисовка объекта
    id = 0 # Номер id объекта обновляется при появлении нового объекта в кадре

    #Инициализируем координаты рамки и id которое увеличивается каждый раз
    #когда появляется новый экземпляр класса BOX
    def __init__(self,x,y,w,h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.id = Box.id
        Box.id += 1
    # Возвращение размеров прямоугольника
    def shape(self):
        return self.x, self.y, self.w, self.h
    def rectangle(self):
        return [self.x, self.y, self.w + self.x, self.h + self.y]
    def print_info(self):
        print(f"ID {self.id}: {self.x}, {self.y}, {self.w}, {self.h}")


    #Условия при которых происходит отрисовка рамки во круг объекта
    @classmethod
    def det_area_create(cls, contour):
        detection = []
        if len(contour) != 0:
            for cnt in contour:
                area = cv2.contourArea(cnt) #Вычесление площади вокруг объекта
            #При условии если area > lim_detArea происходит отрисовка рамки во круг ообъекта
                if area > cls.lim_detArea:
                    x, y, w, h = cv2.boundingRect(cnt)
                    detection.append(cls(x,y,w,h))
        return detection

    @classmethod
    def trackingCreation(cls, detection, tracking):
        if len(tracking) == 0:
            return detection
        if len(detection) == 0:
            return []
        # Формирование матрицы весов IoU - intersection over union
        IoU = np.zeros((len(detection), len(tracking)), dtype=np.float32)



        #Выполнение перебора всех распознаных объектов (detection) и всех отслеживаемых объектов (tracking)
        # и для каждой пары объектов вычисляется площадь пересечения их областей (Intersection over Union).
        for i, det in enumerate(detection):
            for j, tra in enumerate(tracking):
                IoU[i][j] = HA.IntersectionOverUnion(det,tra)
        print("IoU\n", IoU)
        # Применение венгерского метода к матрице весов IoU.(hungarian из HungarianAlgorithm)
        matches, unmatched_detections, unmatched_trackers = HA.hungarian(IoU,tracking,detection)
        # matches - матрица однозначных соответствий
        # [индекс распознаного, индекс индекс отслеживаемого]
        # [idx_detect, idx_tracker] из IoU
        print('matches\n',matches)
        # unmatched_detections - индексы распознаного объекта,
        # которым нет соответсвия в отслеживаемых - НОВЫЙ  бъект во фрейме
        print('unmatched_trackers', len(tracking), '\n', unmatched_trackers)
        # unmatched_trackers - индексы отслеживаемых, которых нет в распознаных
        # - ОБЪЕКТ исчез в данном фрейме
        print('unmatched_trackers', len(detection), '\n', unmatched_detections)

        for mtc in matches:
            #Мы приводим идентификатор детектированного
            # объекта в соответствие с сопоставленным отслеживаемым объектом
            detection[mtc[0]].id = tracking[mtc[1]].id

            tracking[mtc[1]] = detection[mtc[0]]

        # Нераспознанный объект становится новым отслеживаемым
        for i in unmatched_detections:
            tracking.append(detection[i])

        delta = 0
        u_t = np.sort(unmatched_trackers)
        for i in u_t:
            del tracking[i - delta]
            delta += 1

        return tracking

    @classmethod
    #Задаем параметры рамки и точки-центра рамки
    def drawing_box(cls, frame, boxes):
        if len(boxes) > 0:
            for box in boxes:
                x, y, w, h = box.shape()
                cx = x + w // 2
                cy = y + h // 2
                cv2.putText(frame, str(box.id),(cx, cy -7), 0,0.5, cls.Red_Clr, 2)
                cv2.rectangle(frame,(x,y),(x+w, y+h), cls.Green_Clr,2)
                cv2.circle(frame, (cx,cy), 2, cls.Red_Clr, -1)
        return frame