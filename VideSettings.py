import cv2

class Processing:
    #Инициализируем метод удаления фона и захват видео
    def __init__(self, f_name):
        self.mog = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.stream = cv2.VideoCapture(f_name)
    def __del__(self):
        self.stream.release()

    def detect(self):
        ret, frame = self.stream.read()

        if not ret:
            raise Exception('File cannot be open')

        mask = self.mog.apply(frame) #Применяем маску к frame
        _, thresh = cv2.threshold(mask, 250,255,cv2.THRESH_BINARY)
        thresh = cv2.medianBlur(thresh, 15)
        thresh = cv2.dilate(thresh, None, iterations=4)
        cntr, hirarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return cntr, frame, thresh