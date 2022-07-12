from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton



class MainWindow(QWidget):
    def __init__(self, size=(640, 360), window_opacity=0.6):
        super(MainWindow, self).__init__()
        self.setWindowFlags(
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.X11BypassWindowManagerHint
        )
        self.setGeometry(
            QtWidgets.QStyle.alignedRect(
                QtCore.Qt.LeftToRight, QtCore.Qt.AlignRight,
                QtCore.QSize(size[0], size[1]),
                QtWidgets.qApp.desktop().availableGeometry()
        ))
        self.setWindowOpacity(window_opacity)

        self.vbl = QVBoxLayout()
        self.vbl.setContentsMargins(0,0,0,0)
        self.vbl.setSpacing(0)

        # QLabel widget for image container
        self.feed_label = QLabel()
        self.vbl.addWidget(self.feed_label)

        # Button widget for termination
        self.StopBTN = QPushButton("Stop")
        self.StopBTN.clicked.connect(self.stopApp)
        self.StopBTN.setContentsMargins(0,0,0,0)
        self.vbl.addWidget(self.StopBTN)

    def overlayUpdateCallback(self, image : QImage):
        self.feed_label.setPixmap(QPixmap.fromImage(image))

    def stopApp(self):
        self.main_thread.stop()



if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import pyqtSignal, QThread
    import cv2
    class MainThread(QThread):
        OverlayUpdate = pyqtSignal(QImage)
        def run(self):
            self.ThreadActive = True
            self.main_loop()

        def stop(self):
            self.ThreadActive = False
            self.quit()
            QtWidgets.qApp.quit()

        def display_image(self, color_image):
            pic = QImage(color_image.data, color_image.shape[1], color_image.shape[0], QImage.Format_BGR888).scaled(640, 360)
            self.OverlayUpdate.emit(pic)    # calls the function linked through OverlayUpdate.connect()

        # Basic Loop that captures image from camera
        def main_loop(self):
            global cap
            ctr = 0
            while self.ThreadActive:
                ret, frame = cap.read()
                if ret:
                    color_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    color_image = cv2.flip(color_image, 1)
                    pic = QImage(color_image.data, color_image.shape[1], color_image.shape[0], QImage.Format_RGB888).scaled(640, 360)
                    self.OverlayUpdate.emit(pic)    # calls the function linked through OverlayUpdate.connect()
                    ctr += 1
                if ctr > 100:
                    self.stop()

    CAMERA_ID = 1 # 0 is usually for built-in/USB webcam, varies for RealSense Camera
    cap = cv2.VideoCapture(CAMERA_ID)

    app = QApplication([])
    main_window = MainWindow()
    main_window.setLayout(main_window.vbl)
    main_window.show()

    # Spawn QThread
    main_window.main_thread = MainThread()
    main_window.main_thread.start()
    main_window.main_thread.OverlayUpdate.connect(main_window.overlayUpdateCallback)

    app.exec_()