"""
@author: SKS

Prototype UI for SAMs.

"""

import os
import sys
import csv
import cv2
from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import *
from PyQt4.QtCore import *
import sqlite3


# Global Variables

base_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(base_path,'data')
dataset_path = os.path.join(data_path,'datasets')
image_path = os.path.join(data_path,'images')
model_path = os.path.join(data_path,'models')
tmp_path = os.path.join(base_path,'tmp')
camera_port = 0
large_text_size = 28
medium_text_size = 18
small_text_size = 10
demoVar = "Physics Class"
window_width = 600
window_height = 250
conn = sqlite3.connect('sams.db')
cursor = conn.cursor()
new_user_added = False



class WindowMain(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("SAMs - Smart Attendance Management system")
        self.resize(window_width, window_height)
        self.centerWindow()
        self.vbox = QVBoxLayout()
        self.vbox.setAlignment(Qt.AlignCenter)

        self.button1 = QPushButton("Register", self)
        #self.button1.move(250, 100)
        self.button1.clicked.connect(self.onclick_button1)

        self.button2 = QPushButton("Take Attendance", self)
        #self.button2.move(250, 150)
        self.button2.clicked.connect(self.onclick_button2)

        self.vbox.addWidget(self.button1)
        self.vbox.addWidget(self.button2)
        self.setLayout(self.vbox)

    def onclick_button1(self):
        # Register the student
        # win.hide()
        winRegister.show()

    def onclick_button2(self):
        # Take Attendance
        # win.hide()
        winTakeAttnSelectClass.show()

    def centerWindow(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


class WindowRegister(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Register")
        self.resize(window_width, window_height)
        self.centerWindow()
        self.fbox = QFormLayout()

        self.textLabel1 = QLabel("Subject Name: ", self)
        self.textLabel2 = QLabel("Subject Code: ", self)
        self.textLabel3 = QLabel("No. of Students: ", self)
        
        self.lineEdit1 = QLineEdit(self)
        self.lineEdit2 = QLineEdit(self)
        self.lineEdit3 = QLineEdit(self)

        self.button1 = QPushButton("Start", self)
        self.button1.clicked.connect(self.onclick_button1)

        self.fbox.addRow(self.textLabel1,self.lineEdit1)
        self.fbox.addRow(self.textLabel2,self.lineEdit2)
        self.fbox.addRow(self.textLabel3,self.lineEdit3)
        self.fbox.addRow(self.button1)
        self.fbox.setAlignment(Qt.AlignCenter)
        self.fbox.setContentsMargins(100, 50, 100, 50)
        self.fbox.setSpacing(10)
        self.setLayout(self.fbox)

    def onclick_button1(self):
        # Start the Registration process
        winRegister.hide()
        self.lineEdit1.clear()
        self.lineEdit2.clear()
        winRegisterStudentDetails.show()

    def centerWindow(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


class WindowRegisterStudentDetails(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Enter Student Details")
        self.resize(window_width, window_height)
        self.centerWindow()
        self.fbox = QFormLayout()

        self.textLabel1 = QLabel("Enrollment No: ", self)
        self.textLabel2 = QLabel("Student Name: ", self)
        self.lineEdit1 = QLineEdit(self)
        self.lineEdit2 = QLineEdit(self)
        self.button1 = QPushButton("Next", self)
        self.button1.clicked.connect(self.onclick_button1)

        self.fbox.addRow(self.textLabel1,self.lineEdit1)
        self.fbox.addRow(self.textLabel2,self.lineEdit2)
        self.fbox.addRow(self.button1)
        self.fbox.setAlignment(Qt.AlignCenter)
        self.fbox.setContentsMargins(100, 50, 100, 50)
        self.fbox.setSpacing(10)
        self.setLayout(self.fbox)

    def onclick_button1(self):
        # Forward to next page
        winRegisterStudentDetails.hide()
        self.lineEdit1.clear()
        self.lineEdit2.clear()

        winRegisterStudentPhotos.show()

    def centerWindow(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

class WindowRegisterStudentPhotos(QWidget):
    def __init__(self):
        super().__init__()
        self.capturing=False
        self.resize(700,460)
        self.video_size = QSize(400, 300)
        self.snapshot_size = QSize(80, 80)
        self.store_dir= os.path.join(image_path,'class1')
        self.cascPath = 'haarcascade_frontalface_default.xml'
        self.faceCascade = cv2.CascadeClassifier(self.cascPath)
        self.snapshotCnt=0
        self.maxSnapshotCnt=8
        self.captureCompleted = False
        self.uploadCompleted = False
        self.trained = False
        self.initUI()

    def initUI(self):
        self.topleft = QFrame()        
        self.imageLabel=QLabel()
        self.imageLabel.setScaledContents(True)
        self.topleft.setObjectName('gframe')
        self.topleft.setContentsMargins(50,10,50,10)
        self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.vbox1 = QVBoxLayout()
        self.vbox1.addWidget(self.imageLabel)
        self.topleft.setLayout(self.vbox1)

        self.topright = QFrame()
        self.snpGrid = QGridLayout()
        
        self.snpGrid.setSpacing(2)
        self.snpGrid.setContentsMargins(2,2,2,2)
        
        self.topright.setLayout(self.snpGrid)
        self.hbox = QHBoxLayout()
        self.startButton = QPushButton('Start')
        self.stopButton = QPushButton('Stop')
        self.takeSnapshotButton = QPushButton('Take Snapshot')
        self.messageLbl = QLabel('')
        font1 = QFont('Helvetica', small_text_size)
        self.messageLbl.setFont(font1)

        self.startButton.clicked.connect(self.startCapture)
        self.stopButton.clicked.connect(self.stopCapture)
        self.takeSnapshotButton.clicked.connect(self.takeSnapshot)
        self.hbox.addWidget(self.startButton)
        self.hbox.addWidget(self.stopButton)
        self.hbox.addWidget(self.takeSnapshotButton)
        
        self.mhbox = QHBoxLayout()
        self.mhbox.setAlignment(Qt.AlignCenter)
        self.mhbox.addWidget(self.messageLbl)

        self.bvbox = QVBoxLayout()
        self.bvbox.addLayout(self.mhbox)
        self.bvbox.addLayout(self.hbox)
        self.bvbox.setSpacing(10)
        
        self.bottom = QFrame()
        self.bottom.setLayout(self.bvbox)
        self.bottom.setObjectName("gframe")

        self.splitter1 = QSplitter(Qt.Horizontal)
        self.splitter1.addWidget(self.topleft)
        self.splitter1.addWidget(self.topright)
        self.splitter1.setSizes([5,2])

        self.splitter2 = QSplitter(Qt.Vertical)
        self.splitter2.addWidget(self.splitter1)
        self.splitter2.addWidget(self.bottom)
        self.splitter2.setSizes([375,75])
        self.hbox1=QHBoxLayout()
        self.hbox1.addWidget(self.splitter2)
        self.setLayout(self.hbox1)
        self.initGrid()

    def initDir(self):
        self.store_dir= os.path.join(image_path,'class1')
        if os.path.isdir(self.store_dir)==False:
            try:
                original_umask = os.umask(0)
                os.makedirs(self.store_dir)
            finally:
                os.umask(original_umask)

    def initGrid(self):
        range_x=int((self.maxSnapshotCnt+1)/2)
        self.snpLabels =[]
        for i in range(self.maxSnapshotCnt):
            self.snpLabels.append(QLabel())
            self.snpLabels[i].setScaledContents(True)
            self.snpLabels[i].setFixedSize(self.snapshot_size)
            self.snpLabels[i].setObjectName("gframe")

        range_y =2
        pos = [(i,j) for i in range(range_x) for j in range(range_y)]
        
        for p, lbl in zip(pos, self.snpLabels):
            self.snpGrid.addWidget(lbl,*p)


    def display_video_stream(self):
        r , frame = self.capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        frame = cv2.cvtColor(frame, cv2.cv.CV_BGR2RGB)
        frame = cv2.flip(frame, 1)
        image = QImage(frame, frame.shape[1], frame.shape[0], 
                       frame.strides[0], QImage.Format_RGB888)
        
        self.imageLabel.setPixmap(QPixmap.fromImage(image))



    def startCapture(self):
        self.initDir()
        self.capturing = True
        self.capture = cv2.VideoCapture(camera_port)
        # self.capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, self.video_size.width())
        # self.capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, self.video_size.height())

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.display_video_stream)
        self.timer.start(30)

    def stopCapture(self):
        #print "pressed End"
        if self.capturing == True:
            self.capturing = False
            self.capture.release()
            self.timer.stop()
            cv2.destroyAllWindows()

    def takeSnapshot(self):

        if self.capturing == False:
            self.messageLbl.setText('Warning: Start the camera')
            return

        if self.snapshotCnt == self.maxSnapshotCnt:
            self.messageLbl.setText('Warning: All snapshots taken, no need to take more now!')
            return                 
        
        if (self.capturing == True)  and (self.snapshotCnt < self.maxSnapshotCnt):
            try:
                r , frame = self.capture.read()
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(40, 40),
                    flags=cv2.cv.CV_HAAR_SCALE_IMAGE
                )
                if len(faces)==0:
                    return
                max_area = 0
                mx = 0
                my = 0 
                mh = 0 
                mw = 0
                for (x, y, w, h) in faces:
                    if w*h > max_area:
                        mx = x
                        my = y
                        mh = h
                        mw = w
                        max_area=w*h    
                
                image_crop = frame[my:my+mh,mx:mx+mw]
                self.snapshotCnt=self.snapshotCnt+1
                self.messageLbl.setText('Process: Total snapshots captured: %d (Remaining: %d)' % (self.snapshotCnt,self.maxSnapshotCnt-self.snapshotCnt))
                file_name = 'img_%d.jpg'% (self.snapshotCnt)
                file = os.path.join(self.store_dir,file_name)
                cv2.imwrite(file, image_crop)
                self.snpLabels[self.snapshotCnt-1].setPixmap(QPixmap(file))

            except Exception as e:
                self.messageLbl.setText('Error: Snapshot capturing failed')
                print("Snapshot capturing failed...\n Errors:")
                print(e)

        if(self.snapshotCnt == self.maxSnapshotCnt):
            self.captureCompleted=True
            self.stopCapture()


# class WindowRegisterStudentPhotos(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.initUI()

#     def initUI(self):
#         self.setWindowTitle("Take Photographs")
#         self.resize(600, 250)
#         self.centerWindow()

#         textLabel1 = QLabel("", self)
#         textLabel1.resize(220, 220)
#         textLabel1.move(100, 20)
#         textLabel1.setPixmap(QPixmap(os.getcwd() + "/1.jpg"))

#         button1 = QPushButton("Start", self)
#         button1.move(450, 150)
#         button1.clicked.connect(self.onclick_button1)

#         button2 = QPushButton("Stop", self)
#         button2.move(450, 200)
#         button2.clicked.connect(self.onclick_button2)

#     def onclick_button1(self):
#         # Start taking photos for Database
#         pass

#     def onclick_button2(self):
#         # Stop taking photos
#         self.showCompletionMsg()

#     def centerWindow(self):
#         qr = self.frameGeometry()
#         cp = QDesktopWidget().availableGeometry().center()
#         qr.moveCenter(cp)
#         self.move(qr.topLeft())

#     def showCompletionMsg(self):
#         reply = QMessageBox.question(self, "Done!", "Add data for another Student?",
#                                            QMessageBox.Yes | QMessageBox.No)
#         if reply == QMessageBox.No:
#             # Go to main page
#             winRegisterStudentPhotos.hide()
#         else:
#             winRegisterStudentPhotos.hide()
#             winRegisterStudentDetails.show()


class WindowTakeAttnSelectClass(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Take Attendance")
        self.resize(600, 250)
        self.centerWindow()

        self.listWidget = QListWidget(self)
        self.listWidget.addItem("Physics Class")
        self.listWidget.addItem("Biology Class")
        self.listWidget.move(150, 100)
        self.listWidget.resize(200, 100)

        button1 = QPushButton("Start", self)
        button1.move(450, 200)
        button1.clicked.connect(self.onclick_button1)

    def onclick_button1(self):
        # Check which Class is selected!
        winTakeAttnSelectClass.hide()
        winTakeAttnGroupPhotos.show()

        demoVar = self.listWidget.currentItem().text()
        winTakeAttnGroupPhotos.setWindowTitle("Take Attendance Photos for " + demoVar)
        print(self.listWidget.currentItem().text())

    def centerWindow(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


class WindowTakeAttnGroupPhotos(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Take Attendance Photos for " + demoVar)
        self.resize(600, 250)
        self.centerWindow()

        textLabel1 = QLabel("", self)
        textLabel1.resize(300, 220)
        textLabel1.move(50, 20)
        textLabel1.setPixmap(QPixmap(os.getcwd() + "/2.jpg"))

        button1 = QPushButton("Take Photo", self)
        button1.move(450, 150)
        button1.clicked.connect(self.onclick_button1)

        button2 = QPushButton("Done", self)
        button2.move(450, 200)
        button2.clicked.connect(self.onclick_button2)

    def onclick_button1(self):
        # Just click a photo
        pass

    def onclick_button2(self):
        # Done taking photos
        winTakeAttnGroupPhotos.hide()
        winResults.show()

    def centerWindow(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


class WindowResults(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Attendance Results")
        self.resize(600, 250)
        self.centerWindow()

    def centerWindow(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

class MainWindow:
    def __init__(self):
        self.win =  WindowMain()   
        self.winRegister = WindowRegister()
        self.winTakeAttnSelectClass = WindowTakeAttnSelectClass()
        self.winRegisterStudentDetails = WindowRegisterStudentDetails()
        self.winRegisterStudentPhotos = WindowRegisterStudentPhotos()
        self.winTakeAttnGroupPhotos = WindowTakeAttnGroupPhotos()
        self.winResults = WindowResults()
        self.win.show()        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win =  WindowMain()   
    winRegister = WindowRegister()
    winTakeAttnSelectClass = WindowTakeAttnSelectClass()
    winRegisterStudentDetails = WindowRegisterStudentDetails()
    winRegisterStudentPhotos = WindowRegisterStudentPhotos()
    winTakeAttnGroupPhotos = WindowTakeAttnGroupPhotos()
    winResults = WindowResults()
    win.show()        
    sys.exit(app.exec_())
