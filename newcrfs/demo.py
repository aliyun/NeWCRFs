from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import sys
import time
import argparse
import numpy as np

import cv2
from scipy import ndimage
from skimage.transform import resize
import matplotlib.pyplot as plt

plasma = plt.get_cmap('plasma')
greys = plt.get_cmap('Greys')

# UI and OpenGL
from PySide2 import QtCore, QtGui, QtWidgets, QtOpenGL
from OpenGL import GL, GLU
from OpenGL.arrays import vbo
from OpenGL.GL import shaders
import glm

from utils import post_process_depth, flip_lr
from networks.NewCRFDepth import NewCRFDepth


# Argument Parser
parser = argparse.ArgumentParser(description='NeWCRFs Live 3D')
parser.add_argument('--model_name',      type=str,   help='model name', default='newcrfs')
parser.add_argument('--encoder',         type=str,   help='type of encoder, base07, large07', default='large07')
parser.add_argument('--max_depth',       type=float, help='maximum depth in estimation', default=10)
parser.add_argument('--checkpoint_path', type=str,   help='path to a checkpoint to load', required=True)
parser.add_argument('--input_height',    type=int,   help='input height', default=480)
parser.add_argument('--input_width',     type=int,   help='input width',  default=640)
parser.add_argument('--dataset',         type=str,   help='dataset this model trained on',  default='nyu')
parser.add_argument('--crop',            type=str,   help='crop: kbcrop, edge, non',  default='non')
parser.add_argument('--video',           type=str,   help='video path',  default='')

args = parser.parse_args()

# Image shapes
height_rgb, width_rgb = args.input_height, args.input_width
height_depth, width_depth = height_rgb, width_rgb


# =============== Intrinsics rectify ==================
# Open this if you have the real intrinsics
Use_intrs_remap = False
# Intrinsic parameters for your own webcam camera
camera_matrix = np.zeros(shape=(3, 3))
camera_matrix[0, 0] = 5.4765313594010649e+02
camera_matrix[0, 2] = 3.2516069906172453e+02
camera_matrix[1, 1] = 5.4801781476172562e+02
camera_matrix[1, 2] = 2.4794113960783835e+02
camera_matrix[2, 2] = 1
dist_coeffs = np.array([ 3.7230261423972011e-02, -1.6171708069773008e-01, -3.5260752900266357e-04, 1.7161234226767313e-04, 1.0192711400840315e-01 ])
# Parameters for a model trained on NYU Depth V2
new_camera_matrix = np.zeros(shape=(3, 3))
new_camera_matrix[0, 0] = 518.8579
new_camera_matrix[0, 2] = 320
new_camera_matrix[1, 1] = 518.8579
new_camera_matrix[1, 2] = 240
new_camera_matrix[2, 2] = 1

R = np.identity(3, dtype=np.float)
map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, R, new_camera_matrix, (width_rgb, height_rgb), cv2.CV_32FC1)


def load_model():
    args.mode = 'test'
    model = NewCRFDepth(version='large07', inv_depth=False, max_depth=args.max_depth)
    model = torch.nn.DataParallel(model)

    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()
    
    return model

# Function timing
ticTime = time.time()


def tic():
    global ticTime;
    ticTime = time.time()


def toc():
    print('{0} seconds.'.format(time.time() - ticTime))


# Conversion from Numpy to QImage and back
def np_to_qimage(a):
    im = a.copy()
    return QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888).copy()


def qimage_to_np(img):
    img = img.convertToFormat(QtGui.QImage.Format.Format_ARGB32)
    return np.array(img.constBits()).reshape(img.height(), img.width(), 4)


# Compute edge magnitudes
def edges(d):
    dx = ndimage.sobel(d, 0)  # horizontal derivative
    dy = ndimage.sobel(d, 1)  # vertical derivative
    return np.abs(dx) + np.abs(dy)


# Main window
class Window(QtWidgets.QWidget):
    updateInput = QtCore.Signal()
    
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.model = None
        self.capture = None
        self.glWidget = GLWidget()
        
        mainLayout = QtWidgets.QVBoxLayout()
        
        # Input / output views
        viewsLayout = QtWidgets.QGridLayout()
        self.inputViewer = QtWidgets.QLabel("[Click to start]")
        self.inputViewer.setPixmap(QtGui.QPixmap(width_rgb, height_rgb))
        self.outputViewer = QtWidgets.QLabel("[Click to start]")
        self.outputViewer.setPixmap(QtGui.QPixmap(width_rgb, height_rgb))
        
        imgsFrame = QtWidgets.QFrame()
        inputsLayout = QtWidgets.QVBoxLayout()
        imgsFrame.setLayout(inputsLayout)
        inputsLayout.addWidget(self.inputViewer)
        inputsLayout.addWidget(self.outputViewer)
        
        viewsLayout.addWidget(imgsFrame, 0, 0)
        viewsLayout.addWidget(self.glWidget, 0, 1)
        viewsLayout.setColumnStretch(1, 10)
        mainLayout.addLayout(viewsLayout)
        
        # Load depth estimation model
        toolsLayout = QtWidgets.QHBoxLayout()

        self.button2 = QtWidgets.QPushButton("Webcam")
        self.button2.clicked.connect(self.loadCamera)
        toolsLayout.addWidget(self.button2)

        self.button3 = QtWidgets.QPushButton("Video")
        self.button3.clicked.connect(self.loadVideoFile)
        toolsLayout.addWidget(self.button3)
        
        self.button4 = QtWidgets.QPushButton("Pause")
        self.button4.clicked.connect(self.loadImage)
        toolsLayout.addWidget(self.button4)
        
        self.button6 = QtWidgets.QPushButton("Refresh")
        self.button6.clicked.connect(self.updateCloud)
        toolsLayout.addWidget(self.button6)
        
        mainLayout.addLayout(toolsLayout)
        
        self.setLayout(mainLayout)
        self.setWindowTitle(self.tr("NeWCRFs Live"))
        
        # Signals
        self.updateInput.connect(self.update_input)
        
        # Default example
        if self.glWidget.rgb.any() and self.glWidget.depth.any():
            img = (self.glWidget.rgb * 255).astype('uint8')
            self.inputViewer.setPixmap(QtGui.QPixmap.fromImage(np_to_qimage(img)))
            coloredDepth = (plasma(self.glWidget.depth[:, :, 0])[:, :, :3] * 255).astype('uint8')
            self.outputViewer.setPixmap(QtGui.QPixmap.fromImage(np_to_qimage(coloredDepth)))
    
    def loadModel(self):
        print('== loadModel')
        QtGui.QGuiApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        tic()
        self.model = load_model()
        print('Model loaded.')
        toc()
        self.updateCloud()
        QtGui.QGuiApplication.restoreOverrideCursor()
    
    def loadCamera(self):
        print('== loadCamera')
        tic()
        self.model = load_model()
        print('Model loaded.')
        toc()
        self.capture = cv2.VideoCapture(0)
        self.updateInput.emit()
    
    def loadVideoFile(self):
        print('== loadVideoFile')
        self.model = load_model()
        self.capture = cv2.VideoCapture(args.video)
        self.updateInput.emit()
    
    def loadImage(self):
        print('== loadImage')
        self.capture = None
        img = (self.glWidget.rgb * 255).astype('uint8')
        self.inputViewer.setPixmap(QtGui.QPixmap.fromImage(np_to_qimage(img)))
        self.updateCloud()
    
    def loadImageFile(self):
        print('== loadImageFile')
        self.capture = None
        filename = \
        QtWidgets.QFileDialog.getOpenFileName(None, 'Select image', '', self.tr('Image files (*.jpg *.png)'))[0]
        img = QtGui.QImage(filename).scaledToHeight(height_rgb)
        xstart = 0
        if img.width() > width_rgb: xstart = (img.width() - width_rgb) // 2
        img = img.copy(xstart, 0, xstart + width_rgb, height_rgb)
        self.inputViewer.setPixmap(QtGui.QPixmap.fromImage(img))
        print('== loadImageFile')
        self.updateCloud()
    
    def update_input(self):
        print('== update_input')
        # Don't update anymore if no capture device is set
        if self.capture == None:
            return
        
        # Capture a frame
        ret, frame = self.capture.read()
        
        # Loop video playback if current stream is video file
        if not ret:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.capture.read()
        
        # Prepare image and show in UI
        if Use_intrs_remap:
            frame_ud = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
        else:
            frame_ud = cv2.resize(frame, (width_rgb, height_rgb), interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame_ud, cv2.COLOR_BGR2RGB)
        image = np_to_qimage(frame)
        self.inputViewer.setPixmap(QtGui.QPixmap.fromImage(image))
        
        # Update the point cloud
        self.updateCloud()
    
    def updateCloud(self):
        print('== updateCloud')
        rgb8 = qimage_to_np(self.inputViewer.pixmap().toImage())
        self.glWidget.rgb = (rgb8[:, :, :3] / 255)[:, :, ::-1]
        
        if self.model:
            input_image = rgb8[:, :, :3].astype(np.float32)

            # Normalize image
            input_image[:, :, 0] = (input_image[:, :, 0] - 123.68) * 0.017
            input_image[:, :, 1] = (input_image[:, :, 1] - 116.78) * 0.017
            input_image[:, :, 2] = (input_image[:, :, 2] - 103.94) * 0.017

            H, W, _ = input_image.shape
            if args.crop == 'kbcrop':
                top_margin = int(H - 352)
                left_margin = int((W - 1216) / 2)
                input_image_cropped = input_image[top_margin:top_margin + 352, 
                                                  left_margin:left_margin + 1216]
            elif args.crop == 'edge':
                input_image_cropped = input_image[32:-32, 32:-32, :]
            else:
                input_image_cropped = input_image

            input_images = np.expand_dims(input_image_cropped, axis=0)
            input_images = np.transpose(input_images, (0, 3, 1, 2))

            with torch.no_grad():
                image = Variable(torch.from_numpy(input_images)).cuda()
                # Predict
                depth_est = self.model(image)
                post_process = True
                if post_process:
                    image_flipped = flip_lr(image)
                    depth_est_flipped = self.model(image_flipped)
                    depth_cropped = post_process_depth(depth_est, depth_est_flipped)

            depth = np.zeros((height_depth, width_depth), dtype=np.float32)
            if args.crop == 'kbcrop':
                depth[top_margin:top_margin + 352, left_margin:left_margin + 1216] = \
                        depth_cropped[0].cpu().squeeze() / args.max_depth
            elif args.crop == 'edge':
                depth[32:-32, 32:-32] = depth_cropped[0].cpu().squeeze() / args.max_depth
            else:
                depth[:, :] = depth_cropped[0].cpu().squeeze() / args.max_depth

            coloredDepth = (greys(np.log10(depth * args.max_depth))[:, :, :3] * 255).astype('uint8')
            self.outputViewer.setPixmap(QtGui.QPixmap.fromImage(np_to_qimage(coloredDepth)))
            self.glWidget.depth = depth

        else:
            self.glWidget.depth = 0.5 + np.zeros((height_rgb // 2, width_rgb // 2, 1))
        
        self.glWidget.updateRGBD()
        self.glWidget.updateGL()
        
        # Update to next frame if we are live
        QtCore.QTimer.singleShot(10, self.updateInput)


class GLWidget(QtOpenGL.QGLWidget):
    def __init__(self, parent=None):
        QtOpenGL.QGLWidget.__init__(self, parent)
        
        self.object = 0
        self.xRot = 5040
        self.yRot = 40
        self.zRot = 0
        self.zoomLevel = 9
        
        self.lastPos = QtCore.QPoint()
        
        self.green = QtGui.QColor.fromCmykF(0.0, 0.0, 0.0, 1.0)
        self.black = QtGui.QColor.fromCmykF(0.0, 0.0, 0.0, 1.0)
        
        # Precompute for world coordinates
        self.xx, self.yy = self.worldCoords(width=width_rgb, height=height_rgb)
        
        self.rgb = np.zeros((height_rgb, width_rgb, 3), dtype=np.uint8)
        self.depth = np.zeros((height_depth, height_depth), dtype=np.float32)
        
        self.col_vbo = None
        self.pos_vbo = None
        if self.rgb.any() and self.depth.any():
            self.updateRGBD()
    
    def xRotation(self):
        return self.xRot
    
    def yRotation(self):
        return self.yRot
    
    def zRotation(self):
        return self.zRot
    
    def minimumSizeHint(self):
        return QtCore.QSize(height_rgb, width_rgb)
    
    def sizeHint(self):
        return QtCore.QSize(height_rgb, width_rgb)
    
    def setXRotation(self, angle):
        if angle != self.xRot:
            self.xRot = angle
            self.emit(QtCore.SIGNAL("xRotationChanged(int)"), angle)
            self.updateGL()
    
    def setYRotation(self, angle):
        if angle != self.yRot:
            self.yRot = angle
            self.emit(QtCore.SIGNAL("yRotationChanged(int)"), angle)
            self.updateGL()
    
    def setZRotation(self, angle):
        if angle != self.zRot:
            self.zRot = angle
            self.emit(QtCore.SIGNAL("zRotationChanged(int)"), angle)
            self.updateGL()
    
    def resizeGL(self, width, height):
        GL.glViewport(0, 0, width, height)
    
    def mousePressEvent(self, event):
        self.lastPos = QtCore.QPoint(event.pos())
    
    def mouseMoveEvent(self, event):
        dx = -(event.x() - self.lastPos.x())
        dy = (event.y() - self.lastPos.y())
        
        if event.buttons() & QtCore.Qt.LeftButton:
            self.setXRotation(self.xRot + dy)
            self.setYRotation(self.yRot + dx)
        elif event.buttons() & QtCore.Qt.RightButton:
            self.setXRotation(self.xRot + dy)
            self.setZRotation(self.zRot + dx)
        
        self.lastPos = QtCore.QPoint(event.pos())
    
    def wheelEvent(self, event):
        numDegrees = event.delta() / 8
        numSteps = numDegrees / 15
        self.zoomLevel = self.zoomLevel + numSteps
        event.accept()
        self.updateGL()
    
    def initializeGL(self):
        self.qglClearColor(self.black.darker())
        GL.glShadeModel(GL.GL_FLAT)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)
        
        VERTEX_SHADER = shaders.compileShader("""#version 330
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 color;
        uniform mat4 mvp; out vec4 frag_color;
        void main() {gl_Position = mvp * vec4(position, 1.0);frag_color = vec4(color, 1.0);}""", GL.GL_VERTEX_SHADER)
        
        FRAGMENT_SHADER = shaders.compileShader("""#version 330
        in vec4 frag_color; out vec4 out_color;
        void main() {out_color = frag_color;}""", GL.GL_FRAGMENT_SHADER)
        
        self.shaderProgram = shaders.compileProgram(VERTEX_SHADER, FRAGMENT_SHADER)
        
        self.UNIFORM_LOCATIONS = {
            'position': GL.glGetAttribLocation(self.shaderProgram, 'position'),
            'color': GL.glGetAttribLocation(self.shaderProgram, 'color'),
            'mvp': GL.glGetUniformLocation(self.shaderProgram, 'mvp'),
        }
        
        shaders.glUseProgram(self.shaderProgram)
    
    def paintGL(self):
        if self.rgb.any() and self.depth.any():
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            self.drawObject()
    
    def worldCoords(self, width, height):
        cx, cy = width / 2, height / 2
        fx = 518.8579
        fy = 518.8579
        xx, yy = np.tile(range(width), height), np.repeat(range(height), width)
        xx = (xx - cx) / fx
        yy = (yy - cy) / fy
        return xx, yy
    
    def posFromDepth(self, depth):
        length = depth.shape[0] * depth.shape[1]
        
        depth[edges(depth) > 0.3] = 1e6  # Hide depth edges
        z = depth.reshape(length)
        
        return np.dstack((self.xx * z, self.yy * z, z)).reshape((length, 3))
    
    def createPointCloudVBOfromRGBD(self):
        # Create position and color VBOs
        self.pos_vbo = vbo.VBO(data=self.pos, usage=GL.GL_DYNAMIC_DRAW, target=GL.GL_ARRAY_BUFFER)
        self.col_vbo = vbo.VBO(data=self.col, usage=GL.GL_DYNAMIC_DRAW, target=GL.GL_ARRAY_BUFFER)
    
    def updateRGBD(self):
        # RGBD dimensions
        width, height = self.depth.shape[1], self.depth.shape[0]
        
        # Reshape
        points = self.posFromDepth(self.depth.copy())
        colors = resize(self.rgb, (height, width)).reshape((height * width, 3))
        
        # Flatten and convert to float32
        self.pos = points.astype('float32')
        self.col = colors.reshape(height * width, 3).astype('float32')
        
        # Move center of scene
        self.pos = self.pos + glm.vec3(0, -0.06, -0.3)
        
        # Create VBOs
        if not self.col_vbo:
            self.createPointCloudVBOfromRGBD()
    
    def drawObject(self):
        # Update camera
        model, view, proj = glm.mat4(1), glm.mat4(1), glm.perspective(45, self.width() / self.height(), 0.01, 100)
        center, up, eye = glm.vec3(0, -0.075, 0), glm.vec3(0, -1, 0), glm.vec3(0, 0, -0.4 * (self.zoomLevel / 10))
        view = glm.lookAt(eye, center, up)
        model = glm.rotate(model, self.xRot / 160.0, glm.vec3(1, 0, 0))
        model = glm.rotate(model, self.yRot / 160.0, glm.vec3(0, 1, 0))
        model = glm.rotate(model, self.zRot / 160.0, glm.vec3(0, 0, 1))
        mvp = proj * view * model
        GL.glUniformMatrix4fv(self.UNIFORM_LOCATIONS['mvp'], 1, False, glm.value_ptr(mvp))
        
        # Update data
        self.pos_vbo.set_array(self.pos)
        self.col_vbo.set_array(self.col)
        
        # Point size
        GL.glPointSize(2)
        
        # Position
        self.pos_vbo.bind()
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        
        # Color
        self.col_vbo.bind()
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        
        # Draw
        GL.glDrawArrays(GL.GL_POINTS, 0, self.pos.shape[0])

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    res = app.exec_()