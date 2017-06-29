# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 17:35:12 2016

@author: aurelien
"""

from PyQt4 import QtGui,QtCore
import numpy as np
from dahlia_backend import Experiment,saveObject,loadObject,w_trajectory_classification
from dahlia_backend import w_classification
import os
import dahlia_methods as m
import cv2

import scipy.ndimage as ndi

class dahlia_GUI(QtGui.QWidget):
    
    def __init__(self,*args, **kwargs):
        """stores the default values in config.bbn"""
        #GUI itself
        super(dahlia_GUI, self).__init__(*args, **kwargs)        
        self.intro_display = QtGui.QLabel()
        self.intro_display.setText("Select what you want to do:")

        
        self.b_load_experiment = QtGui.QPushButton("Load Experiment")
        self.b_load_experiment.clicked.connect(self.load_experiment)
        
        self.b_new_experiment =  QtGui.QPushButton("New Experiment")       
        self.b_new_experiment.clicked.connect(self.new_experiment)
        
        self.windows = []
        
        self.status_display = QtGui.QLabel()
        self.status_display.setText("Experiment undefined")
        
        if not os.path.isfile("config.bbn"):
            self.default_values={"path":"."}
            saveObject("config.bbn",self.default_values)
        else:
            self.default_values = loadObject("config.bbn")
            print self.default_values
            
        self.experiment=Experiment(self.default_values["path"])
        self.grid = QtGui.QGridLayout()
        self.setLayout(self.grid)
        self.grid.addWidget(self.intro_display,0,0,1,5)
        
        self.grid.addWidget(self.b_load_experiment,1,0)
        self.grid.addWidget(self.b_new_experiment,1,1)
        self.grid.addWidget(self.status_display,2,0,1,5)
        
        #Just to write the thing. Remove after
        self.unlock_menu()
        
    def load_experiment(self):
        """to load an experiment"""
        openfile = str(QtGui.QFileDialog.getExistingDirectory(self, "Select Directory",\
                                                              self.default_values["path"]))
        self.default_values["path"] = str(openfile)
        
        self.experiment = Experiment(str(openfile))
        self.experiment.load()
        
        self.unlock_menu()

        
    def new_experiment(self):
        print self.default_values["path"]
        openfile = str(QtGui.QFileDialog.getExistingDirectory(self, "Select Directory",\
                                                              self.default_values["path"]))
        self.default_values["path"] = str(openfile)
        self.experiment = Experiment(str(openfile))
        self.experiment.save()
        self.unlock_menu()

    def unlock_menu(self):
        self.status_display.setText("Experiment in: "+self.experiment.path)
        self.b_denoising = QtGui.QPushButton("Denoising")
        self.b_denoising.clicked.connect(self.denoising)
        
        self.b_segmentation = QtGui.QPushButton("Segmentation")
        self.b_segmentation.clicked.connect(self.segmentation)
        
        self.b_tracking = QtGui.QPushButton("Tracking")
        self.b_tracking.clicked.connect(self.tracking)
        
        self.b_classification = QtGui.QPushButton("Manual Trajectory Classification")
        self.b_classification.clicked.connect(self.classification)
        
        
        self.b_clustering = QtGui.QPushButton("Data clustering")
        self.b_clustering.clicked.connect(self.data_clustering)
        
        self.grid.addWidget(self.b_denoising,3,0)
        self.grid.addWidget(self.b_segmentation,3,1)
        self.grid.addWidget(self.b_tracking,3,2)
        
        self.grid.addWidget(self.b_classification,4,0)
        self.grid.addWidget(self.b_clustering,4,1)
        
    def segmentation(self):
        
        self.dialog_segmentation = DialogSegmentation(self)
        self.dialog_segmentation.show()
      
    def closeEvent(self, *args, **kwargs):
        saveObject("config.bbn",self.default_values)
        self.experiment.save()
        super(dahlia_GUI, self).closeEvent(*args, **kwargs)
        
        
    def denoising(self):
        if self.experiment.body_path!=None:
            print "warninn"   #To change
        self.dialog_denoising = DialogDenoising(self)
        #self.windows.append(self.dialog_denoising)
        self.dialog_denoising.show()
        
    def tracking(self):
        self.experiment.process_tracking()
        
    def classification(self):
        name = os.path.join(self.path.join(self.experiment.path,"traj_selection"))
        w_trajectory_classification(self.experiment,name)
        
    def data_clustering(self):
        self.my_dialog =  MyDialog(self)
        self.my_dialog.show()
        path_list = []    #Contains the paths to the classified trajectories
        w_classification(path_list)
        
class DialogDenoising(QtGui.QDialog):
    def __init__(self, parent=None):
        super(DialogDenoising, self).__init__(parent)

        """self.buttonBox = QtGui.QDialogButtonBox(self)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)"""
        self.parent = parent
        self.experiment = parent.experiment
        
        self.textBrowser = QtGui.QLabel()
        self.textBrowser.setText("""
        This function uses wavelets to denoise the image stack as well as
        local histogram equalization to compensate for
        irregular illumination. To get it to work properly,
        it is necessary to know approximately what fraction of the
        image is background, ie it is 0.75 if the background is made
        of the 75% darkest pixels.
                                 """)
        self.fraction = QtGui.QLineEdit('0.75')
        self.fraction.editingFinished.connect(self.fractionValueChanged)
        self.fraction_label = QtGui.QLabel()
        self.fraction_label.setText("Fraction of background pixels")
        
        self.test_button = QtGui.QPushButton("test")
        self.test_button.clicked.connect(self.test)
        
        self.equalize_histo = QtGui.QCheckBox()
        self.equalize_histo_label = QtGui.QLabel("Equalize histogram")
        self.equalize_histo.setChecked(True)
        
        self.buttonBox = QtGui.QDialogButtonBox(self)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Ok|QtGui.QDialogButtonBox.Cancel)
    
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(self.textBrowser,0,0,1,5)
        grid.addWidget(self.fraction,1,1)
        grid.addWidget(self.fraction_label,1,0)
        grid.addWidget(self.test_button,1,2)
        grid.addWidget(self.equalize_histo,2,1)
        grid.addWidget(self.equalize_histo_label,2,0)
        
        grid.addWidget(self.buttonBox,3,0,1,3)
        
    def accept(self):
        target_dir = self.experiment.path+"_denoised"
        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)
        else:
            print "Warning: directory already existing"
        print target_dir
        print float(self.fraction.text())
        m.denoiseStack(self.experiment.path,target_dir,\
                       frac =  float(self.fraction.text()),histo = self.equalize_histo.isChecked())
        self.experiment.path = target_dir
        super(DialogDenoising, self).accept()
        
    def reject(self):
        super(DialogDenoising, self).reject()
    
    def fractionValueChanged(self):
        value = float(self.fraction.text())
        if value>0.99:
            value=0.99
        if value<0:
            value=0.001
        self.fraction.setText(str(value))
        
    def test(self):
        im = m.open_frame(self.parent.experiment.path,self.parent.experiment.n_frames/2)
        denoised = m.wlt_total(im,histo=self.equalize_histo.isChecked())
        m.si(denoised)
        
class DialogSegmentation(QtGui.QDialog):
    def __init__(self, parent=None):
        super(DialogSegmentation, self).__init__(parent)

        """self.buttonBox = QtGui.QDialogButtonBox(self)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)"""
        self.parent = parent
        self.experiment = parent.experiment
        
        self.textBrowser = QtGui.QLabel()
        self.textBrowser.setText("""
        This function segments a stack of images.
                                 """)
        thresholds = ["hysteresis","simple","li","mean","otsu"]
        
        self.low_threshold = QtGui.QLineEdit('6')
        self.low_threshold.editingFinished.connect(self.low_thresholdValueChanged)
        self.low_threshold_label = QtGui.QLabel()
        self.low_threshold_label.setText("Lower threshold (for simple and hysteresis only):")
        
        self.high_threshold = QtGui.QLineEdit('10')
        self.high_threshold.editingFinished.connect(self.high_thresholdValueChanged)
        self.high_threshold_label = QtGui.QLabel()
        self.high_threshold_label.setText("Higher threshold (for hysteresis only):")
        
        self.test_button = QtGui.QPushButton("test")
        self.test_button.clicked.connect(self.test)
        
        
        self.spinBox = QtGui.QSpinBox()
        self.spinBox.setMinimum(0)
        self.spinBox.setValue(2)
        
        self.spinBox_label = QtGui.QLabel()
        self.spinBox_label.setText("Number of morphological operations o separate cell bodies from protrusions:")
        self.morphotest_button =  QtGui.QPushButton("test morphological separation")
        self.morphotest_button.clicked.connect(self.test_morphology)
        
        self.buttonBox = QtGui.QDialogButtonBox(self)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Ok|QtGui.QDialogButtonBox.Cancel)
    
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        
        self.comboBox = QtGui.QComboBox(self)
        for elt in thresholds:
            self.comboBox.addItem(elt)

        #self.comboBox.activated[str].connect(self.style_choice)
        
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(self.textBrowser,0,0,1,5)
        grid.addWidget(self.comboBox,0,6)
        grid.addWidget(self.low_threshold_label,1,0)
        grid.addWidget(self.low_threshold,1,1)
        
        grid.addWidget(self.high_threshold,1,3)
        grid.addWidget(self.high_threshold_label,1,2)
        
        grid.addWidget(self.test_button,2,2)
        
        grid.addWidget(self.spinBox_label,3,0)
        grid.addWidget(self.spinBox,3,1)
        grid.addWidget(self.morphotest_button,3,2)
        
        grid.addWidget(self.buttonBox,4,0,1,3)
        
    def low_thresholdValueChanged(self):
        value = float(self.low_threshold.text())
        if value>254:
            value=254
        if value<0:
            value=0
        self.low_threshold.setText(str(value))
    def high_thresholdValueChanged(self):
        value = float(self.high_threshold.text())
        if value>254:
            value=254
        if value<0:
            value=0
        self.high_threshold.setText(str(value))
        
    def test(self):
        im = m.open_frame(self.parent.experiment.path,self.parent.experiment.n_frames/2)
        parameters = [float(self.low_threshold.text()),float(self.high_threshold.text())]
        print "test"
        segmented = m.threshold_image(im,self.comboBox.currentText(),parameters)
        
        m.si2(im,segmented,"original","thresholded")
        
    def test_morphology(self):
        """Tests morphological operations with previously defined parameters"""
        im = m.open_frame(self.parent.experiment.path,self.parent.experiment.n_frames/2)
        parameters = [float(self.low_threshold.text()),float(self.high_threshold.text())]
        segmented = m.threshold_image(im,self.comboBox.currentText(),parameters)

        n_iterations=self.spinBox.value()

        body,arm = m.arm_from_threshold(segmented,n_iterations)
        out = m.cv_overlay_mask2image(body,im,color="green")
        out = m.cv_overlay_mask2image(arm,out,color="red")
        m.si(out,title=str(n_iterations)+" iterations")
    
    def accept(self):
        """Proceeds to segmentation and morphological separation of the whole stack"""
        raw_path = self.experiment.path
        raw_path = raw_path[:-9]  #minus _denoised
        self.body_path = raw_path+"_bodies"
        self.arm_path = raw_path+"_arms"
        
        self.experiment.arm_path = self.arm_path
        self.experiment.body_path = self.body_path
        
        if not os.path.isdir(self.body_path):
            os.mkdir(self.body_path)
        if not os.path.isdir(self.arm_path):
            os.mkdir(self.arm_path)
        n_iterations=self.spinBox.value()
        
        for i in range(1,self.experiment.n_frames+1):
            im = m.open_frame(self.experiment.path,i)
            parameters = [float(self.low_threshold.text()),float(self.high_threshold.text())]
            segmented = m.threshold_image(im,self.comboBox.currentText(),parameters)

            body,arms = m.arm_from_threshold(segmented,n_iterations)
            label_arm ,nr_elts_arms = ndi.label(arms)
            label_body,nr = ndi.label(body)
            
            label_body = label_body.astype(np.uint8)
            label_arm = m.filter_by_size(label_arm,60)
            label_arm = label_arm.astype(np.uint8)
            
            cv2.imwrite(os.path.join(self.body_path,str(i)+".png"),label_body)
            cv2.imwrite(os.path.join(self.arm_path,str(i)+".png"),label_arm)
            print "file ",i
        self.experiment.save()
        super(DialogSegmentation, self).accept()
        
    def reject(self):
        super(DialogSegmentation, self).reject()
    
class MyDialog(QtGui.QDialog):
    def __init__(self, parent=None):
        super(MyDialog, self).__init__(parent)

        self.buttonBox = QtGui.QDialogButtonBox(self)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)

        self.textBrowser = QtGui.QTextBrowser(self)
        self.textBrowser.append("This is a QTextBrowser!")

        self.verticalLayout = QtGui.QVBoxLayout(self)
        self.verticalLayout.addWidget(self.textBrowser)
        self.verticalLayout.addWidget(self.buttonBox)
    
app = QtGui.QApplication([])
win = dahlia_GUI()

win.show()
app.exec_()