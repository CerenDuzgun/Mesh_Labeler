import sys, os

# from PyQt5 import *
from PyQt5 import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QShortcut, QAbstractItemView
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import pandas as pd
import numpy as np
import vtk
from vtk.util import numpy_support
from vedo import *
import vtkmodules

# from UI_files.MainWindow import *
from UI_files.MainWindow_v4 import *


class Mesh_Labeler(QtWidgets.QMainWindow, Ui_MainWindow):
    """
    05/21/2024
    Mesh Labeler v4.3.1
    vedo: 2023.4.6
    vtk: 9.2.6
    chagned:
        1. added filling function ("shift" + left click)
        2. added flagepole-type "show label" function (key: "s")
    """

    def __init__(self, parent=None):
        super(Mesh_Labeler, self).__init__(parent)
        self.setupUi(self)

        # enable drag and drop operation
        self.setAcceptDrops(True)

        ##################
        # set a vtk Widget
        ##################
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.vtkWidget.setSizePolicy(sizePolicy)
        self.vtkWidget.setMinimumSize(QtCore.QSize(580, 430))
        self.vtk_verticalLayout = QtWidgets.QVBoxLayout(self.frame)
        self.vtk_verticalLayout.addWidget(self.vtkWidget)
        self.vp = Plotter(qt_widget=self.vtkWidget)

        ###################################################################################
        # read colormap.csv, resize self.tableWidget, and show colormap in self.tableWidget
        ###################################################################################
        if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "colormap.csv")):
            csv_relative_path = "colormap.csv" # launch when developing
        elif os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../colormap.csv")):
            csv_relative_path = "../colormap.csv" # launch when pynsist

        self.colormap_csv = pd.read_csv(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_relative_path)
        ).values  # used when pynsist
        self.label_id = self.colormap_csv[:, 0]  # label value (i.e., int number)
        self.label_colormap = self.colormap_csv[:, 1:4]  # colormap
        self.label_description = self.colormap_csv[:, 4]  # label description
        self.tableWidget.setColumnWidth(0, 35)  # label
        self.tableWidget.setColumnWidth(1, 35)  # color
        self.tableWidget.setColumnWidth(2, 110)  # description
        self.tableWidget.setRowCount(len(self.label_id))  # update tableWidget
        self.vedo_colorlist = []

        for i_row in range(len(self.label_id)):
            newItem = QtWidgets.QTableWidgetItem("{}".format(self.label_id[i_row]))
            self.tableWidget.setItem(i_row, 0, newItem)  # column 0: label ID
            self.tableWidget.setItem(
                i_row, 1, QtWidgets.QTableWidgetItem("")
            )  # column 1: empty
            self.tableWidget.item(i_row, 1).setBackground(
                QtGui.QColor(
                    self.label_colormap[i_row, 0],
                    self.label_colormap[i_row, 1],
                    self.label_colormap[i_row, 2],
                )
            )  # column 1: change background based on colormap
            self.tableWidget.setItem(
                i_row,
                2,
                QtWidgets.QTableWidgetItem("{}".format(self.label_description[i_row])),
            )  # column 2: description
            self.vedo_colorlist.append(
                (self.label_id[i_row], self.label_colormap[i_row])
            )
        self.vedo_colorlist.append(
            (np.max(self.label_id) + 1, [169, 169, 169])
        )  # add one more label which represents selected cells

        self.lut = build_lut(
            colorlist=self.vedo_colorlist,
            # vmin=0,
            # vmax=np.max(self.label_id)+2,  # start from zero
            # above_color="black",
            # below_color="white",
            interpolate=False,
        )

        ###########################
        # initialize some variables
        ###########################
        self.mesh_exist = False
        self.mesh_wireframe_show = False # no wireframe as default
        self.opened_mesh_path = os.getcwd()
        self.existed_opened_mesh_path = os.getcwd()
        self.reset_plotters()

        self.spinBox_brush_active_label.setRange(
            min(self.label_id), max(self.label_id)
        )  # set min and max label id for spinBox
        self.spinBox_swap_original_label.setRange(
            min(self.label_id), max(self.label_id)
        )
        self.spinBox_swap_new_label.setRange(min(self.label_id), max(self.label_id))
        # default active label = 0
        self.brush_active_label = [0]  
        self.swap_original_label = [0]
        self.swap_new_label = [0]

        # brush method variables
        self.brush_mode = False  # un-do enable
        self.brush_clicked = False
        self.brush_selected_pts = []  # un-do enable
        self.brush_erased_pts = []  # un-do enable
        self.flattened_selected_pt_ids = []  # un-do enable
        self.flattened_erased_pt_ids = []  # un-do enable
        self.brush_erased_pts = []  # un-do enable
        self.selected_cell_ids = []  # un-do enable
        self.undo_brush_mode = False
        self.undo_brush_selected_pts = []  # un-do enable
        self.undo_brush_erased_pts = []  # un-do enable
        self.undo_flattened_selected_pt_ids = []  # un-do enable
        self.undo_flattened_erased_pt_ids = []  # un-do enable
        self.undo_brush_erased_pts = []  # un-do enable
        self.undo_selected_cell_ids = []  # un-do enable
        self.caption_mode = False
        self.caption_meshes = []
        self.tooth_legend = []

        self.brush_radius = 1.0
        self.brush_ball_switch = False  # this switch to control to show a transparent ball or not (default false in v4.0)
        self.brush_ball = Sphere(
            pos=[0, 0, 0], r=self.brush_radius, c="grey", alpha=0.0
        ).pickable(False)
        self.ctrl_pressed = False

        # 04/23/2021; new for v3.0 (landmarking)
        self.doubleSpinBox_R.setValue(0.5)  # default radius = 0.5
        self.landmark_radius = self.doubleSpinBox_R.value()
        self.landmark_color = "red"
        self.selected_landmark_color = "blue"
        self.landmark_exist = False
        self.vedo_landmarks = []
        self.landmarks = []
        self.landmark_names = []
        self.selected_lmk_idx = None
        self.opened_landmarking_path = os.getcwd()
        self.existed_opened_landmarking_path = os.getcwd()
        self.landmark_selecting = False
        self.vedo_landmarks = []
        self.tableWidget_landmarking_focus = False

        ########################
        # setup signals (button, tab, and spinBox)
        ########################
        self.pushButton_load.setToolTip("Load a mesh (.vtp, .stl, .obj, or .ply)")
        self.pushButton_load.setStyleSheet(
            "QPushButton{font: 75 20px}, QToolTip{color: k ; font: 12px}"
        )  # set different font on tool tip
        self.pushButton_save.setToolTip("Save a mesh (.vtp, .stl, or .obj)")
        self.pushButton_save.setStyleSheet(
            "QPushButton{font: 75 20px}, QToolTip{color: k ; font: 12px}"
        )  # set different font on tool tip
        self.pushButton_load.clicked.connect(self.load_mesh)
        self.pushButton_save.clicked.connect(self.save_mesh)

        # if changing tab, show tab information
        self.tabWidget.currentChanged.connect(
            self.show_tab_info
        )  
        self.spinBox_brush_active_label.setKeyboardTracking(False)
        self.spinBox_swap_original_label.setKeyboardTracking(False)
        self.spinBox_swap_new_label.setKeyboardTracking(False)
        self.spinBox_brush_active_label.valueChanged.connect(
            self.brush_active_label_changed
        )
        self.spinBox_swap_original_label.valueChanged.connect(
            self.swap_original_label_changed
        )
        self.spinBox_swap_new_label.valueChanged.connect(self.swap_new_label_changed)
        self.pushButton_swap_change.clicked.connect(self.swap_assign_new_label)
        # 04/23/2021; new for v3.0
        self.pushButton_load_fcsv.clicked.connect(self.load_landmarking)
        self.pushButton_save_fcsv.clicked.connect(self.save_landmarking)
        self.pushButton_add_landmark.clicked.connect(self.add_new_landmark)
        self.pushButton_delete_landmark.clicked.connect(self.delete_landmark)
        self.pushButton_rename.clicked.connect(self.rename_landmark)
        self.pushButton_relocate.clicked.connect(self.relocate_landmark)
        self.pushButton_reset.clicked.connect(self.reset_landmark)
        self.tableWidget_landmakring.itemSelectionChanged.connect(
            self.selected_landmark_column
        )
        self.pushButton_move_right.setIcon(
            self.style().standardIcon(getattr(QtWidgets.QStyle, "SP_ArrowRight"))
        )
        self.pushButton_move_right.clicked.connect(self.landmark_order_move_right)
        self.pushButton_move_left.setIcon(
            self.style().standardIcon(getattr(QtWidgets.QStyle, "SP_ArrowLeft"))
        )
        self.pushButton_move_left.clicked.connect(self.landmark_order_move_left)
        self.doubleSpinBox_R.valueChanged.connect(self.spin_R_changed)

        # # get rendering and add callback functions
        self.add_callbacks()
        self.shortcut_lmk_relocation = QShortcut(QKeySequence("Ctrl+r"), self)
        self.shortcut_lmk_relocation.activated.connect(self.shortcut_function_landmark_relocation)
        self.shortcut_fcsv_save = QShortcut(QKeySequence("Ctrl+s"), self)
        self.shortcut_fcsv_save.activated.connect(self.shortcut_function_fcsv_save)
        self.shortcut_toggle_wireframe = QShortcut(QKeySequence("l"), self)
        self.shortcut_toggle_wireframe.activated.connect(self.shortcut_function_toggle_wireframe)
        self.shortcut_click_right = QShortcut(QKeySequence("right"), self)
        self.shortcut_click_right.activated.connect(self.shortcut_function_select_right_landmark)
        self.shortcut_click_left = QShortcut(QKeySequence("left"), self)
        self.shortcut_click_left.activated.connect(self.shortcut_function_select_left_landmark)

        #############
        # show window
        #############
        self.showMaximized()
        # self.show()

    ########################
    # general functions
    ########################
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        for f in files:
            self.opened_mesh_path = f
            try:
                if self.opened_mesh_path[-4:].lower() in [
                    ".vtp",
                    ".stl",
                    ".obj",
                    ".ply",
                ]:  # check file extension for mesh
                    self.reset_plotters()
                    self.plot_mesh()
                    self.existed_opened_mesh_path = self.opened_mesh_path
                    self.setWindowTitle(
                        "Mesh Labeler (open source) -- {}".format(
                            self.existed_opened_mesh_path
                        )
                    )
                    self.vtkWidget.setFocus()
                elif self.opened_mesh_path[-5:].lower() in [
                    ".fcsv"
                ]:  # check file extension for landmarking
                    self.tabWidget.setCurrentIndex(2)
                    if self.mesh_exist:  # only enable when mesh exists
                        self.opened_landmarking_path = self.opened_mesh_path

                        # read fcsv file
                        self.load_fcsv()

                        if self.landmark_exist:  # fcsv load successfullly
                            self.existed_opened_landmarking_path = (
                                self.opened_landmarking_path
                            )
                            # display landmarking info on tableWidget
                            self.display_landmark_table()
                            self.tableWidget_landmakring.setCurrentCell(0, 0) # deafult select the first ladnmark

                            # plot landmarks
                            self.vp.remove(
                                self.vedo_landmarks
                            )  # clean previous landmarks if existing
                            self.vedo_landmarks = []
                            for i_landmark in range(len(self.landmarks)):
                                # self.vedo_landmarks.append(Point(self.landmarks[i_landmark], r=self.landmark_radius).c(self.landmark_color))
                                self.vedo_landmarks.append(
                                    Sphere(
                                        self.landmarks[i_landmark],
                                        r=self.landmark_radius,
                                    ).c(self.landmark_color)
                                )
                            self.vp.add(self.vedo_landmarks)

                            self.setWindowTitle(
                                "Mesh Labeler (open source) -- {} -- {}".format(
                                    self.existed_opened_mesh_path,
                                    self.existed_opened_landmarking_path,
                                )
                            )
                            self.vtkWidget.setFocus()
                    else:
                        self.show_messageBox(
                            "No mesh available! Please load a mesh first!"
                        )
            except:
                None  # it won't happen, because we only allow to load four types of mesh file

    def onClose(self):
        # Disable the interactor before closing to prevent it
        # from trying to act on already deleted items
        self.vtkWidget.close()

    def closeEvent(self, event):  # new in v3.0
        reply = QMessageBox.question(
            self, "Quit", "Are you sure to quit?", QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def add_callbacks(self):
        """
        all callback functions to create the GUI interaction
        """
        # segmentation
        self.vp.add_callback("KeyPressEvent", self.segmentation_keypress)

        # in order to get both "press" and "release" events, need to use the original VTK function (i.e., AddObserver)
        # self.vp.interactor.AddObserver("RightButtonPressEvent", self.brush_onRightClick)
        self.vp.add_callback("RightButtonPressEvent", self.brush_onRightClick)
        self.vp.add_callback("MouseMove", self.brush_dragging)
        self.vp.interactor.AddObserver(
            "RightButtonReleaseEvent", self.brush_onRightRelease
        )
        self.vp.add_callback("MouseWheelForwardEvent", self.brush_increase_radius)
        self.vp.add_callback("MouseWheelBackwardEvent", self.brush_decrease_radius)
        self.vp.add_callback("LeftButtonPressEvent", self.brush_filling)
        self.vp.interactor.AddObserver("KeyPressEvent", self.press_ctrl)
        self.vp.interactor.AddObserver("KeyReleaseEvent", self.release_ctrl)
        self.vp.interactor.AddObserver("KeyPressEvent", self.press_shift)
        self.vp.interactor.AddObserver("KeyReleaseEvent", self.release_shift)

        # landmarking
        self.vp.add_callback("LMK_LeftButtonPressEvent", self.landmarking_onLeftClick)
        self.vp.add_callback("LMK_RightButtonPressEvent", self.landmarking_onRightClick)

        # general
        # self.vp.interactor.AddObserver("KeyPressEvent", self.toggle_wireframe)
        # self.vp.interactor.AddObserver("KeyPressEvent", self.shortcut_keypress)

    def set_mesh_color(self):
        self.mesh.cmap(
            input_cmap=self.lut,
            input_array=self.mesh.celldata["Label"],
            on="cells",
            n_colors=len(self.vedo_colorlist),
        )#.add_scalarbar()
        self.mesh.mapper().SetScalarRange(0, np.max(self.label_id) + 2) # keep "np.max(self.label_id) + 1 " for selected color

    def plot_mesh(self):
        """
        call in load_mesh, using vedo to read mesh file
        """
        self.mesh_exist = True  # new in v3.0
        self.mesh = load(self.opened_mesh_path)
        # self.mesh = load("Example_01.vtp")
        self.mesh_cms = Points(self.mesh.cell_centers())
        self.mesh.linecolor('black').linewidth(
            0.0
        )  # .lighting('off')#.lineColor('black')#.lineWidth(0.1)

        self.normals = self.mesh.compute_normals()

        # check if the input mesh has cell array 'Label'; if not, assign zero for all cells
        if not "Label" in self.mesh.celldata.keys():
            self.mesh.celldata["Label"] = np.zeros(
                [
                    self.mesh.ncells,
                ],
                dtype=np.int32,
            )

        self.set_mesh_color()
        # self.mesh.mapper().SetScalarRange(0, len(self.vedo_colorlist))

        self.temp_labels = self.mesh.clone().celldata[
            "Label"
        ]  # save new labels to backup

        self.undo_labels = self.mesh.clone().celldata[
            "Label"
        ]  # another backup label for un-do procedure

        self.undo_temp_labels = self.mesh.clone().celldata[
            "Label"
        ]  # another backup label for un-do procedure

        # added by 06/21/2022; fixed the bug if there is another cell array prior to 'Label'
        self.mesh.celldata.select("Label")

        # self.vp.show(self.mesh, interactorStyle=0)
        self.vp.show(self.mesh, interactive=False)

        self.show_tab_info()  # show tab information in status bar

    def reset_plotters(self):
        # self.vp.close()
        # self.vp = Plotter(qt_widget=self.vtkWidget)
        self.vp.clear()
        self.brush_mode = False
        self.caption_mode = False

        self.selected_cell_ids = []

    def selected_pt_ids_to_cell_ids(self, selected_ids):
        """
        General funciton that return cell_ids when pt_ids are given
        """
        selected_cell_ids = []
        for i in selected_ids:
            selected_cell_ids.append(self.mesh.connected_cells(i, return_ids=True))
        flat_list = []
        for sublist in selected_cell_ids:
            for item in sublist:
                flat_list.append(item)
        selected_cell_ids = np.unique(np.asarray(flat_list))
        return selected_cell_ids

    ########################
    # landmarking functions
    ########################
    def load_fcsv(self):  # new in v3.0
        try:
            LPS = False  # LPS coordinate; default: False
            # check if it's LPS coordinate system
            with open(self.opened_landmarking_path, "r") as f:
                line = f.readline()
                line = f.readline()
                if "LPS" in line:
                    LPS = True

            # reader for RAS coordinate system
            data_type = {0: object, 1: np.float32, 2: np.float32, 3: np.float32, 4: np.float32, 5: np.float32, 6: np.float32, 7: np.float32, 8: np.float32, 9: np.float32, 10: np.float32, 11: object, 12: object, 13: object}
            header = None  # according to 3D slicer format
            skiprows = 3  # according to 3D Slicer format
            lmk_df = pd.read_csv(
                self.opened_landmarking_path, header=header, skiprows=skiprows, dtype=data_type
            )
        
            num_landmarks = len(lmk_df)
            landmarks = np.zeros([num_landmarks, 3], dtype=np.float32)

            landmarks = lmk_df[[1, 2, 3]].values
            landmark_names = lmk_df[[11]].values.tolist()

            if LPS:
                landmarks[:, 0:2] *= -1  # flip the first two axes

            self.landmarks = landmarks
            self.landmark_names = landmark_names
            self.landmark_exist = True
        except:
            self.show_messageBox(
                "Check fcsv format!\nYou might see an example on the repository."
            )

    def display_landmark_table(self):  # new in v3.0
        self.tableWidget_landmakring.setRowCount(2)  # update tableWidget
        self.tableWidget_landmakring.setColumnCount(
            len(self.landmarks)
        )  # update tableWidget

        self.tableWidget_landmakring.setRowHeight(0, 30)  # header
        self.tableWidget_landmakring.setRowHeight(1, 30)  # header

        # self.tableWidget_landmakring.setItem(0, 0, QtWidgets.QTableWidgetItem("Name"))
        # self.tableWidget_landmakring.setItem(1, 0, QtWidgets.QTableWidgetItem("Position"))

        for i_col in range(len(self.landmarks)):
            self.tableWidget_landmakring.setItem(
                0,
                i_col,
                QtWidgets.QTableWidgetItem("{}".format(self.landmark_names[i_col][0])),
            )
            self.tableWidget_landmakring.setItem(
                1,
                i_col,
                QtWidgets.QTableWidgetItem(
                    "({:.2f}, {:.2f}, {:.2f})".format(
                        self.landmarks[i_col, 0],
                        self.landmarks[i_col, 1],
                        self.landmarks[i_col, 2],
                    )
                ),
            )

    def selected_landmark_column(self):  # new in v3.0
        idx = self.tableWidget_landmakring.selectedIndexes()
        try:
            col_idx = idx[0].column()
            self.selected_lmk_idx = col_idx

            # reset to red for all landmarks
            for i in range(len(self.vedo_landmarks)):
                self.vedo_landmarks[i].c(self.landmark_color)

            # show selection
            if self.selected_lmk_idx != None:
                self.vedo_landmarks[self.selected_lmk_idx].c(
                    self.selected_landmark_color
                )
                self.vp.render(resetcam=False)
        except:
            self.selected_lmk_idx = None  # idx == []

    def rename_landmark(self):
        if self.landmark_exist:
            if self.selected_lmk_idx != None:  # selection cannot be empty
                new_name, done = QtWidgets.QInputDialog.getText(
                    self, "Rename", "Enter the new landmark name:"
                )
                if done:
                    self.landmark_names[self.selected_lmk_idx] = [new_name]
                    self.display_landmark_table()

    def relocate_landmark(self):
        if self.landmark_exist:
            if self.selected_lmk_idx != None:  # selection cannot be empty
                if self.tabWidget.currentIndex() == 2:  # in landmarking mode
                    self.landmark_selecting = True

    def reset_landmark(self):
        if self.landmark_exist:
            if self.selected_lmk_idx != None:  # selection cannot be empty
                if self.tabWidget.currentIndex() == 2:  # in landmarking mode
                    self.landmarks[self.selected_lmk_idx] = np.array([np.nan, np.nan, np.nan])

                    # update landmark plot and table
                    self.display_landmark_table()
                    self.vp.remove(self.vedo_landmarks)
                    self.vedo_landmarks[self.selected_lmk_idx] = Sphere(
                        self.landmarks[self.selected_lmk_idx], r=self.landmark_radius
                    ).c(self.selected_landmark_color)
                    self.vp.add(self.vedo_landmarks)
                    self.vp.render(resetcam=False)

    def landmarking_onLeftClick(self, evt):
        if self.tabWidget.currentIndex() == 2:  # in landmarking mode
            if self.landmark_selecting:
                # assign new landmark position
                self.landmarks[self.selected_lmk_idx] = self.mesh.picked3d

                # update landmark plot and table
                self.display_landmark_table()
                self.vp.remove(self.vedo_landmarks)
                self.vedo_landmarks[self.selected_lmk_idx] = Sphere(
                    self.landmarks[self.selected_lmk_idx], r=self.landmark_radius
                ).c(self.selected_landmark_color)
                self.vp.add(self.vedo_landmarks)
                self.vp.render(resetcam=False)

                self.landmark_selecting = False

    def landmarking_onRightClick(self, evt):
        if self.tabWidget.currentIndex() == 2:  # in landmarking mode
            self.tableWidget_landmakring.clearSelection()
            self.landmark_selecting = False
            # reset to red for all landmarks
            for i in range(len(self.vedo_landmarks)):
                self.vedo_landmarks[i].c(self.landmark_color)

    def add_new_landmark(self):
        if self.landmark_exist:  # add a new landmark on the existing landmark table
            if len(self.landmarks) > 0:
                new_name, done = QtWidgets.QInputDialog.getText(
                    self, "Add landmark", "Enter a name for the new landmark:"
                )
                if done:
                    # add new name
                    self.landmark_names.append([new_name])
                    # add new location
                    new_location = np.zeros([1, 3])  # default origin
                    self.landmarks = np.concatenate(
                        (self.landmarks, new_location), axis=0
                    )
                    # create new vedo_landmark
                    # self.vedo_landmarks.append(Point(self.landmarks[-1], r=self.landmark_radius).c(self.landmark_color))
                    self.vedo_landmarks.append(
                        Sphere(self.landmarks[-1], r=self.landmark_radius).c(
                            self.landmark_color
                        )
                    )
                    self.display_landmark_table()
                    self.tableWidget_landmakring.selectColumn(
                        len(self.landmarks) - 1
                    )  # select last column (new)
        else:  # create a new landmark table
            if self.mesh_exist:  # lock this function until mesh is existed
                new_name, done = QtWidgets.QInputDialog.getText(
                    self, "Add landmark", "Enter a name for the new landmark:"
                )
                if done:
                    # add new name
                    self.landmark_names.append([new_name])
                    # add new location
                    self.landmarks = np.zeros([1, 3])  # default origin
                    # create new vedo_landmark
                    # self.vedo_landmarks.append(Point(self.landmarks[-1], r=self.landmark_radius).c(self.landmark_color))
                    self.vedo_landmarks.append(
                        Sphere(self.landmarks[-1], r=self.landmark_radius).c(
                            self.landmark_color
                        )
                    )
                    self.landmark_exist = True  # turn on landmark_exist
                    self.display_landmark_table()
                    self.tableWidget_landmakring.selectColumn(0)  # select column 0

    def delete_landmark(self):
        if self.landmark_exist:  # add a new landmark on the existing landmark table
            if self.selected_lmk_idx != None:  # selection cannot be empty
                buttonReply = QMessageBox.question(
                    self,
                    "Delete",
                    "Do you want to delete?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if buttonReply == QMessageBox.Yes:
                    if len(self.landmark_names) > 1:  # multiple landmarks
                        del self.landmark_names[
                            self.selected_lmk_idx
                        ]  # remove self.landmark_names
                        self.landmarks = np.delete(
                            self.landmarks, self.selected_lmk_idx, 0
                        )  # remove self.landmarks
                        self.vp.remove(
                            self.vedo_landmarks[self.selected_lmk_idx]
                        )  # remove plot
                        self.vp.render()
                        del self.vedo_landmarks[
                            self.selected_lmk_idx
                        ]  # remove vedo_landmarks
                    else:  # only a landmark
                        self.vp.remove(
                            self.vedo_landmarks[self.selected_lmk_idx]
                        )  # remove plot
                        self.vp.render()
                        # reset everything
                        self.landmark_exist = False
                        self.vedo_landmarks = []
                        self.landmarks = []
                        self.landmark_names = []
                        self.selected_lmk_idx = None
                        self.landmark_selecting = False
                        self.vedo_landmarks = []
                    self.display_landmark_table()
                    self.tableWidget_landmakring.clearSelection()  # clean selection
                    self.selected_lmk_idx = None  # reset to no select_idx

    def landmark_order_move_right(self):
        if self.landmark_exist:  # add a new landmark on the existing landmark table
            if self.selected_lmk_idx != None:  # selection cannot be empty
                if (
                    self.selected_lmk_idx != len(self.landmarks) - 1
                ):  # not the most right
                    # list[pos1], list[pos2] = list[pos2], list[pos1]
                    (
                        self.landmark_names[self.selected_lmk_idx],
                        self.landmark_names[self.selected_lmk_idx + 1],
                    ) = (
                        self.landmark_names[self.selected_lmk_idx + 1],
                        self.landmark_names[self.selected_lmk_idx],
                    )
                    self.landmarks[
                        [self.selected_lmk_idx, self.selected_lmk_idx + 1]
                    ] = self.landmarks[
                        [self.selected_lmk_idx + 1, self.selected_lmk_idx]
                    ]
                    (
                        self.vedo_landmarks[self.selected_lmk_idx],
                        self.vedo_landmarks[self.selected_lmk_idx + 1],
                    ) = (
                        self.vedo_landmarks[self.selected_lmk_idx + 1],
                        self.vedo_landmarks[self.selected_lmk_idx],
                    )
                    self.selected_lmk_idx += 1
                    self.tableWidget_landmakring.selectColumn(self.selected_lmk_idx)
                    self.vp.render()
                    self.display_landmark_table()

    def landmark_order_move_left(self):
        if self.landmark_exist:  # add a new landmark on the existing landmark table
            if self.selected_lmk_idx != None:  # selection cannot be empty
                if self.selected_lmk_idx != 0:  # not the most left
                    # list[pos1], list[pos2] = list[pos2], list[pos1]
                    (
                        self.landmark_names[self.selected_lmk_idx],
                        self.landmark_names[self.selected_lmk_idx - 1],
                    ) = (
                        self.landmark_names[self.selected_lmk_idx - 1],
                        self.landmark_names[self.selected_lmk_idx],
                    )
                    self.landmarks[
                        [self.selected_lmk_idx, self.selected_lmk_idx - 1]
                    ] = self.landmarks[
                        [self.selected_lmk_idx - 1, self.selected_lmk_idx]
                    ]
                    (
                        self.vedo_landmarks[self.selected_lmk_idx],
                        self.vedo_landmarks[self.selected_lmk_idx - 1],
                    ) = (
                        self.vedo_landmarks[self.selected_lmk_idx - 1],
                        self.vedo_landmarks[self.selected_lmk_idx],
                    )
                    self.selected_lmk_idx -= 1
                    self.tableWidget_landmakring.selectColumn(self.selected_lmk_idx)
                    self.vp.render()
                    self.display_landmark_table()

    ########################
    # labeling functions
    ########################
    def undo_backup(self):
        """
        step 1 to perform undo process (ctrl + z)
        """
        # save all current status into backup
        self.undo_brush_mode = self.brush_mode
        self.undo_brush_selected_pts = self.brush_selected_pts.copy()
        self.undo_brush_erased_pts = self.brush_erased_pts.copy()
        self.undo_flattened_selected_pt_ids = self.flattened_selected_pt_ids.copy()
        self.undo_flattened_erased_pt_ids = self.flattened_erased_pt_ids.copy()
        self.undo_brush_erased_pts = self.brush_erased_pts.copy()
        self.undo_selected_cell_ids = self.selected_cell_ids.copy()
        self.undo_labels = self.mesh.celldata[
            "Label"
        ].copy()  # backup label for un-do procedure
        self.undo_temp_labels = (
            self.temp_labels.copy()
        )  # backup temp_label for un-do procedure

    def undo_recover(self):
        """
        step 2 to perform undo process (ctrl + z)
        """
        # recover all variables by the undo_backup
        self.brush_mode = self.undo_brush_mode
        self.brush_selected_pts = self.undo_brush_selected_pts.copy()
        self.brush_erased_pts = self.undo_brush_erased_pts.copy()
        self.flattened_selected_pt_ids = self.undo_flattened_selected_pt_ids.copy()
        self.flattened_erased_pt_ids = self.undo_flattened_erased_pt_ids.copy()
        self.brush_erased_pts = self.undo_brush_erased_pts.copy()
        self.selected_cell_ids = self.undo_selected_cell_ids.copy()
        self.mesh.celldata["Label"] = self.undo_labels.copy()
        self.temp_labels = self.undo_temp_labels.copy()
        self.vp.show(self.mesh, resetcam=False)

    def assign_active_label_to_selection(self):
        if self.tabWidget.currentIndex() == 0:  # in segmentation mode
            self.mesh.celldata["Label"][
                self.selected_cell_ids
            ] = self.brush_active_label[0]
            self.temp_labels = self.mesh.clone().celldata[
                "Label"
            ]  # save new labels to backup labels
            self.set_mesh_color()
            self.vp.show(self.mesh, resetcam=False)

    def clean_segmentation_selection(self):
        try:
            if self.brush_ball_switch:
                self.brush_ball.__init__(
                    pos=[0, 0, 0], r=self.brush_radius, c="grey", alpha=0.0
                )
            self.vp.remove("tmp_select")
            self.vp.render(resetcam=False)

            self.brush_selected_pts = []
            self.brush_erased_pts = []
            self.flattened_selected_pt_ids = []
            self.flattened_erased_pt_ids = []
            self.selected_cell_ids = []
            self.brush_mode = False

            self.mesh.celldata["Label"] = self.temp_labels  # restore temp_labels
            self.set_mesh_color()
            self.vp.show(self.mesh, resetcam=False)
        except:
            None  # to prevent crash before loading mesh and change to One-Way Swap Method

    def segmentation_keypress(self, evt):
        if self.tabWidget.currentIndex() == 0:  # if in segmentation mode
            if evt.keypress in ["b", "B"]:
                self.brush_mode = not self.brush_mode  # toggle brush mode

                if self.brush_mode == True:
                    self.brush_selected_pts = []
                    self.flattened_selected_pt_ids = []
                    self.selected_cell_ids = []
                    self.ctrl_pressed = False
                    self.brush_erased_pts = []
                    self.flattened_erased_pt_ids = []

                if self.brush_mode == False:
                    self.clean_segmentation_selection()
                self.instruction = "Brush Mode: {}".format(self.brush_mode)
                self.statusBar().showMessage(self.instruction)

            elif evt.keypress in ["e", "E"]:  # click 'e' to execute
                if (
                    len(self.selected_cell_ids) != 0
                ):  # as long as there has a selected cell
                    # undo backup
                    self.undo_backup()

                    self.assign_active_label_to_selection()

                    # reset everything, but still in brush mode
                    self.brush_clicked = False
                    self.ctrl_pressed = False
                    self.brush_selected_pts = []
                    self.brush_erased_pts = []
                    self.flattened_selected_pt_ids = []
                    self.flattened_erased_pt_ids = []
                    self.selected_cell_ids = []
                    self.mesh.celldata[
                        "Label"
                    ] = self.temp_labels  # restore temp_labels
                    self.set_mesh_color()
                    self.vp.show(self.mesh, resetcam=False)

                    # show instruction
                    self.instruction = "Execution is completed"
                    self.statusBar().showMessage(self.instruction)

            elif evt.keypress in ["c", "C"]:  # click c to clean all selection
                # undo backup
                self.undo_backup()

                self.clean_segmentation_selection()
                self.instruction = "Clean everything"
                self.statusBar().showMessage(self.instruction)

            elif evt.keypress in ["s", "S"]:  # click s to show label
                self.caption_mode = not self.caption_mode  # toggle brush mode

                if self.caption_mode:
                    unique_labels = np.unique(self.mesh.celldata["Label"])
                    for i_label in unique_labels:
                        if i_label != 0:
                            i_seg = self.mesh.clone().threshold('Label', above=i_label-0.5, below=i_label+0.5, on='cells').alpha(0)
                            self.caption_meshes.append(i_seg)
                            if i_seg.ncells > 0:
                                i_cap = i_seg.caption(
                                    f"{int(i_label)}",
                                    point=i_seg.center_of_mass(),
                                    size=(0.3, 0.06),
                                    padding=0,
                                    font="VictorMono",
                                    alpha=1,)
                                i_cap.name = f"cap_{i_label}" # <-- this is assigned name for i_seg instead of i_cap
                                self.tooth_legend.append(i_cap)
                    
                    self.vp.add(self.tooth_legend).render()
                else:
                    self.vp.remove(self.caption_meshes)
                    self.vp.remove(self.tooth_legend)
                    self.vp.render()
                    # reset
                    self.caption_meshes = []
                    self.tooth_legend = []
                        

    def brush_onRightClick(self, evt):
        if self.tabWidget.currentIndex() == 0:  # if in segmentation mode
            if self.brush_mode == True:
                if not self.brush_clicked:
                    self.brush_clicked = (
                        True  # this will tricker mouse move call_back function
                    )
                    self.vp.show(resetcam=False)

                    # update and show brush ball (the first action when clicking)
                    p = evt.picked3d
                    if p is None:
                        return

                    tmp_pt = self.mesh.closest_point(
                        p, radius=self.brush_radius, return_point_id=True
                    )

                    tmp_cells = self.selected_pt_ids_to_cell_ids(tmp_pt)
                    if len(tmp_cells) > 0:
                        mesh_center_points = self.mesh.cell_centers()
                        tmp_cell_pts = mesh_center_points[tmp_cells]
                        tmp_pts = Points(tmp_cell_pts).c("red").pickable(False)
                        tmp_pts.name = "tmp_select"
                        self.vp.remove("tmp_select").add(tmp_pts).render()

                    if self.brush_ball_switch:
                        tmp_pt = self.mesh.closest_point(p, n=1)
                        self.brush_ball.__init__(
                            pos=tmp_pt, r=self.brush_radius, c="grey", alpha=0.3
                        )
                        self.vp.add(self.brush_ball).render()

                    # undo backup
                    self.undo_backup()

                    # show selected area
                    pt_ids = self.mesh.closest_point(
                        p, radius=self.brush_radius, return_point_id=True
                    )

                    # collect all selected pts
                    if self.ctrl_pressed == False:  # if no ctrl pressed
                        self.brush_selected_pts = list(self.flattened_selected_pt_ids)
                        for i in pt_ids:
                            if i not in self.brush_selected_pts:
                                self.brush_selected_pts.append(i)
                        self.flattened_selected_pt_ids = np.asarray(
                            self.brush_selected_pts, dtype=np.int32
                        )
                    else:  # if ctrl pressed
                        self.brush_erased_pts = list(self.flattened_erased_pt_ids)
                        for i in pt_ids:
                            if i not in self.brush_erased_pts:
                                self.brush_erased_pts.append(i)
                        self.flattened_erased_pt_ids = np.asarray(
                            self.brush_erased_pts, dtype=np.int32
                        )

                    # final selection = selected pts - erased pts
                    self.flattened_selected_pt_ids = np.asarray(
                        [
                            i
                            for i in list(self.flattened_selected_pt_ids)
                            if i not in list(self.flattened_erased_pt_ids)
                        ],
                        dtype=np.int32,
                    )

                    # convert final selected pt_ids -> selected cell_ids
                    self.selected_cell_ids = self.selected_pt_ids_to_cell_ids(
                        self.flattened_selected_pt_ids
                    )

                    self.mesh.celldata["Label"] = self.temp_labels
                    if len(self.selected_cell_ids) > 0:  # avoid IndexError
                        self.mesh.celldata["Label"][self.selected_cell_ids] = (
                            np.max(self.label_id) + 1
                        )  # assign selection as the selection color (i.e., np.max(self.label_id) + 1)

                    self.set_mesh_color()
                    self.vp.show(self.mesh, resetcam=False)

    def brush_onRightRelease(self, iren, evt):
        if self.tabWidget.currentIndex() == 0:  # if in segmentation mode
            if self.brush_mode == True:
                if self.brush_clicked:
                    # turn off brush_clicked
                    self.brush_clicked = (
                        False  # this will stop mouse move call_back function
                    )

    def brush_increase_radius(self, evt):
        if self.tabWidget.currentIndex() == 0:  # if in segmentation mode
            if self.brush_mode == True:
                if self.ctrl_pressed:
                    self.brush_radius += 0.1
                    self.vp.show(resetcam=False)

                    # show brush ball
                    p = evt.picked3d
                    if p is None:
                        return

                    tmp_pt = self.mesh.closest_point(
                        p, radius=self.brush_radius, return_point_id=True
                    )
                    tmp_cells = self.selected_pt_ids_to_cell_ids(tmp_pt)
                    if len(tmp_cells) > 0:
                        mesh_center_points = self.mesh.cell_centers()
                        tmp_cell_pts = mesh_center_points[tmp_cells]
                        tmp_pts = Points(tmp_cell_pts).c("red").pickable(False)
                        tmp_pts.name = "tmp_select"
                        self.vp.remove("tmp_select").add(tmp_pts).render()

                    if self.brush_ball_switch:
                        tmp_pt = self.mesh.closest_point(p, n=1)
                        self.brush_ball.__init__(
                            pos=tmp_pt, r=self.brush_radius, c="grey", alpha=0.3
                        )
                        self.vp.render()

    def brush_decrease_radius(self, evt):
        if self.tabWidget.currentIndex() == 0:  # if in segmentation mode
            if self.brush_mode == True:
                if self.ctrl_pressed:
                    if self.brush_radius > 0.1:
                        self.brush_radius -= 0.1
                    else:
                        self.brush_radius = 0.1

                    self.vp.show(resetcam=False)

                    # show brush ball
                    p = evt.picked3d
                    if p is None:
                        return

                    tmp_pt = self.mesh.closest_point(
                        p, radius=self.brush_radius, return_point_id=True
                    )
                    tmp_cells = self.selected_pt_ids_to_cell_ids(tmp_pt)
                    if len(tmp_cells) > 0:
                        mesh_center_points = self.mesh.cell_centers()
                        tmp_cell_pts = mesh_center_points[tmp_cells]
                        tmp_pts = Points(tmp_cell_pts).c("red").pickable(False)
                        tmp_pts.name = "tmp_select"
                        self.vp.remove("tmp_select").add(tmp_pts).render()

                    if self.brush_ball_switch:
                        tmp_pt = self.mesh.closest_point(p, n=1)
                        self.brush_ball.__init__(
                            pos=tmp_pt, r=self.brush_radius, c="grey", alpha=0.3
                        )
                        self.vp.render()


    def press_shift(self, obj, evt):
        if obj.GetShiftKey() == 1:
            self.shift_pressed = True


    def release_shift(self, obj, evt):
        if obj.GetShiftKey() == 0:
            self.shift_pressed = False


    def brush_filling(self, evt):
        if self.tabWidget.currentIndex() == 0:  # if in segmentation mode
            if self.brush_mode == True:
                if self.shift_pressed:
                    
                    # show brush ball
                    p = evt.picked3d
                    if p is None:
                        return
                    
                    print('shift pressed for filling model', p)
                    selected_filling_pt_id = self.mesh.closest_point(p, return_point_id=True)
                    selected_filling_pt_ids = [selected_filling_pt_id]

                    # initial the 1st iteration
                    selected_filling_cell_ids = []
                    tmp_selected_filling_cell_ids = []
                    i_cell_ids = self.mesh.connected_cells(selected_filling_pt_id, return_ids=True)
                    for j in i_cell_ids:
                        if self.mesh.celldata["Label"][j] == 0 and j not in selected_filling_cell_ids:
                            selected_filling_cell_ids.append(j)

                    mesh_cells = self.mesh.cells()
                    mesh_cells = np.array(mesh_cells)

                    # start the self-iteration
                    while len(selected_filling_cell_ids) != len(tmp_selected_filling_cell_ids):
                        # find the different elements between selected_filling_cell_ids and tmp_selected_filling_cell_ids
                        diff_cells = list(set(selected_filling_cell_ids) - set(tmp_selected_filling_cell_ids))
                        # clone the selected_filling_cell_ids list to tmp_selected_filling_cell_ids at the beginning of each iteration
                        tmp_selected_filling_cell_ids = selected_filling_cell_ids.copy()
                        
                        next_round_cell_pts = mesh_cells[diff_cells]
                        next_round_cell_pts = np.unique(next_round_cell_pts)
                        for i_pt in next_round_cell_pts:
                            if i_pt not in selected_filling_pt_ids:
                                selected_filling_pt_ids.append(i_pt)
                                ii_cell_ids = self.mesh.connected_cells(i_pt, return_ids=True)
                                for j in ii_cell_ids:
                                    if self.mesh.celldata["Label"][j] == 0 and j not in selected_filling_cell_ids:
                                        selected_filling_cell_ids.append(j)

                    self.selected_cell_ids = selected_filling_cell_ids
                    # selected_filling_cell_ids is the result of filling
                    self.mesh.celldata["Label"] = self.temp_labels
                    if len(self.selected_cell_ids) > 0:  # avoid IndexError
                        self.mesh.celldata["Label"][self.selected_cell_ids] = (
                            np.max(self.label_id) + 1
                        )  # assign selection as the selection color (i.e., np.max(self.label_id) + 1)

                    self.set_mesh_color()
                    self.vp.show(self.mesh, resetcam=False)


    def press_ctrl(self, obj, evt):
        if obj.GetControlKey() == 1:
            self.ctrl_pressed = True
            if self.tabWidget.currentIndex() == 0:  # if in segmentation mode
                # reset brush erased pts
                self.brush_erased_pts = []
                self.flattened_erased_pt_ids = []
                # un-do process, when ctrl + z is pressed
                if obj.GetKeySym() in ["z", "Z"]:  
                    self.undo_recover()


    def release_ctrl(self, obj, evt):
        if obj.GetControlKey() == 0:
            self.ctrl_pressed = False
            if self.tabWidget.currentIndex() == 0:  # if in segmentation mode
                # reset brush erased pts (important! cannot remove!)
                self.brush_erased_pts = []
                self.flattened_erased_pt_ids = []


    def brush_dragging(self, evt):
        if self.brush_mode:
            if self.brush_clicked:
                p = evt.picked3d
                if p is None:
                    return

                # update and show brush ball
                tmp_pt = self.mesh.closest_point(
                    p, radius=self.brush_radius, return_point_id=True
                )
                tmp_cells = self.selected_pt_ids_to_cell_ids(tmp_pt)
                if len(tmp_cells) > 0:
                    mesh_center_points = self.mesh.cell_centers()
                    tmp_cell_pts = mesh_center_points[tmp_cells]
                    tmp_pts = Points(tmp_cell_pts).c("red").pickable(False)
                    tmp_pts.name = "tmp_select"
                    self.vp.remove("tmp_select").add(tmp_pts).render()

                if self.brush_ball_switch:
                    tmp_pt = self.mesh.closest_point(p, n=1)
                    self.brush_ball.__init__(
                        pos=tmp_pt, r=self.brush_radius, c="grey", alpha=0.3
                    )
                    self.vp.add(self.brush_ball).render()

                # show selected area
                pt_ids = self.mesh.closest_point(
                    p, radius=self.brush_radius, return_point_id=True
                )

                # collect all selected pts
                if self.ctrl_pressed == False:  # if no ctrl pressed
                    self.brush_selected_pts = list(self.flattened_selected_pt_ids)
                    for i in pt_ids:
                        if i not in self.brush_selected_pts:
                            self.brush_selected_pts.append(i)
                    self.flattened_selected_pt_ids = np.asarray(
                        self.brush_selected_pts, dtype=np.int32
                    )
                else:  # if ctrl pressed
                    self.brush_erased_pts = list(self.flattened_erased_pt_ids)
                    for i in pt_ids:
                        if i not in self.brush_erased_pts:
                            self.brush_erased_pts.append(i)
                    self.flattened_erased_pt_ids = np.asarray(
                        self.brush_erased_pts, dtype=np.int32
                    )

                # final selection = selected pts - erased pts
                self.flattened_selected_pt_ids = np.asarray(
                    [
                        i
                        for i in list(self.flattened_selected_pt_ids)
                        if i not in list(self.flattened_erased_pt_ids)
                    ],
                    dtype=np.int32,
                )

                # convert final selected pt_ids -> selected cell_ids
                self.selected_cell_ids = self.selected_pt_ids_to_cell_ids(
                    self.flattened_selected_pt_ids
                )

                self.mesh.celldata["Label"] = self.temp_labels
                if len(self.selected_cell_ids) > 0:  # avoid IndexError
                    self.mesh.celldata["Label"][self.selected_cell_ids] = (
                        np.max(self.label_id) + 1
                    )  # assign selection as the selection color (i.e., np.max(self.label_id) + 1)

                self.set_mesh_color()
                self.vp.show(self.mesh, resetcam=False)

            else:
                p = evt.picked3d
                if p is None:
                    return
                # show and update the brush ball (transparent ball)
                tmp_pt = self.mesh.closest_point(
                    p, radius=self.brush_radius, return_point_id=True
                )
                tmp_cells = self.selected_pt_ids_to_cell_ids(tmp_pt)
                if len(tmp_cells) > 0:
                    mesh_center_points = self.mesh.cell_centers()
                    tmp_cell_pts = mesh_center_points[tmp_cells]
                    tmp_pts = Points(tmp_cell_pts).c("red").pickable(False)
                    tmp_pts.name = "tmp_select"
                    self.vp.remove("tmp_select").add(tmp_pts).render()

                if self.brush_ball_switch:
                    tmp_pt = self.mesh.closest_point(p, n=1)
                    self.brush_ball.__init__(
                        pos=tmp_pt, r=self.brush_radius, c="grey", alpha=0.3
                    )
                    self.vp.add(self.brush_ball).render()


    def shortcut_function_toggle_wireframe(self):
        try:
            if self.mesh_exist:
                self.mesh_wireframe_show = not self.mesh_wireframe_show
                if self.mesh_wireframe_show == True:
                    self.mesh.lw(0.1)
                else:
                    self.mesh.lw(0)
                self.vp.render(resetcam=False)
        except:
            None

    def shortcut_function_fcsv_save(self):
        # print('ctrl+s pressed')
        self.save_landmarking()


    def shortcut_function_landmark_relocation(self):
        # print('ctrl+r pressed')
        self.relocate_landmark()

    def shortcut_function_select_right_landmark(self):
        if self.tabWidget.currentIndex() == 2: # in landmarking mode
            # self.landmark_order_move_right()
            idx = self.tableWidget_landmakring.selectedIndexes()

            if idx:
                # Assuming single selection, get the first (and only) selected item
                current_index = idx[0]
                col = current_index.column()

                # Check if the right column is within the range of the table
                if col < self.tableWidget_landmakring.columnCount() - 1:
                    right_col = col + 1
                    # Set the new selection
                    self.tableWidget_landmakring.setCurrentCell(0, right_col) # only 1 row, so row = 0
            else: # no selection
                self.tableWidget_landmakring.setCurrentCell(0, self.tableWidget_landmakring.columnCount()-1) # only 1 row, so row = 0

    def shortcut_function_select_left_landmark(self):
        if self.tabWidget.currentIndex() == 2: # in landmarking mode
            # self.landmark_order_move_left()
            idx = self.tableWidget_landmakring.selectedIndexes()

            if idx:
                # Assuming single selection, get the first (and only) selected item
                current_index = idx[0]
                col = current_index.column()

                # Check if the left column is within the range of the table
                if col > 0:
                    left_col = col - 1
                    # Set the new selection
                    self.tableWidget_landmakring.setCurrentCell(0, left_col) # only 1 row, so row = 0
            else: # no selection
                self.tableWidget_landmakring.setCurrentCell(0, 0) # only 1 row, so row = 0

    ########################
    # slot function(s)
    ########################
    @Qt.pyqtSlot()
    def spin_R_changed(self):
        self.landmark_radius = self.doubleSpinBox_R.value()

        # remove old landmarks
        self.vp.remove(self.vedo_landmarks)

        # create new landmarks based on new radius
        self.vedo_landmarks = []
        for i_landmark in range(len(self.landmarks)):
            self.vedo_landmarks.append(
                Sphere(self.landmarks[i_landmark], r=self.landmark_radius).c(
                    self.landmark_color
                )
            )
        self.vp.add(self.vedo_landmarks)
        self.vp.render()

        # recover selection
        # show selection
        if self.selected_lmk_idx != None:
            self.vedo_landmarks[self.selected_lmk_idx].c(self.selected_landmark_color)
            self.vp.show()

    @Qt.pyqtSlot()
    def brush_active_label_changed(self):
        if self.spinBox_brush_active_label.value() in self.label_id:
            self.brush_active_label = [
                self.spinBox_brush_active_label.value()
            ]  # update spline active label
            self.vtkWidget.setFocus()
        else:
            self.show_messageBox('Label ID does not exist!')
            # set the value back to the previous one
            self.spinBox_brush_active_label.setValue(self.brush_active_label[0])

    @Qt.pyqtSlot()
    def swap_original_label_changed(self):
        if self.spinBox_swap_original_label.value() in self.label_id:
            self.swap_original_label = [
                self.spinBox_swap_original_label.value()
            ]  # update swap original label
        else:
            self.show_messageBox('Label ID does not exist!')
            # set the value back to the previous one
            self.spinBox_swap_original_label.setValue(self.swap_original_label[0])

    @Qt.pyqtSlot()
    def swap_new_label_changed(self):
        if self.spinBox_swap_new_label.value() in self.label_id:
            self.swap_new_label = [
                self.spinBox_swap_new_label.value()
            ]  # update swap new label
        else:
            self.show_messageBox('Label ID does not exist!')
            # set the value back to the previous one
            self.spinBox_swap_new_label.setValue(self.swap_new_label[0])

    @Qt.pyqtSlot()
    def swap_assign_new_label(self):
        self.mesh.celldata["Label"][
            self.mesh.celldata["Label"] == self.swap_original_label[0]
        ] = self.swap_new_label[0]
        self.temp_labels = self.mesh.clone().celldata[
            "Label"
        ]  # save new labels to backup labels
        self.set_mesh_color()
        self.vp.show(self.mesh, resetcam=False)
        self.instruction = "Change is completed"
        self.statusBar().showMessage(self.instruction)
        self.vtkWidget.setFocus()

    @Qt.pyqtSlot()
    def show_tab_info(self):
        if (
            self.tabWidget.currentIndex() == 0
        ):  # show segmentation insturction if in tab_segmentation
            self.instruction = 'Welcome to Segmentation Method. Press "b" to enter/exit brush Mode; press "c" to clean selection; and press "e" to execute change.'
            self.statusBar().showMessage(self.instruction)

            # turn off landmarks if exist
            self.vp.remove(self.vedo_landmarks)
            self.vp.render()

            self.vtkWidget.setFocus()

        elif self.tabWidget.currentIndex() == 1:  # show swap insturction if in tab_swap
            self.instruction = 'Welcome to One-Way Swap Method. Given "Original label" and "New label", then click "Change!" All cells with "Original label" will be replaced by "New label"'
            self.statusBar().showMessage(self.instruction)

            # reset variables in segmentation method
            self.clean_segmentation_selection()

            # turn off landmarks if exist
            self.vp.remove(self.vedo_landmarks)
            self.vp.render()

            self.vtkWidget.setFocus()

        # new in v3.0
        elif (
            self.tabWidget.currentIndex() == 2
        ):  # show landmarking insturction if in tab_landmarking
            self.instruction = "Welcome to landmarking mode. Please create/load new fcsv file for landmark identification"
            self.statusBar().showMessage(self.instruction)

            # reset variables in segmentation method
            self.clean_segmentation_selection()

            # show landmarks
            self.vp.add(self.vedo_landmarks)
            self.vp.render()

            self.vtkWidget.setFocus()

    @Qt.pyqtSlot()
    def load_mesh(self):
        self.opened_mesh_path, _ = QFileDialog.getOpenFileName(
            None, "Open File", self.opened_mesh_path, "Mesh Files (*.vtp *.stl *.obj *.ply)"
        )

        try:
            if self.opened_mesh_path[-4:].lower() in [
                ".vtp",
                ".stl",
                ".obj",
                ".ply",
            ]:  # check file extension
                self.reset_plotters()
                self.plot_mesh()
                self.existed_opened_mesh_path = self.opened_mesh_path
                self.setWindowTitle(
                    "Mesh Labeler (open source) -- {}".format(
                        self.existed_opened_mesh_path
                    )
                )
                self.vtkWidget.setFocus()
        except:
            None  # it won't happen, because we only allow to load four types of mesh file

    @Qt.pyqtSlot()
    def save_mesh(self):
        try:
            if self.mesh_exist:
                # clean labeling selection
                self.clean_segmentation_selection()

                if self.comboBox_save_type.currentText() == "VTP":
                    self.save_data_path, _ = QFileDialog.getSaveFileName(
                        None, "Save File", self.existed_opened_mesh_path[:-4], "*.vtp"
                    )
                    if self.save_data_path:  # not empty
                        # remove unnecessary arraies
                        self.saved_mesh = self.mesh.clone()
                        # self.saved_mesh.polydata().GetPointData().RemoveArray('Normals')
                        # self.saved_mesh.polydata().GetCellData().RemoveArray('Normals')
                        # if self.existed_opened_mesh_path[-4:] == '.obj':
                        #     self.saved_mesh.polydata().GetCellData().RemoveArray('GroupIds')
                        # if self.existed_opened_mesh_path[-4:] == '.ply':
                        #     self.saved_mesh.polydata().GetPointData().RemoveArray('RGBA')
                        file_io.write(self.saved_mesh, self.save_data_path, binary=True)
                        self.existed_opened_mesh_path = self.save_data_path
                        self.setWindowTitle(
                            "Mesh Labeler (open source) -- {}".format(
                                self.save_data_path
                            )
                        )
                        # reset, don't call reset_plotters() becuase we don't want to delete current plotter
                        self.brush_mode = False
                        self.selected_cell_ids = []

                elif self.comboBox_save_type.currentText() in ["STL", "OBJ"]:
                    self.save_data_path, _ = QFileDialog.getSaveFileName(
                        None, "Save File", self.existed_opened_mesh_path[:-4]
                    )
                    if self.save_data_path:  # not empty
                        # remove unnecessary arraies
                        self.saved_mesh = self.mesh.clone()
                        # self.saved_mesh.polydata().GetPointData().RemoveArray('Normals')
                        # self.saved_mesh.polydata().GetCellData().RemoveArray('Normals')
                        # if self.existed_opened_mesh_path[-4:] == '.obj':
                        #     self.saved_mesh.polydata().GetCellData().RemoveArray('GroupIds')
                        # if self.existed_opened_mesh_path[-4:] == '.ply':
                        #     self.saved_mesh.polydata().GetPointData().RemoveArray('RGBA')

                        # iterate all single label
                        label_classes = np.unique(
                            self.saved_mesh.celldata["Label"]
                        ).astype(np.int32)
                        for i_label in label_classes:
                            lb = i_label - 0.5
                            ub = i_label + 0.5
                            tmp_mesh = self.saved_mesh.clone()
                            tmp_mesh = tmp_mesh.threshold(
                                "Label", above=lb, below=ub, on="cells"
                            )
                            file_io.write(
                                tmp_mesh,
                                self.save_data_path
                                + "_label_{}.{}".format(
                                    i_label,
                                    self.comboBox_save_type.currentText().lower(),
                                ),
                                binary=True,
                            )
                        self.existed_opened_mesh_path = self.save_data_path
                        self.setWindowTitle(
                            "Mesh Labeler (open source) -- {}_label_*.{}".format(
                                self.save_data_path,
                                self.comboBox_save_type.currentText().lower(),
                            )
                        )
                        # reset, don't call reset_plotters() becuase we don't want to delete current plotter
                        self.brush_mode = False
                        self.selected_cell_ids = []

                # update status in statusBar
                self.statusBar().showMessage("File(s) saved")
                self.vtkWidget.setFocus()
        except:
            self.show_messageBox("No mesh available! Please load a mesh first!")

    @Qt.pyqtSlot()
    def load_landmarking(self):
        if self.mesh_exist:  # only enable when mesh exists
            self.opened_landmarking_path, _ = QFileDialog.getOpenFileName(
                None, "Open File", self.opened_landmarking_path, "*.fcsv"
            )

            if self.opened_landmarking_path[-5:].lower() in [
                ".fcsv"
            ]:  # check file extension
                # read fcsv file
                self.load_fcsv()

                if self.landmark_exist:  # fcsv load successfullly
                    self.existed_opened_landmarking_path = self.opened_landmarking_path
                    # display landmarking info on tableWidget
                    self.display_landmark_table()
                    self.tableWidget_landmakring.setCurrentCell(0, 0) # deafult select the first ladnmark

                    # plot landmarks
                    self.vp.remove(
                        self.vedo_landmarks
                    )  # clean previous landmarks if existing
                    self.vedo_landmarks = []
                    for i_landmark in range(len(self.landmarks)):
                        # self.vedo_landmarks.append(Point(self.landmarks[i_landmark], r=self.landmark_radius).c(self.landmark_color))
                        self.vedo_landmarks.append(
                            Sphere(
                                self.landmarks[i_landmark], r=self.landmark_radius
                            ).c(self.landmark_color)
                        )
                    self.vp.add(self.vedo_landmarks)

                    self.setWindowTitle(
                        "Mesh Labeler (open source) -- {} -- {}".format(
                            self.existed_opened_mesh_path,
                            self.existed_opened_landmarking_path,
                        )
                    )
                    self.vtkWidget.setFocus()
        else:
            self.show_messageBox("No mesh available! Please load a mesh first!")

    @Qt.pyqtSlot()
    def save_landmarking(self):
        if self.mesh_exist:  # only enable when mesh exists
            if self.landmark_exist:  # only enable when landmark exists
                self.save_landmarking_data_path, _ = QFileDialog.getSaveFileName(
                    None,
                    "Save File",
                    self.existed_opened_landmarking_path[:-5],
                    "*.fcsv",
                )
                if self.save_landmarking_data_path:  # not empty
                    # write fcsv file
                    with open(self.save_landmarking_data_path, "w") as file:
                        file.write("# Markups fiducial file version = 4.10\n")
                        file.write("# CoordinateSystem = RAS\n")
                        file.write(
                            "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n"
                        )
                        for i in range(len(self.landmarks)):
                            file.write(
                                "vtkMRMLMarkupsFiducialNode_{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},,{12}\n".format(
                                    i,
                                    self.landmarks[i, 0],
                                    self.landmarks[i, 1],
                                    self.landmarks[i, 2],
                                    0.0,
                                    0.0,
                                    0.0,
                                    1.0,
                                    1,
                                    1,
                                    1,
                                    self.landmark_names[i][0],
                                    "vtkMRMLModelNode4",
                                )
                            )

                    self.existed_opened_landmarking_path = (
                        self.save_landmarking_data_path
                    )
                    self.setWindowTitle(
                        "Mesh Labeler (open source) -- {} -- {}".format(
                            self.existed_opened_mesh_path,
                            self.save_landmarking_data_path,
                        )
                    )
                    self.statusBar().showMessage("Landmarking file saved")
                    self.vtkWidget.setFocus()
            else:
                self.show_messageBox(
                    "No landmark available! Please create/load a landmark first!"
                )
        else:
            self.show_messageBox("No mesh available! Please load a mesh first!")

    def show_messageBox(self, shown_str):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Critical)
        msgBox.setText(shown_str)
        msgBox.setWindowTitle("Error")
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.exec()


def main():
    app = Qt.QApplication(sys.argv)
    window = Mesh_Labeler()
    app.aboutToQuit.connect(window.onClose)  # <-- connect the onClose event
    app.exec_()


if __name__ == "__main__":
    main()
