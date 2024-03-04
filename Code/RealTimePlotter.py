from PyQt6 import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from PyQt6.QtGui import QIcon, QAction

class MainWindow(QtWidgets.QMainWindow):
    # The class for plotting real-time data

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        # setting the initial geometry of window 
        self.setGeometry(1025, 25, 500, 380) 
        self.setWindowTitle("Pressure")
        # plot
        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)
        
        # setting the axis label
        self.graphWidget.setLabel(
            "left",
            '<span style="color: black; font-size: 18px">Pressure</span>'
        )
        self.graphWidget.setLabel(
            "bottom",
            '<span style="color: black; font-size: 18px">Time</span>'
        )

        self.x = list(range(100))  # 100 time points
        self.y = [0 for _ in range(100)]  # 100 data points

        self.graphWidget.setBackground('w')

        pen = pg.mkPen(color=(255, 0, 0), width=3)
        self.data_line =  self.graphWidget.plot(self.x, self.y, pen=pen)
    
    def set_x_y_size(self, x_left, y_top, width, height):  # location and dimension
        self.setGeometry(x_left, y_top, width, height)

    def set_title(self, title):  # title
        self.graphWidget.setTitle(title, color="k", size="20pt")

    def update_plot_data(self, data):

        self.x = self.x[1:]  # Remove the first y element.
        self.x.append(self.x[-1] + 1)  # Add a new value 1 higher than the last.

        self.y = self.y[1:]  # Remove the first
        self.y.append(data)  # Add a new recent value.

        self.data_line.setData(self.x, self.y)  # Update the data.

    def pressure_diff(self):
        return self.y[-1]-self.y[-2]  # difference between the current data point and the previous 

class MainWindow_wo_x_lim(QtWidgets.QMainWindow):
    # The class for plotting stress-strain data

    def __init__(self, *args, **kwargs):
        super(MainWindow_wo_x_lim, self).__init__(*args, **kwargs)

        self.init_UI()
        self.init_variables()
        self.init_plot_style()
        
    def init_UI(self):
        # Clear
        clearAct = QAction(QIcon('exit.png'), '&Clear', self)
        clearAct.setShortcut('Ctrl+C')
        clearAct.setStatusTip('Clear data point')
        clearAct.triggered.connect(self.clear_data_points)

        self.statusBar()

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&Tool(s)')
        fileMenu.addAction(clearAct)

        # plot
        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)

        # setting the initial geometry of window 
        self.setGeometry(1025, 435, 500, 380)
        self.setWindowTitle('Stress-Strain Plot')
        self.show()

    def init_variables(self):
        # initialize list
        self.x = []  
        self.y = []  
        self.x_temp = []
        self.y_temp = []
        self.x_temp_log_LR_all = []
        self.y_log_LR_all = []
        self.x_temp_log_LR = []
        self.y_log_LR = []
        self.x_temp_LR_all = []
        self.y_LR_all = []
        self.x_temp_LR = []
        self.y_LR = []
        self.m_LR = 0

    def init_plot_style(self):
        self.graphWidget.setBackground('w')  # Background color
        self.graphWidget.addLegend()  # legend
        # setting the axis label
        self.graphWidget.setLabel("left", '<span style="color: black; font-size: 18px">Stress</span>')
        self.graphWidget.setLabel("bottom", '<span style="color: black; font-size: 18px">Strain</span>')
        
        # Data points
        pen = pg.mkPen(color=(255, 255, 255), width=3)  # line color to white --> invisible
        self.data_line =  self.graphWidget.plot(self.x, self.y, pen=pen, symbol="o", symbolSize=5, symbolBrush="k", name="Data point(s)",)
        # log LR 
        # pen = pg.mkPen(color=(255, 0, 0), width=3)
        # self.data_line_linear_log_reg_all = self.graphWidget.plot(self.x_temp_log_LR_all, self.y_log_LR_all, pen=pen, name="log LR (overall)",)
        pen = pg.mkPen(color=(0, 0, 255), width=3)
        self.data_line_linear_log_reg = self.graphWidget.plot(self.x_temp_log_LR, self.y_log_LR, pen=pen, name="log LR (each press)",)
        # # LR 
        # pen = pg.mkPen(color=(255, 0, 0), width=3)
        # self.data_line_linear_reg_all = self.graphWidget.plot(self.x_temp_LR_all, self.y_LR_all, pen=pen, name="LR (overall)",)
        pen = pg.mkPen(color=(255, 0, 0), width=3)
        self.data_line_linear_reg = self.graphWidget.plot(self.x_temp_LR, self.y_LR, pen=pen, name="LR (each press)",)

    def set_x_y_size(self, x_left, y_top, width, height):  # location and dimension
        self.setGeometry(x_left, y_top, width, height)

    def set_title(self, title):  # title 
        self.graphWidget.setTitle(title, color="k", size="20pt")

    def update_plot_data(self, x, y):
        self.x.append(x)  # Add a new recent value.
        self.y.append(y)  # Add a new recent value.
        self.x_temp.append(x)
        self.y_temp.append(y)
        self.data_line.setData(self.x, self.y)  # Update the data.

    def return_data_each_press(self):
        return self.x_temp, self.y_temp

    def regression_each_press(self, model='LR', plot_bool=False, clear_bool=True, return_stiff_bool=True):
        if len(self.x_temp)!=0 and len(self.y_temp)!=0:
            X_train = np.array(self.x_temp)
            y_train = np.array(self.y_temp)
            zero_idx = np.where(X_train==0)
            X_train = np.delete(X_train, zero_idx)
            y_train = np.ravel(np.delete(y_train, zero_idx))
            if len(X_train)!=0 and len(y_train)!=0:
                X_test = np.linspace(np.min(X_train)/2,np.max(X_train),50)
                
                # Linear regression
                if model == 'LR':
                    self.m_LR, b = np.polyfit(X_train, y_train, 1)
                    y_pred = self.m_LR*X_test+b
                    if plot_bool:
                            self.data_line_linear_reg.setData(X_test.reshape(-1,).tolist(), y_pred.reshape(-1,).tolist())  # Update the data.
                
                # Log Linear regression
                if model == 'logLR':
                    self.m_LR, b = np.polyfit(np.log(X_train), y_train, 1)
                    y_pred = self.m_LR*np.log(X_test)+b
                    if plot_bool:
                            self.data_line_linear_log_reg.setData(X_test.reshape(-1,).tolist(), y_pred.reshape(-1,).tolist())  # Update the data.

        # clear cache 
        if clear_bool:
            self.x_temp = []
            self.y_temp = []    
        if return_stiff_bool:
            return self.m_LR
    # def regression_all(self, model='LR', plot_bool=False):
        # if len(self.x)!=0 and len(self.y)!=0:
        #     X_train = np.array(self.x)
        #     y_train = np.array(self.y)
        #     zero_idx = np.where(X_train==0)
        #     X_train = np.delete(X_train, zero_idx).reshape(-1, 1)
        #     y_train = np.ravel(np.delete(y_train, zero_idx))
        #     if len(X_train)!=0 and len(y_train)!=0:
        #         X_test = np.linspace(np.min(X_train)/2,np.max(X_train),50).reshape(-1, 1)
                
        #         # linear log regression
        #         if model == 'logLR':
        #             LR_log_all = LinearRegression()
        #             LR_log_all.fit(np.log(X_train), y_train)
        #             y_pred = LR_log_all.predict(np.log(X_test))
        #             if plot_bool:
        #                 self.data_line_linear_log_reg_all.setData(X_test.reshape(-1,).tolist(), y_pred.reshape(-1,).tolist())  # Update the data.
                
        #         # linear regression
        #         if model == 'LR':
        #             LR_all = LinearRegression()
        #             LR_all.fit(X_train, y_train)
        #             y_pred = LR_all.predict(X_test)
        #             stiff = LR_all.coef_
        #             if plot_bool:
        #                 self.data_line_linear_reg_all.setData(X_test.reshape(-1,).tolist(), y_pred.reshape(-1,).tolist())  # Update the data.
        #             return stiff
                
        # return 0
    
    def clear_data_points(self):
        self.x = []
        self.y = []
        self.x_temp = []
        self.y_temp = []
        self.data_line.setData(self.x, self.y)