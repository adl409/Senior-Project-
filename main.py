import sys
import os
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6 import QtCore
from eegBuilderBackend import *


from layerClasses import *

# Defaults for 
defaults = {
    "Standard Neural Network": [Flatten(0),Dense(0, 64), Dense(0,64,"relu")],
    "Convolutional NN": [Convolution_2d(0,filters=16, kernel_size=(3,3)), Max_Pool_2d(0,pool_size=(2,2)),Convolution_2d(0,filters=32, kernel_size=3), Max_Pool_2d(0,pool_size=(2,2)),Convolution_2d(0,filters=64, kernel_size=3), Max_Pool_2d(0,pool_size=(2,2)), Flatten(0), Dense(0,64), Dense(0,32) ],
    "Recurrent NN": [SimpleRNN(0,16)]
}

# Mapping for Parameters
mapping = {
    None: "None",
    True: "True",
    False: "False",
    "glorot_normal": "glorot normal",
    "glorot_uniform": "glorot uniform",
    "zeros": "zeros",
    "orthogonal": "orthogonal",
    "elu": "elu",
    "exponential": "exponential",
    "gelu": "gelu",
    "linear": "linear",
    "relu": "relu",
    "relu6": "relu6",
    "leaky_relu": "leaky_relu",
    "mish": "mish",
    "selu": "selu",
    "sigmoid": "sigmoid",
    "hard_sigmoid": "hard sigmoid",
    "silu": "silu",
    "hard_silu": "hard silu",
    "softmax": "softmax",
    "softplus": "softplus",
    "tanh": "tanh",
    "valid": "valid",
    "same": "same"
}

# Dictionary that hold configuration values and will eventually be passed to the backend
class neuralnetModel(QtCore.QAbstractListModel):
    def __init__(self, *args, data=None, **kwargs):
        super(neuralnetModel, self).__init__(*args, **kwargs)
        self.datadict = {
            "input_parameters": {
                "input_shape1" : 0,
                "input_shape2" : 0,
                "root_directory" : "",
                "normalized" : False
            },
            "test_parameters": {
                "test_split" : 0.20,
                "epochs" : 10,
                "shuffle" : True
            },
            "model_1" : {
                "save_model" : True, 
                "save_file" : "",
                "active": True,
                "layers": []
            }
        }

    def data(self, index, role):
        if role == QtCore.Qt.DisplayRole:
            text = str(index.row()+1) + ". " + self.datadict.get("model_1").get("layers")[index.row()].displayName
            return text

    def rowCount(self, index):
        return len(self.datadict.get("model_1").get("layers"))
    
nnet = neuralnetModel()
index = -1

class TestConfigDialog(QDialog):

    def onValueChanged(self, value):
        # Update the label text to display the percentage
        self.label.setText(f'Data Testing Split: {value}%')
        #keeps selection to only ticks
        aligned_value = round(value / 5) * 5
        self.slider.setValue(aligned_value)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Testing")

        layout = QVBoxLayout(self)

        #Epoch testing
        self.epoch_group = QGroupBox("\u24D8 Epochs")
        self.epoch_group.setToolTip("Number of times the full training data set is passed to the model for training")
        self.epoch_group.setMaximumWidth(70)
        self.epoch_group.setMinimumWidth(50)
        self.epoch_layout = QHBoxLayout(self.epoch_group)
        self.epoch_input = QLineEdit()
        self.epoch_input.setText(str(nnet.datadict["test_parameters"]["epochs"]))
        self.epoch_input.setMaximumWidth(50)
        self.epoch_input.setMinimumWidth(50)

        validator = QRegularExpressionValidator(QRegularExpression("[0-9]{0,4}"))
        self.epoch_input.setValidator(validator)



        self.epoch_layout.addWidget(self.epoch_input)

        layout.addWidget(self.epoch_group)

        #Shuffling
        self.checkbox = QCheckBox("\u24D8 Shuffle")
        self.checkbox.setToolTip("Should the data be shuffled?")
        layout.addWidget(self.checkbox)

        if nnet.datadict["test_parameters"]["shuffle"]:
            self.checkbox.setChecked(True)

        self.checkbox2 = QCheckBox("\u24D8 Save File")
        self.checkbox2.setToolTip("Do you want to save the model as a file?")
        layout.addWidget(self.checkbox2)
        if nnet.datadict["model_1"]["save_model"]:
            self.checkbox2.setChecked(True)

        self.label2 = QLabel('\u24D8 File Name')
        self.label2.setToolTip("If save file is checked, insert a name for the file.")
        self.model_input = QLineEdit()
        self.model_input.setText(nnet.datadict["model_1"]["save_file"])
        layout.addWidget(self.label2)
        layout.addWidget(self.model_input)

        #slider for testing split

        labelInfo = '\u24D8 Data Testing Split: ' + str(int(nnet.datadict["test_parameters"]["test_split"]*100)) + "%"

        self.label = QLabel(labelInfo)
        self.label.setToolTip("Percentage of data used for testing (not training)")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(20)
        self.slider.setMaximum(80)
        self.slider.setValue(int(nnet.datadict["test_parameters"]["test_split"]*100))
        self.slider.valueChanged.connect(self.onValueChanged)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.setTickInterval(5)


        layout.addWidget(self.slider)

        #Dialog buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Save)
        layout.addWidget(self.buttons)

        #Connect buttons
        self.buttons.accepted.connect(self.saveConfig)
        self.buttons.rejected.connect(self.reject)

        self.setMinimumSize(300, 100)

    def saveConfig(self):

        nnet.datadict["test_parameters"]["test_split"] = self.slider.value() / 100
        nnet.datadict["test_parameters"]["epochs"] = int(float(self.epoch_input.text()))
        nnet.datadict["test_parameters"]["shuffle"] = self.checkbox.isChecked()
        nnet.datadict["model_1"]["save_model"] = self.checkbox2.isChecked()
        nnet.datadict["model_1"]["save_file"] = str(self.model_input.text())
        
        self.accept()

class ConfigureAddLayerDialog(QDialog):
    def __init__(self, layer_type, layer_location, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Layer")

        self.layer_type = layer_type  # Store the selected layer type
        self.config = []              # Store the configuration for the Layer
        self.layer_location = layer_location

        layout = QVBoxLayout(self)

        # Create and customize widgets based on layer type
        if self.layer_type =="none":
            return None
        elif self.layer_type == "Dense":
            self.configureDenseLayer(layout)
        elif self.layer_type == "Flatten":
            self.configureFlattenLayer(layout)
        elif self.layer_type == "Zero Padding 2d":
            self.configureZeroPadding2dLayer(layout)
        elif self.layer_type == "Average Pooling 2d":
            self.configureAveragePooling2dLayer(layout)
        elif self.layer_type == "Max Pooling 2d":
            self.configureMaxPooling2dLayer(layout)
        elif self.layer_type == "Convolution 2d":
            self.configureConvolution2dLayer(layout)
        elif self.layer_type == "Convolution 2d Transpose":
            self.configureConvolution2dTransposeLayer(layout)
        elif self.layer_type == "Simple RNN":
            self.configureSimpleRNNLayer(layout)
        elif self.layer_type == "LSTM":
            self.configureLSTMLayer(layout)
        elif self.layer_type == "GRU":
            self.configureGRULayer(layout)
        elif self.layer_type == "Dropout":
            self.configureDropoutLayer(layout)
        # Add conditions for other layer types

        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Save)
        layout.addWidget(buttons)

        buttons.accepted.connect(self.saveLayer)    # Saving configuration function
        buttons.rejected.connect(self.reject)
        
    def configureDropoutLayer(self, layout):
        
        dropout_desc_label = QLabel("Dropout - Layer for applying dropout regularization")
        layout.addWidget(dropout_desc_label)
        
        # Validator for rate text box input (regex for float between 0 and 1)
        rate_validator = QRegularExpressionValidator(QRegularExpression(r'^0(\.\d+)?|1(\.0+)?$'))

        rate_label = QLabel("\u24D8 Rate:")
        rate_label.setToolTip("Float between 0 and 1, fraction of the input units to drop")

        # Add text box for rate
        dropout_rate = QLineEdit()
        dropout_rate.setPlaceholderText("Rate")
        dropout_rate.setValidator(rate_validator)
        dropout_rate.setStyleSheet("border-style: outset;border-width: 2px;")
        layout.addWidget(rate_label)
        layout.addWidget(dropout_rate)

        self.config.append(dropout_rate)

        # Validator for seed text box input (regex for integers)
        seed_validator = QRegularExpressionValidator(QRegularExpression(r'^\d+$'))

        seed_label = QLabel("\u24D8 Seed:")
        seed_label.setToolTip("Integer, random seed for reproducibility")

        # Add text box for seed
        dropout_seed = QLineEdit()
        dropout_seed.setPlaceholderText("Seed")
        dropout_seed.setValidator(seed_validator)
        dropout_seed.setStyleSheet("border-style: outset;border-width: 2px;")
        layout.addWidget(seed_label)
        layout.addWidget(dropout_seed)

        self.config.append(dropout_seed)

        # Add widgets specific to configuring Dropout layer
        pass

        
    
    def configureDenseLayer(self, layout):

        dense_desc_label = QLabel("Dense - 1D layer of fully connected neurons")
        layout.addWidget(dense_desc_label)
        
        #validator for text box input (regex)
        validator = QRegularExpressionValidator(QRegularExpression(r'^\d{1,4}$'))

        units_label = QLabel("\u24D8 Units:")
        units_label.setToolTip("Positive integer, dimensionality of the output space")

        #add text box
        dense_units = QLineEdit()
        dense_units.setPlaceholderText("Units")
        dense_units.setValidator(validator)
        dense_units.setStyleSheet("border-style: outset;border-width: 2px;")
        layout.addWidget(units_label)
        layout.addWidget(dense_units)

        self.config.append(dense_units)

        #activation dropdown
        activation_label = QLabel("\u24D8 Activation Type:")
        activation_label.setToolTip("What activation function to use. If you picked None, no activation is applied")
        layout.addWidget(activation_label)
        dense_activation = QComboBox()
        dense_activation.addItems(["None","elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "sigmoid", "hard sigmoid", "silu", "hard silu", "softmax", "softplus", "tanh"])
        layout.addWidget(dense_activation)

        self.config.append(dense_activation)

        #use bias dropdown
        bias_label = QLabel("\u24D8 Use Bias:")
        bias_label.setToolTip("Boolean, determines whether the layer uses a bias vector")
        layout.addWidget(bias_label)
        use_bias = QComboBox()
        use_bias.addItems(["True", "False"])
        layout.addWidget(use_bias)

        self.config.append(use_bias)

        #kernel initializer
        k_initializer_label = QLabel("\u24D8 Kernel Initializer:")
        k_initializer_label.setToolTip("Initializer for the kernel weights matrix")
        layout.addWidget(k_initializer_label)
        kernel_initializer = QComboBox()
        kernel_initializer.addItems(["zeros", "glorot normal", "glorot uniform"])
        layout.addWidget(kernel_initializer)

        self.config.append(kernel_initializer)

        #bias initializer
        b_initializer_label = QLabel("\u24D8 Bias Initializer:")
        b_initializer_label.setToolTip("Initializer for the bias vector")
        layout.addWidget(b_initializer_label)
        bias_initializer = QComboBox()
        bias_initializer.addItems(["zeros", "glorot normal", "glorot uniform"])
        layout.addWidget(bias_initializer)
        
        self.config.append(bias_initializer)

        # Add widgets specific to configuring Dense layer
        pass

    def configureFlattenLayer(self, layout):

        flatten_desc_label = QLabel("Flatten - Transforms 2d input into 1d layer")
        layout.addWidget(flatten_desc_label)

        flatten_label = QLabel("Flatten Layer (no extra parameters needed)")
        layout.addWidget(flatten_label)

        # Add widgets specific to configuring layer
        pass

    def configureZeroPadding2dLayer(self, layout):

        #validator for text box input (regex)
        validator = QRegularExpressionValidator(QRegularExpression(r'^\d{1,2}$'))


        padding_label = QLabel("\u24D8 Padding: (x, y)")
        padding_label.setToolTip("Tuple of 2 ints indicating how many rows/columns to add to input.")
        layout.addWidget(padding_label)

        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        zp2d_padding_x = QLineEdit()
        zp2d_padding_x.setValidator(validator)
        zp2d_padding_x.setPlaceholderText("Padding x value")
        zp2d_padding_x.setText("1")
        layout.addWidget(zp2d_padding_x)

        self.config.append(zp2d_padding_x)

        zp2d_padding_y = QLineEdit()
        zp2d_padding_y.setValidator(validator)
        zp2d_padding_y.setPlaceholderText("Padding y value")
        zp2d_padding_y.setText("1")
        layout.addWidget(zp2d_padding_y)

        self.config.append(zp2d_padding_y)

        #PROBABLY DOESN'T WORK RIGHT
        zp2d_padding = []
        zp2d_padding.append(zp2d_padding_x.text())
        zp2d_padding.append(zp2d_padding_y.text())

        #print(zp2d_padding)



        # Add widgets specific to configuring layer
        pass

    def configureAveragePooling2dLayer(self, layout):

        avg_pool_desc_label = QLabel("Average Pooling 2D - Reduces input size by averaging a group of inputs in a 2D space and storing them as a single input")
        layout.addWidget(avg_pool_desc_label)

        #validator for text box input (regex)
        validator = QRegularExpressionValidator(QRegularExpression(r'^\d{1,2}$'))

        pool_size_label = QLabel("\u24D8 Pool Size: (x, y)")
        pool_size_label.setToolTip("Tuple of 2 integers, factors by which to downscale (dim1, dim2)")
        layout.addWidget(pool_size_label)

        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        ap2d_poolsize_x = QLineEdit()
        ap2d_poolsize_x.setValidator(validator)
        ap2d_poolsize_x.setPlaceholderText("Pool Size x value")
        ap2d_poolsize_x.setStyleSheet("border-style: outset;border-width: 2px;")
        layout.addWidget(ap2d_poolsize_x)

        self.config.append(ap2d_poolsize_x)

        ap2d_poolsize_y = QLineEdit()
        ap2d_poolsize_y.setValidator(validator)
        ap2d_poolsize_y.setPlaceholderText("Pool Size y value")
        ap2d_poolsize_y.setStyleSheet("border-style: outset;border-width: 2px;")
        layout.addWidget(ap2d_poolsize_y)

        self.config.append(ap2d_poolsize_y)

        #PROBABLY DOESN'T WORK RIGHT
        ap2d_poolsize = []
        ap2d_poolsize.append(ap2d_poolsize_x.text())
        ap2d_poolsize.append(ap2d_poolsize_y.text())


        strides_label = QLabel("\u24D8 Strides: (x, y)")
        strides_label.setToolTip("Tuple of 2 integers. Distance filter moves in x and y directions")
        layout.addWidget(strides_label)

        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        ap2d_strides_x = QLineEdit()
        ap2d_strides_x.setValidator(validator)
        ap2d_strides_x.setPlaceholderText("Strides x value")
        ap2d_strides_x.setStyleSheet("border-style: outset;border-width: 2px;")
        layout.addWidget(ap2d_strides_x)

        self.config.append(ap2d_strides_x)

        ap2d_strides_y = QLineEdit()
        ap2d_strides_y.setValidator(validator)
        ap2d_strides_y.setPlaceholderText("Strides y value")
        ap2d_strides_y.setStyleSheet("border-style: outset;border-width: 2px;")
        layout.addWidget(ap2d_strides_y)

        self.config.append(ap2d_strides_y)

        #PROBABLY DOESN'T WORK RIGHT
        ap2d_strides = []
        ap2d_strides.append(ap2d_strides_x.text())
        ap2d_strides.append(ap2d_strides_y.text())

        #padding dropdown
        padding_label = QLabel("\u24D8 Padding:")
        padding_label.setToolTip("Either valid or same (case-insensitive), valid means no padding. Same results in padding evenly to the left/right or up/down of the input such that output has the same heigh/width dimension as the input")
        layout.addWidget(padding_label)
        ap2d_padding = QComboBox()
        ap2d_padding.addItems(["valid", "same"])
        layout.addWidget(ap2d_padding)

        self.config.append(ap2d_padding)


        # Add widgets specific to configuring layer
        pass

    def configureMaxPooling2dLayer(self, layout):

        max_pool_desc_label = QLabel("Max pooling 2d - Reduces input size by taking the maximum of a group of inputs in a 2D space and storing them as a single input")
        layout.addWidget(max_pool_desc_label)

        #validator for text box input (regex)
        validator = QRegularExpressionValidator(QRegularExpression(r'^\d{1,2}$'))


        pool_size_label = QLabel("\u24D8 Pool Size: (x, y)")
        pool_size_label.setToolTip("Tuple of 2 integers, factors by which to downscale (dim1, dim2)")
        layout.addWidget(pool_size_label)

        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        mp2d_poolsize_x = QLineEdit()
        mp2d_poolsize_x.setValidator(validator)
        mp2d_poolsize_x.setPlaceholderText("Pool Size x value")
        mp2d_poolsize_x.setStyleSheet("border-style: outset;border-width: 2px;")
        layout.addWidget(mp2d_poolsize_x)

        self.config.append(mp2d_poolsize_x)

        mp2d_poolsize_y = QLineEdit()
        mp2d_poolsize_y.setValidator(validator)
        mp2d_poolsize_y.setPlaceholderText("Pool Size y value")
        mp2d_poolsize_y.setStyleSheet("border-style: outset;border-width: 2px;")
        layout.addWidget(mp2d_poolsize_y)

        self.config.append(mp2d_poolsize_y)

        #PROBABLY DOESN'T WORK RIGHT
        mp2d_poolsize = []
        mp2d_poolsize.append(mp2d_poolsize_x.text())
        mp2d_poolsize.append(mp2d_poolsize_y.text())


        strides_label = QLabel("\u24D8 Strides: (x, y)")
        strides_label.setToolTip("Tuple of 2 integers. Distance filter moves in x and y directions")
        layout.addWidget(strides_label)

        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        mp2d_strides_x = QLineEdit()
        mp2d_strides_x.setValidator(validator)
        mp2d_strides_x.setPlaceholderText("Strides x value")
        mp2d_strides_x.setStyleSheet("border-style: outset;border-width: 2px;")
        layout.addWidget(mp2d_strides_x)

        self.config.append(mp2d_strides_x)

        mp2d_strides_y = QLineEdit()
        mp2d_strides_y.setValidator(validator)
        mp2d_strides_y.setPlaceholderText("Strides y value")
        mp2d_strides_y.setStyleSheet("border-style: outset;border-width: 2px;")
        layout.addWidget(mp2d_strides_y)

        self.config.append(mp2d_strides_y)

        #PROBABLY DOESN'T WORK RIGHT
        mp2d_strides = []
        mp2d_strides.append(mp2d_strides_x.text())
        mp2d_strides.append(mp2d_strides_y.text())

        #padding dropdown
        padding_label = QLabel("\u24D8 Padding:")
        padding_label.setToolTip("Either valid or same (case-insensitive), valid means no padding. Same results in padding evenly to the left/right or up/down of the input such that output has the same heigh/width dimension as the input")
        layout.addWidget(padding_label)
        mp2d_padding = QComboBox()
        mp2d_padding.addItems(["valid", "same"])
        layout.addWidget(mp2d_padding)

        self.config.append(mp2d_padding)

        # Add widgets specific to configuring layer
        pass

    def configureConvolution2dLayer(self, layout):

        convolution_2d_desc_label = QLabel("Convolutional 2d - Passes filter over 2D input, creating smaller 2D output that highlights features")
        layout.addWidget(convolution_2d_desc_label)

        #validator for text box input (regex)
        validator = QRegularExpressionValidator(QRegularExpression(r'^\d{1,2}$'))

        #filters
        filters_label = QLabel("\u24D8 Filters:")
        filters_label.setToolTip("Integer, the dimension of the output space (the number of filters in the convolution)")
        layout.addWidget(filters_label)

        #add text box
        c2d_filter = QLineEdit()
        c2d_filter.setPlaceholderText("Filters")
        c2d_filter.setValidator(validator)
        c2d_filter.setStyleSheet("border-style: outset;border-width: 2px;")
        layout.addWidget(c2d_filter)

        self.config.append(c2d_filter)

        #kernel size
        kernelsize_label = QLabel("\u24D8 Kernel Size: (x, y)")
        kernelsize_label.setToolTip("Tuple of 2 integer, specifying the size of the convolution window")
        layout.addWidget(kernelsize_label)


        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        c2d_kernelsize_x = QLineEdit()
        c2d_kernelsize_x.setValidator(validator)
        c2d_kernelsize_x.setPlaceholderText("Kernel Size x value")
        c2d_kernelsize_x.setStyleSheet("border-style: outset;border-width: 2px;")
        layout.addWidget(c2d_kernelsize_x)

        self.config.append(c2d_kernelsize_x)

        c2d_kernelsize_y = QLineEdit()
        c2d_kernelsize_y.setValidator(validator)
        c2d_kernelsize_y.setPlaceholderText("Kernel Size y value")
        c2d_kernelsize_y.setStyleSheet("border-style: outset;border-width: 2px;")
        layout.addWidget(c2d_kernelsize_y)

        self.config.append(c2d_kernelsize_y)

        #strides
        strides_label = QLabel("\u24D8 Strides: (x, y)")
        strides_label.setToolTip("Tuple of 2 integer, specifying the stride length of the convolution. Note - strides > 1 is incompatible with dilation rate > 1")
        layout.addWidget(strides_label)


        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        c2d_strides_x = QLineEdit()
        c2d_strides_x.setValidator(validator)
        c2d_strides_x.setPlaceholderText("Strides x value")
        c2d_strides_x.setText("1")
        layout.addWidget(c2d_strides_x)

        self.config.append(c2d_strides_x)

        c2d_strides_y = QLineEdit()
        c2d_strides_y.setValidator(validator)
        c2d_strides_y.setPlaceholderText("Strides y value")
        c2d_strides_y.setText("1")
        layout.addWidget(c2d_strides_y)

        self.config.append(c2d_strides_y)

        #PROBABLY DOESN'T WORK RIGHT
        c2d_strides = []
        c2d_strides.append(c2d_strides_x.text())
        c2d_strides.append(c2d_strides_y.text())

        #padding dropdown
        padding_label = QLabel("\u24D8 Padding:")
        padding_label.setToolTip("Either valid or same (case-insensitive), valid means no padding. Same results in padding evenly to the left/right or up/down of the input such that output has the same heigh/width dimension as the input")
        layout.addWidget(padding_label)
        c2d_padding = QComboBox()
        c2d_padding.addItems(["valid", "same"])
        layout.addWidget(c2d_padding)

        self.config.append(c2d_padding)

        #dialation rate
        dialationrate_label = QLabel("\u24D8 Dilation Rate: (x, y)")
        dialationrate_label.setToolTip("Tuple of 2 integers, specifying the dilation rate to use for dilated convolution")
        layout.addWidget(dialationrate_label)

        #add text box``
        #ADD THING TO GET MAD IF BOX IS EMPTY

        c2d_dialationrate_x = QLineEdit()
        c2d_dialationrate_x.setValidator(validator)
        c2d_dialationrate_x.setPlaceholderText("Dilation Rate x value")
        c2d_dialationrate_x.setText("1")
        layout.addWidget(c2d_dialationrate_x)

        self.config.append(c2d_dialationrate_x)

        c2d_dialationrate_y = QLineEdit()
        c2d_dialationrate_y.setValidator(validator)
        c2d_dialationrate_y.setPlaceholderText("Dilation Rate y value")
        c2d_dialationrate_y.setText("1")
        layout.addWidget(c2d_dialationrate_y)

        self.config.append(c2d_dialationrate_y)

        #groups
        groups_label = QLabel("\u24D8 Groups:")
        groups_label.setToolTip("A positive int specifying the number of groups in which the int is split along the channel axis")
        layout.addWidget(groups_label)

        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY
        c2d_groups = QLineEdit()
        c2d_groups.setPlaceholderText("Groups")
        c2d_groups.setValidator(validator)
        c2d_groups.setText("1")
        layout.addWidget(c2d_groups)

        self.config.append(c2d_groups)

        #activation dropdown
        activation_label = QLabel("\u24D8 Activation Type:")
        activation_label.setToolTip("Activation function. If None, no activation is applied")
        layout.addWidget(activation_label)
        c2d_activation = QComboBox()
        c2d_activation.addItems(["None","elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "sigmoid", "hard sigmoid", "silu", "hard silu", "softmax", "softplus", "tanh"])
        layout.addWidget(c2d_activation)

        self.config.append(c2d_activation)

        #use bias dropdown
        bias_label = QLabel("\u24D8 Use Bias:")
        bias_label.setToolTip("Bool, if True, bias will be added to the output.")
        layout.addWidget(bias_label)
        use_bias = QComboBox()
        use_bias.addItems(["True", "False"])
        layout.addWidget(use_bias)

        self.config.append(use_bias)

        #kernel initializer
        k_initializer_label = QLabel("\u24D8 Kernel Initializer:")
        k_initializer_label.setToolTip("Initializer for the convolution kernel")
        layout.addWidget(k_initializer_label)
        kernel_initializer = QComboBox()
        kernel_initializer.addItems(["glorot uniform", "zeros", "glorot normal", "orthogonal"])
        layout.addWidget(kernel_initializer)

        self.config.append(kernel_initializer)

        #bias initializer
        b_initializer_label = QLabel("\u24D8 Bias Initializer:")
        b_initializer_label.setToolTip("Initializer for the bias vector")
        layout.addWidget(b_initializer_label)
        bias_initializer = QComboBox()
        bias_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(bias_initializer)

        self.config.append(bias_initializer)

        # Add widgets specific to configuring layer
        pass

    def configureConvolution2dTransposeLayer(self, layout):

        #validator for text box input (regex)
        validator = QRegularExpressionValidator(QRegularExpression(r'^\d{1,2}$'))

        #filters
        filters_label = QLabel("Filters:")
        layout.addWidget(filters_label)

        #add text box
        c2dt_filter = QLineEdit()
        c2dt_filter.setPlaceholderText("Filters")
        c2dt_filter.setValidator(validator)
        c2dt_filter.setStyleSheet("border-style: outset;border-width: 2px;")
        layout.addWidget(c2dt_filter)

        self.config.append(c2dt_filter)

        #kernel size
        kernelsize_label = QLabel("Kernel Size:")
        layout.addWidget(kernelsize_label)


        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        c2dt_kernelsize_x = QLineEdit()
        c2dt_kernelsize_x.setValidator(validator)
        c2dt_kernelsize_x.setPlaceholderText("Kernel Size value")
        c2dt_kernelsize_x.setStyleSheet("border-style: outset;border-width: 2px;")
        layout.addWidget(c2dt_kernelsize_x)

        self.config.append(c2dt_kernelsize_x)

        #strides
        strides_label = QLabel("Strides:")
        layout.addWidget(strides_label)


        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        c2dt_strides_x = QLineEdit()
        c2dt_strides_x.setValidator(validator)
        c2dt_strides_x.setPlaceholderText("Strides value")
        c2dt_strides_x.setText("1")
        layout.addWidget(c2dt_strides_x)

        self.config.append(c2dt_strides_x)

        #PROBABLY DOESN'T WORK RIGHT
        c2dt_strides = []
        c2dt_strides.append(c2dt_strides_x.text())

        #padding dropdown
        padding_label = QLabel("Padding:")
        layout.addWidget(padding_label)
        c2dt_padding = QComboBox()
        c2dt_padding.addItems(["valid", "same"])
        layout.addWidget(c2dt_padding)

        self.config.append(c2dt_padding)

        #dialation rate
        dialationrate_label = QLabel("Dilation Rate:")
        layout.addWidget(dialationrate_label)

        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        c2dt_dialationrate_x = QLineEdit()
        c2dt_dialationrate_x.setValidator(validator)
        c2dt_dialationrate_x.setPlaceholderText("Dilation Rate value")
        c2dt_dialationrate_x.setText("1")
        layout.addWidget(c2dt_dialationrate_x)

        self.config.append(c2dt_dialationrate_x)

        #groups
        groups_label = QLabel("Groups:")
        layout.addWidget(groups_label)

        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY
        c2dt_groups = QLineEdit()
        c2dt_groups.setPlaceholderText("Groups")
        c2dt_groups.setValidator(validator)
        c2dt_groups.setText("1")
        layout.addWidget(c2dt_groups)

        self.config.append(c2dt_groups)

        #activation dropdown
        activation_label = QLabel("Activation Type:")
        layout.addWidget(activation_label)
        c2dt_activation = QComboBox()
        c2dt_activation.addItems(["None","elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "sigmoid", "hard sigmoid", "silu", "hard silu", "softmax", "softplus", "tanh"])
        layout.addWidget(c2dt_activation)

        self.config.append(c2dt_activation)

        #use bias dropdown
        bias_label = QLabel("Use Bias:")
        layout.addWidget(bias_label)
        use_bias = QComboBox()
        use_bias.addItems(["True", "False"])
        layout.addWidget(use_bias)

        self.config.append(use_bias)

        #kernel initializer
        k_initializer_label = QLabel("Kernel Initializer:")
        layout.addWidget(k_initializer_label)
        kernel_initializer = QComboBox()
        kernel_initializer.addItems(["glorot uniform", "zeros", "glorot normal", "orthogonal"])
        layout.addWidget(kernel_initializer)

        self.config.append(kernel_initializer)

        #bias initializer
        b_initializer_label = QLabel("Bias Initializer:")
        layout.addWidget(b_initializer_label)
        bias_initializer = QComboBox()
        bias_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(bias_initializer)

        self.config.append(bias_initializer)



        # Add widgets specific to configuring layer
        pass


    def configureSimpleRNNLayer(self, layout):

        simple_rnn_desc_label = QLabel("Basic Recurrent Neural Network, utilizes previously seen data during training to better forecast future sequential data")
        layout.addWidget(simple_rnn_desc_label)

        #validator for text box input (regex)
        validator = QRegularExpressionValidator(QRegularExpression(r'^\d{1,2}$'))

        float_validator = QRegularExpressionValidator(QRegularExpression('^(\d)*(\.)?([0-9]{1})?$'))

        #units
        units_label = QLabel("\u24D8 Units:")
        units_label.setToolTip("Positive integer, dimensionality of the output space")
        layout.addWidget(units_label)

        #add text box
        srnn_units = QLineEdit()
        srnn_units.setPlaceholderText("Units")
        srnn_units.setValidator(validator)
        layout.addWidget(srnn_units)

        self.config.append(srnn_units)

        #activation dropdown
        activation_label = QLabel("\u24D8 Activation Type:")
        activation_label.setToolTip("Activation function to use")
        layout.addWidget(activation_label)
        srnn_activation = QComboBox()
        srnn_activation.addItems(["tanh", "elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "sigmoid", "hard sigmoid", "silu", "hard silu", "softmax", "softplus"])
        layout.addWidget(srnn_activation)

        self.config.append(srnn_activation)

        #use bias dropdown
        bias_label = QLabel("\u24D8 Use Bias:")
        bias_label.setToolTip("Boolean, whether the layer uses a bias vector")
        layout.addWidget(bias_label)
        use_bias = QComboBox()
        use_bias.addItems(["True", "False"])
        layout.addWidget(use_bias)

        self.config.append(use_bias)

        #kernel initializer
        k_initializer_label = QLabel("\u24D8 Kernel Initializer:")
        k_initializer_label.setToolTip("Initializer for the kernel weights matrix, used for the linear transformation of the inputs")
        layout.addWidget(k_initializer_label)
        kernel_initializer = QComboBox()
        kernel_initializer.addItems(["glorot uniform", "zeros", "glorot normal", "orthogonal"])
        layout.addWidget(kernel_initializer)

        self.config.append(kernel_initializer)

        #recurrent initializer
        r_initializer_label = QLabel("\u24D8 Recurrent Initializer:")
        r_initializer_label.setToolTip("Initializer for the recurrent_kernel weights matrix, used for the linear transformation of the recurrent state")
        layout.addWidget(r_initializer_label)
        recurrent_initializer = QComboBox()
        recurrent_initializer.addItems(["orthogonal", "zeros", "glorot normal", "glorot uniform"])
        layout.addWidget(recurrent_initializer)

        self.config.append(recurrent_initializer)

        #bias initializer
        b_initializer_label = QLabel("\u24D8 Bias Initializer:")
        b_initializer_label.setToolTip("Initializer for the bias vector")
        layout.addWidget(b_initializer_label)
        bias_initializer = QComboBox()
        bias_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(bias_initializer)

        self.config.append(bias_initializer)

        #dropout
        dropout_label = QLabel("\u24D8 Dropout:")
        dropout_label.setToolTip("Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs. Default is 0.")
        layout.addWidget(dropout_label)

        #add text box
        #GET MAD IF USER PUTS IN FUNKY STUFF (anything besides numbers and a decimal)
        srnn_dropout = QLineEdit()
        srnn_dropout.setPlaceholderText("Dropout")
        # srnn_dropout.setValidator(float_validator)
        layout.addWidget(srnn_dropout)

        self.config.append(srnn_dropout)

        #recurrent dropout
        recurrent_dropout_label = QLabel("\u24D8 Recurrent Dropout:")
        recurrent_dropout_label.setToolTip("Recurrent Dropout - Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state. Default: 0.")
        layout.addWidget(recurrent_dropout_label)

        #add text box
        #GET MAD IF USER PUTS IN FUNKY STUFF (anything besides numbers and a decimal)
        srnn_recurrent_dropout = QLineEdit()
        srnn_recurrent_dropout.setPlaceholderText("Recurrent Dropout")
        # srnn_recurrent_dropout.setValidator(float_validator)
        layout.addWidget(srnn_recurrent_dropout)

        self.config.append(srnn_recurrent_dropout)

        # Add widgets specific to configuring layer
        pass

    def configureLSTMLayer(self, layout):

        lstm_desc_label = QLabel("Long Short-Term Memory layer - Subset of RNN able to handle long-term dependencies, takes in 2D input and outputs 1D data")
        layout.addWidget(lstm_desc_label)

        #validator for text box input (regex)
        validator = QRegularExpressionValidator(QRegularExpression(r'^\d{1,2}$'))

        float_validator = QRegularExpressionValidator(QRegularExpression('^(\d)*(\.)?([0-9]{1})?$'))

        #units
        units_label = QLabel("\u24D8 Units:")
        units_label.setToolTip("Positive integer, dimensionality of the output space")
        layout.addWidget(units_label)

        #add text box
        lstm_units = QLineEdit()
        lstm_units.setPlaceholderText("Units")
        lstm_units.setValidator(validator)
        layout.addWidget(lstm_units)

        self.config.append(lstm_units)

        #activation dropdown
        activation_label = QLabel("\u24D8 Activation Type:")
        activation_label.setToolTip("Activation function to use.")
        layout.addWidget(activation_label)
        lstm_activation = QComboBox()
        lstm_activation.addItems(["tanh", "elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "sigmoid", "hard sigmoid", "silu", "hard silu", "softmax", "softplus"])
        layout.addWidget(lstm_activation)

        self.config.append(lstm_activation)

        #recurrent activation dropdown
        recurrent_activation_label = QLabel("\u24D8 Recurrent Activation Type:")
        recurrent_activation_label.setToolTip("Activation function to use for the recurrent step.")
        layout.addWidget(recurrent_activation_label)
        lstm_recurrent_activation = QComboBox()
        lstm_recurrent_activation.addItems(["sigmoid", "elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "hard sigmoid", "silu", "hard silu", "softmax", "softplus", "tanh"])
        layout.addWidget(lstm_recurrent_activation)

        self.config.append(lstm_recurrent_activation)

        #use bias dropdown
        bias_label = QLabel("\u24D8 Use Bias:")
        bias_label.setToolTip("Boolean, whether the layer should use a bias vector.")
        layout.addWidget(bias_label)
        use_bias = QComboBox()
        use_bias.addItems(["True", "False"])
        layout.addWidget(use_bias)

        self.config.append(use_bias)

        #kernel initializer
        k_initializer_label = QLabel("\u24D8 Kernel Initializer:")
        k_initializer_label.setToolTip("Initializer for the kernel weights matrix, used for the linear transformation of the inputs.")
        layout.addWidget(k_initializer_label)
        kernel_initializer = QComboBox()
        kernel_initializer.addItems(["glorot uniform", "zeros", "glorot normal", "orthogonal"])
        layout.addWidget(kernel_initializer)

        self.config.append(kernel_initializer)

        #recurrent initializer
        r_initializer_label = QLabel("\u24D8 Recurrent Initializer:")
        r_initializer_label.setToolTip("Initializer for the recurrent kernel weights matrix, used for the linear transformation of the recurrent state.")
        layout.addWidget(r_initializer_label)
        recurrent_initializer = QComboBox()
        recurrent_initializer.addItems(["orthogonal", "zeros", "glorot normal", "glorot uniform"])
        layout.addWidget(recurrent_initializer)

        self.config.append(recurrent_initializer)

        #bias initializer
        b_initializer_label = QLabel("\u24D8 Bias Initializer:")
        b_initializer_label.setToolTip("Initializer for the bias vector")
        layout.addWidget(b_initializer_label)
        bias_initializer = QComboBox()
        bias_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(bias_initializer)

        self.config.append(bias_initializer)

        #unit forget bias dropdown
        unit_forget_bias_label = QLabel("\u24D8 Unit Forget Bias:")
        unit_forget_bias_label.setToolTip("Boolean. If True, add 1 to the bias of the forget gate at initialization. Setting it to True will also force bias initializer = “zeros”.")
        layout.addWidget(unit_forget_bias_label)
        unit_forget_bias = QComboBox()
        unit_forget_bias.addItems(["True", "False"])
        layout.addWidget(unit_forget_bias)

        self.config.append(unit_forget_bias)


        #dropout
        dropout_label = QLabel("\u24D8 Dropout:")
        dropout_label.setToolTip("Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs. Default is 0.")
        layout.addWidget(dropout_label)

        #add text box
        #GET MAD IF USER PUTS IN FUNKY STUFF (anything besides numbers and a decimal)
        lstm_dropout = QLineEdit()
        lstm_dropout.setPlaceholderText("Dropout")
        lstm_dropout.setValidator(float_validator)
        layout.addWidget(lstm_dropout)

        self.config.append(lstm_dropout)

        #recurrent dropout
        recurrent_dropout_label = QLabel("\u24D8 Recurrent Dropout:")
        recurrent_dropout_label.setToolTip("Recurrent Dropout - Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state. Default: 0.")
        layout.addWidget(recurrent_dropout_label)

        #add text box
        #GET MAD IF USER PUTS IN FUNKY STUFF (anything besides numbers and a decimal)
        lstm_recurrent_dropout = QLineEdit()
        lstm_recurrent_dropout.setPlaceholderText("Recurrent Dropout")
        # lstm_recurrent_dropout.setValidator(float_validator)
        layout.addWidget(lstm_recurrent_dropout)

        self.config.append(lstm_recurrent_dropout)

        # Add widgets specific to configuring layer
        pass

    def configureGRULayer(self, layout):

        gru_desc_label = QLabel("Gated Recurrent Unit - Subset of RNN that allows for selective storing and replacing of previously seen data")
        layout.addWidget(gru_desc_label)

        #validator for text box input (regex)
        validator = QRegularExpressionValidator(QRegularExpression(r'^\d{1,2}$'))

        float_validator = QRegularExpressionValidator(QRegularExpression('^(\d)*(\.)?([0-9]{1})?$'))

        #units
        units_label = QLabel("\u24D8 Units:")
        units_label.setToolTip("Positive integer, dimensionality of the output space")
        layout.addWidget(units_label)

        #add text box
        gru_units = QLineEdit()
        gru_units.setPlaceholderText("Units")
        gru_units.setValidator(validator)
        layout.addWidget(gru_units)

        self.config.append(gru_units)

        #activation dropdown
        activation_label = QLabel("\u24D8 Activation Type:")
        activation_label.setToolTip("Activation function to use.")
        layout.addWidget(activation_label)
        gru_activation = QComboBox()
        gru_activation.addItems(["tanh", "elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "sigmoid", "hard sigmoid", "silu", "hard silu", "softmax", "softplus"])
        layout.addWidget(gru_activation)

        self.config.append(gru_activation)

        #recurrent activation dropdown
        recurrent_activation_label = QLabel("\u24D8 Recurrent Activation Type:")
        recurrent_activation_label.setToolTip("Activation function to use for the recurrent step.")
        layout.addWidget(recurrent_activation_label)
        gru_recurrent_activation = QComboBox()
        gru_recurrent_activation.addItems(["sigmoid", "elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "hard sigmoid", "silu", "hard silu", "softmax", "softplus", "tanh"])
        layout.addWidget(gru_recurrent_activation)

        self.config.append(gru_recurrent_activation)

        #use bias dropdown
        bias_label = QLabel("\u24D8 Use Bias:")
        bias_label.setToolTip("Boolean, whether the layer should use a bias vector.")
        layout.addWidget(bias_label)
        use_bias = QComboBox()
        use_bias.addItems(["True", "False"])
        layout.addWidget(use_bias)

        self.config.append(use_bias)

        #kernel initializer
        k_initializer_label = QLabel("\u24D8 Kernel Initializer:")
        k_initializer_label.setToolTip("Initializer for the kernel weights matrix, used for the linear transformation of the inputs.")
        layout.addWidget(k_initializer_label)
        kernel_initializer = QComboBox()
        kernel_initializer.addItems(["glorot uniform", "zeros", "glorot normal", "orthogonal"])
        layout.addWidget(kernel_initializer)

        self.config.append(kernel_initializer)

        #recurrent initializer
        r_initializer_label = QLabel("\u24D8 Recurrent Initializer:")
        r_initializer_label.setToolTip("Initializer for the recurrent kernel weights matrix, used for the linear transformation of the recurrent state.")
        layout.addWidget(r_initializer_label)
        recurrent_initializer = QComboBox()
        recurrent_initializer.addItems(["orthogonal", "zeros", "glorot normal", "glorot uniform"])
        layout.addWidget(recurrent_initializer)

        self.config.append(recurrent_initializer)

        #bias initializer
        b_initializer_label = QLabel("\u24D8 Bias Initializer:")
        b_initializer_label.setToolTip("Initializer for the bias vector")
        layout.addWidget(b_initializer_label)
        bias_initializer = QComboBox()
        bias_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        layout.addWidget(bias_initializer)

        self.config.append(bias_initializer)

        #dropout
        dropout_label = QLabel("\u24D8 Dropout:")
        dropout_label.setToolTip("Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs. Default is 0.")
        layout.addWidget(dropout_label)

        #add text box
        #GET MAD IF USER PUTS IN FUNKY STUFF (anything besides numbers and a decimal)
        gru_dropout = QLineEdit()
        gru_dropout.setPlaceholderText("Dropout")
        gru_dropout.setValidator(float_validator)
        layout.addWidget(gru_dropout)

        self.config.append(gru_dropout)

        #recurrent dropout
        recurrent_dropout_label = QLabel("\u24D8 Recurrent Dropout:")
        recurrent_dropout_label.setToolTip("Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state. Default: 0")
        layout.addWidget(recurrent_dropout_label)

        #add text box
        #GET MAD IF USER PUTS IN FUNKY STUFF (anything besides numbers and a decimal)
        gru_recurrent_dropout = QLineEdit()
        gru_recurrent_dropout.setPlaceholderText("Recurrent Dropout")
        gru_recurrent_dropout.setValidator(float_validator)
        layout.addWidget(gru_recurrent_dropout)

        self.config.append(gru_recurrent_dropout)

        #reset after dropdown
        reset_after_label = QLabel("\u24D8 Reset After:")
        reset_after_label.setToolTip("GRU convention (whether to apply reset gate after or before matrix multiplication). False is before, True is after.")
        layout.addWidget(reset_after_label)
        reset_after = QComboBox()
        reset_after.addItems(["True", "False"])
        layout.addWidget(reset_after)

        self.config.append(reset_after)

        # Add widgets specific to configuring layer
        pass

    def RNNlayerCompatibilityChecker(self):
        for layer in nnet.datadict["model_1"]["layers"]:
            if isinstance(layer, (SimpleRNN, LSTM, GRU)):
                return True
        return False
    
    def CNNlayerCompatibilityChecker(self):
        for layer in nnet.datadict["model_1"]["layers"]:
            if isinstance(layer, (Zero_Padding_2d, Average_Pooling_2d, Max_Pool_2d, Convolution_2d, Convolution_2d_Transpose)):
                return True
        return False
    
    def flattenChecker(self, index):
        layers = nnet.datadict["model_1"]["layers"]
        total_layers = len(layers)

        # If adding at the end or as the only layer, check the previous layer if any
        if index == 0:
            # It's the first layer in the list, typically allowed to be a Flatten
            return True

        # Check the next layer, if there is one
        if index != total_layers:
            next_layer = layers[index]
            # If the next layer is not Dense or Flatten, return False
            if not isinstance(next_layer, (Dense, Flatten)):
                return False

        # Check if the previous layer is Dense or Flatten
        prev_layer = layers[index - 1]
        return isinstance(prev_layer, (Dense, Flatten))

    def saveLayer(self):

        # Append to main dictionary the layer type the user wants to add
        length = len(nnet.datadict["model_1"]["layers"])
        
        global index
        
        insertion_index = index 
        if self.layer_location == "After selected":
            insertion_index = index 
        if self.layer_location == "Beginning of list":
            insertion_index = 0
        else:
            len(nnet.datadict["model_1"]["layers"])
    
        if self.layer_type == "Flatten":
            if not self.flattenChecker(insertion_index):
                QMessageBox.warning(self, "Invalid Layer Placement", "A Flatten layer must only be placed before a Dense or another Flatten layer.")
                return

        if self.layer_type == "Max Pooling 2d" and self.RNNlayerCompatibilityChecker():
            QMessageBox.warning(self, "Invalid Layer", "Cannot add Max pooling 2d layer because an RNN layer is in (Simple RNN, LSTM, or GRU)")
            return
        
        if self.layer_type == "Average Pooling 2d" and self.RNNlayerCompatibilityChecker():
            QMessageBox.warning(self, "Invalid Layer", "Cannot add Average pooling 2d layer because an RNN layer is in (Simple RNN, LSTM, or GRU)")
            return
        
        if self.layer_type == "Convolution 2d" and self.RNNlayerCompatibilityChecker():
            QMessageBox.warning(self, "Invalid Layer", "Cannot add Convolution 2d layer because an RNN layer is in (Simple RNN, LSTM, or GRU)")
            return
        
        if self.layer_type == "Convolution 2d Transpose" and self.RNNlayerCompatibilityChecker():
            QMessageBox.warning(self, "Invalid Layer", "Cannot add Convolution 2d transpose layer because an RNN layer is in (Simple RNN, LSTM, or GRU)")
            return
        
        if (self.layer_type == "Simple RNN" and self.CNNlayerCompatibilityChecker()) or (self.layer_type == "Simple RNN" and self.RNNlayerCompatibilityChecker()):
            QMessageBox.warning(self, "Invalid Layer", "Cannot add Simple RNN layer because a recurrent, convolutional, or pooling layer is present.")
            return
        
        if (self.layer_type == "LSTM" and self.CNNlayerCompatibilityChecker()) or (self.layer_type == "LSTM" and self.RNNlayerCompatibilityChecker()):
            QMessageBox.warning(self, "Invalid Layer", "Cannot add LSTM layer because a recurrent, convolutional, or pooling layer is present.")
            return
        
        if (self.layer_type == "GRU" and self.CNNlayerCompatibilityChecker()) or (self.layer_type == "GRU" and self.RNNlayerCompatibilityChecker()):
            QMessageBox.warning(self, "Invalid Layer", "Cannot add GRU layer because a recurrent, convolutional, or pooling layer is present.")
            return

        if(self.layer_location == "Beginning of list"):
            if self.layer_type == "Dense":
                nnet.datadict["model_1"]["layers"].insert(0,
                    Dense(
                        0, 
                        int(float(self.config[0].text())), 
                        list(mapping.keys())[list(mapping.values()).index(self.config[1].currentText())],
                        list(mapping.keys())[list(mapping.values()).index(self.config[2].currentText())],
                        list(mapping.keys())[list(mapping.values()).index(self.config[3].currentText())],
                        list(mapping.keys())[list(mapping.values()).index(self.config[4].currentText())]
                        )
                    )
            elif self.layer_type == "Flatten":
                nnet.datadict["model_1"]["layers"].insert(0,
                    Flatten(
                        0
                        )
                    )
            elif self.layer_type == "Dropout":
                nnet.datadict["model_1"]["layers"].insert(0,
                    Dropout(
                        0,
                        (int(float(self.config[0].text()))), int(self.config[1].text())
                        )
                    )
            elif self.layer_type == "Zero Padding 2d":
                nnet.datadict["model_1"]["layers"].insert(0,
                    Zero_Padding_2d(
                        0, 
                        (int(float(self.config[0].text())), int(float(self.config[1].text())))
                        )
                    )
            elif self.layer_type == "Average Pooling 2d":
                nnet.datadict["model_1"]["layers"].insert(0,
                    Average_Pooling_2d(
                        0, 
                        (int(float(self.config[0].text())), int(float(self.config[1].text()))), 
                        (int(float(self.config[2].text())), int(float(self.config[3].text()))), 
                        list(mapping.keys())[list(mapping.values()).index(self.config[4].currentText())]
                        )
                    )
            elif self.layer_type == "Max Pooling 2d":
                nnet.datadict["model_1"]["layers"].insert(0,
                    Max_Pool_2d(
                        0, 
                        (int(float(self.config[0].text())), int(float(self.config[1].text()))), 
                        (int(float(self.config[2].text())), int(float(self.config[3].text()))), 
                        list(mapping.keys())[list(mapping.values()).index(self.config[4].currentText())]
                        )
                    )
            elif self.layer_type == "Convolution 2d":
                nnet.datadict["model_1"]["layers"].insert(0,
                    Convolution_2d(
                        0, 
                        int(float(self.config[0].text())), 
                        (int(float(self.config[1].text())), int(float(self.config[2].text()))), 
                        (int(float(self.config[3].text())), int(float(self.config[4].text()))), 
                        list(mapping.keys())[list(mapping.values()).index(self.config[5].currentText())], 
                        (int(float(self.config[6].text())), int(float(self.config[7].text()))), 
                        int(float(self.config[8].text())), 
                        list(mapping.keys())[list(mapping.values()).index(self.config[9].currentText())],
                        list(mapping.keys())[list(mapping.values()).index(self.config[10].currentText())], 
                        list(mapping.keys())[list(mapping.values()).index(self.config[11].currentText())],
                        list(mapping.keys())[list(mapping.values()).index(self.config[12].currentText())]
                        )
                    )
            # elif self.layer_type == "Convolution 2d Transpose":
            #     nnet.datadict["model_1"]["layers"].insert(0,
            #         Convolution_2d_Transpose(
            #             length, 
            #             int(float(self.config[0].text())), 
            #             (int(float(self.config[1].text()))), 
            #             (int(float(self.config[2].text()))), 
            #             self.config[3].currentText(), 
            #             (int(float(self.config[4].text()))), 
            #             int(float(self.config[5].text())), 
            #             self.config[6].currentText(), 
            #             self.config[7].currentText(), 
            #             self.config[8].currentText(), 
            #             self.config[9].currentText()
            #             )
            #         )
            elif self.layer_type == "Simple RNN":
                nnet.datadict["model_1"]["layers"].insert(0,
                    SimpleRNN(
                        0, 
                        int(float(self.config[0].text())), 
                        list(mapping.keys())[list(mapping.values()).index(self.config[1].currentText())],
                        list(mapping.keys())[list(mapping.values()).index(self.config[2].currentText())],
                        list(mapping.keys())[list(mapping.values()).index(self.config[3].currentText())],
                        list(mapping.keys())[list(mapping.values()).index(self.config[4].currentText())],
                        list(mapping.keys())[list(mapping.values()).index(self.config[5].currentText())],
                        float(self.config[6].text()), 
                        float(self.config[7].text())
                        )
                    )
            elif self.layer_type == "LSTM":
                nnet.datadict["model_1"]["layers"].insert(0,
                    LSTM(
                        0, 
                        int(float(self.config[0].text())), 
                        list(mapping.keys())[list(mapping.values()).index(self.config[1].currentText())],
                        list(mapping.keys())[list(mapping.values()).index(self.config[2].currentText())],
                        list(mapping.keys())[list(mapping.values()).index(self.config[3].currentText())],
                        list(mapping.keys())[list(mapping.values()).index(self.config[4].currentText())],
                        list(mapping.keys())[list(mapping.values()).index(self.config[5].currentText())],
                        list(mapping.keys())[list(mapping.values()).index(self.config[6].currentText())],
                        list(mapping.keys())[list(mapping.values()).index(self.config[7].currentText())], 
                        float(self.config[8].text()), 
                        float(self.config[9].text())
                        )
                    )
            elif self.layer_type == "GRU":
                nnet.datadict["model_1"]["layers"].insert(0,
                    GRU(
                        0, 
                        int(float(self.config[0].text())), 
                        list(mapping.keys())[list(mapping.values()).index(self.config[1].currentText())],
                        list(mapping.keys())[list(mapping.values()).index(self.config[2].currentText())],
                        list(mapping.keys())[list(mapping.values()).index(self.config[3].currentText())],
                        list(mapping.keys())[list(mapping.values()).index(self.config[4].currentText())],
                        list(mapping.keys())[list(mapping.values()).index(self.config[5].currentText())],
                        list(mapping.keys())[list(mapping.values()).index(self.config[6].currentText())], 
                        float(self.config[7].text()), 
                        float(self.config[8].text()), 
                        None,
                        list(mapping.keys())[list(mapping.values()).index(self.config[9].currentText())]
                        )
                    )
        elif(self.layer_location == "End of list" or index == -1):
            if self.layer_type == "Dense":
                nnet.datadict["model_1"]["layers"].append(
                    Dense(
                        length, 
                        int(float(self.config[0].text())), 
                        list(mapping.keys())[list(mapping.values()).index(self.config[1].currentText())],
                        list(mapping.keys())[list(mapping.values()).index(self.config[2].currentText())],
                        list(mapping.keys())[list(mapping.values()).index(self.config[3].currentText())],
                        list(mapping.keys())[list(mapping.values()).index(self.config[4].currentText())]
                        )
                    )
            elif self.layer_type == "Flatten":
                nnet.datadict["model_1"]["layers"].append(
                    Flatten(
                        length
                        )
                    )
            elif self.layer_type == "Dropout":
                nnet.datadict["model_1"]["layers"].append(
                    Dropout(
                        length,
                        (int(float(self.config[0].text()))), int(self.config[1].text())
                        )
                    )
            elif self.layer_type == "Zero Padding 2d":
                nnet.datadict["model_1"]["layers"].append(
                    Zero_Padding_2d(
                        length, 
                        (int(float(self.config[0].text())), int(float(self.config[1].text())))
                        )
                    )
            elif self.layer_type == "Average Pooling 2d":
                nnet.datadict["model_1"]["layers"].append(
                    Average_Pooling_2d(
                        length, 
                        (int(float(self.config[0].text())), int(float(self.config[1].text()))), 
                        (int(float(self.config[2].text())), int(float(self.config[3].text()))), 
                        list(mapping.keys())[list(mapping.values()).index(self.config[4].currentText())]
                        )
                    )
            elif self.layer_type == "Max Pooling 2d":
                nnet.datadict["model_1"]["layers"].append(
                    Max_Pool_2d(
                        length, 
                        (int(float(self.config[0].text())), int(float(self.config[1].text()))), 
                        (int(float(self.config[2].text())), int(float(self.config[3].text()))), 
                        list(mapping.keys())[list(mapping.values()).index(self.config[4].currentText())]
                        )
                    )
            elif self.layer_type == "Convolution 2d":
                nnet.datadict["model_1"]["layers"].append(
                    Convolution_2d(
                        length, 
                        int(float(self.config[0].text())), 
                        (int(float(self.config[1].text())), int(float(self.config[2].text()))), 
                        (int(float(self.config[3].text())), int(float(self.config[4].text()))), 
                        list(mapping.keys())[list(mapping.values()).index(self.config[5].currentText())], 
                        (int(float(self.config[6].text())), int(float(self.config[7].text()))), 
                        int(float(self.config[8].text())), 
                        list(mapping.keys())[list(mapping.values()).index(self.config[9].currentText())], 
                        list(mapping.keys())[list(mapping.values()).index(self.config[10].currentText())], 
                        list(mapping.keys())[list(mapping.values()).index(self.config[11].currentText())], 
                        list(mapping.keys())[list(mapping.values()).index(self.config[12].currentText())]
                        )
                    )
            # elif self.layer_type == "Convolution 2d Transpose":
            #     nnet.datadict["model_1"]["layers"].append(
            #         Convolution_2d_Transpose(
            #             length, 
            #             int(float(self.config[0].text())), 
            #             (int(float(self.config[1].text()))), 
            #             (int(float(self.config[2].text()))), 
            #             self.config[3].currentText(), 
            #             (int(float(self.config[4].text()))), 
            #             int(float(self.config[5].text())), 
            #             self.config[6].currentText(), 
            #             self.config[7].currentText(), 
            #             self.config[8].currentText(), 
            #             self.config[9].currentText()
            #             )
            #         )
            elif self.layer_type == "Simple RNN":
                nnet.datadict["model_1"]["layers"].append(
                    SimpleRNN(
                        length, 
                        int(float(self.config[0].text())), 
                        list(mapping.keys())[list(mapping.values()).index(self.config[1].currentText())], 
                        list(mapping.keys())[list(mapping.values()).index(self.config[2].currentText())], 
                        list(mapping.keys())[list(mapping.values()).index(self.config[3].currentText())], 
                        list(mapping.keys())[list(mapping.values()).index(self.config[4].currentText())], 
                        list(mapping.keys())[list(mapping.values()).index(self.config[5].currentText())], 
                        float(self.config[6].text()), 
                        float(self.config[7].text())
                        )
                    )
            elif self.layer_type == "LSTM":
                nnet.datadict["model_1"]["layers"].append(
                    LSTM(
                        length, 
                        int(float(self.config[0].text())), 
                        list(mapping.keys())[list(mapping.values()).index(self.config[1].currentText())], 
                        list(mapping.keys())[list(mapping.values()).index(self.config[2].currentText())], 
                        list(mapping.keys())[list(mapping.values()).index(self.config[3].currentText())], 
                        list(mapping.keys())[list(mapping.values()).index(self.config[4].currentText())], 
                        list(mapping.keys())[list(mapping.values()).index(self.config[5].currentText())],
                        list(mapping.keys())[list(mapping.values()).index(self.config[6].currentText())],
                        list(mapping.keys())[list(mapping.values()).index(self.config[7].currentText())], 
                        float(self.config[8].text()), 
                        float(self.config[9].text())
                        )
                    )
            elif self.layer_type == "GRU":
                nnet.datadict["model_1"]["layers"].append(
                    GRU(
                        length, 
                        int(float(self.config[0].text())), 
                        list(mapping.keys())[list(mapping.values()).index(self.config[1].currentText())],
                        list(mapping.keys())[list(mapping.values()).index(self.config[2].currentText())],
                        list(mapping.keys())[list(mapping.values()).index(self.config[3].currentText())],
                        list(mapping.keys())[list(mapping.values()).index(self.config[4].currentText())],
                        list(mapping.keys())[list(mapping.values()).index(self.config[5].currentText())],
                        list(mapping.keys())[list(mapping.values()).index(self.config[6].currentText())],
                        float(self.config[7].text()), 
                        float(self.config[8].text()), 
                        None,
                        list(mapping.keys())[list(mapping.values()).index(self.config[9].currentText())]
                        )
                    )     
        elif(self.layer_location == "After selected"):
            if self.layer_type == "Dense":
                nnet.datadict["model_1"]["layers"].insert(index + 1,
                    Dense(
                        length, 
                        int(float(self.config[0].text())), 
                        list(mapping.keys())[list(mapping.values()).index(self.config[1].currentText())],
                        list(mapping.keys())[list(mapping.values()).index(self.config[2].currentText())],
                        list(mapping.keys())[list(mapping.values()).index(self.config[3].currentText())],
                        list(mapping.keys())[list(mapping.values()).index(self.config[4].currentText())]
                        )
                    )
            elif self.layer_type == "dropout":
                nnet.datadict["model_1"]["layers"].insert(index + 1,
                    Dropout(
                        length,
                        (int(float(self.config[0].text()))), int(self.config[1].text())
                        )
                    )
            elif self.layer_type == "Flatten":
                nnet.datadict["model_1"]["layers"].insert(index + 1,
                    Flatten(
                        length
                        )
                    )
            elif self.layer_type == "Zero Padding 2d":
                nnet.datadict["model_1"]["layers"].insert(index + 1,
                    Zero_Padding_2d(
                        length, 
                        (int(float(self.config[0].text())), int(float(self.config[1].text())))
                        )
                    )
            elif self.layer_type == "Average Pooling 2d":
                nnet.datadict["model_1"]["layers"].insert(index + 1,
                    Average_Pooling_2d(
                        length, 
                        (int(float(self.config[0].text())), int(float(self.config[1].text()))), 
                        (int(float(self.config[2].text())), int(float(self.config[3].text()))), 
                        list(mapping.keys())[list(mapping.values()).index(self.config[4].currentText())]
                        )
                    )
            elif self.layer_type == "Max Pooling 2d":
                nnet.datadict["model_1"]["layers"].insert(index + 1,
                    Max_Pool_2d(
                        length, 
                        (int(float(self.config[0].text())), int(float(self.config[1].text()))), 
                        (int(float(self.config[2].text())), int(float(self.config[3].text()))), 
                        list(mapping.keys())[list(mapping.values()).index(self.config[4].currentText())]
                        )
                    )
            elif self.layer_type == "Convolution 2d":
                nnet.datadict["model_1"]["layers"].insert(index + 1,
                    Convolution_2d(
                        length, 
                        int(float(self.config[0].text())), 
                        (int(float(self.config[1].text())), int(float(self.config[2].text()))), 
                        (int(float(self.config[3].text())), int(float(self.config[4].text()))), 
                        list(mapping.keys())[list(mapping.values()).index(self.config[5].currentText())], 
                        (int(float(self.config[6].text())), int(float(self.config[7].text()))), 
                        int(float(self.config[8].text())), 
                        list(mapping.keys())[list(mapping.values()).index(self.config[9].currentText())], 
                        list(mapping.keys())[list(mapping.values()).index(self.config[10].currentText())],
                        list(mapping.keys())[list(mapping.values()).index(self.config[11].currentText())],
                        list(mapping.keys())[list(mapping.values()).index(self.config[12].currentText())]
                        )
                    )
            
            # elif self.layer_type == "Convolution 2d Transpose":
            #     nnet.datadict["model_1"]["layers"].insert(index + 1,
            #         Convolution_2d_Transpose(
            #             length, 
            #             int(float(self.config[0].text())), 
            #             (int(float(self.config[1].text()))), 
            #             (int(float(self.config[2].text()))), 
            #             self.config[3].currentText(), 
            #             (int(float(self.config[4].text()))), 
            #             int(float(self.config[5].text())), 
            #             self.config[6].currentText(), 
            #             self.config[7].currentText(), 
            #             self.config[8].currentText(), 
            #             self.config[9].currentText()
            #             )
            #         )

            elif self.layer_type == "Simple RNN":
                nnet.datadict["model_1"]["layers"].insert(index + 1,
                    SimpleRNN(
                        length, 
                        int(float(self.config[0].text())), 
                        list(mapping.keys())[list(mapping.values()).index(self.config[1].currentText())], 
                        list(mapping.keys())[list(mapping.values()).index(self.config[2].currentText())], 
                        list(mapping.keys())[list(mapping.values()).index(self.config[3].currentText())], 
                        list(mapping.keys())[list(mapping.values()).index(self.config[4].currentText())], 
                        list(mapping.keys())[list(mapping.values()).index(self.config[5].currentText())], 
                        float(self.config[6].text()), 
                        float(self.config[7].text()), 
                        int(float(self.config[8].text()))
                        )
                    )
            elif self.layer_type == "LSTM":
                nnet.datadict["model_1"]["layers"].insert(index + 1,
                    LSTM(
                        length, 
                        int(float(self.config[0].text())), 
                        list(mapping.keys())[list(mapping.values()).index(self.config[1].currentText())], 
                        list(mapping.keys())[list(mapping.values()).index(self.config[2].currentText())], 
                        list(mapping.keys())[list(mapping.values()).index(self.config[3].currentText())], 
                        list(mapping.keys())[list(mapping.values()).index(self.config[4].currentText())], 
                        list(mapping.keys())[list(mapping.values()).index(self.config[5].currentText())], 
                        list(mapping.keys())[list(mapping.values()).index(self.config[6].currentText())], 
                        list(mapping.keys())[list(mapping.values()).index(self.config[7].currentText())], 
                        float(self.config[8].text()), 
                        float(self.config[9].text()), 
                        int(float(self.config[10].text()))
                        )
                    )
            elif self.layer_type == "GRU":
                nnet.datadict["model_1"]["layers"].insert(index + 1,
                    GRU(
                        length, 
                        int(float(self.config[0].text())), 
                        list(mapping.keys())[list(mapping.values()).index(self.config[1].currentText())], 
                        list(mapping.keys())[list(mapping.values()).index(self.config[2].currentText())], 
                        list(mapping.keys())[list(mapping.values()).index(self.config[3].currentText())], 
                        list(mapping.keys())[list(mapping.values()).index(self.config[4].currentText())], 
                        list(mapping.keys())[list(mapping.values()).index(self.config[5].currentText())], 
                        list(mapping.keys())[list(mapping.values()).index(self.config[6].currentText())], 
                        float(self.config[7].text()), 
                        float(self.config[8].text()), 
                        None, 
                        list(mapping.keys())[list(mapping.values()).index(self.config[10].currentText())]
                        )
                    )
        
        nnet.layoutChanged.emit()
        self.accept()

class AddLayerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Layer")

        layout = QVBoxLayout(self)

        self.label = QLabel("\u24D8 Layer Type:")
        self.label.setToolTip("Type of layer you would like to add")
        layout.addWidget(self.label)

        self.comboBox = QComboBox()
        self.comboBox.addItems(["Dense", "Flatten", "Zero Padding 2d", "Average Pooling 2d", "Max Pooling 2d", "Convolution 2d", "Simple RNN", "LSTM", "GRU", "Dropout"])
        layout.addWidget(self.comboBox)

        self.label2 = QLabel("\u24D8 Layer Location:")
        self.label2.setToolTip("Where do you want to insert this layer")
        layout.addWidget(self.label2)

        self.comboBox2 = QComboBox()
        self.comboBox2.addItems(["End of list", "Beginning of list", "After selected"])
        layout.addWidget(self.comboBox2)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Save)
        layout.addWidget(self.buttons)

        self.buttons.accepted.connect(self.openConfigureAddLayerDialog)
        self.buttons.rejected.connect(self.reject)

        self.comboBox.currentTextChanged.connect(self.updateConfigureDialog)

        self.setMinimumSize(300, 100)

    def openConfigureAddLayerDialog(self):
        dialog = ConfigureAddLayerDialog(self.comboBox.currentText(), self.comboBox2.currentText(), self)
        dialog.exec()
        self.accept()

    def updateConfigureDialog(self, layer_type):
        # Update the configuration dialog when the layer type is changed
        pass

class EditLayerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Layer")

        self.layer = nnet.datadict["model_1"]["layers"][index] 
        self.layer_type = self.layer.layerType
        self.config = []              # Store the configuration for the Layer
        #self.layer_location = layer_location

        layout = QVBoxLayout(self)

        # Create and customize widgets based on layer type
        if self.layer_type =="none":
            return None
        elif self.layer_type == "dense":
            self.configureDenseLayer(layout)
        elif self.layer_type == "flatten":
            self.configureFlattenLayer(layout)
        elif self.layer_type == "zero_padding_2d":
            self.configureZeroPadding2dLayer(layout)
        elif self.layer_type == "average_pooling_2d":
            self.configureAveragePooling2dLayer(layout)
        elif self.layer_type == "max_pool_2d":
            self.configureMaxPooling2dLayer(layout)
        elif self.layer_type == "convolution_2d":
            self.configureConvolution2dLayer(layout)
        elif self.layer_type == "convolution_2d_transpose":
            self.configureConvolution2dTransposeLayer(layout)
        elif self.layer_type == "simplernn":
            self.configureSimpleRNNLayer(layout)
        elif self.layer_type == "lstm":
            self.configureLSTMLayer(layout)
        elif self.layer_type == "gru":
            self.configureGRULayer(layout)
        elif self.layer_type == "dropout":
            self.configureDropoutLayer(layout)
        # Add conditions for other layer types

        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Save)
        layout.addWidget(buttons)

        buttons.accepted.connect(self.saveLayer)
        buttons.rejected.connect(self.reject)

    def configureDropoutLayer(self, layout):
        
        dropout_desc_label = QLabel("Dropout - Layer for applying dropout regularization")
        layout.addWidget(dropout_desc_label)
        
        # Validator for rate text box input (regex for float between 0 and 1)
        rate_validator = QRegularExpressionValidator(QRegularExpression(r'^0(\.\d+)?|1(\.0+)?$'))

        rate_label = QLabel("\u24D8 Rate:")
        rate_label.setToolTip("Float between 0 and 1, fraction of the input units to drop")

        # Add text box for rate
        dropout_rate = QLineEdit()
        dropout_rate.setPlaceholderText("Rate")
        dropout_rate.setValidator(rate_validator)
        dropout_rate.setStyleSheet("border-style: outset;border-width: 2px;")
        layout.addWidget(rate_label)
        layout.addWidget(dropout_rate)

        self.config.append(dropout_rate)

        # Validator for seed text box input (regex for integers)
        seed_validator = QRegularExpressionValidator(QRegularExpression(r'^\d+$'))

        seed_label = QLabel("\u24D8 Seed:")
        seed_label.setToolTip("Integer, random seed for reproducibility")

        # Add text box for seed
        dropout_seed = QLineEdit()
        dropout_seed.setPlaceholderText("Seed")
        dropout_seed.setValidator(seed_validator)
        dropout_seed.setStyleSheet("border-style: outset;border-width: 2px;")
        layout.addWidget(seed_label)
        layout.addWidget(dropout_seed)

        self.config.append(dropout_seed)

        # Add widgets specific to configuring Dropout layer
        pass
        
    def configureDenseLayer(self, layout):

        dense_desc_label = QLabel("Dense - 1D layer of fully connected neurons")
        layout.addWidget(dense_desc_label)

        #validator for text box input (regex)
        validator = QRegularExpressionValidator(QRegularExpression(r'^\d{1,4}$'))

        units_label = QLabel("\u24D8 Units:")
        units_label.setToolTip("Positive integer, dimensionality of the output space")

        #add text box
        dense_units = QLineEdit()
        dense_units.setPlaceholderText("Units")
        dense_units.setValidator(validator)
        dense_units.setStyleSheet("border-style: outset;border-width: 2px;")
        dense_units.setText(str(self.layer.units))
        layout.addWidget(units_label)
        layout.addWidget(dense_units)

        self.config.append(dense_units)

        #activation dropdown
        activation_label = QLabel("\u24D8 Activation Type:")
        activation_label.setToolTip("What activation function to use. If you picked None, no activation is applied")
        layout.addWidget(activation_label)
        dense_activation = QComboBox()
        dense_activation.addItems(["None","elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "sigmoid", "hard sigmoid", "silu", "hard silu", "softmax", "softplus", "tanh"])
        dense_activation.setCurrentText(mapping[self.layer.activation])
        layout.addWidget(dense_activation)

        self.config.append(dense_activation)

        #use bias dropdown
        bias_label = QLabel("\u24D8 Use Bias:")
        bias_label.setToolTip("Boolean, determines whether the layer uses a bias vector")
        layout.addWidget(bias_label)
        use_bias = QComboBox()
        use_bias.addItems(["True", "False"])
        if(self.layer.use_bias):
            use_bias.setCurrentText("True")
        else:
            use_bias.setCurrentText("False")
        layout.addWidget(use_bias)

        self.config.append(use_bias)

        #kernel initializer
        k_initializer_label = QLabel("\u24D8 Kernel Initializer:")
        k_initializer_label.setToolTip("Initializer for the kernel weights matrix")
        layout.addWidget(k_initializer_label)
        kernel_initializer = QComboBox()
        kernel_initializer.addItems(["zeros", "glorot normal", "glorot uniform"])
        kernel_initializer.setCurrentText(mapping[self.layer.kernel_initializer])
        layout.addWidget(kernel_initializer)

        self.config.append(kernel_initializer)

        #bias initializer
        b_initializer_label = QLabel("\u24D8 Bias Initializer:")
        b_initializer_label.setToolTip("Initializer for the bias vector")
        layout.addWidget(b_initializer_label)
        bias_initializer = QComboBox()
        bias_initializer.addItems(["zeros", "glorot normal", "glorot uniform"])
        bias_initializer.setCurrentText(mapping[self.layer.bias_initializer])
        layout.addWidget(bias_initializer)
        
        self.config.append(bias_initializer)

        # Add widgets specific to configuring Dense layer
        pass

    def configureFlattenLayer(self, layout):

        flatten_desc_label = QLabel("Flatten - Transforms 2d input into 1d layer")
        layout.addWidget(flatten_desc_label)

        flatten_label = QLabel("Flatten Layer (no extra parameters needed)")
        layout.addWidget(flatten_label)

        # Add widgets specific to configuring layer
        pass

    def configureZeroPadding2dLayer(self, layout):

        #validator for text box input (regex)
        validator = QRegularExpressionValidator(QRegularExpression(r'^\d{1,2}$'))


        padding_label = QLabel("\u24D8 Padding: (x, y)")
        padding_label.setToolTip("Tuple of 2 ints indicating how many rows/columns to add to input.")
        layout.addWidget(padding_label)

        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        zp2d_padding_x = QLineEdit()
        zp2d_padding_x.setValidator(validator)
        zp2d_padding_x.setPlaceholderText("Padding x value")
        zp2d_padding_x.setText(str(self.layer.padding[0]))
        layout.addWidget(zp2d_padding_x)

        self.config.append(zp2d_padding_x)

        zp2d_padding_y = QLineEdit()
        zp2d_padding_y.setValidator(validator)
        zp2d_padding_y.setPlaceholderText("Padding y value")
        zp2d_padding_y.setText(str(self.layer.padding[1]))
        layout.addWidget(zp2d_padding_y)

        self.config.append(zp2d_padding_y)

        #PROBABLY DOESN'T WORK RIGHT
        zp2d_padding = []
        zp2d_padding.append(zp2d_padding_x.text())
        zp2d_padding.append(zp2d_padding_y.text())

        #print(zp2d_padding)



        # Add widgets specific to configuring layer
        pass

    def configureAveragePooling2dLayer(self, layout):

        avg_pool_desc_label = QLabel("Average Pooling 2D - Reduces input size by averaging a group of inputs in a 2D space and storing them as a single input")
        layout.addWidget(avg_pool_desc_label)

        #validator for text box input (regex)
        validator = QRegularExpressionValidator(QRegularExpression(r'^\d{1,2}$'))


        pool_size_label = QLabel("\u24D8 Pool Size: (x, y)")
        pool_size_label.setToolTip("Tuple of 2 integers, factors by which to downscale (dim1, dim2)")
        layout.addWidget(pool_size_label)

        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        ap2d_poolsize_x = QLineEdit()
        ap2d_poolsize_x.setValidator(validator)
        ap2d_poolsize_x.setPlaceholderText("Pool Size x value")
        ap2d_poolsize_x.setStyleSheet("border-style: outset;border-width: 2px;")
        ap2d_poolsize_x.setText(str(self.layer.pool_size[0]))
        layout.addWidget(ap2d_poolsize_x)

        self.config.append(ap2d_poolsize_x)

        ap2d_poolsize_y = QLineEdit()
        ap2d_poolsize_y.setValidator(validator)
        ap2d_poolsize_y.setPlaceholderText("Pool Size y value")
        ap2d_poolsize_y.setStyleSheet("border-style: outset;border-width: 2px;")
        ap2d_poolsize_y.setText(str(self.layer.pool_size[1]))
        layout.addWidget(ap2d_poolsize_y)

        self.config.append(ap2d_poolsize_y)

        #PROBABLY DOESN'T WORK RIGHT
        ap2d_poolsize = []
        ap2d_poolsize.append(ap2d_poolsize_x.text())
        ap2d_poolsize.append(ap2d_poolsize_y.text())


        strides_label = QLabel("\u24D8 Strides: (x, y)")
        strides_label.setToolTip("Tuple of 2 integers. Distance filter moves in x and y directions")
        layout.addWidget(strides_label)

        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        ap2d_strides_x = QLineEdit()
        ap2d_strides_x.setValidator(validator)
        ap2d_strides_x.setPlaceholderText("Strides x value")
        ap2d_strides_x.setStyleSheet("border-style: outset;border-width: 2px;")
        layout.addWidget(ap2d_strides_x)

        self.config.append(ap2d_strides_x)

        ap2d_strides_y = QLineEdit()
        ap2d_strides_y.setValidator(validator)
        ap2d_strides_y.setPlaceholderText("Strides y value")
        ap2d_strides_y.setStyleSheet("border-style: outset;border-width: 2px;")
        layout.addWidget(ap2d_strides_y)

        if(self.layer.strides is not None):
            ap2d_strides_x.setText(str(self.layer.strides[0]))
            ap2d_strides_y.setText(str(self.layer.strides[1]))

        self.config.append(ap2d_strides_y)

        #PROBABLY DOESN'T WORK RIGHT
        ap2d_strides = []
        ap2d_strides.append(ap2d_strides_x.text())
        ap2d_strides.append(ap2d_strides_y.text())

        #padding dropdown
        padding_label = QLabel("\u24D8 Padding:")
        padding_label.setToolTip("Either valid or same (case-insensitive), valid means no padding. Same results in padding evenly to the left/right or up/down of the input such that output has the same heigh/width dimension as the input")
        layout.addWidget(padding_label)
        ap2d_padding = QComboBox()
        ap2d_padding.addItems(["valid", "same"])
        ap2d_padding.setCurrentText(mapping[self.layer.padding])
  
        layout.addWidget(ap2d_padding)

        self.config.append(ap2d_padding)


        # Add widgets specific to configuring layer
        pass

    def configureMaxPooling2dLayer(self, layout):

        max_pool_desc_label = QLabel("Max pooling 2d - Reduces input size by taking the maximum of a group of inputs in a 2D space and storing them as a single input")
        layout.addWidget(max_pool_desc_label)

        #validator for text box input (regex)
        validator = QRegularExpressionValidator(QRegularExpression(r'^\d{1,2}$'))


        pool_size_label = QLabel("\u24D8 Pool Size: (x, y)")
        pool_size_label.setToolTip("Tuple of 2 integers, factors by which to downscale (dim1, dim2)")
        layout.addWidget(pool_size_label)

        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        mp2d_poolsize_x = QLineEdit()
        mp2d_poolsize_x.setValidator(validator)
        mp2d_poolsize_x.setPlaceholderText("Pool Size x value")
        mp2d_poolsize_x.setStyleSheet("border-style: outset;border-width: 2px;")
        mp2d_poolsize_x.setText(str(self.layer.pool_size[0]))
        layout.addWidget(mp2d_poolsize_x)

        self.config.append(mp2d_poolsize_x)

        mp2d_poolsize_y = QLineEdit()
        mp2d_poolsize_y.setValidator(validator)
        mp2d_poolsize_y.setPlaceholderText("Pool Size y value")
        mp2d_poolsize_y.setStyleSheet("border-style: outset;border-width: 2px;")
        mp2d_poolsize_y.setText(str(self.layer.pool_size[1]))
        layout.addWidget(mp2d_poolsize_y)

        self.config.append(mp2d_poolsize_y)

        #PROBABLY DOESN'T WORK RIGHT
        mp2d_poolsize = []
        mp2d_poolsize.append(mp2d_poolsize_x.text())
        mp2d_poolsize.append(mp2d_poolsize_y.text())


        strides_label = QLabel("\u24D8 Strides: (x, y)")
        strides_label.setToolTip("Tuple of 2 integers. Distance filter moves in x and y directions")
        layout.addWidget(strides_label)

        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        mp2d_strides_x = QLineEdit()
        mp2d_strides_x.setValidator(validator)
        mp2d_strides_x.setPlaceholderText("Strides x value")
        mp2d_strides_x.setStyleSheet("border-style: outset;border-width: 2px;")
        layout.addWidget(mp2d_strides_x)

        self.config.append(mp2d_strides_x)

        mp2d_strides_y = QLineEdit()
        mp2d_strides_y.setValidator(validator)
        mp2d_strides_y.setPlaceholderText("Strides y value")
        mp2d_strides_y.setStyleSheet("border-style: outset;border-width: 2px;")
        layout.addWidget(mp2d_strides_y)

        if(self.layer.strides is not None):
            mp2d_strides_x.setText(str(self.layer.strides[0]))
            mp2d_strides_y.setText(str(self.layer.strides[1]))

        self.config.append(mp2d_strides_y)

        #PROBABLY DOESN'T WORK RIGHT
        mp2d_strides = []
        mp2d_strides.append(mp2d_strides_x.text())
        mp2d_strides.append(mp2d_strides_y.text())

        #padding dropdown
        padding_label = QLabel("\u24D8 Padding:")
        padding_label.setToolTip("Either valid or same (case-insensitive), valid means no padding. Same results in padding evenly to the left/right or up/down of the input such that output has the same heigh/width dimension as the input")
        layout.addWidget(padding_label)
        mp2d_padding = QComboBox()
        mp2d_padding.addItems(["valid", "same"])
        mp2d_padding.setCurrentText(mapping[self.layer.padding])

        layout.addWidget(mp2d_padding)

        self.config.append(mp2d_padding)

        # Add widgets specific to configuring layer
        pass

    def configureConvolution2dLayer(self, layout):

        convolution_2d_desc_label = QLabel("Convolutional 2d - Passes filter over 2D input, creating smaller 2D output that highlights features")
        layout.addWidget(convolution_2d_desc_label)

        #validator for text box input (regex)
        validator = QRegularExpressionValidator(QRegularExpression(r'^\d{1,2}$'))

        #filters
        filters_label = QLabel("\u24D8 Filters:")
        filters_label.setToolTip("Integer, the dimension of the output space (the number of filters in the convolution)")
        layout.addWidget(filters_label)

        #add text box
        c2d_filter = QLineEdit()
        c2d_filter.setPlaceholderText("Filters")
        c2d_filter.setValidator(validator)
        c2d_filter.setStyleSheet("border-style: outset;border-width: 2px;")
        if(self.layer.filter is not None):
            c2d_filter.setText(str(self.layer.filter))
        
        layout.addWidget(c2d_filter)

        self.config.append(c2d_filter)

        #kernel size
        kernelsize_label = QLabel("\u24D8 Kernel Size: (x, y)")
        kernelsize_label.setToolTip("Tuple of 2 integer, specifying the size of the convolution window")
        layout.addWidget(kernelsize_label)


        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        c2d_kernelsize_x = QLineEdit()
        c2d_kernelsize_x.setValidator(validator)
        c2d_kernelsize_x.setPlaceholderText("Kernel Size x value")
        c2d_kernelsize_x.setStyleSheet("border-style: outset;border-width: 2px;")
        layout.addWidget(c2d_kernelsize_x)

        self.config.append(c2d_kernelsize_x)

        c2d_kernelsize_y = QLineEdit()
        c2d_kernelsize_y.setValidator(validator)
        c2d_kernelsize_y.setPlaceholderText("Kernel Size y value")
        c2d_kernelsize_y.setStyleSheet("border-style: outset;border-width: 2px;")
        layout.addWidget(c2d_kernelsize_y)

        if(self.layer.kernel_size is not None):
            c2d_kernelsize_x.setText(str(self.layer.kernel_size[0]))
            c2d_kernelsize_y.setText(str(self.layer.kernel_size[1]))

        self.config.append(c2d_kernelsize_y)

        #strides
        strides_label = QLabel("\u24D8 Strides: (x, y)")
        strides_label.setToolTip("Tuple of 2 integer, specifying the stride length of the convolution. Note - strides > 1 is incompatible with dilation rate > 1")
        layout.addWidget(strides_label)


        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        c2d_strides_x = QLineEdit()
        c2d_strides_x.setValidator(validator)
        c2d_strides_x.setPlaceholderText("Strides x value")
        c2d_strides_x.setText(str(self.layer.strides[0]))
        layout.addWidget(c2d_strides_x)

        self.config.append(c2d_strides_x)

        c2d_strides_y = QLineEdit()
        c2d_strides_y.setValidator(validator)
        c2d_strides_y.setPlaceholderText("Strides y value")
        c2d_strides_y.setText(str(self.layer.strides[1]))  
        layout.addWidget(c2d_strides_y)

      

        self.config.append(c2d_strides_y)

        #PROBABLY DOESN'T WORK RIGHT
        c2d_strides = []
        c2d_strides.append(c2d_strides_x.text())
        c2d_strides.append(c2d_strides_y.text())

        #padding dropdown
        padding_label = QLabel("\u24D8 Padding:")
        padding_label.setToolTip("Either valid or same (case-insensitive), valid means no padding. Same results in padding evenly to the left/right or up/down of the input such that output has the same heigh/width dimension as the input")
        layout.addWidget(padding_label)
        c2d_padding = QComboBox()
        c2d_padding.addItems(["valid", "same"])
        c2d_padding.setCurrentText(mapping[self.layer.padding])

        layout.addWidget(c2d_padding)

        self.config.append(c2d_padding)

        #dialation rate
        dialationrate_label = QLabel("\u24D8 Dilation Rate: (x, y)")
        dialationrate_label.setToolTip("Tuple of 2 integers, specifying the dilation rate to use for dilated convolution")
        layout.addWidget(dialationrate_label)

        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY

        c2d_dialationrate_x = QLineEdit()
        c2d_dialationrate_x.setValidator(validator)
        c2d_dialationrate_x.setPlaceholderText("Dilation Rate x value")
        layout.addWidget(c2d_dialationrate_x)

        self.config.append(c2d_dialationrate_x)

        c2d_dialationrate_y = QLineEdit()
        c2d_dialationrate_y.setValidator(validator)
        c2d_dialationrate_y.setPlaceholderText("Dilation Rate y value")
        layout.addWidget(c2d_dialationrate_y)

        if self.layer.dialation_rate:
            c2d_dialationrate_x.setText(str(self.layer.dialation_rate[0]))
            c2d_dialationrate_y.setText(str(self.layer.dialation_rate[1]))

        self.config.append(c2d_dialationrate_y)

        #groups
        groups_label = QLabel("\u24D8 Groups:")
        groups_label.setToolTip("A positive int specifying the number of groups in which the int is split along the channel axis")
        layout.addWidget(groups_label)

        #add text box
        #ADD THING TO GET MAD IF BOX IS EMPTY
        c2d_groups = QLineEdit()
        c2d_groups.setPlaceholderText("Groups")
        c2d_groups.setValidator(validator)
        c2d_groups.setText("1")
        c2d_groups.setText(str(self.layer.groups))

        layout.addWidget(c2d_groups)

        self.config.append(c2d_groups)

        #activation dropdown
        activation_label = QLabel("\u24D8 Activation Type:")
        activation_label.setToolTip("Activation function. If None, no activation is applied")
        layout.addWidget(activation_label)
        c2d_activation = QComboBox()
        c2d_activation.addItems(["None","elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "sigmoid", "hard sigmoid", "silu", "hard silu", "softmax", "softplus", "tanh"])
        c2d_activation.setCurrentText(mapping[self.layer.activation])

        layout.addWidget(c2d_activation)

        self.config.append(c2d_activation)

        #use bias dropdown
        bias_label = QLabel("\u24D8 Use Bias:")
        bias_label.setToolTip("Bool, if True, bias will be added to the output.")
        layout.addWidget(bias_label)
        use_bias = QComboBox()
        use_bias.addItems(["True", "False"])
        use_bias.setCurrentText(mapping[self.layer.use_bias]) 

        layout.addWidget(use_bias)

        self.config.append(use_bias)

        #kernel initializer
        k_initializer_label = QLabel("\u24D8 Kernel Initializer:")
        k_initializer_label.setToolTip("Initializer for the convolution kernel")
        layout.addWidget(k_initializer_label)
        kernel_initializer = QComboBox()
        kernel_initializer.addItems(["glorot uniform", "zeros", "glorot normal", "orthogonal"])
        kernel_initializer.setCurrentText(mapping[self.layer.kernel_initializer])

        layout.addWidget(kernel_initializer)

        self.config.append(kernel_initializer)

        #bias initializer
        b_initializer_label = QLabel("\u24D8 Bias Initializer:")
        b_initializer_label.setToolTip("Initializer for the bias vector")
        layout.addWidget(b_initializer_label)
        bias_initializer = QComboBox()
        bias_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        bias_initializer.setCurrentText(mapping[self.layer.bias_initializer])

        layout.addWidget(bias_initializer)

        self.config.append(bias_initializer)

        # Add widgets specific to configuring layer
        pass

    # def configureConvolution2dTransposeLayer(self, layout):

    #     #validator for text box input (regex)
    #     validator = QRegularExpressionValidator(QRegularExpression(r'^\d{1,2}$'))

    #     #filters
    #     filters_label = QLabel("\u24D8 Filters:")
    #     filters_label.setToolTip("Integer, the dimension of the output space (the number of filters in the convolution)")
    #     layout.addWidget(filters_label)

    #     #add text box
    #     c2dt_filter = QLineEdit()
    #     c2dt_filter.setPlaceholderText("Filters")
    #     c2dt_filter.setValidator(validator)
    #     c2dt_filter.setStyleSheet("border-style: outset;border-width: 2px;")
    #     if(self.layer.filter is not None):
    #         c2dt_filter.setText(str(self.layer.filter))
    #     layout.addWidget(c2dt_filter)

    #     self.config.append(c2dt_filter)

    #     #kernel size
    #     kernelsize_label = QLabel("Kernel Size:")

    #     layout.addWidget(kernelsize_label)


    #     #add text box
    #     #ADD THING TO GET MAD IF BOX IS EMPTY

    #     c2dt_kernelsize_x = QLineEdit()
    #     c2dt_kernelsize_x.setValidator(validator)
    #     c2dt_kernelsize_x.setPlaceholderText("Kernel Size value")
    #     c2dt_kernelsize_x.setStyleSheet("border-style: outset;border-width: 2px;")
    #     layout.addWidget(c2dt_kernelsize_x)

    #     self.config.append(c2dt_kernelsize_x)

    #     #strides
    #     strides_label = QLabel("Strides:")
    #     layout.addWidget(strides_label)


    #     #add text box
    #     #ADD THING TO GET MAD IF BOX IS EMPTY

    #     c2dt_strides_x = QLineEdit()
    #     c2dt_strides_x.setValidator(validator)
    #     c2dt_strides_x.setPlaceholderText("Strides value")
    #     c2dt_strides_x.setText(str(self.layer.strides))
    #     layout.addWidget(c2dt_strides_x)

    #     self.config.append(c2dt_strides_x)

    #     #PROBABLY DOESN'T WORK RIGHT
    #     c2dt_strides = []
    #     c2dt_strides.append(c2dt_strides_x.text())

    #     #padding dropdown
    #     padding_label = QLabel("Padding:")
    #     layout.addWidget(padding_label)
    #     c2dt_padding = QComboBox()
    #     c2dt_padding.addItems(["valid", "same"])
    #     if self.layer.padding:
    #         index = c2dt_padding.findText(self.layer.padding, QtCore.Qt.MatchFixedString)
    #     if index >= 0:
    #         c2dt_padding.setCurrentIndex(index)

    #     layout.addWidget(c2dt_padding)

    #     self.config.append(c2dt_padding)

    #     #dialation rate
    #     dialationrate_label = QLabel("Dilation Rate:")
    #     layout.addWidget(dialationrate_label)

    #     #add text box
    #     #ADD THING TO GET MAD IF BOX IS EMPTY

    #     c2dt_dialationrate_x = QLineEdit()
    #     c2dt_dialationrate_x.setValidator(validator)
    #     c2dt_dialationrate_x.setPlaceholderText("Dilation Rate value")
    #     layout.addWidget(c2dt_dialationrate_x)

    #     self.config.append(c2dt_dialationrate_x)

    #     if self.layer.dialation_rate:
    #         c2dt_dialationrate_x.setText(str(self.layer.dialation_rate))


    #     #groups
    #     groups_label = QLabel("Groups:")
    #     layout.addWidget(groups_label)

    #     #add text box
    #     #ADD THING TO GET MAD IF BOX IS EMPTY
    #     c2dt_groups = QLineEdit()
    #     c2dt_groups.setPlaceholderText("Groups")
    #     c2dt_groups.setValidator(validator)
    #     c2dt_groups.setText(str(self.layer.groups))
    #     layout.addWidget(c2dt_groups)

    #     self.config.append(c2dt_groups)

    #     #activation dropdown
    #     activation_label = QLabel("Activation Type:")
    #     layout.addWidget(activation_label)
    #     c2dt_activation = QComboBox()
    #     c2dt_activation.addItems(["None","elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "sigmoid", "hard sigmoid", "silu", "hard silu", "softmax", "softplus", "tanh"])
    #     layout.addWidget(c2dt_activation)

    #     self.config.append(c2dt_activation)

    #     #use bias dropdown
    #     bias_label = QLabel("Use Bias:")
    #     layout.addWidget(bias_label)
    #     use_bias = QComboBox()
    #     use_bias.addItems(["True", "False"])
    #     if self.layer.use_bias:
    #         index = use_bias.findText(self.layer.use_bias, QtCore.Qt.MatchFixedString)
    #     if index >= 0:
    #         use_bias.setCurrentIndex(index)

    #     layout.addWidget(use_bias)

    #     self.config.append(use_bias)

    #     #kernel initializer
    #     k_initializer_label = QLabel("Kernel Initializer:")
    #     layout.addWidget(k_initializer_label)
    #     kernel_initializer = QComboBox()
    #     kernel_initializer.addItems(["glorot uniform", "zeros", "glorot normal", "orthogonal"])
    #     if self.layer.kernel_initializer:
    #         index = kernel_initializer.findText(self.layer.kernel_initializer, QtCore.Qt.MatchFixedString)
    #     if index >= 0:
    #         kernel_initializer.setCurrentIndex(index)

    #     layout.addWidget(kernel_initializer)

    #     self.config.append(kernel_initializer)

    #     #bias initializer
    #     b_initializer_label = QLabel("Bias Initializer:")
    #     layout.addWidget(b_initializer_label)
    #     bias_initializer = QComboBox()
    #     bias_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
    #     layout.addWidget(bias_initializer)
    #     if self.layer.bias_initializer:
    #         index = bias_initializer.findText(self.layer.bias_initializer, QtCore.Qt.MatchFixedString)
    #     if index >= 0:
    #         bias_initializer.setCurrentIndex(index)

    #     self.config.append(bias_initializer)



    #     # Add widgets specific to configuring layer
    #     pass

    def configureSimpleRNNLayer(self, layout):

        simple_rnn_desc_label = QLabel("Basic Recurrent Neural Network, utilizes previously seen data during training to better forecast future sequential data")
        layout.addWidget(simple_rnn_desc_label)

        #validator for text box input (regex)
        validator = QRegularExpressionValidator(QRegularExpression(r'^\d{1,2}$'))

        float_validator = QRegularExpressionValidator(QRegularExpression('^(\d)*(\.)?([0-9]{1})?$'))

        #units
        units_label = QLabel("\u24D8 Units:")
        units_label.setToolTip("Positive integer, dimensionality of the output space")
        layout.addWidget(units_label)

        #add text box
        srnn_units = QLineEdit()
        srnn_units.setPlaceholderText("Units")
        srnn_units.setValidator(validator)
        srnn_units.setText(str(self.layer.units))
        layout.addWidget(srnn_units)

        self.config.append(srnn_units)

        #activation dropdown
        activation_label = QLabel("\u24D8 Activation Type:")
        activation_label.setToolTip("Activation function to use")
        layout.addWidget(activation_label)
        srnn_activation = QComboBox()
        srnn_activation.addItems(["elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "sigmoid", "hard sigmoid", "silu", "hard silu", "softmax", "softplus", "tanh"])
        srnn_activation.setCurrentText(mapping[self.layer.activation])

        layout.addWidget(srnn_activation)

        self.config.append(srnn_activation)

        #use bias dropdown
        bias_label = QLabel("\u24D8 Use Bias:")
        bias_label.setToolTip("Boolean, whether the layer uses a bias vector")
        layout.addWidget(bias_label)
        use_bias = QComboBox()
        use_bias.addItems(["True", "False"])
        use_bias.setCurrentText(mapping[self.layer.use_bias])

        layout.addWidget(use_bias)

        self.config.append(use_bias)

        #kernel initializer
        k_initializer_label = QLabel("\u24D8 Kernel Initializer:")
        k_initializer_label.setToolTip("Initializer for the kernel weights matrix, used for the linear transformation of the inputs")
        layout.addWidget(k_initializer_label)
        kernel_initializer = QComboBox()
        kernel_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        kernel_initializer.setCurrentText(mapping[self.layer.kernel_initializer])

        layout.addWidget(kernel_initializer)

        self.config.append(kernel_initializer)

        #recurrent initializer
        r_initializer_label = QLabel("\u24D8 Recurrent Initializer:")
        r_initializer_label.setToolTip("Initializer for the recurrent_kernel weights matrix, used for the linear transformation of the recurrent state")
        layout.addWidget(r_initializer_label)
        recurrent_initializer = QComboBox()
        recurrent_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        recurrent_initializer.setCurrentText(self.layer.recurrent_initializer)

        layout.addWidget(recurrent_initializer)

        self.config.append(recurrent_initializer)

        #bias initializer
        b_initializer_label = QLabel("\u24D8 Bias Initializer:")
        b_initializer_label.setToolTip("Initializer for the bias vector")
        layout.addWidget(b_initializer_label)
        bias_initializer = QComboBox()
        bias_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        bias_initializer.setCurrentText(mapping[self.layer.bias_initializer])
        
        layout.addWidget(bias_initializer)

        self.config.append(bias_initializer)

        #dropout
        dropout_label = QLabel("\u24D8 Dropout:")
        dropout_label.setToolTip("Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs. Default is 0.")
        layout.addWidget(dropout_label)

        #add text box
        #GET MAD IF USER PUTS IN FUNKY STUFF (anything besides numbers and a decimal)
        srnn_dropout = QLineEdit()
        srnn_dropout.setPlaceholderText("Dropout")
        srnn_dropout.setValidator(float_validator)
        srnn_dropout.setText(str(self.layer.dropout))
        layout.addWidget(srnn_dropout)

        self.config.append(srnn_dropout)

        #recurrent dropout
        recurrent_dropout_label = QLabel("\u24D8 Recurrent Dropout:")
        recurrent_dropout_label.setToolTip("Recurrent Dropout - Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state. Default: 0.")
        layout.addWidget(recurrent_dropout_label)

        #add text box
        #GET MAD IF USER PUTS IN FUNKY STUFF (anything besides numbers and a decimal)
        srnn_recurrent_dropout = QLineEdit()
        srnn_recurrent_dropout.setPlaceholderText("Recurrent Dropout")
        srnn_recurrent_dropout.setText(str(self.layer.recurrent_dropout))
        srnn_recurrent_dropout.setValidator(float_validator)
        layout.addWidget(srnn_recurrent_dropout)

        self.config.append(srnn_recurrent_dropout)

        # Add widgets specific to configuring layer
        pass

    def configureLSTMLayer(self, layout):

        lstm_desc_label = QLabel("Long Short-Term Memory layer - Subset of RNN able to handle long-term dependencies, takes in 2D input and outputs 1D data")
        layout.addWidget(lstm_desc_label)

        #validator for text box input (regex)
        validator = QRegularExpressionValidator(QRegularExpression(r'^\d{1,2}$'))

        float_validator = QRegularExpressionValidator(QRegularExpression('^(\d)*(\.)?([0-9]{1})?$'))

        #units
        units_label = QLabel("\u24D8 Units:")
        units_label.setToolTip("Positive integer, dimensionality of the output space")
        layout.addWidget(units_label)

        #add text box
        lstm_units = QLineEdit()
        lstm_units.setPlaceholderText("Units")
        lstm_units.setValidator(validator)
        lstm_units.setText(str(self.layer.units))
        layout.addWidget(lstm_units)

        self.config.append(lstm_units)

        #activation dropdown
        activation_label = QLabel("\u24D8 Activation Type:")
        activation_label.setToolTip("Activation function to use.")
        layout.addWidget(activation_label)
        lstm_activation = QComboBox()
        lstm_activation.addItems(["elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "sigmoid", "hard sigmoid", "silu", "hard silu", "softmax", "softplus", "tanh"])
        lstm_activation.setCurrentText(mapping[self.layer.activation])
       
        layout.addWidget(lstm_activation)

        self.config.append(lstm_activation)

        #recurrent activation dropdown
        recurrent_activation_label = QLabel("\u24D8 Recurrent Activation Type:")
        recurrent_activation_label.setToolTip("Activation function to use for the recurrent step.")
        layout.addWidget(recurrent_activation_label)
        lstm_recurrent_activation = QComboBox()
        lstm_recurrent_activation.addItems(["elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "sigmoid", "hard sigmoid", "silu", "hard silu", "softmax", "softplus", "tanh"])
        lstm_recurrent_activation.setCurrentText(mapping[self.layer.recurrent_activation])     
        
        layout.addWidget(lstm_recurrent_activation)

        self.config.append(lstm_recurrent_activation)

        #use bias dropdown
        bias_label = QLabel("\u24D8 Use Bias:")
        bias_label.setToolTip("Boolean, whether the layer should use a bias vector.")
        layout.addWidget(bias_label)
        use_bias = QComboBox()
        use_bias.addItems(["True", "False"])
        use_bias.setCurrentText(mapping[self.layer.use_bias])

        layout.addWidget(use_bias)

        self.config.append(use_bias)

        #kernel initializer
        k_initializer_label = QLabel("\u24D8 Kernel Initializer:")
        k_initializer_label.setToolTip("Initializer for the kernel weights matrix, used for the linear transformation of the inputs.")
        layout.addWidget(k_initializer_label)
        kernel_initializer = QComboBox()
        kernel_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        kernel_initializer.setCurrentText(mapping[self.layer.kernel_initializer])
       
        layout.addWidget(kernel_initializer)

        self.config.append(kernel_initializer)

        #recurrent initializer
        r_initializer_label = QLabel("\u24D8 Recurrent Initializer:")
        r_initializer_label.setToolTip("Initializer for the recurrent kernel weights matrix, used for the linear transformation of the recurrent state.")
        layout.addWidget(r_initializer_label)
        recurrent_initializer = QComboBox()
        recurrent_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        recurrent_initializer.setCurrentText(mapping[self.layer.kernel_initializer])

        layout.addWidget(recurrent_initializer)

        self.config.append(recurrent_initializer)

        #bias initializer
        b_initializer_label = QLabel("\u24D8 Bias Initializer:")
        b_initializer_label.setToolTip("Initializer for the bias vector")
        layout.addWidget(b_initializer_label)
        bias_initializer = QComboBox()
        bias_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        bias_initializer.setCurrentText(mapping[self.layer.bias_initializer]) 
        
        layout.addWidget(bias_initializer)

        self.config.append(bias_initializer)

        #unit forget bias dropdown
        unit_forget_bias_label = QLabel("\u24D8 Unit Forget Bias:")
        unit_forget_bias_label.setToolTip("Boolean. If True, add 1 to the bias of the forget gate at initialization. Setting it to True will also force bias initializer = “zeros”.")
        layout.addWidget(unit_forget_bias_label)
        unit_forget_bias = QComboBox()
        unit_forget_bias.addItems(["True", "False"])
        unit_forget_bias.setCurrentText(mapping[self.layer.unit_forget_bias])

        layout.addWidget(unit_forget_bias)

        self.config.append(unit_forget_bias)


        #dropout
        dropout_label = QLabel("\u24D8 Dropout:")
        dropout_label.setToolTip("Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs. Default is 0.")
        layout.addWidget(dropout_label)

        #add text box
        #GET MAD IF USER PUTS IN FUNKY STUFF (anything besides numbers and a decimal)
        lstm_dropout = QLineEdit()
        lstm_dropout.setPlaceholderText("Dropout")
        lstm_dropout.setValidator(float_validator)
        lstm_dropout.setText(str(self.layer.dropout))
        layout.addWidget(lstm_dropout)

        self.config.append(lstm_dropout)

        #recurrent dropout
        recurrent_dropout_label = QLabel("\u24D8 Recurrent Dropout:")
        recurrent_dropout_label.setToolTip("Recurrent Dropout - Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state. Default: 0.")
        layout.addWidget(recurrent_dropout_label)

        #add text box
        #GET MAD IF USER PUTS IN FUNKY STUFF (anything besides numbers and a decimal)
        lstm_recurrent_dropout = QLineEdit()
        lstm_recurrent_dropout.setPlaceholderText("Recurrent Dropout")
        lstm_recurrent_dropout.setValidator(float_validator)
        lstm_recurrent_dropout.setText(str(self.layer.recurrent_dropout))
        layout.addWidget(lstm_recurrent_dropout)

        self.config.append(lstm_recurrent_dropout)

        # Add widgets specific to configuring layer
        pass

    def configureGRULayer(self, layout):

        gru_desc_label = QLabel("Gated Recurrent Unit - Subset of RNN that allows for selective storing and replacing of previously seen data")
        layout.addWidget(gru_desc_label)

        #validator for text box input (regex)
        validator = QRegularExpressionValidator(QRegularExpression(r'^\d{1,2}$'))

        float_validator = QRegularExpressionValidator(QRegularExpression('^(\d)*(\.)?([0-9]{1})?$'))

        #units
        units_label = QLabel("\u24D8 Units:")
        units_label.setToolTip("Positive integer, dimensionality of the output space")
        layout.addWidget(units_label)

        #add text box
        gru_units = QLineEdit()
        gru_units.setPlaceholderText("Units")
        gru_units.setValidator(validator)
        gru_units.setText(str(self.layer.units))
        layout.addWidget(gru_units)

        self.config.append(gru_units)

        #activation dropdown
        activation_label = QLabel("\u24D8 Activation Type:")
        activation_label.setToolTip("Activation function to use.")
        layout.addWidget(activation_label)
        gru_activation = QComboBox()
        gru_activation.addItems(["elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "sigmoid", "hard sigmoid", "silu", "hard silu", "softmax", "softplus", "tanh"])
        gru_activation.setCurrentText(mapping[self.layer.activation])

        layout.addWidget(gru_activation)

        self.config.append(gru_activation)

        #recurrent activation dropdown
        recurrent_activation_label = QLabel("\u24D8 Recurrent Activation Type:")
        recurrent_activation_label.setToolTip("Activation function to use for the recurrent step.")
        layout.addWidget(recurrent_activation_label)
        gru_recurrent_activation = QComboBox()
        gru_recurrent_activation.addItems(["elu", "exponential", "gelu", "linear", "relu", "relu6", "leaky_relu", "mish", "selu", "sigmoid", "hard sigmoid", "silu", "hard silu", "softmax", "softplus", "tanh"])
        gru_recurrent_activation.setCurrentText(mapping[self.layer.recurrent_activation])

        layout.addWidget(gru_recurrent_activation)

        self.config.append(gru_recurrent_activation)

        #use bias dropdown
        bias_label = QLabel("\u24D8 Use Bias:")
        bias_label.setToolTip("Boolean, whether the layer should use a bias vector.")
        layout.addWidget(bias_label)
        use_bias = QComboBox()
        use_bias.addItems(["True", "False"])
        use_bias.setCurrentText(mapping[self.layer.use_bias])

        layout.addWidget(use_bias)

        self.config.append(use_bias)

        #kernel initializer
        k_initializer_label = QLabel("\u24D8 Kernel Initializer:")
        k_initializer_label.setToolTip("Initializer for the kernel weights matrix, used for the linear transformation of the inputs.")
        layout.addWidget(k_initializer_label)
        kernel_initializer = QComboBox()
        kernel_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        kernel_initializer.setCurrentText(mapping[self.layer.kernel_initializer])        
        
        layout.addWidget(kernel_initializer)

        self.config.append(kernel_initializer)

        #recurrent initializer
        r_initializer_label = QLabel("\u24D8 Recurrent Initializer:")
        r_initializer_label.setToolTip("Initializer for the recurrent kernel weights matrix, used for the linear transformation of the recurrent state.")
        layout.addWidget(r_initializer_label)
        recurrent_initializer = QComboBox()
        recurrent_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        recurrent_initializer.setCurrentText(mapping[self.layer.recurrent_initializer])

        layout.addWidget(recurrent_initializer)

        self.config.append(recurrent_initializer)

        #bias initializer
        b_initializer_label = QLabel("\u24D8 Bias Initializer:")
        b_initializer_label.setToolTip("Initializer for the bias vector")
        layout.addWidget(b_initializer_label)
        bias_initializer = QComboBox()
        bias_initializer.addItems(["zeros", "glorot normal", "glorot uniform", "orthogonal"])
        bias_initializer.setCurrentText(mapping[self.layer.bias_initializer])      
        
        layout.addWidget(bias_initializer)

        self.config.append(bias_initializer)

        #dropout
        dropout_label = QLabel("\u24D8 Dropout:")
        dropout_label.setToolTip("Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs. Default is 0.")
        layout.addWidget(dropout_label)

        #add text box
        #GET MAD IF USER PUTS IN FUNKY STUFF (anything besides numbers and a decimal)
        gru_dropout = QLineEdit()
        gru_dropout.setPlaceholderText("Dropout")
        gru_dropout.setValidator(float_validator)
        gru_dropout.setText(str(self.layer.dropout))
        layout.addWidget(gru_dropout)

        self.config.append(gru_dropout)

        #recurrent dropout
        recurrent_dropout_label = QLabel("\u24D8 Recurrent Dropout:")
        recurrent_dropout_label.setToolTip("Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state. Default: 0")
        layout.addWidget(recurrent_dropout_label)

        #add text box
        #GET MAD IF USER PUTS IN FUNKY STUFF (anything besides numbers and a decimal)
        gru_recurrent_dropout = QLineEdit()
        gru_recurrent_dropout.setPlaceholderText("Recurrent Dropout")
        gru_recurrent_dropout.setValidator(float_validator)
        gru_recurrent_dropout.setText(str(self.layer.recurrent_dropout))
        layout.addWidget(gru_recurrent_dropout)

        self.config.append(gru_recurrent_dropout)

        #reset after dropdown
        reset_after_label = QLabel("\u24D8 Reset After:")
        reset_after_label.setToolTip("GRU convention (whether to apply reset gate after or before matrix multiplication). False is before, True is after.")
        layout.addWidget(reset_after_label)
        reset_after = QComboBox()
        reset_after.addItems(["True", "False"])
        reset_after.setCurrentText(mapping[self.layer.reset_after])

        layout.addWidget(reset_after)

        self.config.append(reset_after)

        # Add widgets specific to configuring layer
        pass

    def saveLayer(self):
        # Append to main dictionary the layer type the user wants to add
        length = len(nnet.datadict["model_1"]["layers"])

        global index


        if self.layer_type == "dense":
            nnet.datadict["model_1"]["layers"].insert(index,
                Dense(
                    length, 
                    int(float(self.config[0].text())), 
                    list(mapping.keys())[list(mapping.values()).index(self.config[1].currentText())],
                    list(mapping.keys())[list(mapping.values()).index(self.config[2].currentText())],
                    list(mapping.keys())[list(mapping.values()).index(self.config[3].currentText())],
                    list(mapping.keys())[list(mapping.values()).index(self.config[4].currentText())]
                    )
                )
        elif self.layer_type == "dropout":
            nnet.datadict["model_1"]["layers"].insert(index,
                Dropout(
                    length,
                    (int(float(self.config[0].text()))), int(self.config[1].text())
                    )
                )
            
        elif self.layer_type == "zero_padding_2d":
            nnet.datadict["model_1"]["layers"].insert(index,
                Zero_Padding_2d(
                    length, 
                    (int(float(self.config[0].text())), int(float(self.config[1].text())))
                    )
                )
        elif self.layer_type == "average_pooling_2d":
            nnet.datadict["model_1"]["layers"].insert(index,
                Average_Pooling_2d(
                    length, 
                    (int(float(self.config[0].text())), int(float(self.config[1].text()))), 
                    (int(float(self.config[2].text())), int(float(self.config[3].text()))), 
                    list(mapping.keys())[list(mapping.values()).index(self.config[4].currentText())]
                    )
                )
        elif self.layer_type == "max_pool_2d":
            nnet.datadict["model_1"]["layers"].insert(index,
                Max_Pool_2d(
                    length, 
                    (int(float(self.config[0].text())), int(float(self.config[1].text()))), 
                    (int(float(self.config[2].text())), int(float(self.config[3].text()))), 
                    list(mapping.keys())[list(mapping.values()).index(self.config[4].currentText())]
                    )
                )
        elif self.layer_type == "convolution_2d":
            nnet.datadict["model_1"]["layers"].insert(index,
                Convolution_2d(
                    length, 
                    int(float(self.config[0].text())), 
                    (int(float(self.config[1].text())), int(float(self.config[2].text()))), 
                    (int(float(self.config[3].text())), int(float(self.config[4].text()))), 
                    list(mapping.keys())[list(mapping.values()).index(self.config[5].currentText())], 
                    (int(float(self.config[6].text())), int(float(self.config[7].text()))), 
                    int(float(self.config[8].text())), 
                    list(mapping.keys())[list(mapping.values()).index(self.config[9].currentText())], 
                    list(mapping.keys())[list(mapping.values()).index(self.config[10].currentText())],
                    list(mapping.keys())[list(mapping.values()).index(self.config[11].currentText())],
                    list(mapping.keys())[list(mapping.values()).index(self.config[12].currentText())]
                    )
                )
        # elif self.layer_type == "convolution_2d_transpose":
        #     nnet.datadict["model_1"]["layers"].insert(index,
        #         Convolution_2d_Transpose(
        #             length, 
        #             int(float(self.config[0].text())), 
        #             (int(float(self.config[1].text()))), 
        #             (int(float(self.config[2].text()))), 
        #             self.config[3].currentText(), 
        #             (int(float(self.config[4].text()))), 
        #             int(float(self.config[5].text())), 
        #             self.config[6].currentText(), 
        #             self.config[7].currentText(), 
        #             self.config[8].currentText(), 
        #             self.config[9].currentText()
        #             )
        #         )
        elif self.layer_type == "simplernn":
            nnet.datadict["model_1"]["layers"].insert(index,
                SimpleRNN(
                    length, 
                    int(float(self.config[0].text())), 
                    list(mapping.keys())[list(mapping.values()).index(self.config[1].currentText())],
                    list(mapping.keys())[list(mapping.values()).index(self.config[2].currentText())],
                    list(mapping.keys())[list(mapping.values()).index(self.config[3].currentText())],
                    list(mapping.keys())[list(mapping.values()).index(self.config[4].currentText())],
                    list(mapping.keys())[list(mapping.values()).index(self.config[5].currentText())], 
                    float(self.config[6].text()), 
                    float(self.config[7].text())
                    )
                )
        elif self.layer_type == "lstm":
            nnet.datadict["model_1"]["layers"].insert(index,
                LSTM(
                    length, 
                    int(float(self.config[0].text())), 
                    list(mapping.keys())[list(mapping.values()).index(self.config[1].currentText())], 
                    list(mapping.keys())[list(mapping.values()).index(self.config[2].currentText())],
                    list(mapping.keys())[list(mapping.values()).index(self.config[3].currentText())],
                    list(mapping.keys())[list(mapping.values()).index(self.config[4].currentText())],
                    list(mapping.keys())[list(mapping.values()).index(self.config[5].currentText())],
                    list(mapping.keys())[list(mapping.values()).index(self.config[6].currentText())],
                    list(mapping.keys())[list(mapping.values()).index(self.config[7].currentText())],
                    float(self.config[8].text()), 
                    float(self.config[9].text())
                    )
                )
        elif self.layer_type == "gru":
            nnet.datadict["model_1"]["layers"].insert(index,
                GRU(
                    length, 
                    int(float(self.config[0].text())), 
                    list(mapping.keys())[list(mapping.values()).index(self.config[1].currentText())],
                    list(mapping.keys())[list(mapping.values()).index(self.config[2].currentText())],
                    list(mapping.keys())[list(mapping.values()).index(self.config[3].currentText())],
                    list(mapping.keys())[list(mapping.values()).index(self.config[4].currentText())],
                    list(mapping.keys())[list(mapping.values()).index(self.config[5].currentText())],
                    list(mapping.keys())[list(mapping.values()).index(self.config[6].currentText())],
                    float(self.config[7].text()), 
                    float(self.config[8].text()), 
                    None,
                    list(mapping.keys())[list(mapping.values()).index(self.config[9].currentText())]
                    )
                )    
        
        del nnet.datadict["model_1"]["layers"][index+1]


        nnet.layoutChanged.emit()
        self.accept()

class UsePresetDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Use Preset")

        layout = QVBoxLayout(self)

        #Dropdown menu for selecting options
        self.comboBox = QComboBox()

        #Options for it
        self.comboBox.addItems(["Standard Neural Network", "Convolutional NN", "Recurrent NN"])
        layout.addWidget(self.comboBox)

        #Dialog buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Save)
        layout.addWidget(self.buttons)

        #Connect buttons
        self.buttons.accepted.connect(self.confirmPreset)
        self.buttons.rejected.connect(self.reject)

        self.setMinimumSize(300, 100)

    def confirmPreset(self):
        dialog = ConfirmPresetDialog(self.comboBox.currentText(), self)
        dialog.exec()
        self.accept()

class ConfirmPresetDialog(QDialog):

    def __init__(self, preset, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Confirm Preset")

        layout = QVBoxLayout(self)

        self.preset = preset

        #slider for testing split
        self.label = QLabel('Are you sure you want to use this preset?\nDoing so will delete your previous layers')
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        #Dialog buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Save)
        layout.addWidget(self.buttons)

        #Connect buttons
        self.buttons.accepted.connect(self.usePreset)
        self.buttons.rejected.connect(self.reject)

        self.setMinimumSize(300, 100)

    def usePreset(self):

        nnet.datadict["model_1"]["layers"] = []
        for layer in defaults[self.preset]:
            nnet.datadict["model_1"]["layers"].append(layer)
        nnet.layoutChanged.emit()
        self.accept()

class ConfigureInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Input")

        self.config = []

        layout = QVBoxLayout(self)

        # File selection components
        file_selection_layout = QHBoxLayout()
        if nnet.datadict["input_parameters"]["root_directory"] == "":

            self.file_path_label = QLabel("\u24D8 No Directory Selected")
        else:
            self.file_path_label = QLabel("\u24D8" + nnet.datadict["input_parameters"]["root_directory"])
        self.file_path_label.setToolTip("Select the root directory of the input data.")
        self.file_path_label.setWordWrap(True)  # Allow label to wrap if path is too long
        select_file_button = QPushButton("Select Directory")
        file_selection_layout.addWidget(select_file_button)
        file_selection_layout.addWidget(self.file_path_label, 1)  # The '1' makes the label expandable

        frame = QFrame()
        label = QLabel()

        pixmap = QPixmap('directoryformat.jpg')
        scaled_pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)

        frame_layout = QVBoxLayout(frame)
        frame_layout.addWidget(label)
        layout.addWidget(frame)

        label2 = QLabel("Select root data folder. Data files must be in correct format. \nAny number of participants, results, or trials can be used. \nResult names will be used as data labels.")
        layout.addWidget(label2)

        select_file_button.clicked.connect(self.selectDirectory)

        layout.addLayout(file_selection_layout)

        # Input Shape components grouped together
        input_shape_group = QGroupBox("\u24D8 Input Shape")
        input_shape_group.setToolTip("Input shape of the CSV files. Rows x Columns. (All files must have same layout)")
        input_shape_layout = QHBoxLayout(input_shape_group)

        width_input = QLineEdit()
     
        height_input = QLineEdit()

        validator = QRegularExpressionValidator(QRegularExpression("[0-9]{0,4}"))
        width_input.setValidator(validator)
        height_input.setValidator(validator)

        self.config.append(width_input)
        self.config.append(height_input)

        input_shape_layout.addWidget(width_input)
        width_input.setText(str(nnet.datadict["input_parameters"]["input_shape1"]))
        input_shape_layout.addWidget(QLabel("X"))
        input_shape_layout.addWidget(height_input)
        height_input.setText(str(nnet.datadict["input_parameters"]["input_shape2"]))

        layout.addWidget(input_shape_group)

            # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)

        buttons.accepted.connect(self.saveConfig) # Calls the saveConfig function once save button is pressed
        buttons.rejected.connect(self.reject)
        
    def selectDirectory(self):
        # Prompts a select directory screen
        file_path = QFileDialog.getExistingDirectory(self, "Select directory")
        if file_path:  # Only update the label if a file path was selected
            self.file_path_label.setText(file_path)
            nnet.datadict["input_parameters"]["root_directory"] = file_path # Saves file path to main dictionary

    def saveConfig(self):
        # Saves the Input shape size and normalization to main dictionary.
        nnet.datadict["input_parameters"]["input_shape1"] = self.config[0].text() if self.config[0].text() != "" else nnet.datadict["input_parameters"]["input_shape1"]
        nnet.datadict["input_parameters"]["input_shape2"] = self.config[1].text() if self.config[1].text() != "" else nnet.datadict["input_parameters"]["input_shape2"]

        self.accept()

class ConfirmRemoveDialog(QDialog):

    def __init__(self, idx, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Confirm Remove")

        layout = QVBoxLayout(self)

        #slider for testing split
        self.label = QLabel('Are You Sure?')
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        #Dialog buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Save)
        layout.addWidget(self.buttons)

        #Connect buttons
        self.buttons.accepted.connect(self.deleteLayer)
        self.buttons.rejected.connect(self.reject)

        self.setMinimumSize(300, 100)

    def deleteLayer(self):

        del nnet.datadict["model_1"]["layers"][index]
        nnet.layoutChanged.emit()
        self.accept()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Main widget for the QMainWindow.
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        # Main layout for the central widget.
        main_layout = QVBoxLayout(main_widget)

        self.todoView = QListView()
        self.todoView.setStyleSheet("font: 20pt")
        self.todoView.setModel(nnet)
        main_layout.addWidget(self.todoView)

        # Buttons
        buttons_layout = QHBoxLayout()
        self.add_layer_button = QPushButton("Add Layer")
        self.add_layer_button.setToolTip("Add a layer to the Model.")
        self.edit_layer_button = QPushButton("Edit Layer")
        self.edit_layer_button.setToolTip("Edit the selected layer.")
        self.remove_layer_button = QPushButton("Remove Layer")
        self.remove_layer_button.setToolTip("Remove the selected layer from the model")
        self.use_preset_button = QPushButton("Use Preset")
        self.use_preset_button.setToolTip("Use a built-in model and import it into the screen.")
        self.configure_input_button = QPushButton("Configure Input")
        self.configure_input_button.setToolTip("Configure the model parameters")
        self.test_input_button = QPushButton("Configure Testing")
        self.test_input_button.setToolTip("Configure the testing parameters")
        buttons = [self.add_layer_button, self.remove_layer_button, self.edit_layer_button, self.use_preset_button, self.configure_input_button, self.test_input_button]
        for button in buttons:
            if isinstance(button, QPushButton):
                buttons_layout.addWidget(button)

        # Add buttons layout below the tab widget.
        main_layout.addLayout(buttons_layout)

        self.train_button = QPushButton("Train Model")
        self.train_button.setToolTip("Train the model")

        main_layout.addWidget(self.train_button)

        # Connect the "Configure Input" button
        self.configure_input_button.clicked.connect(self.openConfigureInputModal)

        # Connect the "Use Preset" button
        self.use_preset_button.clicked.connect(self.openUsePresetModal)


        # Connect the "Add Layer" button
        self.add_layer_button.clicked.connect(self.openAddLayerModal)
        
        # Connect the "Configure Testing" Button
        self.test_input_button.clicked.connect(self.openTestConfigModal)

        # Connect the "Remove Layer" button
        self.remove_layer_button.clicked.connect(self.deleteLayer)

        self.edit_layer_button.clicked.connect(self.openEditLayerModal)

        self.train_button.clicked.connect(self.trainModel)


        # Adjust the main window's size to ensure content is visible.
        self.setMinimumSize(800, 600)

    def openConfigureInputModal(self):
        dialog = ConfigureInputDialog(self)
        dialog.exec()

    def openUsePresetModal(self):
        dialog = UsePresetDialog(self)
        dialog.exec()

    def openAddLayerModal(self):
        indexes = self.todoView.selectedIndexes()
        global index
        if indexes:
            # Indexes is a list of a single item in single-select mode.
            index = indexes[0].row()
        else:
            index = -1

        dialog = AddLayerDialog(self)
        dialog.exec()
    
    def openTestConfigModal(self):
        dialog = TestConfigDialog(self)
        dialog.exec()

    def deleteLayer(self):
        indexes = self.todoView.selectedIndexes()
        global index
        if indexes:
            # Indexes is a list of a single item in single-select mode.
            index = indexes[0].row()
            dialog = ConfirmRemoveDialog(self)
            dialog.exec()
        else:
            index = -1

    def openEditLayerModal(self):
        indexes = self.todoView.selectedIndexes()
        global index
        if indexes:
            # Indexes is a list of a single item in single-select mode.
            index = indexes[0].row()
            dialog = EditLayerDialog(self)
            dialog.exec()
        else:
            index = -1

    def trainModel(self):
        process_model(nnet.datadict)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle("EEG App")
    window.show()
    sys.exit(app.exec())
