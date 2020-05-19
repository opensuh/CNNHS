from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense
from tensorflow.keras import Model, Input
from config import DROPOUT_RATIO


class CNN_HS(Model):
    def __init__(self):
        super(CNN_HS,self).__init__()
        self.conv1 = Conv2D(20, kernel_size=(10, 10), strides=(2,2), activation='relu', input_shape=(128,128,3))
        self.conv2 = Conv2D(40, kernel_size=(5, 5), strides=(2,2), activation='relu')
        self.conv3 = Conv2D(20, kernel_size=(3, 3), strides=(1,1), activation='relu')
        self.flatten = Flatten()
        self.dense1 = Dense(500, activation='relu', name='feature_extraction')
        self.dropout = Dropout(DROPOUT_RATIO)
        self.dense2 = Dense(50, activation='relu')
        self.dense3 = Dense(2, activation='softmax')
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)
        return self.dense3(x)
    def FeatureExtraction(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        return self.dense1(x)