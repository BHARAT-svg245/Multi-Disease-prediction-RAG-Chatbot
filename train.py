import os
os.environ["TF_CPP_MIN_LOC_LEVEL"]="3"
os.environ["TF_ENABLE_ONEDNN_OPTS"]="1"

import tensorflow as tf
from tensorflow import keras
from keras import layers,regularizers,models
from keras.utils import image_dataset_from_directory

train_dir="put here directory where your training images are present"
train_dataset=image_dataset_from_directory(train_dir,image_size=(256,256),color_mode="rgb",batch_size=32,shuffle=True,label_mode="int")

# #Normalising the layer
normal_layer=models.Sequential(
    [
        layers.Rescaling(1/255.0)
    ]
)
#Augmentation layer
augmentation_layer=models.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.3)
    ]
)

# Loading the transfer learning model weigths 
from keras.applications import MobileNetV2,InceptionV3
base_model=MobileNetV2(weights="imagenet",include_top=False,input_shape=(256,256,3))

# here all the layers of the Mobilenet is freezes.

#base_model=InceptionV3(weights="imagenet",include_top=False,input_shape=(256,256,3))  
#base_model.trainable=False

for layer in base_model.layers[:-30]:       # here, i unfreeze the last 30 layers of the Mobilenetv2 model for increasing the training and testing accuracy.
    layer.trainable=False 


#models architecture
model=models.Sequential(
    [
        layers.Input(shape=(256,256,3)),   # by default whatever is the shape of your images, it will make them 256*256.
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(64,activation="relu",kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(4,activation="softmax")    # put here the number of classes your images have
    ]
)
# compiling the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.00001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

file_name="filename given by you in .keras formate"
#making the callbacks
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
early_stop=EarlyStopping(monitor="val_loss",patience=3)
checkpt=ModelCheckpoint(filepath=file_name,monitor="val_accuracy",save_best_only=True,mode="max",verbose=2)

# Making the training, validation and testing sets.
def Split(dataset,train_split):
    data_size=len(dataset)
    train_size=int(data_size*train_split)
    train_set=dataset.take(train_size)
    val_set=dataset.skip(train_size)
    return train_set,val_set

#getting the each set
train_ds,val_ds=Split(train_dataset,0.8)

# GPU optimisation
train_set = train_ds.shuffle(500).prefetch(buffer_size=tf.data.AUTOTUNE)
val_set = val_ds.shuffle(500).prefetch(buffer_size=tf.data.AUTOTUNE)

# Fitting the model with MobilenetV2
h=model.fit(train_set,epochs=10,validation_data=val_set,callbacks=[early_stop,checkpt])

