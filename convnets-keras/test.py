from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from convnetskeras.convnets import preprocess_image_batch, convnet

from alexnet_base import *
from utils import *

batch_size = 16
input_size = (3,227,227)
nb_classes = 6
mean_flag = True # if False, then the mean subtraction layer is not prepended

#code ported from https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator()
                                  

train_generator = train_datagen.flow_from_directory(
        '../Data/Train',  
        batch_size=batch_size,
        shuffle=True,
        target_size=input_size[1:],
        class_mode='categorical')  

validation_generator = test_datagen.flow_from_directory(
        '../Data/Test',  
        batch_size=batch_size,
        target_size=input_size[1:],
        shuffle=True,
        class_mode='categorical')

alexnet = get_alexnet(input_size,nb_classes,mean_flag)
alexnet.load_weights('weights/alexnet_weights.h5', by_name=True)

print alexnet.summary()

layers = ['dense_3_new','dense_2','dense_1','conv_5_1','conv_4_1','conv_3','conv_2_1','conv_1']
epochs = [10,10,10,10,10,10,10,10]
lr = [1e-2,1e-3,1e-3,1e-3,1e-3,1e-3,1e-3,1e-3]

history_finetune = []

for i,layer in enumerate(layers):
        
    alexnet = unfreeze_layer_onwards(alexnet,layer)    
   
    sgd = SGD(lr=lr[i], decay=1e-6, momentum=0.9, nesterov=True)
    alexnet.compile(loss='mse',
              optimizer=sgd,
              metrics=['accuracy'])
    
    for epoch in range(epochs[i]):    
        h = alexnet.fit_generator(train_generator,
                                        verbose=1,
                                        validation_data=validation_generator,
                                        steps_per_epoch=319, #samples/batch_size
                                        epochs=1,
                                        validation_steps=566)
        
        history_finetune = append_history(history_finetune,h)

alexnet.save('weights/alexnet_weights_grapevine.h5')
