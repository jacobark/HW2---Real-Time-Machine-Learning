import tensorflow as tf
import matplotlib.pyplot as plt
from ptflops import get_model_complexity_info
from keras_flops import get_flops

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#scaling image values
train_images = train_images / 255
test_images = test_images / 255
print(train_images.shape)

x_val = train_images[-10000:]
y_val = train_labels[-10000:]

x_train = train_images[:-10000]
y_train = train_labels[:-10000]

#PART 1:
#****************************8
'''This is the baseline LeNet Model From class'''

LeNet = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid', padding='same', input_shape=(28,28,1)),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation='sigmoid'),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation='sigmoid'),
    tf.keras.layers.Dense(84, activation='sigmoid'),
    tf.keras.layers.Dense(10)])

LeNet.compile(optimizer='SGD',
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

lenet = LeNet.fit(x_train, y_train, batch_size=16, epochs=20, validation_data=(x_val, y_val))
print("Baseline LeNet Model")
flops = get_flops(LeNet, batch_size=1)
print(f"FLOPS: {flops / 10 ** 9:.03} G")
plt.plot(lenet.history['accuracy'])
plt.plot(lenet.history['val_accuracy'])
plt.plot(lenet.history['loss'])
plt.title('Base LeNet Model')
plt.xlabel('epochs')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')
plt.show()
#**********************************
'''My Modernized LeNet Model'''
ModernLeNet = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid', padding='same', input_shape=(28,28,1)),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation='sigmoid'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    #tf.keras.layers.Dense(120, activation='sigmoid'),
    tf.keras.layers.Activation('relu'),
    #tf.keras.layers.Dense(84, activation='sigmoid'),
    tf.keras.layers.Dense(10)])

ModernLeNet.compile(optimizer='SGD',
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

history = ModernLeNet.fit(x_train, y_train, batch_size=16, epochs=20, validation_data=(x_val, y_val))
print("My more modern LeNet model")

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.title('My Modern LeNet Model')
plt.xlabel('epochs')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')
plt.show()

#******************************************************
#PART 2

#1 - Adjusting Convolution Window size
'''I ran the LeNet model with a lower and higher Convolution size
    compared to the baseline LeNet Model from lecture'''

#lower conv window size model
LeNet = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=6, kernel_size=3, activation='sigmoid', padding='same', input_shape=(28,28,1)),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='sigmoid'),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation='sigmoid'),
    tf.keras.layers.Dense(84, activation='sigmoid'),
    tf.keras.layers.Dense(10)])

LeNet.compile(optimizer='SGD',
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

lenetLow = LeNet.fit(x_train, y_train, batch_size=16, epochs=20, validation_data=(x_val, y_val))
print("smaller convolution window size model")
flops = get_flops(LeNet, batch_size=1)
print(f"FLOPS: {flops / 10 ** 9:.03} G")
print(LeNet.summary())

#higher conv window size model
LeNet = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=6, kernel_size=7, activation='sigmoid', padding='same', input_shape=(28,28,1)),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=7, activation='sigmoid'),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation='sigmoid'),
    tf.keras.layers.Dense(84, activation='sigmoid'),
    tf.keras.layers.Dense(10)])

LeNet.compile(optimizer='SGD',
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

lenetHigh = LeNet.fit(x_train, y_train, batch_size=16, epochs=20, validation_data=(x_val, y_val))
print("Larger conv window size model")
flops = get_flops(LeNet, batch_size=1)
print(f"FLOPS: {flops / 10 ** 9:.03} G")
print(LeNet.summary())

#plotting lower and higher convoltions windows with baseline results

plt.subplot(3,1,1)
plt.plot(lenetLow.history['accuracy'])
plt.plot(lenetLow.history['val_accuracy'])
plt.plot(lenetLow.history['loss'])
plt.title('Lower Conv Window LeNet')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')

plt.subplot(3,1,2)
plt.plot(lenet.history['accuracy'])
plt.plot(lenet.history['val_accuracy'])
plt.plot(lenet.history['loss'])
plt.title('Baseline LeNet Model')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')

plt.subplot(3,1,3)
plt.plot(lenetHigh.history['accuracy'])
plt.plot(lenetHigh.history['val_accuracy'])
plt.plot(lenetHigh.history['loss'])
plt.title('Higher Conv Window LeNet')
plt.xlabel('epochs')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')

plt.show()

#******************************
#2 Adjustin the number of output channels
'''I ran the LeNet model with a lower and higher output channels
    compared to the baseline LeNet Model from lecture'''
#lower Output channels model
LeNet = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=3, kernel_size=5, activation='sigmoid', padding='same', input_shape=(28,28,1)),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=8, kernel_size=5, activation='sigmoid'),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(60, activation='sigmoid'),
    tf.keras.layers.Dense(42, activation='sigmoid'),
    tf.keras.layers.Dense(10)])

LeNet.compile(optimizer='SGD',
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

lenetLow = LeNet.fit(x_train, y_train, batch_size=16, epochs=20, validation_data=(x_val, y_val))
print("Less output channels model")
flops = get_flops(LeNet, batch_size=1)
print(f"FLOPS: {flops / 10 ** 9:.03} G")
print(LeNet.summary())

#higher Ouptu channels model
LeNet = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=12, kernel_size=5, activation='sigmoid', padding='same', input_shape=(28,28,1)),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=5, activation='sigmoid'),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(240, activation='sigmoid'),
    tf.keras.layers.Dense(168, activation='sigmoid'),
    tf.keras.layers.Dense(10)])

LeNet.compile(optimizer='SGD',
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

lenetHigh = LeNet.fit(x_train, y_train, batch_size=16, epochs=20, validation_data=(x_val, y_val))
print("More output channels model")
flops = get_flops(LeNet, batch_size=1)
print(f"FLOPS: {flops / 10 ** 9:.03} G")
print(LeNet.summary())

#plotting lower and higher convoltions windows with baseline results

plt.subplot(3,1,1)
plt.plot(lenetLow.history['accuracy'])
plt.plot(lenetLow.history['val_accuracy'])
plt.plot(lenetLow.history['loss'])
plt.title('Lower Output Channels LeNet')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')

plt.subplot(3,1,2)
plt.plot(lenet.history['accuracy'])
plt.plot(lenet.history['val_accuracy'])
plt.plot(lenet.history['loss'])
plt.title('Baseline LeNet Model')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')

plt.subplot(3,1,3)
plt.plot(lenetHigh.history['accuracy'])
plt.plot(lenetHigh.history['val_accuracy'])
plt.plot(lenetHigh.history['loss'])
plt.title('Higher Output Channels LeNet')
plt.xlabel('epochs')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')

plt.show()


#**********************************************
#3 Adjusting the number of convolution layers
'''I ran the LeNet model with a lower and higher convolution layers
    compared to the baseline LeNet Model from lecture'''
#lower conv model
LeNet = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid', padding='same', input_shape=(28,28,1)),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation='sigmoid'),
    tf.keras.layers.Dense(84, activation='sigmoid'),
    tf.keras.layers.Dense(10)])

LeNet.compile(optimizer='SGD',
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

lenetLow = LeNet.fit(x_train, y_train, batch_size=16, epochs=20, validation_data=(x_val, y_val))
print("less convolutions model")
flops = get_flops(LeNet, batch_size=1)
print(f"FLOPS: {flops / 10 ** 9:.03} G")
print(LeNet.summary())

#higher conv model
LeNet = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid', padding='same', input_shape=(28,28,1)),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation='sigmoid'),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation='sigmoid'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation='sigmoid'),
    tf.keras.layers.Dense(84, activation='sigmoid'),
    tf.keras.layers.Dense(10)])

LeNet.compile(optimizer='SGD',
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

lenetHigh = LeNet.fit(x_train, y_train, batch_size=16, epochs=20, validation_data=(x_val, y_val))
print("More convolutions model")
flops = get_flops(LeNet, batch_size=1)
print(f"FLOPS: {flops / 10 ** 9:.03} G")
print(LeNet.summary())

#plotting lower and higher convoltions windows with baseline results

plt.subplot(3,1,1)
plt.plot(lenetLow.history['accuracy'])
plt.plot(lenetLow.history['val_accuracy'])
plt.plot(lenetLow.history['loss'])
plt.title('Less Convolutions LeNet')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')

plt.subplot(3,1,2)
plt.plot(lenet.history['accuracy'])
plt.plot(lenet.history['val_accuracy'])
plt.plot(lenet.history['loss'])
plt.title('Baseline LeNet Model')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')

plt.subplot(3,1,3)
plt.plot(lenetHigh.history['accuracy'])
plt.plot(lenetHigh.history['val_accuracy'])
plt.plot(lenetHigh.history['loss'])
plt.title('More Convolutions LeNet')
plt.xlabel('epochs')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')

plt.show()


#**************************************************
#4 Adjust the number of fully connected layers
'''I ran the LeNet model with a lower and higher fully connected layers
    compared to the baseline LeNet Model from lecture'''
#lower Fully connected layers model
LeNet = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid', padding='same', input_shape=(28,28,1)),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation='sigmoid'),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(84, activation='sigmoid'),
    tf.keras.layers.Dense(10)])

LeNet.compile(optimizer='SGD',
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

lenetLow = LeNet.fit(x_train, y_train, batch_size=16, epochs=20, validation_data=(x_val, y_val))
print("less fully connecyed layers model")
flops = get_flops(LeNet, batch_size=1)
print(f"FLOPS: {flops / 10 ** 9:.03} G")
print(LeNet.summary())

#higher Fully Connected layers model
LeNet = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid', padding='same', input_shape=(28,28,1)),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation='sigmoid'),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(200, activation='sigmoid'),
    tf.keras.layers.Dense(120, activation='sigmoid'),
    tf.keras.layers.Dense(84, activation='sigmoid'),
    tf.keras.layers.Dense(10)])

LeNet.compile(optimizer='SGD',
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

lenetHigh = LeNet.fit(x_train, y_train, batch_size=16, epochs=20, validation_data=(x_val, y_val))
print("More fully connected layers model")
flops = get_flops(LeNet, batch_size=1)
print(f"FLOPS: {flops / 10 ** 9:.03} G")
print(LeNet.summary())

#plotting lower and higher convoltions windows with baseline results

plt.subplot(3,1,1)
plt.plot(lenetLow.history['accuracy'])
plt.plot(lenetLow.history['val_accuracy'])
plt.plot(lenetLow.history['loss'])
plt.title('Lower Fully Connected Layers LeNet')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')

plt.subplot(3,1,2)
plt.plot(lenet.history['accuracy'])
plt.plot(lenet.history['val_accuracy'])
plt.plot(lenet.history['loss'])
plt.title('Baseline LeNet Model')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')

plt.subplot(3,1,3)
plt.plot(lenetHigh.history['accuracy'])
plt.plot(lenetHigh.history['val_accuracy'])
plt.plot(lenetHigh.history['loss'])
plt.title('Higher Fully Connected Layers LeNet')
plt.xlabel('epochs')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')

plt.show()


#************************************************
#5 Ajusting Learning rates
'''I ran the LeNet model with a lower and higher learning rate
    compared to the baseline LeNet Model from lecture'''
#lower learning rate model
LeNet = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=6, kernel_size=3, activation='sigmoid', padding='same', input_shape=(28,28,1)),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='sigmoid'),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation='sigmoid'),
    tf.keras.layers.Dense(84, activation='sigmoid'),
    tf.keras.layers.Dense(10)])

LeNet.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-4),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

lenetLow = LeNet.fit(x_train, y_train, batch_size=16, epochs=20, validation_data=(x_val, y_val))
print("lower learning rate model")
flops = get_flops(LeNet, batch_size=1)
print(f"FLOPS: {flops / 10 ** 9:.03} G")
print(LeNet.summary())

#higher Learning rate model
LeNet = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=6, kernel_size=7, activation='sigmoid', padding='same', input_shape=(28,28,1)),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=7, activation='sigmoid'),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation='sigmoid'),
    tf.keras.layers.Dense(84, activation='sigmoid'),
    tf.keras.layers.Dense(10)])

LeNet.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-1),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

lenetHigh = LeNet.fit(x_train, y_train, batch_size=16, epochs=20, validation_data=(x_val, y_val))
print("higher learning rate model")
BestModel = lenetHigh
flops = get_flops(LeNet, batch_size=1)
print(f"FLOPS: {flops / 10 ** 9:.03} G")
print(LeNet.summary())

#plotting lower and higher convoltions windows with baseline results

plt.subplot(3,1,1)
plt.plot(lenetLow.history['accuracy'])
plt.plot(lenetLow.history['val_accuracy'])
plt.plot(lenetLow.history['loss'])
plt.title('Lower Learning Rate LeNet')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')

plt.subplot(3,1,2)
plt.plot(lenet.history['accuracy'])
plt.plot(lenet.history['val_accuracy'])
plt.plot(lenet.history['loss'])
plt.title('Baseline LeNet Model')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')

plt.subplot(3,1,3)
plt.plot(lenetHigh.history['accuracy'])
plt.plot(lenetHigh.history['val_accuracy'])
plt.plot(lenetHigh.history['loss'])
plt.title('Higher Learning Rate LeNet')
plt.xlabel('epochs')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')

plt.show()


#******************************************************
#PART 3
'''Applying dropout across all experiments in PART 2'''


#1 - Adjusting Convolution Window size with dropout
'''I ran the LeNet model with a lower and higher Convolution size
    compared to the baseline LeNet Model from lecture'''

#lower conv window size model
LeNet = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=6, kernel_size=3, activation='sigmoid', padding='same', input_shape=(28,28,1)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='sigmoid'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation='sigmoid'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(84, activation='sigmoid'),
    tf.keras.layers.Dense(10)])

LeNet.compile(optimizer='SGD',
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

lenetLow = LeNet.fit(x_train, y_train, batch_size=16, epochs=20, validation_data=(x_val, y_val))
print("lower conv window size with dropout model")

#higher conv window size model
LeNet = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=6, kernel_size=7, activation='sigmoid', padding='same', input_shape=(28,28,1)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=7, activation='sigmoid'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation='sigmoid'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(84, activation='sigmoid'),
    tf.keras.layers.Dense(10)])

LeNet.compile(optimizer='SGD',
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

lenetHigh = LeNet.fit(x_train, y_train, batch_size=16, epochs=20, validation_data=(x_val, y_val))
print("larger conv windows ize with dropout model")

#plotting lower and higher convoltions windows with baseline results

plt.subplot(3,1,1)
plt.plot(lenetLow.history['accuracy'])
plt.plot(lenetLow.history['val_accuracy'])
plt.plot(lenetLow.history['loss'])
plt.title('Lower Conv Window LeNet')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')

plt.subplot(3,1,2)
plt.plot(BestModel.history['accuracy'])
plt.plot(BestModel.history['val_accuracy'])
plt.plot(BestModel.history['loss'])
plt.title('Best LeNet Model from Part 2')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')

plt.subplot(3,1,3)
plt.plot(lenetHigh.history['accuracy'])
plt.plot(lenetHigh.history['val_accuracy'])
plt.plot(lenetHigh.history['loss'])
plt.title('Higher Conv Window LeNet')
plt.xlabel('epochs')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')

plt.show()

#******************************
#2 Adjustin the number of output channels with dropout
'''I ran the LeNet model with a lower and higher output channels
    compared to the baseline LeNet Model from lecture'''
#lower Output channels model
LeNet = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=3, kernel_size=5, activation='sigmoid', padding='same', input_shape=(28,28,1)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=8, kernel_size=5, activation='sigmoid'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(60, activation='sigmoid'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(42, activation='sigmoid'),
    tf.keras.layers.Dense(10)])

LeNet.compile(optimizer='SGD',
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

lenetLow = LeNet.fit(x_train, y_train, batch_size=16, epochs=20, validation_data=(x_val, y_val))
print("lower output channels with dropout model")

#higher Ouptu channels model
LeNet = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=12, kernel_size=5, activation='sigmoid', padding='same', input_shape=(28,28,1)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=5, activation='sigmoid'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(240, activation='sigmoid'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(168, activation='sigmoid'),
    tf.keras.layers.Dense(10)])

LeNet.compile(optimizer='SGD',
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

lenetHigh = LeNet.fit(x_train, y_train, batch_size=16, epochs=20, validation_data=(x_val, y_val))
print("higher output channels model with dropout")

#plotting 

plt.subplot(3,1,1)
plt.plot(lenetLow.history['accuracy'])
plt.plot(lenetLow.history['val_accuracy'])
plt.plot(lenetLow.history['loss'])
plt.title('Lower Output Channels LeNet')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')

plt.subplot(3,1,2)
plt.plot(BestModel.history['accuracy'])
plt.plot(BestModel.history['val_accuracy'])
plt.plot(BestModel.history['loss'])
plt.title('Best LeNet Model from Part 2')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')

plt.subplot(3,1,3)
plt.plot(lenetHigh.history['accuracy'])
plt.plot(lenetHigh.history['val_accuracy'])
plt.plot(lenetHigh.history['loss'])
plt.title('Higher Output Channels LeNet')
plt.xlabel('epochs')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')

plt.show()

#**********************************************
#3 Adjusting the number of convolution layers with dropout
'''I ran the LeNet model with a lower and higher convolution layers
    compared to the baseline LeNet Model from lecture'''
#lower conv model
LeNet = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid', padding='same', input_shape=(28,28,1)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation='sigmoid'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(84, activation='sigmoid'),
    tf.keras.layers.Dense(10)])

LeNet.compile(optimizer='SGD',
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

lenetLow = LeNet.fit(x_train, y_train, batch_size=16, epochs=20, validation_data=(x_val, y_val))
print("less convolutions model with dropout")

#higher conv model
LeNet = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid', padding='same', input_shape=(28,28,1)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation='sigmoid'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation='sigmoid'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation='sigmoid'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(84, activation='sigmoid'),
    tf.keras.layers.Dense(10)])

LeNet.compile(optimizer='SGD',
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

lenetHigh = LeNet.fit(x_train, y_train, batch_size=16, epochs=20, validation_data=(x_val, y_val))
print("More conv model with ddropout")

#plotting lower and higher convoltions windows with baseline results

plt.subplot(3,1,1)
plt.plot(lenetLow.history['accuracy'])
plt.plot(lenetLow.history['val_accuracy'])
plt.plot(lenetLow.history['loss'])
plt.title('Less Convolutions LeNet')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')

plt.subplot(3,1,2)
plt.plot(BestModel.history['accuracy'])
plt.plot(BestModel.history['val_accuracy'])
plt.plot(BestModel.history['loss'])
plt.title('Best LeNet Model from Part 2')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')

plt.subplot(3,1,3)
plt.plot(lenetHigh.history['accuracy'])
plt.plot(lenetHigh.history['val_accuracy'])
plt.plot(lenetHigh.history['loss'])
plt.title('More Convolutions LeNet')
plt.xlabel('epochs')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')

plt.show()


#**************************************************
#4 Adjust the number of fully connected layers with dropout
'''I ran the LeNet model with a lower and higher fully connected layers
    compared to the baseline LeNet Model from lecture'''
#lower Fully connected layers model
LeNet = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid', padding='same', input_shape=(28,28,1)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation='sigmoid'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(84, activation='sigmoid'),
    tf.keras.layers.Dense(10)])

LeNet.compile(optimizer='SGD',
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

lenetLow = LeNet.fit(x_train, y_train, batch_size=16, epochs=20, validation_data=(x_val, y_val))
print("Less fully connected moel with dropout")

#higher Fully Connected layers model
LeNet = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid', padding='same', input_shape=(28,28,1)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation='sigmoid'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(200, activation='sigmoid'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(120, activation='sigmoid'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(84, activation='sigmoid'),
    tf.keras.layers.Dense(10)])

LeNet.compile(optimizer='SGD',
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

lenetHigh = LeNet.fit(x_train, y_train, batch_size=16, epochs=20, validation_data=(x_val, y_val))
print("More fully connected model with dropout")

#plotting lower and higher convoltions windows with baseline results

plt.subplot(3,1,1)
plt.plot(lenetLow.history['accuracy'])
plt.plot(lenetLow.history['val_accuracy'])
plt.plot(lenetLow.history['loss'])
plt.title('Lower Fully Connected Layers LeNet')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')

plt.subplot(3,1,2)
plt.plot(BestModel.history['accuracy'])
plt.plot(BestModel.history['val_accuracy'])
plt.plot(BestModel.history['loss'])
plt.title('Best LeNet Model from Part 2')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')

plt.subplot(3,1,3)
plt.plot(lenetHigh.history['accuracy'])
plt.plot(lenetHigh.history['val_accuracy'])
plt.plot(lenetHigh.history['loss'])
plt.title('Higher Fully Connected Layers LeNet')
plt.xlabel('epochs')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')

plt.show()

#************************************************
#5 Ajusting Learning rates with dropout
'''I ran the LeNet model with a lower and higher learning rate
    compared to the baseline LeNet Model from lecture'''
#lower learning rate model
LeNet = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid', padding='same', input_shape=(28,28,1)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation='sigmoid'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation='sigmoid'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(84, activation='sigmoid'),
    tf.keras.layers.Dense(10)])

LeNet.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-4),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

lenetLow = LeNet.fit(x_train, y_train, batch_size=16, epochs=20, validation_data=(x_val, y_val))
print("lower learning rate mode with dropout")

#higher Learning rate model
LeNet = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid', padding='same', input_shape=(28,28,1)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation='sigmoid'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation='sigmoid'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(84, activation='sigmoid'),
    tf.keras.layers.Dense(10)])

LeNet.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-1),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

lenetHigh = LeNet.fit(x_train, y_train, batch_size=16, epochs=20, validation_data=(x_val, y_val))
print("Higher learning rate model with dropout")
BestModel2 = lenetHigh


#plotting lower and higher convoltions windows with baseline results

plt.subplot(3,1,1)
plt.plot(lenetLow.history['accuracy'])
plt.plot(lenetLow.history['val_accuracy'])
plt.plot(lenetLow.history['loss'])
plt.title('Lower Learning Rate LeNet')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')

plt.subplot(3,1,2)
plt.plot(BestModel.history['accuracy'])
plt.plot(BestModel.history['val_accuracy'])
plt.plot(BestModel.history['loss'])
plt.title('Best LeNet Model from Part 2')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')

plt.subplot(3,1,3)
plt.plot(lenetHigh.history['accuracy'])
plt.plot(lenetHigh.history['val_accuracy'])
plt.plot(lenetHigh.history['loss'])
plt.title('Higher Learning Rate LeNet')
plt.xlabel('epochs')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')

plt.show()

#******************************************************
#PART 4

#******************************************************
#PART 4


'''Base AlexNet Model'''

AlexNet = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4, activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10)])



AlexNet.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-1),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
alexnet = AlexNet.fit(x_train, y_train, batch_size=16, epochs=20, validation_data=(x_val, y_val))
print("Baseline AlexNet Model")

plt.plot(alexnet.history['accuracy'])
plt.plot(alexnet.history['val_accuracy'])
plt.plot(alexnet.history['loss'])
plt.title('Base AlexNet Model')
plt.xlabel('epochs')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')
plt.show()

#********************************************
'''My Simplified AlexNet Model'''

AlexNet = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=12, kernel_size=11, strides=4, activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10)])



AlexNet.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-1),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
alexnet = AlexNet.fit(x_train, y_train, batch_size=16, epochs=20, validation_data=(x_val, y_val))
print("Baseline AlexNet Model")

plt.plot(alexnet.history['accuracy'])
plt.plot(alexnet.history['val_accuracy'])
plt.plot(alexnet.history['loss'])
plt.title('Base AlexNet Model')
plt.xlabel('epochs')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')
plt.show()

