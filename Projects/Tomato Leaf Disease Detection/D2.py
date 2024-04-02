from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_size = [244, 244]
batch_size = 32

# Load pre-trained VGG16 model
vgg = VGG16(input_shape=image_size + [3], weights='imagenet', include_top=False)

# Freeze VGG16 layers
for layer in vgg.layers:
    layer.trainable = False

from glob import glob
folders = glob('D:/Deep Learning/Tomato Leaf Disease Detection/tomato/train')

# Define the output layer
x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation='softmax')(x)

# Create the model
model = Model(inputs=vgg.input, outputs=prediction)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Create data generators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'D:/Deep Learning/Tomato Leaf Disease Detection/tomato/train',
    target_size=(244, 244),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    'D:/Deep Learning/Tomato Leaf Disease Detection/tomato/val',
    target_size=(244, 244),
    batch_size=batch_size,
    class_mode='categorical'
)

# Train the model
hitory = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=20,
    steps_per_epoch=len(train_generator),
    validation_steps=len(test_generator)
)