from keras.models import load_model
from cv2 import resize, imread, getRotationMatrix2D, GaussianBlur, imwrite, warpAffine, INTER_CUBIC, morphologyEx, MORPH_CLOSE
from numpy import reshape, ones, uint8
from os import walk
from PIL import Image
from random import randint
from tensorflow.keras.optimizers import RMSprop
from ClassNames import classes_names


validation_path = 'Validation-set/'

number_of_model = raw_input("Insert the number of model you want tested: ")

model_name = 'TrafficRecognitionModel_' + number_of_model + '.h5'

model = load_model(model_name, compile=False)

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc']
)

if number_of_model == '1':
    img_width, img_height = 128, 128
elif number_of_model == '6' or number_of_model == '7' or number_of_model == '8':
    img_width, img_height = 32, 32
else:
    img_width, img_height = 64, 64

for _ in range(5):
    (_, _, images) = walk(validation_path).next()

    random_index = randint(0, len(images) - 1)
    random_picture = images[random_index]

    while random_picture == '.DS_Store':
        random_index = randint(0, len(images) - 1)
        random_picture = images[random_index]

    full_image_path = validation_path + random_picture

    image = imread(full_image_path)

    rows,cols = image.shape[0], image.shape[1]
    scaleX = 1 + randint(1, 4) / 10
    scaleY = 1 + randint(1, 4) / 10
    kernel = ones((3,3), uint8)

    # image = resize(image, None, fx=scaleX, fy=scaleY, interpolation=INTER_CUBIC)
    # image = morphologyEx(image, MORPH_CLOSE, kernel)

    temp_image_path = validation_path + "temp.jpg"

    imwrite(temp_image_path, image)

    image = imread(temp_image_path)
    Image.open(temp_image_path).show()
    image = resize(image, (img_height, img_width))
    image = reshape(image, [1, img_height, img_width, 3])

    classes = model.predict_classes(image)

    print classes_names[classes[0]]