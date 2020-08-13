import scipy.io as sio
import os
from PIL import Image


def image_crop(CoordinateFile, ImageQuantity, AddFolder, OpenFolder, SaveFolder):
    for i in range(1, AddFolder + 1):
        FileName = f'{SaveFolder}{i}/'
        if not os.path.exists(FileName):
            os.makedirs(FileName)
    Coordinate = sio.loadmat(CoordinateFile)
    for i in range(ImageQuantity):
        Name = Coordinate['annotations'][0][i][5].tolist()[0]   # File name of each picture
        pic = Image.open(OpenFolder+Name)
        x1 = int(Coordinate['annotations'][0][i][0])
        y1 = int(Coordinate['annotations'][0][i][1])
        x2 = int(Coordinate['annotations'][0][i][2])
        y2 = int(Coordinate['annotations'][0][i][3])
        Class = int(Coordinate['annotations'][0][i][4])
        new_image = pic.crop((x1, y1, x2, y2))
        new_image.save(f'{SaveFolder}{Class}/{Name}')

        # new_image_tp = new_image.transpose(Image.FLIP_LEFT_RIGHT)
        # new_image_tp.save(f'{SaveFolder}{Class}/{ImageQuantity + i + 1}.jpg')


if __name__ == '__main__':
    image_crop(CoordinateFile='./devkit/cars_train_annos.mat',
               ImageQuantity=8144,
               AddFolder=196,
               OpenFolder='./cars_train/',
               SaveFolder='./cars_train_crop/')

    image_crop(CoordinateFile='./devkit/cars_test_annos.mat',
               ImageQuantity=8041,
               AddFolder=196,
               OpenFolder='./cars_test/',
               SaveFolder='./cars_test_crop/')
