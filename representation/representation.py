import matplotlib.pyplot as plt

def smooth_curve(points, factor=0.8):
    """
    Smoothing the curve of a plot
    :param points: the points to smooth
    :param factor: the factor of smoothing
    :return: the smoothing points
    """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def show(listHistory, smooth):
    """
    show in matplotlib resuts the results of training models with data augmentation
    :param listHistory:
    :param smooth: boolean to knwo if the user want to have his points smoothed or not
    :return:
    """
    acc = []
    val_acc = []
    loss = []
    val_loss = []


    for i in range(len(listHistory)):
        acc.extend(listHistory[i].history['acc'])
        val_acc.extend(listHistory[i].history['val_acc'])
        loss.extend(listHistory[i].history['loss'])
        val_loss.extend(listHistory[i].history['val_loss'])

    epochs = range(1, len(acc) + 1)
    plt.figure()

    plt.subplot(2, 1, 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    if smooth:
        plt.plot(epochs, smooth_curve(val_acc), 'r', label='Validation acc')
    else :
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training accuracy and Validation accuracy')
    plt.legend()


    plt.subplot(2, 1, 2)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    if smooth:
        plt.plot(epochs, smooth_curve(val_loss), 'r', label='Validation loss')
    else:
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Validation loss and Training loss')
    plt.legend()
    plt.show()

def showPictures(pictures):
    """
    show data generated pictures just to check if they are correct
    :param pictures:
    :return:
    """
    plt.figure()
    for i in range(1,pictures.shape[0] + 1):
        plt.subplot(2,2,i)
        image = pictures[0]
        print(image.shape)
        plt.imshow(image)

    plt.show()