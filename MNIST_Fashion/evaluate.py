import matplotlib.pyplot as plt 

def plot_accuracy(history):
    # Extract the training accuracy from the history object
    training_accuracy = history.history['accuracy']

    # Extract the test (validation) accuracy from the history object
    test_accuracy = history.history['val_accuracy']

    # Create a range of epochs
    epochs = range(1, len(training_accuracy) + 1)

    # Plot training and test accuracy
    plt.figure()
    plt.plot(epochs, training_accuracy, 'b', label='Training accuracy')
    plt.plot(epochs, test_accuracy, 'r', label='Test accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def plot_loss(history):
    # Extract the training loss from the history object
    training_loss = history.history['loss']

    # Extract the test (validation) loss from the history object
    test_loss = history.history['val_loss']

    # Create a range of epochs
    epochs = range(1, len(training_loss) + 1)

    # Plot training and test loss
    plt.figure()
    plt.plot(epochs, training_loss, 'b', label='Training loss')
    plt.plot(epochs, test_loss, 'r', label='Test loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()