
def main():
    import numpy as np
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    from keras import optimizers
    
    basepath = "F:/2023 project code/Knee Osteoarthritis Detection/Knee Osteoarthritis Detection 100% Code"
    
    # Initializing the LSTM model
    classifier = Sequential()
    
    # Reshaping the input data for LSTM
    classifier.add(LSTM(64, input_shape=(64, 64*3)))
    
    # Adding a dropout layer
    classifier.add(Dropout(0.5))
    
    # Adding the output layer
    classifier.add(Dense(5, activation='softmax'))  # change class no.
    
    # Compiling the LSTM model
    classifier.compile(
        optimizer=optimizers.SGD(lr=0.01),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    
    # Preparing the data for LSTM
    from keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    training_set = train_datagen.flow_from_directory(
        basepath + '/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')
    
    test_set = test_datagen.flow_from_directory(
        basepath + '/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')
    
    # Fitting the LSTM model to the data
    steps_per_epoch = int(np.ceil(training_set.samples / 32))
    val_steps = int(np.ceil(test_set.samples / 32))
    
    
    
    model = classifier.fit_generator(
        training_set,
        steps_per_epoch=steps_per_epoch,
        epochs=1526,
        validation_data=test_set,
        validation_steps=val_steps)
    
    # Saving the model
    classifier.save(basepath + '/model3.h5')
    
    # Evaluating the model
    scores = classifier.evaluate(test_set, verbose=1)
    B = "Testing Accuracy: %.2f%%" % (scores[1]*100)
    print(B)
    scores = classifier.evaluate(training_set, verbose=1)
    C = "Training Accuracy: %.2f%%" % (scores[1]*100)
    print(C)
    
    msg = B+'\n'+C
    
    # Plotting the accuracy and loss
    import matplotlib.pyplot as plt
    
    # Summarize history for accuracy
    plt.plot(model.history['accuracy'])
    plt.plot(model.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(basepath + "/accuracy.png", bbox_inches='tight')
    plt.show()
    
    # Summarize history for loss
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(basepath + "/loss.png", bbox_inches='tight')
    plt.show()
    
    return msg



