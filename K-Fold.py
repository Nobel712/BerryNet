from sklearn.model_selection import KFold
import numpy as np

# Define the K-fold Cross Validator
kfold = KFold(n_splits=10, shuffle=True)

# Lists to store accuracy values for each fold
train_accuracies = []
val_accuracies = []


fold_no = 1
for train, test in kfold.split(X, y):
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    history = model.fit(X[train], y[train], validation_data=(X[test], y[test]),
                        epochs=20, verbose=0, batch_size=32)

    # Calculate and print the training accuracy
    train_loss, train_accuracy = model.evaluate(X[train], y[train], verbose=0)
    print(f'Fold {fold_no} Training Loss: {train_loss}')
    print(f'Fold {fold_no} Training Accuracy: {train_accuracy}')
    train_accuracies.append(train_accuracy)

    # Calculate and print the validation accuracy
    val_loss, val_accuracy = model.evaluate(X[test], y[test], verbose=0)
    print(f'Fold {fold_no} Validation Loss: {val_loss}')
    print(f'Fold {fold_no} Validation Accuracy: {val_accuracy}')
    val_accuracies.append(val_accuracy)

    # Increase fold number
    fold_no += 1
