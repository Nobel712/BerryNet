import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

model.compile(loss='categorical_crossentropy',optimizer = tf.keras.optimizers.Nadam(0.0001), metrics= ['accuracy'])
tensorboard = TensorBoard(log_dir = 'logs')
checkpoint = ModelCheckpoint("effnet.h5",monitor="val_accuracy",save_best_only=True,mode="auto",verbose=1)
reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.3, patience = 2, min_delta = 0.001,
                              mode='auto',verbose=1)
history = model.fit(X_train,y_train,validation_data=(X_test,y_test), epochs =20, verbose=1, batch_size=8,
                   callbacks=[tensorboard,checkpoint,reduce_lr])

# Assuming 'model' is your Keras functional model
x_train_each_class = ['01 ChiangMai60','02 RedKing','03 WhiteKing','04 BlackOodTurkey','05 TaiwanStraberry','06 BlackAustralia','07 Buriram60','08 Kamphaengsaeng42','09 TaiwanMeacho','10 ChiangMaiBuriram60']

# Convert multiclass labels to one-hot encoding
y_test_bin = label_binarize(y_test, classes=np.arange(5))

# Get predicted probabilities for each class
y_pred_proba = model.predict(X_test)

# Calculate ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(5):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves for each class
plt.figure(figsize=(6, 5))
for i in range(5):
    plt.plot(fpr[i], tpr[i], label=f'Class {x_train_each_class[i]} (AUC= {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")
plt.show()


y_class=[np.argmax(x) for x in Y_pred]
predictions = model.predict(x_test_each_class)
predicted_class = np.argmax(predictions, axis=1)
predicted_class
print(classification_report(y_test_new,y_class))


cm_data = tf.math.confusion_matrix(labels=y_test_new,predictions=y_class)
cm = pd.DataFrame(cm_data, columns=labels, index =labels)
cm.index.name = 'Actual'
cm.columns.name = 'Predicted'
plt.figure(figsize = (5,3))
plt.title('Confusion Matrix', fontsize = 20)
sns.set(font_scale=1.2)
cm = cm.astype('float') / cm.sum(axis=1)
sns.heatmap(cm, cbar=True, cmap=plt.cm.Blues, annot=True, annot_kws={"size": 16},fmt = '.2f')
