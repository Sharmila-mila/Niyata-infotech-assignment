import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report)

class_names = ['Airplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    y_pred = np.argmax(predictions, axis=1)

    y_true = y_test.flatten()

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(y_true,y_pred,average='weighted')

    recall = recall_score(y_true,y_pred,average='weighted')

    f1 = f1_score(y_true,y_pred,average='weighted')
    print("MODEL EVALUATION")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("CLASSIFICATION REPORT")
    print(classification_report(y_true,y_pred,target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))

    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=class_names,yticklabels=class_names)

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig("outputs/confusion_matrix.png")

    plt.show()
