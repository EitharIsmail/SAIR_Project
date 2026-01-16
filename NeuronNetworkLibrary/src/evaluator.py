import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Evaluator:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def compute_confusion_matrix(self, y_true, y_pred):
        y_true = y_true.flatten().astype(int)
        y_pred = y_pred.flatten().astype(int)
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        return TP, TN, FP, FN

    def compute_multiclass_matrix(self, y_true_idx, y_pred_idx, num_classes):
        matrix = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(y_true_idx, y_pred_idx):
            # Added bounds check to prevent index errors
            if t < num_classes and p < num_classes:
                matrix[t, p] += 1
        return matrix

    def evaluate(self, x_test, y_test):
        # 1. Get Predictions
        y_pred = self.pipeline.predict(x_test)
        
        # 2. Determine if Binary or Multiclass
        # If output dim is 1 or it's a 1D array, it's binary
        is_binary = (y_test.shape[1] == 1) if y_test.ndim > 1 else True

        if is_binary:
            acc = np.mean(y_pred.flatten() == y_test.flatten())
        else:
            true_indices = np.argmax(y_test, axis=1)
            # Ensure y_pred is also converted to indices if it isn't already
            pred_indices = y_pred.flatten() if y_pred.ndim == 1 else np.argmax(y_pred, axis=1)
            acc = np.mean(pred_indices == true_indices)

        # 3. Retrieve History safely
        train_loss = self.pipeline.loss_history[-1] if getattr(self.pipeline, 'loss_history', None) else 0.0
        train_acc  = self.pipeline.accuracy_history[-1] if getattr(self.pipeline, 'accuracy_history', None) else 0.0
        
        val_loss_hist = getattr(self.pipeline, 'val_loss_history', [])
        val_acc_hist = getattr(self.pipeline, 'val_accuracy_history', [])

        val_loss_str = f"{val_loss_hist[-1]:.4f}" if len(val_loss_hist) > 0 else "N/A"
        val_acc_str  = f"{val_acc_hist[-1]:.2f}%" if len(val_acc_hist) > 0 else "N/A"

        print("\n" + "="*55)
        print(f"      FINAL PERFORMANCE REPORT ({'Binary' if is_binary else 'Multi-Class'})")
        print("="*55)
        print(f"{'Metric':<10} | {'Train':<10} | {'Val':<10} | {'Test (New)':<10}")
        print("-" * 55)
        print(f"{'Loss':<10} | {train_loss:.4f}     | {val_loss_str:<10} | N/A")
        print(f"{'Accuracy':<10} | {train_acc:.2f}%    | {val_acc_str:<10} | {acc*100:.2f}%")
        
        metrics = {'accuracy': acc}

        if is_binary:
            TP, TN, FP, FN = self.compute_confusion_matrix(y_test, y_pred)
            epsilon = 1e-15
            precision = TP / (TP + FP + epsilon)
            recall = TP / (TP + FN + epsilon)
            f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
            
            print("\n--- CLASSIFICATION METRICS (Test Set) ---")
            print(f"Confusion Matrix:\n [[TN={TN}  FP={FP}]\n  [FN={FN}  TP={TP}]]")
            print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1_score:.4f}")
            metrics['f1_score'] = f1_score
        else:
            print("\n--- CLASSIFICATION METRICS (Test Set) ---")
            print(f"Overall Test Accuracy: {acc*100:.2f}%")

        print("="*55 + "\n")
        return metrics

    def plot_confusion_matrix(self, x_test, y_test):
        y_pred = self.pipeline.predict(x_test)
        is_binary = (y_test.shape[1] == 1) if y_test.ndim > 1 else True

        if is_binary:
            TP, TN, FP, FN = self.compute_confusion_matrix(y_test, y_pred)
            cm = np.array([[TN, FP], [FN, TP]])
            labels = ['Negative', 'Positive']
        else:
            num_classes = y_test.shape[1]
            y_true_idx = np.argmax(y_test, axis=1)
            y_pred_idx = y_pred.flatten() if y_pred.ndim == 1 else np.argmax(y_pred, axis=1)
            cm = self.compute_multiclass_matrix(y_true_idx, y_pred_idx, num_classes)
            labels = [str(i) for i in range(num_classes)]

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.show()

    # (Keep your plot_history and plot_roc_curve methods as they were)