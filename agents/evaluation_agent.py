from sklearn.metrics import accuracy_score, classification_report

class EvaluationAgent:

    def evaluate(self, y_test, predictions):

        accuracy = accuracy_score(y_test, predictions)
        print("\nEvaluation Agent Report")
        print("Accuracy:", accuracy)

        print("\nClassification Report:")
        print(classification_report(y_test, predictions))