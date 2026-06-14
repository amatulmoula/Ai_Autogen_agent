from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class MLAgent:

    def train_model(self, X, y):

        # split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # create model
        model = RandomForestClassifier()

        # train
        model.fit(X_train, y_train)

        # predict
        pred = model.predict(X_test)

        # accuracy
        accuracy = accuracy_score(y_test, pred)

        print("\nML Agent Training Completed")
        print("Model Accuracy:", accuracy)

        return model