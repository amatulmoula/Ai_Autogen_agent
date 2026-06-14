import pandas as pd

class DataAgent:

    def load_data(self, path):

        df = pd.read_csv(path)

        print("Data loaded successfully")

        return df