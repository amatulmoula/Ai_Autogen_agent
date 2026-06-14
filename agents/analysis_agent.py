import pandas as pd

class AnalysisAgent:

    def analyze_crime_types(self, df):

        crime_counts = df['Primary Type'].value_counts().head(10)

        print("\nTop 10 Crime Types:")
        print(crime_counts)

        return crime_counts


    def analyze_by_hour(self, df):

        hour_counts = df['Hour'].value_counts().sort_index()

        print("\nCrime distribution by Hour:")
        print(hour_counts)

        return hour_counts