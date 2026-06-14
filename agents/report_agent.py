class ReportAgent:

    def generate_report(self, df):

        print("\n===== Crime Data Report =====")

        print("\nTop Crime Types:")
        print(df['Primary Type'].value_counts().head())

        print("\nCrime by Hour:")
        print(df['Hour'].value_counts().sort_index())

        print("\nTotal Records:", df.shape[0])