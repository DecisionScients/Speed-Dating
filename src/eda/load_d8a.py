# --------------------------------------------------------------------------- #
#                               D8A Class                                     #
# --------------------------------------------------------------------------- #
import pandas as pd
class LoadD8A:

    def __init__(self, filepath=None):
        """Initializes class and saves filepath

        Args:
        filepath(str): Absolute filepath including filename for raw data

        """
        if filepath is None:
            self.filepath = "./data/raw/speed_dating_raw_2004.csv"
        else:
            self.filepath = filepath

    def load(self):
        """Loads raw data into a Pandas DataFrame object"""
        df = pd.read_csv(self.filepath, encoding='ISO-8859-1')
        return(df)