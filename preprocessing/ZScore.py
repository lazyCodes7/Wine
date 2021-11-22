from scipy.stats import zscore
import numpy as np
class ZScore:
    def __init__(self, df, threshold = 3):
        self.threshold = threshold
        self.df = df
    def transform(self):
        self.zscores = zscore(self.df)
        abs_z_scores = np.abs(self.zscores)
        filtered_entries = (abs_z_scores < self.threshold).all(axis=1)
        cleaned_df = self.df[filtered_entries]
        return cleaned_df
        
            
