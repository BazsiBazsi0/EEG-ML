from imblearn.over_sampling import SMOTE

class SmoteHelper:
    @staticmethod
    def smote_processor(self, x, y):
        sm = SMOTE(random_state=42)
        x_resampled, y_resampled = sm.fit_resample(x, y)
        return x_resampled, y_resampled