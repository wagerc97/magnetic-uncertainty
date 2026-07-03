import numpy as np
import os
from residual_styled import plot_cv_predictions_hexbin

def test_plot_cv_predictions_hexbin():
    y_true = np.random.rand(100)
    y_pred = y_true + np.random.normal(0, 0.1, 100)
    save_path = "test_hexbin_plot.png"
    title = "Test Hexbin Plot"
    
    try:
        plot_cv_predictions_hexbin(y_pred, y_true, save_path, title, show=False)
        if os.path.exists(save_path):
            print(f"Successfully created plot: {save_path}")
            os.remove(save_path)
        else:
            print(f"Failed to create plot: {save_path}")
    except Exception as e:
        print(f"Error during plotting: {e}")

if __name__ == "__main__":
    test_plot_cv_predictions_hexbin()
