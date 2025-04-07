import numpy as np

class Muiltple_Linear_Regression:
    
    # The fit function trains the Model on the vecotrs X and y
    def fit(self,X,y):
        self.X = X
        self.y = y
        
        # This is the equations that follows the Muiltple Linear regression Model  
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias column
        self.B = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y  # Safer with pinv
        self.X_b = X_b  # Store for prediction if needed
        self.equ = f"{self.B[0]:.3f}X_0"


        # To make a string which follows he mathematical equation of the Muiltple Linear Regression Model Trained on X and y
        for i in range(1, len(self.B)):
            self.equ += f" + {self.B[i]:.3f}X_{i}"


    # The predict function predicts the values according to the data trained by the fir function
    def predict(self, X_test) -> np.ndarray:
        X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
        return X_test_b @ self.B

    
    # The getEquation functions gets the Muiltple Linear equation for the regression on the data X and y
    def getEquation(self) -> str:
        return self.equ
    
    
