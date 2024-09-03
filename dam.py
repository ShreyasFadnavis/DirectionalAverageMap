import numpy as np
from dipy.reconst.base import ReconstModel, ReconstFit
from dipy.data import fetch_cenir_multib, read_cenir_multib
from dipy.reconst.multi_voxel import multi_voxel_fit
from numpy import polyfit


class DirectionalAverageModel(ReconstModel):
    """
    A custom diffusion model that computes the directional average of diffusion 
    MRI signals across unique b-value shells.

    Attributes:
        gtab : GradientTable
            The gradient table containing diffusion MRI acquisition parameters.
        bvals : np.ndarray
            Rounded b-values to the nearest 1000.
        unique_bvals : np.ndarray
            Unique b-values used in the diffusion MRI data.
        xb : np.ndarray
            Predefined log values for modeling signal attenuation.
    """

    def __init__(self, gtab):
        """
        Initializes the DirectionalAverageModel with the provided gradient table.

        Parameters:
            gtab : GradientTable
                The gradient table containing diffusion MRI acquisition parameters.
        """
        super().__init__(gtab)
        self.gtab = gtab
        # Round the bvals to the nearest 1000
        self.bvals = np.array([round(x, -3) for x in gtab.bvals])
        # Get unique bvals
        self.unique_bvals = np.unique(self.bvals)
        # Predefined log values for modeling attenuation
        # Adapt xb to the number of shells detected
        self.xb = -np.log(np.arange(1, len(self.unique_bvals)))  # Log values for attenuation

    @multi_voxel_fit
    def fit(self, data):
        """
        Fits the DirectionalAverageModel to the provided diffusion MRI data.

        Parameters:
            data : np.ndarray
                The diffusion MRI data to fit the model on.

        Returns:
            DirectionalAverageFit
                An object containing the fitted parameters of the model.
        """
        # Mean signal for the non-diffusion weighted images (b0)
        s0 = data[self.bvals == self.unique_bvals[0]].mean()

        # Compute signal for each shell relative to s0 dynamically
        S_bvals = [data[self.bvals == bval].mean() / s0 for bval in self.unique_bvals[1:]]

        # Logarithm of signal attenuation
        S_log = np.log(S_bvals)

        # Fit a line to the log-attenuated signal
        A, B = polyfit(self.xb[:len(S_log)], S_log, deg=1)

        return DirectionalAverageFit(A, B)


class DirectionalAverageFit(ReconstFit):
    """
    A class representing the fit of the DirectionalAverageModel.

    Attributes:
        P : float
            The slope of the fitted line in the log signal space.
        V : float
            The intercept of the fitted line in the log signal space.
    """

    def __init__(self, P, V):
        """
        Initializes the DirectionalAverageFit with the provided parameters.

        Parameters:
            P : float
                The slope of the fitted line in the log signal space.
            V : float
                The intercept of the fitted line in the log signal space.
        """
        self.P = P
        self.V = V

    def predict(self, gtab):
        """
        Predicts the signal attenuation using the fitted model.

        Parameters:
            gtab : GradientTable
                The gradient table containing diffusion MRI acquisition parameters.

        Returns:
            np.ndarray
                The predicted signal attenuation based on the fitted parameters.
        """
        # Using the fitted parameters P and V to predict the signal
        bvals = np.array([round(x, -3) for x in gtab.bvals])
        unique_bvals = np.unique(bvals)
        xb = -np.log(np.arange(1, len(unique_bvals)))  # Log values for attenuation
        S_predicted = np.exp(self.P * xb + self.V)
        return S_predicted


if __name__ == "__main__":
    # Fetch and read the CENIR dataset
    fetch_cenir_multib(with_raw=False)
    img, gtab = read_cenir_multib()
    data = img.get_fdata()

    # Initialize the model
    dam = DirectionalAverageModel(gtab=gtab)

    # Fit the model
    daf = dam.fit(data)

    # Example usage of the fitted model
    predicted_signal = daf.predict(gtab)
    print("Fitted Parameters:", daf.P, daf.V)
    print("Predicted Signal:", predicted_signal)
