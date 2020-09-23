import math
import numpy as np

class Bezier:
    """
    Bezier curve interpolation around points

        Attributes:
            n: The number of points around which it will be interpolated
            points: The coordinate of the points around which it will be interpolated
            curve_pts_num: The number of points on a Bezier curve between two points
    """

    n = None
    points = None
    curve_pts_num = None
    curve_pts = None

    def __init__(self, n, points, curve_pts_num):
        """
        Initializes the class

            Parameters:
                n: The number of points around which it will be interpolated
                points: The coordinate of the points around which it will be interpolated
                curve_pts_num: The number of points on a Bezier curve between two points
        """
        self.n = n
        self.points = points
        self.curve_pts_num = curve_pts_num
        self.fixVariables()

    def fixVariables(self):
        """
        Fixes the type of the variables
        """

        print(type(self.points))

    def createCoefficientMatrix(self):
        """
            Creates the coefficient matrix for the Bezier curve interpolation

                Returns:
                    numpy.ndarray: The coefficient matrix
        """
        C = np.zeros((self.n, self.n))

        for i in range(self.n):
            r = i + 1 if i + 1 < self.n else (i + 1) % self.n
            row = np.zeros(self.n)
            row[i], row[r] = 1, 2
            C[i] = row

        return C

    def createEndPointVector(self):
        """
        Creates the column vector which contains the end points of each curve connecting two points

            Returns:
                numpy.ndarray: The column vector
        """
        P = np.zeros((self.n, 2))

        for i in range(self.n):
            l = i + 1 if i + 1 < self.n else (i + 1) % self.n
            r = i + 2 if i + 2 < self.n else (i + 2) % self.n

            val = 2 * self.points[l] + self.points[r]
            P[i] = val
        
        return P


    def findPoints(self):
        """
        Finds the points on the smooth curve
        """
        self.createCoefficientMatrix()
        self.createEndPointVector()
    
if __name__ == "__main__":
    # Test some methods

    points = np.array([
        [150, 200],
        [200, 150], 
        [150, 100],
        [100, 150]
    ])
    b = Bezier(4, points, 30)
    b.findPoints()