import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
from PIL import Image
from bresenham import bresenham
import sys
import random
import lloyd
import noise

class Bezier:
    """
    Bezier curve interpolation around points

        Attributes:
            n: The number of points around which it will be interpolated
            points: The coordinate of the points around which it will be interpolated
            curve_pts_num: The number of points on a Bezier curve between two points
    """
    size = None
    n = None
    points = None
    curve_pts_num = None
    curve_pts = None
    C = None
    P = None
    A = None
    B = None
    mask = None
    maxN = None
    maxS = None
    num = 0

    def __init__(self, size, n, points, curve_pts_num):
        """
        Initializes the class

            Parameters:
                size: Size of the field
                n: The number of points around which it will be interpolated
                points: The coordinate of the points around which it will be interpolated
                curve_pts_num: The number of points on a Bezier curve between two points
        """
        self.size = size
        self.n = n
        self.points = points
        self.curve_pts_num = curve_pts_num
        self.maxN = self.size[1] // 2
        self.maxS = self.size[1] // 2
        self.fixVariables()

    def fixVariables(self):
        """
        Fixes the type of the variables
        """
        if type(self.points) != np.ndarray:
            self.points = np.array(self.points)

    def createCoefficientMatrix(self):
        """
            Creates the coefficient matrix for the Bezier curve interpolation
        """
        C = np.zeros((self.n, self.n))

        for i in range(self.n):
            r = i + 1 if i + 1 < self.n else (i + 1) % self.n
            row = np.zeros(self.n)
            row[i], row[r] = 1, 2
            C[i] = row

        self.C = C
        #return C

    def createEndPointVector(self):
        """
        Creates the column vector which contains the end points of each curve connecting two points
        """
        P = np.zeros((self.n, 2))

        for i in range(self.n):
            l = i + 1 if i + 1 < self.n else (i + 1) % self.n
            r = i + 2 if i + 2 < self.n else (i + 2) % self.n

            val = 2 * self.points[l] + self.points[r]
            P[i] = val
        
        self.P = P
        #return P

    def findControlPoints(self):
        """
        Find the control points for the Bezier curve
        """
        A = np.linalg.solve(self.C, self.P)

        B = np.zeros_like(A)

        for i in range(self.n):
            l = i + 1 if i + 1 < self.n else (i + 1) % self.n
            B[i] = 2 * self.points[l] - A[l]
        
        self.A = A
        self.B = B
        #print(B)

    def findPoints(self):
        """
        Finds the points on the smooth curve
        """
        self.createCoefficientMatrix()
        self.createEndPointVector()

        self.findControlPoints()
        
        all_pts = []

        for i in range(self.n):
            next_i = i + 1 if i + 1 < self.n else (i + 1) % self.n
            dpts = np.linspace(0, 1, self.curve_pts_num)
            for j in dpts:
                pt = np.power(1 - j, 3) * self.points[i] + 3 * j * np.power(1 - j, 2) * self.A[i] + 3 * (1 - j) * np.power(j, 2) * self.B[i] + np.power(j, 3) * self.points[next_i]
                all_pts.append(pt.tolist())

        self.curve_pts = np.array(all_pts)

    def createMask(self):
        mask = np.zeros(self.size)
        for i in range(self.curve_pts.shape[0]):
            next_p = i + 1 if i + 1 < self.curve_pts.shape[0] else (i + 1) % self.curve_pts.shape[0]

            pts = list(bresenham(int(self.curve_pts[i][0]), int(self.curve_pts[i][1]), int(self.curve_pts[next_p][0]), int(self.curve_pts[next_p][1])))

            mask[int(self.curve_pts[i][1])][int(self.curve_pts[i][0])] = 1
            mask[int(self.curve_pts[next_p][1])][int(self.curve_pts[next_p][0])] = 1

            for j in pts:
                mask[j[1]][j[0]] = 1
                #if j[1] < self.maxN:
                #    self.maxN = j[1]
                #if j[1] > self.maxS:
                #    self.maxS = j[1]
            
        #rv = mask[::, ::-1]
        #Image.fromarray((mask * 255).astype('uint8')).show()
        #rv = np.flipud(mask)
        #print(">> ", self.maxN, " ", self.maxS)
        rv = mask[::-1]
        #print()
        #rv = mask

        self.mask = rv

        locy, locx = self.mask.shape[0] // 2, self.mask.shape[1] // 2

        queue = []
        queue.append([locx, locy])

        
        while len(queue) > 0:
            pt = queue.pop()
            self.mask[pt[1]][pt[0]] = 1
            #print(">>> ", pt[1], " ", pt)

            x = pt[0]
            y = pt[1]

            if x - 1 >= 0 and self.mask[y][x - 1] != 1:
                queue.append([x - 1, y])
            if x + 1 < self.mask.shape[1] and self.mask[y][x + 1] != 1:
                queue.append([x + 1, y])
            if y - 1 >= 0 and self.mask[y - 1][x] != 1:
                queue.append([x, y - 1])
            if y + 1 < self.mask.shape[0] and self.mask[y + 1][x] != 1:
                queue.append([x, y + 1])


        #self.assignPoint(locx, locy)

        Image.fromarray((self.mask * 255).astype('uint8')).save("island_3.png")

    def assignPoint(self, locx, locy):
        '''
        for i in range(self.size[0]):
            fill = False
            #if i == self.maxN:
            #    continue
            #if i == self.maxS:
            #    break
            for j in range(self.size[1] - 1):
                #print("=> ", fill, " ", i, " ", j, " ", self.mask[i][j])
                if self.mask[i][j] == 1 and self.mask[i][j + 1] != 1:
                    fill ^= True
                elif self.mask[i][j] != 1 and fill == True:
                    self.mask[i][j] = 1    
        '''
        '''
        #print("-> ", locx, " ", locy)
        self.num += 1
        print(self.num)
        self.mask[locy][locx] = 1
        if locx + 1 < self.size[0] and self.mask[locy][locx + 1] != 1:
            self.assignPoint(locx + 1, locy)
        if locx - 1 >= 0 and self.mask[locy][locx - 1] != 1:
           self.assignPoint(locx - 1, locy) 
        if locy + 1 < self.size[1] and self.mask[locy + 1][locx] != 1:
           self.assignPoint(locx, locy + 1)
        if locy - 1 >= 0 and self.mask[locy - 1][locx] != 1:
           self.assignPoint(locx, locy - 1)
        '''

    def getMask(self):
        return self.mask
        

    def draw(self):
        """
        Draws a plot of the curve and the points
        """
        x, y = self.curve_pts[:,0], self.curve_pts[:,1]
        px, py = self.points[:,0], self.points[:,1]

        plt.plot(x, y, "b-")
        plt.plot(px, py, "ko")
        plt.axes().set_aspect('equal')
        #plt.show()




class Graph:
    points = None
    adj = None
    adj_m = None

    def __init__(self, points, adj):
        self.points = points
        self.adj = adj
    
    def generateAdjMatrix(self):
        self.adj_m = np.zeros((len(self.points), len(self.points)))

        for i in self.adj:
            self.adj_m[i[0]][i[1]] = 1
            self.adj_m[i[1]][i[0]] = 1
        
    def getPointsInsideMask(self, mask):

        flipped_mask = mask[::-1]

        points_inside = []
        
        for i in self.points:
            if i[0] >=mask.shape[1] or i[1] >= mask.shape[0]:
                continue

            if flipped_mask[int(i[1])][int(i[0])] == 1:
                points_inside.append(i.tolist())

        print(points_inside)







class MapRegion:
    num_points = None
    size = None
    vor = None
    vor_points = None

    def __init__(self, size, num_points):
        self.size = size
        self.num_points = num_points
    
    def generate(self):
        points = []
        for i in range(self.num_points):
            x = random.randint(0, self.size[0])
            y = random.randint(0, self.size[1])
            points.append([x, y])

        relaxed_points = lloyd.Field(np.array(points))
        relaxed_points.relax()
        relaxed_points = relaxed_points.get_points()
        self.vor_points = relaxed_points

        vor = Voronoi(np.array(relaxed_points))
        self.vor = vor
        #fig = voronoi_plot_2d(vor)
        #plt.show()

    def returnData(self):
        return self.vor_points, self.vor.ridge_points

        
class PerlinNoise:
    octaves = None
    persistence = None
    scale = None
    lacunarity = None
    size = None
    perlin_noise = None

    def __init__(self, size, scale, octaves, persistence, lacunarity):
        self.size = size
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity

    def generateNoise(self):
        perlin_noise = np.zeros(self.size)

        for i in range(self.size[0]):
            for j in range(self.size[1]):
                perlin_noise[i][j] = noise.pnoise2(i/self.scale, 
                                            j/self.scale, 
                                            octaves=self.octaves, 
                                            persistence=self.persistence, 
                                            lacunarity=self.lacunarity, 
                                            repeatx=self.size[0], 
                                            repeaty=self.size[1], 
                                            base=0)
        
        self.perlin_noise = perlin_noise
    
    def getNoise(self):
        return self.perlin_noise

class DrawMap:
    size = None
    def __init__(self, size):
        self.size = size

    def drawMap(self, island, hills):
        hills = np.int_(hills * 20)
        #print(hills)
        t_mask = np.zeros(self.size)
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                combine = (hills[i][j] * island[i][j] + 1) * island[i][j]
                t_mask[i][j] = combine if combine > 0 else 0
        
        return t_mask
        #Image.fromarray((t_mask * 255).astype('uint8'), mode="L").show()#.save("hills.png")

    def drawMap3D(self, island, hills):
        hills = np.int_(hills * 20)
        
        t_mask = np.zeros(self.size)
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                combine = hills[i][j] * island[i][j]
                t_mask[i][j] = combine if combine > 0 else 0
        
        x, y = np.meshgrid(range(t_mask.shape[1]), range(t_mask.shape[0]))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        #ax.set_aspect('equal')
        ax.set_zlim3d(-1,799)
        ax.plot_surface(x, y, t_mask)
        plt.title("Graph")
        plt.show()


class TestMap:
    #testing
    def __init__(self):
        pass
        #if test == 1:
        #    self.runTest1()

    def runTest1(self):
        points = np.array([
            [400, 505],
            [580, 400],
            [400, 130],
            [280, 400]
        ])
        b = Bezier((800, 800), 4, points, 30)
        b.findPoints()
        b.createMask()

        n = PerlinNoise((800, 800), 80.0, 6, 0.5, 2.0)
        n.generateNoise()

        mp = DrawMap((800, 800))
        return mp.drawMap(b.getMask(), n.getNoise())   

'''
if __name__ == "__main__":
    #sys.setrecursionlimit(800*800)
    # Test some methods

    #points = np.array([
    #    [400, 460],
    #    [580, 400], 
    #    [400, 340],
    #    [385, 400]
    #])

    tt = TestMap()
    tt.runTest1()

 
    points = np.array([
        [400, 505],
        [580, 400],
        [400, 130],
        [280, 400]
    ])

    b = Bezier((800, 800), 4, points, 30)
    b.findPoints()
    b.draw()
    b.createMask()
    mp = MapRegion((800, 800), 250)
    mp.generate()
    pts, adj = mp.returnData()
    g = Graph(pts, adj)
    g.getPointsInsideMask(b.getMask())
    n = PerlinNoise((800, 800), 80.0, 6, 0.5, 2.0)
    n.generateNoise()
    #n.getNoise()

    mp = DrawMap((800, 800))
    mp.drawMap(b.getMask(), n.getNoise())
'''
    