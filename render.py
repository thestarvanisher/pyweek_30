import pygame, sys, random
from pygame.locals import *
from pygame import gfxdraw
from math import sin, cos, tan, sqrt, pi
from math import atan2 as arctan

pygame.mixer.pre_init(44100, -16, 2, 2048)
pygame.init()

info = pygame.display.Info()
WINDOW_SIZE = (info.current_w, info.current_h)
WINDOW = pygame.display.set_mode(WINDOW_SIZE, FULLSCREEN)
pygame.display.set_caption("pyweek-30")
FPS = pygame.time.Clock()

w, h = int(WINDOW_SIZE[0] / 2), int(WINDOW_SIZE[1] / 2)

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)

size = 5
walking_right = [pygame.transform.scale(pygame.image.load(f"data/player/walking/{i}.png").convert_alpha(), (30 * size, 48 * size)) for i in range(6)]
walking_left = [pygame.transform.scale(pygame.transform.flip(pygame.image.load(f"data/player/walking/{i}.png").convert_alpha(), True, False), (30 * size, 48 * size)) for i in range(6)]
walking_up = walking_right
walking_down = walking_left
idle_right = [pygame.transform.scale(pygame.image.load(f"data/player/idle/{i}.png").convert_alpha(), (30 * size, 48 * size)) for i in range(4)]
idle_left = [pygame.transform.scale(pygame.transform.flip(pygame.image.load(f"data/player/idle/{i}.png").convert_alpha(), True, False), (30 * size, 48 * size)) for i in range(4)]
idle_right = [idle_right[i] for i in range(4) for j in range(3)]
idle_left = [idle_left[i] for i in range(4) for j in range(3)]


class Camera:
    def __init__(self, coordinates, alpha=0, beta=-0.8, fov=1500, velocity=0.15):
        self.x, self.y, self.z = coordinates
        self.coordinates = coordinates
        self.alpha, self.beta = alpha, beta
        self.fov = fov
        self.velocity = velocity
        self.c, self.s = cos(self.alpha), sin(self.alpha)

    def move(self, mf, mb, ml, mr):
        if mf:
            self.x += self.velocity * self.c
            self.y += self.velocity * self.s
        if ml:
            self.x -= self.velocity * self.s
            self.y += self.velocity * self.c
        if mb:
            self.x -= self.velocity * self.c
            self.y -= self.velocity * self.s
        if mr:
            self.x += self.velocity * self.s
            self.y -= self.velocity * self.c

    def where_it_is_looking_at(self):
        return cam.x - cam.z / tan(self.beta), cam.y


class Cube:
    def __init__(self, coordinates, color):
        self.x, self.y, self.z = coordinates
        self.coordinates = coordinates
        self.color = color

    def distance(self, player):
        return (player.x-self.x-0.5)**2+(player.y-self.y-0.5)**2+(player.z-self.z-0.5)**2


class Player:
    def __init__(self):
        self.status = "idle_left"
        self.direction = "left"
        self.idle_index = 0
        self.walk_index = 0

    def update_status(self, ml, mr, mu, md):
        if mu:
            self.status = "walking_up"
        elif md:
            self.status = "walking_down"
        elif ml:
            self.direction = "left"
            self.status = "walking_left"
        elif mr:
            self.direction = "right"
            self.status = "walking_right"
        else:
            self.status = "idle_" + self.direction
        self.draw()

    def draw(self):
        if self.status.startswith("walking"):
            i = ["walking_right", "walking_left", "walking_up", "walking_down"].index(self.status)
            j = [walking_right, walking_left, walking_up, walking_down][i]
            pic = j[self.walk_index]
            WINDOW.blit(pic, (w - 15 * size, h - 24 * size))
            self.walk_index += 1
            if self.walk_index == 6:
                self.walk_index = 0
            self.idle_index = 0
        elif self.status.startswith("idle"):
            i = ["idle_right", "idle_left"].index(self.status)
            j = [idle_right, idle_left][i]
            pic = j[self.idle_index]
            WINDOW.blit(pic, (w - 15 * size, h - 24 * size))
            self.idle_index += 1
            if self.idle_index == len(j):
                self.idle_index = 0
            self.walk_index = 0


def terminate():
    pygame.quit()
    sys.exit()


def render(cube, camera, b=0.1):
    def render_point(point, camera):
        def rotate_y(a, x, y, z):
            return z * sin(a) + x * cos(a), y, z * cos(a) - x * sin(a)

        def rotate_z(a, x, y, z):
            return x * cos(a) - y * sin(a), x * sin(a) + y * cos(a), z

        def relative_sa(alpha, beta, x, y, z):
            x, y, z = rotate_z(-alpha, x, y, z)
            x, y, z = rotate_y(beta, x, y, z)
            axy = arctan(y, x)
            az = arctan(z, sqrt(x ** 2 + y ** 2))
            return axy, az

        def sa_to_pix(axy, az):
            if -pi / 2 + b < axy < pi / 2 - b and -pi / 2 + b < az < pi / 2 - b:
                pixel_x, pixel_y = round(-camera.fov * tan(axy)), round(-camera.fov * tan(az) / cos(axy))
                return w + pixel_x, h + pixel_y

        axy, az = relative_sa(camera.alpha, camera.beta, point[0]-camera.x, point[1]-camera.y, point[2]-camera.z)
        return sa_to_pix(axy, az)

    if camera.x-cube.x < 0:
        if camera.y-cube.y < 0:
            if camera.z-cube.z < 0:
                corners = [(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1), (1, 0, 1), (0, 0, 0)]
            elif 0 <= camera.z-cube.z <= 1:
                corners = [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1), (1, 0, 1), (1, 0, 0)]
            elif 1 < camera.z-cube.z:
                corners = [(0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 0, 1), (1, 0, 0), (0, 0, 1)]
        elif 0 <= camera.y-cube.y <= 1:
            if camera.z-cube.z < 0:
                corners = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)]
            elif 0 <= camera.z-cube.z <= 1:
                corners = [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)]
            elif 1 < camera.z-cube.z:
                corners = [(0, 0, 1), (0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 0, 1)]
        elif 1 < camera.y-cube.y:
            if camera.z-cube.z < 0:
                corners = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1), (0, 0, 1), (0, 1, 0)]
            elif 0 <= camera.z-cube.z <= 1:
                corners = [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1), (0, 0, 1), (0, 0, 0)]
            elif 1 < camera.z-cube.z:
                corners = [(0, 1, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1), (0, 0, 1), (0, 0, 0), (0, 1, 1)]
    elif 0 <= camera.x-cube.x <= 1:
        if camera.y-cube.y < 0:
            if camera.z-cube.z < 0:
                corners = [(0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
            elif 0 <= camera.z-cube.z <= 1:
                corners = [(0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0)]
            elif 1 < camera.z-cube.z:
                corners = [(0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1), (1, 0, 0), (0, 0, 0)]
        elif 0 <= camera.y-cube.y <= 1:
            if camera.z-cube.z < 0:
                corners = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
            elif 0 <= camera.z-cube.z <= 1:
                corners = []
            elif 1 < camera.z-cube.z:
                corners = [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
        elif 1 < camera.y-cube.y:
            if camera.z-cube.z < 0:
                corners = [(1, 1, 0), (1, 1, 1), (0, 1, 1), (0, 1, 0), (0, 0, 0), (1, 0, 0)]
            elif 0 <= camera.z-cube.z <= 1:
                corners = [(1, 1, 0), (1, 1, 1), (0, 1, 1), (0, 1, 0)]
            elif 1 < camera.z-cube.z:
                corners = [(1, 1, 1), (1, 0, 1), (0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0)]
    elif 1 < camera.x-cube.x:
        if camera.y-cube.y < 0:
            if camera.z-cube.z < 0:
                corners = [(0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (1, 1, 0), (0, 1, 0), (1, 0, 0)]
            elif 0 <= camera.z-cube.z <= 1:
                corners = [(1, 0, 0), (0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (1, 1, 0)]
            elif 1 < camera.z-cube.z:
                corners = [(1, 0, 0), (0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 1, 0), (1, 0, 1)]
        elif 0 <= camera.y-cube.y <= 1:
            if camera.z-cube.z < 0:
                corners = [(1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0), (0, 1, 0), (0, 0, 0)]
            elif 0 <= camera.z-cube.z <= 1:
                corners = [(1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)]
            elif 1 < camera.z-cube.z:
                corners = [(1, 0, 1), (0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 1, 0), (1, 0, 0)]
        elif 1 < camera.y-cube.y:
            if camera.z-cube.z < 0:
                corners = [(1, 0, 0), (1, 0, 1), (1, 1, 1), (0, 1, 1), (0, 1, 0), (0, 0, 0), (1, 1, 0)]
            elif 0 <= camera.z-cube.z <= 1:
                corners = [(1, 1, 0), (1, 0, 0), (1, 0, 1), (1, 1, 1), (0, 1, 1), (0, 1, 0)]
            elif 1 < camera.z-cube.z:
                corners = [(1, 1, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 1)]

    #corners = [(i/2, j/2, k/2) for i, j, k in corners]

    corners_pixel = [render_point((ix+cube.x, iy+cube.y, iz+cube.z), camera) for ix, iy, iz in corners]

    if corners_pixel and (None not in corners_pixel):
        if len(corners) == 7:
            pygame.gfxdraw.aapolygon(WINDOW, corners_pixel[:-1], cube.color)
            pygame.gfxdraw.filled_polygon(WINDOW, corners_pixel[:-1], cube.color)
        elif len(corners) == 6:
            pygame.gfxdraw.aapolygon(WINDOW, corners_pixel, cube.color)
            pygame.gfxdraw.filled_polygon(WINDOW, corners_pixel, cube.color)
        elif len(corners) == 4:
            pygame.gfxdraw.aapolygon(WINDOW, corners_pixel, cube.color)
            pygame.gfxdraw.filled_polygon(WINDOW, corners_pixel, cube.color)


cam = Camera((0, 0, 18))
moving_forward, moving_back, moving_left, moving_right = False, False, False, False

sand = [(i, i-10, int(i/2)) for i in [200, 190, 180, 170, 160]]
all_cubes_dict = {(i, j, -1):Cube((i, j, -1), random.choice(sand)) for i in range(100) for j in range(100)}

p = Player()

pygame.mouse.set_visible(False)
pygame.mouse.set_pos(w, h)
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            terminate()
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                terminate()
            elif event.key == K_w:
                moving_forward = True
            elif event.key == K_a:
                moving_left = True
            elif event.key == K_s:
                moving_back = True
            elif event.key == K_d:
                moving_right = True
        elif event.type == KEYUP:
            if event.key == K_w:
                moving_forward = False
            elif event.key == K_a:
                moving_left = False
            elif event.key == K_s:
                moving_back = False
            elif event.key == K_d:
                moving_right = False

    cam.move(moving_forward, moving_back, moving_left, moving_right)

    WINDOW.fill((200, 200, 255))

    f1, f2, f3, f4 = -8, 15, -18, 19
    a, b = cam.where_it_is_looking_at()
    a, b = int(a), int(b)
    c = [(a + i, b + j, -1) for i in range(f1, f2) for j in range(f3, f4)]
    cubes = [all_cubes_dict[cc] for cc in c if cc in all_cubes_dict.keys()]
    cubes.sort(key=lambda x: -x.distance(cam))
    for c in cubes:
        render(c, cam)

    p.update_status(moving_left, moving_right, moving_forward, moving_back)
    """t = pygame.font.Font(None, 36).render(f"{FPS.get_fps()}", True, BLACK)
    WINDOW.blit(t, (100, 100))"""
    pygame.display.update()
    FPS.tick(23)