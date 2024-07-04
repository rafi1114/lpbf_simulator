################ Version 6 #############
## Replacing the timer with frame rate ###
##########################################


# Simulation Parameters
RENDER = False
GRID_X, GRID_Y = 200, 200
START_X, START_Y = 0, 0
END_SCREEN = False #If true, will create a plot of the highest temperature recorded in each point 
GEOMETRY = 0 ##### 0 for Square, 1 for Circle and 2 for Triangle

# Laser Parameters
POWER = 2000                ### For now, meltpool temperature when laser is on a spot without residue
SPEED = 10
DIAMETER = 10
FADE_TIME = 100             ### Time(Frames) after which the residue heat will be zero 
FADE_COEFF = 100            ### Reduction in temperature with each frame
HATCH = 10

# Importing Dependencies
import pandas as pd
import numpy as np
import cProfile
import pstats
import io
import matplotlib.pyplot as plt

if RENDER == True:
    import pygame
    from pygame.locals import *

# Creating a circular shape for the boundary
def create_circular_mask(diameter, grid_size):
    mask = np.zeros((grid_size, grid_size), dtype=int)
    center = grid_size // 2
    radius = diameter / 2
    for i in range(grid_size):
        for j in range(grid_size):
            if (i - center) ** 2 + (j - center) ** 2 <= radius ** 2:
                mask[i, j] = 1
    return mask

#creating a triangle shaped boundary
def create_triangle_mask(side_length1, side_length2):
    mask = np.zeros((side_length1, side_length2), dtype=int)
    for i in range(side_length1):
        for j in range(side_length2):
            if j <= side_length2 - i*(side_length1/side_length2):
                mask[i, j] = 1
    return mask

def create_square_mask(side_length1, side_length2):
    mask = np.ones((side_length1, side_length2), dtype=int)
    return mask

class laser:
    def __init__(self):
        self.temperature = POWER
        self.x = START_X
        self.y = START_Y
        self.direction = 'down'
        self.param1 = GRID_X
        self.param2 = GRID_Y
        if GEOMETRY== 2:
            self.shape_mask = create_triangle_mask(self.param1, self.param2)
        elif GEOMETRY == 1:
            self.shape_mask = create_circular_mask(self.param1, self.param2)
        else:
            self.shape_mask = create_square_mask(self.param1, self.param2)

    def move_left(self):
        self.direction = 'left'

    def move_right(self):
        self.direction = 'right'

    def move_up(self):
        self.direction = 'up'

    def move_down(self):
        self.direction = 'down'

    def laser_pass(self):
        if self.direction == 'left':
            self.x -= SPEED
        if self.direction == 'right':
            self.x += SPEED
        if self.direction == 'up':
            self.y -= SPEED
        if self.direction == 'down':
            self.y += SPEED

    def is_within_shape(self, x, y):
        if 0 <= x < GRID_X and 0 <= y < GRID_Y:
            return self.shape_mask[y][x] == 1
        else:
            return False

    def hatching(self):
        if self.x >= GRID_X:
            return
        if not self.is_within_shape(self.x, self.y):
            if self.is_within_shape(self.x + SPEED, self.y + 2 * SPEED) == False and self.is_within_shape(self.x + SPEED, self.y - 2 * SPEED) == False:
                self.y = self.y + 3 * SPEED if self.direction == 'up' else self.y - 3 * SPEED
                self.x += HATCH
                self.direction = 'down' if self.direction == 'up' else 'up'
            elif self.is_within_shape(self.x + SPEED, self.y + SPEED) == False and self.is_within_shape(self.x + SPEED, self.y - SPEED) == False:
                self.y = self.y + 2 * SPEED if self.direction == 'up' else self.y - 2 * SPEED
                self.x += HATCH
                self.direction = 'down' if self.direction == 'up' else 'up'
            elif not self.is_within_shape(self.x + SPEED, self.y):
                self.y = self.y + SPEED if self.direction == 'up' else self.y - SPEED
                self.x += HATCH
                self.direction = 'down' if self.direction == 'up' else 'up'
            else:
                self.x += HATCH
                self.direction = 'down' if self.direction == 'up' else 'up'

class thermal_profile():
    def __init__(self, laser):
        self.laser = laser
        self.duration = np.zeros((GRID_X, GRID_Y)) # Initialized once
        self.start_time = np.full((GRID_X, GRID_Y),np.nan) # Initialized once
        self.residues = np.zeros((GRID_X, GRID_Y))
        self.accum = np.zeros((GRID_X, GRID_Y))
        self.maxt = np.zeros((GRID_X, GRID_Y))

    def calculate_values(self):
        y_indices, x_indices = np.indices((GRID_X, GRID_Y))
        distance = np.sqrt((x_indices - self.laser.x) ** 2 + (y_indices - self.laser.y) ** 2)
        tempx = np.where(distance <= DIAMETER, POWER, 0)
        return tempx, distance

    def residue(self, tempx, xx):
        within_diameter = xx < DIAMETER

        # Update maxt
        self.maxt = np.maximum(self.maxt, self.residues + tempx) 

        # Update accum, start_time and duration 
        accum = np.where(within_diameter & ~np.isnan(self.start_time), self.residues, self.accum) #If laser at spot and the start_timer is not null, then accum must save the previous value of residues
        
        
        start_time = np.where(within_diameter, 1, self.start_time) # If laser at that spot, set start_timer to Current Time

        duration = np.where(~within_diameter & ~np.isnan(self.start_time), self.duration+1, 0)
        
        faded = self.duration >= FADE_TIME
        is_zero = self.duration * FADE_COEFF > accum + POWER / 2

        start_time = np.where(faded, np.nan, start_time)
        accum = np.where((duration == 0) & np.isnan(start_time), 0, accum)
        residues = np.where(np.logical_or(faded, ((duration == 0) & np.isnan(start_time))), 0, self.residues)
        residues = np.where(np.logical_and((duration == 0), ~np.isnan(start_time)), POWER/2 + accum, residues)
        residues = np.where(np.logical_and(~faded, duration != 0) & is_zero, 0, residues)
        residues = np.where(np.logical_and(~faded, duration != 0) & ~is_zero, accum + POWER / 2 - duration* FADE_COEFF, residues)



        self.start_time = start_time
        self.duration = duration
        self.residues = residues
        self.accum = accum

        return self.residues, self.duration, self.start_time, tempx, xx, self.accum, self.maxt

    def visual(self, residue, duration, start_time, tempx, xx, accum, maxt):
        v_duration = []
        v_residue = []
        v_start_time = []
        v_dist = []
        v_tot = []
        v_accum = []
        v_maxt = []

        for i in range(0, GRID_X, int(GRID_X/8)):
            v_d_temp = []
            v_r_temp = []
            v_s_temp = []
            v_dist_temp = []
            v_tot_temp = []
            v_accum_temp = []
            v_maxt_temp = []

            for j in range(0, GRID_Y, int(GRID_Y/8)):
                d_temp = duration[i][j]
                r_temp = residue[i][j]
                s_temp = start_time[i][j]
                tot_temp = residue[i][j] + tempx[i][j]
                dist_temp = xx[i][j]
                accum_time_temp = accum[i][j]
                maxt_temp = maxt[i][j]

                v_d_temp.append(d_temp)
                v_r_temp.append(r_temp)
                v_s_temp.append(s_temp)
                v_dist_temp.append(dist_temp)
                v_tot_temp.append(tot_temp)
                v_accum_temp.append(accum_time_temp)
                v_maxt_temp.append(maxt_temp)

            v_duration.append(v_d_temp)
            v_residue.append(v_r_temp)
            v_start_time.append(v_s_temp)
            v_dist.append(v_dist_temp)
            v_tot.append(v_tot_temp)
            v_accum.append(v_accum_temp)
            v_maxt.append(v_maxt_temp)

        #print(pd.DataFrame(v_dist))
        print(pd.DataFrame(v_duration))
        #print("The residues are: ", "\n", pd.DataFrame(v_residue))
        #print(pd.DataFrame(v_start_time))
        #print(pd.DataFrame(v_tot))
        #print("The accums are: ", pd.DataFrame(v_accum))
        #print("The maxt are: ", pd.DataFrame(v_maxt))

class Render():
    def __init__(self, laser, res, surface):   
        self.laser = laser
        self.res = res
        self.surface = surface          

    def apply_to_surface(self, tempx, xx):
        pixel_array = pygame.PixelArray(self.surface)
        residue, _, _, _, _, accum, _ = self.res.residue(tempx, xx)
        for i in range(0, GRID_X):
            for j in range(0, GRID_Y):
                if not self.laser.is_within_shape(i, j):
                    pixel_array[j, i] = (100, 100, 255)
                else:
                    temp = residue[i][j] + tempx[i][j]
                    color_value = int((temp / 12000) * 255)
                    if color_value > 255:
                        pixel_array[j, i] = (color_value / 2, color_value / 2, 0)
                    else:
                        pixel_array[j, i] = (color_value, 0, 0)
        del pixel_array

    def max_temp_profile(self, tempx, xx):
        pixel_array = pygame.PixelArray(self.surface)
        _, _, _, _, _, _, maxt = self.res.residue(tempx, xx)
        for i in range(0, GRID_X):
            for j in range(0, GRID_Y):
                if not self.laser.is_within_shape(i, j):
                    pixel_array[j, i] = (0, 0, 255)
                else:
                    temp = maxt[i][j]
                    color_value = int((temp / 12000) * 255)
                    if color_value > 255:
                        pixel_array[j, i] = (color_value / 2, color_value / 2, 0)
                    else:
                        pixel_array[j, i] = (color_value, 0, 0)
        del pixel_array

class Process:
    def __init__(self):
        if RENDER:
            pygame.init()
            self.surface = pygame.display.set_mode((GRID_X, GRID_Y))
            self.laser = laser()
            self.profile = thermal_profile(self.laser)
            self.render = Render(self.laser, self.profile, self.surface)
            self.profile.calculate_values()
            self.update_background()
        else:
            self.laser = laser()
            self.profile = thermal_profile(self.laser)
            self.profile.calculate_values()
            self.no_render()


    def update_background(self):
        tempx, xx = self.profile.calculate_values()
        residue, duration, start_time, _, _, accum, maxt = self.profile.residue(tempx, xx)
        self.render.apply_to_surface(tempx, xx)
        self.profile.visual(residue, duration, start_time, tempx, xx, accum, maxt)
        pygame.display.flip()

    def no_render(self):
        tempx, xx = self.profile.calculate_values()
        residue, duration, start_time, _, _, accum, maxt = self.profile.residue(tempx, xx)
        #self.profile.visual(residue, duration, start_time, tempx, xx, accum, maxt)

    def end_screen(self):
        tempx, xx = self.profile.calculate_values()
        _,_,_, _, _, _, maxt = self.profile.residue(tempx, xx)
        plt.imshow(maxt, cmap='magma')
        plt.colorbar()
        plt.show()



    def run(self):
        running = True
        while running:
            if RENDER:
                for event in pygame.event.get():  #If we want to control the laser with our keyboard
                    if event.type == KEYDOWN:
                        if event.key == K_ESCAPE:
                            running = False
                        if event.key == K_LEFT:
                            self.laser.move_left()
                        if event.key == K_RIGHT:
                            self.laser.move_right()
                        if event.key == K_UP:
                            self.laser.move_up()
                        if event.key == K_DOWN:
                            self.laser.move_down()

                    elif event.type == QUIT:
                        running = False

           
                if self.laser.x >= GRID_X:
                    if END_SCREEN:
                        self.end_screen()
                    running = False
                else:
                    self.laser.hatching()
                    self.laser.laser_pass()
                    self.update_background()
            else:
                if self.laser.x >= GRID_X:
                    if END_SCREEN:
                        self.end_screen()
                    running = False
                else:
                    self.laser.hatching()
                    self.laser.laser_pass()
                    self.no_render()

def profile_main():
    pr = cProfile.Profile()
    pr.enable()

    game = Process()
    game.run()

    pr.disable()
    s = io.StringIO()
    sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    with open('profile_results.txt', 'w') as f:
        f.write(s.getvalue())


if __name__ == '__main__':
    profile_main()