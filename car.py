## Car Class

''' Defines the car class.'''
import numpy as np
import pygame as pg
from pygame.locals import *
import network as net
import sys

class Car():
	MAX_SPEED = 150
	omega = .5
	acc = 10
	velocity = [0,0]
	location = [0,0]
	speed = 0
	theta = 0
	
	def __init__(self, control, color, track, index):
		self.control = control(self, track)
		self.color = color
		self.track = track
		self.index = index
  self.length = 4
	
	def accel(self, time):
		self.speed = min(self.speed + self.acc*time, self.MAX_SPEED)
	
	def turn(self, time, direction):
		#direction: bool. True is counter clockwise (pos)
		if self.speed > 0:
			self.theta += self.omega*direction*time
			self.theta -= self.omega*(not direction)*time
	
	def brake(self, time):
		self.speed = max(self.speed - 2*self.acc*time, 0)
	
	def update(self, time, events):
		self.control.update(events)
		for cmd in self.control.cmds:
			if cmd == 'accel':
				self.accel(time)
			elif cmd == 'brake':
				self.brake(time)
			elif cmd == 'turn left':
				self.turn(time, True)
			elif cmd == 'turn right':
				self.turn(time, False)
		if self in self.track.hit:
			self.speed = 0
			self.control.kill()
		self.velocity = [np.cos(self.theta)*self.speed,
						np.sin(self.theta)*self.speed]
		self.location = [self.location[0]+self.velocity[0]*time,
						self.location[1]+self.velocity[1]*time]
	
	def reset(self, track):
		self.control.cmds = []
		self.track = track
		self.control.track = track
		self.speed = 0
	
	def get_reward(self, reward):
		self.control.get_reward(reward)
						
## Controllers: Human to start, then RL, then ENN

class Control():
	dead = False
	
	def __init__(self, car, track):
		self.cmds = []
		self.car = car
		self.track = track
	
	def add_cmd(self, cmd):
		if cmd not in ['accel', 'brake', 'turn left', 'turn right']:
			print "Command not recognized: %s" %cmd
			return
		if cmd in self.cmds: return
		if cmd[:4] == 'turn':
			if 'turn left' in self.cmds or 'turn right' in self.cmds: return
		self.cmds.append(cmd)
	
	def remove_cmd(self, cmd):
		if cmd not in self.cmds:
			# print "Command cannot be removed: %s" %cmd
			return
		self.cmds.remove(cmd)
	
	def update(self, events):
		pass
	
	def get_dist_to_wall(self, dtheta):
		D = 3
		x, y = self.car.location
		theta = self.car.theta + dtheta
		dist = 0
		while not self.track.crash_location([x,y]):
			x += np.cos(theta)*D
			y += np.sin(theta)*D
			dist += D
		return dist
	
	def get_reward(self, reward):
		pass

class HumanControl(Control):
	
	def __init__(self, car, track):
		Control.__init__(self, car, track)
	
	def update(self, events):
		for e in events:
			if e.type == KEYDOWN:
				task = self.add_cmd
			elif e.type == KEYUP:
				task = self.remove_cmd
			else: continue
			if e.key == K_UP:
				task('accel')
			elif e.key == K_DOWN:
				task('brake')
			elif e.key == K_LEFT:
				task('turn right')	  # backwards b/c y is down on screen
			elif e.key == K_RIGHT:
				task('turn left')
			else: continue

class RandomControl(Control):
	
	def __init__(self, car, track):
		Control.__init__(self, car, track)
	
	def update(self, events):
		r = np.random.random()
		if r < .9:
			task = self.add_cmd
			if r < .5:
				task('accel')
			elif r< .65:
				task('turn left')
			elif r< .8:
				task('turn right')
			else:
				task('brake')
		else:
			for cmd in self.cmds:
				self.remove_cmd(cmd)

class NNControl(Control):
	'''A generic NN controller for driving a car'''
	
	def __init__(self, car, track):
		Control.__init__(self, car, track)
		self.nn = net.Network()
		# track.species is what evolves. It's just an instance of net.Species
		self.track.species.add_network(self.nn)
		self.fitness = 0 # fitness is the record of how good this controller,
			# or specifically its neural network, is at doing the task
	
	def update(self, events):
		input = self.get_input()
		output = self.nn.run(input)
		self.cmds = []
		task = self.add_cmd
		# If the net gives a positive value, then this controller adds that to
		# what the car needs to do this time-step
		for i in range(len(output)):
			if output[i] <= 0: continue
			if i == 0: task('accel')
			elif i == 1: task('turn left')
			elif i == 2: task('turn right')
			elif i == 3: task('brake')
	
	def get_input(self):
		# The input to the neural net
		speed = self.car.speed/self.car.MAX_SPEED
		dist_left = self.get_dist_to_wall(2*np.pi/12.)	#30 degrees CCW
		dist_front = self.get_dist_to_wall(0)
		dist_right = self.get_dist_to_wall(-2*np.pi/12.)
		return [speed, dist_left/100., dist_front/100., dist_right/100.]
	
	def get_reward(self, reward):
		self.nn.fitness += reward   # reward is calculated by track to be
			# the distance in rads that the car moved in the last time-step.
			# Final fitness will be the total distance in rads moved.
	
	def get_net_img(self):
		# Just gives an image that helps visualize what the neural net is doing
		def get_col(val):
			return ((val<0)*min(1,-val)*255,(val>0)*min(1,val)*255,0)
		square = 30
		n = self.nn.nodes
		w = self.nn.weights
		img = pg.Surface((200,500))
		img.fill((0,0,0))
		nodes = []
		layer = self.get_input()
		for i in range(len(n)):
			nodes.append([])
			for j in range(n[i]):
				nodes[i].append(pg.Surface((square,square)))
				col = get_col(layer[j])
				nodes[i][j].fill(col)
			if i < len(n)-1:
				layer = np.dot(layer, w[i])
		for i in range(len(nodes)):
			for j in range(len(nodes[i])):
				img.blit(nodes[i][j],(i*(square+10),j*(square+10)))
		for i in range(len(w)):
			s1, s2 = np.shape(w[i])
			for j in range(s1):
				for k in range(s2):
					pg.draw.line(img, get_col(w[i][j,k]),
						((i+0.5)*(square+10)-5, (j+0.5)*(square+10)-5),
						((i+1.5)*(square+10)-5, (k+0.5)*(square+10)-5),
						3)
		return img
	
	

## Track class
''' The track will be the main class, getting initialized and initializing down
from there.'''

class Track(pg.sprite.Sprite):
	'''The Track class runs the race'''
	
	def __init__(self, radii, screen):
		pg.sprite.Sprite.__init__(self)
		'''
		num_cars: the number of cars on the track.
		controllers: a list with length num_cars of the controllers for each car
		colors: sim to controllers
		radius: a list of two polar functions defining the inner and outer radii
			of the track from the origin for theta from 0 to 2pi
		'''
		self.origin = []
		self.rad_in, self.rad_out = radii
		self.img = self.draw()
		self.font = pg.font.SysFont('arial', 20)
	
	def init_cars(self, num_cars, controllers, colors):
		if NNControl in controllers:
			self.species = net.Species(4, 4, .1, .05, .5)
		self.cars = []
		self.hit = []
		for i in range(num_cars):
			self.cars.append(Car(controllers[i], colors[i], self, i))
		self.species.init_population()
		self.species.update_population()
	
	def init_from_species(self, species, cars):
		self.species = species
		self.cars = cars
		self.hit = []
	
	def reset_cars(self):
		self.hit = []
		for c in self.cars:
			c.location[0], c.location[1] = self.starting_location()
			c.theta = -np.pi/2
			c.reset(self)
		
	def run(self, clock, screen, timescale):
		self.reset_cars()
		stopped_cars = 0
		total_time = 0
		self.clock = clock
		text = "Generation " + str(self.species.generation)
		text2 = "Max: " + str(round(self.species.max_fitness,3)) +\
			" Median: " + str(round(self.species.med_fitness,3))
		while len(self.hit) + stopped_cars < len(self.cars) and total_time < 60:
			events = pg.event.get()
			for e in events:
				if e.type == QUIT:
					sys.exit()
			time = 0.050
			if GRAPHICS:
				time = clock.tick() / 1000. * timescale # ms -> seconds
				time = min(.1*timescale, time)
			total_time += time
			if GRAPHICS: self.image = self.img.copy()
			stopped_cars = 0
			for c in self.cars:
				if c in self.hit:
					if GRAPHICS:
      bx = c.location[0] - (np.cos(c.theta) + np.sin(c.theta))*c.length
      ex = c.location[0] + (np.cos(c.theta) + np.sin(c.theta))*c.length
      by = c.location[1] - (-np.cos(c.theta) + np.sin(c.theta))*c.length
      ey = c.location[1] + (-np.cos(c.theta) + np.sin(c.theta))*c.length
      pg.draw.line(self.image, c.color, 
						(bx, by), (ex,ey), 2)
					continue
				if c.speed < 2 and total_time > 5: stopped_cars += 1
				old_loc = [c.location[0], c.location[1]]
				c.update(time, events)
				if self.crash(c):
					self.hit.append(c)
				new_loc = [c.location[0], c.location[1]]
				c.get_reward(self.calc_reward(old_loc, new_loc, time))
				if GRAPHICS: pg.draw.circle(self.image, c.color, 
					(int(c.location[0]), int(c.location[1])), 3, 0)
			if GRAPHICS:
				self.image.blit(self.font.render(text, 0, (255,255,255)), self.origin)
				self.image.blit(self.font.render(text2, 0, (255,255,255)), [self.origin[0]-50, self.origin[1]+50])
				screen.blit(self.image, (0,0))
				screen.blit(self.cars[0].control.get_net_img(), (600,100))
				pg.display.flip()
		print "All cars died!"
	
	def crash_location(self, location):
		theta = self.get_theta(location)
		rad_in, rad_out = self.get_track_radii(theta)
		radius = self.get_radius(location)
		if radius < rad_in or radius > rad_out:
			return True
		return False
	
	def crash(self, car):
		return self.crash_location(car.location)
	
	def calc_reward(self, old, new, time):
		t_old = self.get_theta(old)
		t_new = self.get_theta(new)
		if old[1] != new[1]:
			pass
		dtheta = -t_new + t_old
		if t_old < 0.1 and t_new > np.pi-0.1:
			dtheta += 2*np.pi
		return dtheta
		
	def get_theta(self, location):
		c_x, c_y = location
		o_x, o_y = self.origin
		theta = np.arctan2(c_y-o_y, c_x-o_x)
		return theta
	
	def get_track_radii(self, theta):
		return self.rad_in(theta), self.rad_out(theta)
	
	def get_radius(self, location):
		return np.hypot(location[0]-self.origin[0],
			location[1]-self.origin[1])
			
	def draw(self):
		max_up = 0
		max_right = 0
		max_left = 0
		max_down = 0
		for i in np.linspace(0, 2*np.pi, 1000):
			rad = self.rad_out(i)
			height = np.sin(i)*rad
			width = np.cos(i)*rad
			if i < np.pi:
				if height > max_up: max_up = height
			else:
				if -height > max_down: max_down = -height
			if i < np.pi/2 or i > 3*np.pi/2:
				if width > max_right: max_right = width
			else:
				if -width > max_left: max_left = -width
		self.origin = [max_left, max_down]
		surf = pg.Surface((max_right+max_left, max_up+max_down))
		surf.fill((0,0,0))
		for i in np.linspace(0, 2*np.pi, 1000):
			rad_in = self.rad_in(i)
			rad_out = self.rad_out(i)
			cos = np.cos(i)
			sin = np.sin(i)
			if GRAPHICS:
				pg.draw.circle(surf, (255,255,255),
					(int(self.origin[0] + rad_out*cos),
					int(self.origin[1]+rad_out*sin)),
					1, 0)
				pg.draw.circle(surf, (255,255,255),
					(int(self.origin[0]+rad_in*cos),
					int(self.origin[1]+rad_in*sin)),
					1, 0)
		return surf
	
	def starting_location(self):
		return [self.origin[0]+0.5*(self.rad_in(0)+self.rad_out(0)),
			self.origin[1]]
		
	
if __name__ == '__main__':
	NUM_RUNS = 100
	RUNS_PER_TRACK = 1
	time_scale = 1
	GRAPHICS = 1
	def r(theta):
		# # theta = theta%(2*np.pi)
		# # if (theta > np.pi/4 and theta < np.pi * 3/4) or \
		# #	 (theta > np.pi * 5/4 and theta < np.pi * 7/4):
		# #		 ret = np.abs(200/np.sin(theta))
		# # else: ret = np.abs(200/np.cos(theta))
		# # return ret
		# return 200-40*np.sin(2*theta)*np.cos(3*theta)+60*np.sin(6*theta)**2
		# return 200-40*np.sin(3*theta)
		return 35*(6+np.cos(2*theta)*np.sin(theta)-np.sin(2*theta)*np.cos(4*theta))
	def r2(theta):
		return r(theta)*1.15
	# def r3(theta):
	#	 return 200-40*np.sin(3*theta)
	# def r4(theta):
	#	 return r3(theta)*1.15
	# def r5(theta):
	#	 return 50
	# def r6(theta):
	#	 return 70
	pg.init()
	clock = pg.time.Clock()
	screen = pg.display.set_mode((800,600))
	NUM_CARS = 30
	controllers = [NNControl]*(NUM_CARS-1) + [HumanControl]
	colors = [(255,125,125)]+[(125,125,0)]*(NUM_CARS-2) + [(255,0,255)]
	track = Track([r, r2],screen)
	# track1 = Track([r3, r4], screen)
	# track2 = Track([r5, r6], screen)
	track.init_cars(NUM_CARS, controllers, colors)
	while True:
		for i in range(RUNS_PER_TRACK):
			track.run(clock, screen, time_scale)
			track.species.evolve()
		# track1.init_from_species(track.species, track.cars)
		# for i in range(RUNS_PER_TRACK):
		#	 track1.run(clock, screen, time_scale)
		#	 track1.species.evolve()
		# track2.init_from_species(track1.species, track1.cars)
		# for i in range(RUNS_PER_TRACK):
		#	 track2.run(clock, screen, time_scale)
		#	 track2.species.evolve()
		# track.init_from_species(track2.species, track2.cars)
			
			
			
			
			
			
			
			