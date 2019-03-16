from .utils.coordinate_grid import CartesianPoint
import noise

# TEMP
from PIL import Image

class SimplexTerrain:
	def __init__(self, size, seed, caller, args):
		self.size = size
		self.seed = seed
		self.caller = caller
		self.args = args
		self.heightmap = self.generateHeightmap()
		
	def generateHeightmap(self):
		args = self.args

		# Alas, poor DRY! I knew it, Horatio...
		initFreq = 2048.0; octaves = 16; persistence = 0.51; lacunarity = 3.007
		if 'initFreq' in args: initFreq = args['initFreq']
		if 'octaves' in args: octaves = args['octaves']
		if 'persistence' in args: persistence = args['persistence']
		if 'lacunarity' in args: lacunarity = args['lacunarity']

		heightmap = {}

		for x in range(0,self.size[0]):
			for y in range(0,self.size[1]):
				simplexVal = noise.snoise3(x/initFreq, y/initFreq, self.seed, octaves=octaves, persistence=persistence, lacunarity=lacunarity)
				heightmap[(x,y)] = CartesianPoint((x,y,simplexVal))

		return(heightmap)

	def getGrade(self, x, y):
		height = self.heightmap[(x,y)].coords[2]
		xdiff = -1
		if x+1 < self.size[0]: xdiff = 1
		dx = self.heightmap[(x+xdiff,y)].coords[2]-height
		ydiff = -1
		if y+1 < self.size[1]: ydiff = 1
		dy = self.heightmap[(x,y+ydiff)].coords[2]-height
		grade = (dx**2 + dy**2)**0.5
		return(grade)


	def tempRender(self, layerManager):
		display = Image.new('RGB', self.size)
		print('Readying image...')
		# Print a pretty picture
		for x in range(0,self.size[0]):
			for y in range(0,self.size[1]):
				height = self.heightmap[(x,y)].coords[2]
				rgbVal = int(height*127+128)
				rgb = (0,rgbVal,255-rgbVal)
				display.putpixel((x,y),rgb)
		display.show()
