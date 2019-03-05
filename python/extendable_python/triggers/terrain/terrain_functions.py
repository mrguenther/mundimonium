from .utils.coordinate_grid import CartesianPoint
import noise

class CityTerrain:
	def __init__(self, size, seed, args):
		self.size = size
		self.seed = seed
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

		return (heightmap)