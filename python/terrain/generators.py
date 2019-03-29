import noise

class SimplexGenerator2d:
	"""[summary]
	"""

	def generate(self, size: tuple, zOffset: float = 924.3, initFreq: float = 2048.0, octaves: int = 16, persistence: float = 0.51, lacunarity: float = 3.007):
		heightmap = {}

		for x in range(size[0]):
			for y in range(size[1]):
				heightmap[(x,y)] = noise.snoise3(x/initFreq, y/initFreq, zOffset, octaves=octaves, persistence=persistence, lacunarity=lacunarity)

		return heightmap

