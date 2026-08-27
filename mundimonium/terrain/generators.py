import noise


class SimplexGenerator2d:
	"""[summary]
	"""

	def generate(
		self,
		size: tuple,
		zOffset: float = 924.3,
		initFreq: float = 2048.0,
		octaves: int = 16,
		persistence: float = 0.51,
		lacunarity: float = 3.007
	) -> dict:
		"""[summary]

		Args:
			size (tuple): [description]
			zOffset (float, optional): Defaults to 924.3. [description]
			initFreq (float, optional): Defaults to 2048.0. [description]
			octaves (int, optional): Defaults to 16. [description]
			persistence (float, optional): Defaults to 0.51. [description]
			lacunarity (float, optional): Defaults to 3.007. [description]

		Returns:
			dict: [description]
		"""

		heightDict = {}

		for x in range(size[0]):
			for y in range(size[1]):
				heightDict[(x, y)] = noise.snoise3(	x / initFreq,
													y / initFreq,
													zOffset,
													octaves=octaves,
													persistence=persistence,
													lacunarity=lacunarity)

		return heightDict

