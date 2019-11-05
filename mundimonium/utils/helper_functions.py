

def argc(*args) -> int:
	return sum([arg is not None for arg in args])
