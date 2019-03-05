from ast import literal_eval

defaultParams = {
	# TODO
}
"""Supported operation parameters and default values, using strings as keys."""

def loadParamsFromDict(userParams: dict) -> dict:
	"""
	Merges user-specified parameters with the defaults for unspecified values.

	Args:
		userParams: a dict of parameters specified by the user.

	Returns:
		A dict containing the parameters from the file as well as defaults for any unspecified parameters.

	Raises:
		KeyError: A provided parameter was not found in the supported options.
	"""
	result = copy.copy(defaultParams)
	for key, val in userParams.items():
		if key in defaultParams:
			result[key] = val
		else:
			raise KeyError("Invalid parameter: {:s}".format(key))
	return result
	# Fun alternative implementation:
	# return {key: userParams[key] if key in userParams else val for key, val in defaultParams.items()}

def loadParamsFromFile(filePath: str) -> dict:
	"""
	Loads parameters from a user-specified file and merges them with the defaults for unspecified values.

	Args:
		filePath: A string containing the path to a configuration file.

	Returns:
		A dict containing the parameters from the file as well as defaults for any unspecified parameters.

	Raises:
		KeyError: A provided parameter was not found in the supported options.
	"""
	with open(filePath, "r") as f:
		userParams = literal_eval(f.read())
	return loadParamsFromDict(userParams)
