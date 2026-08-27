

class TimeController:
	"""
	Object for managing the passage of time.
	Something something Dr Who joke something.
	This will function similarly to a game loop,
		in that it stops object motion and whatnot from being tied only to tick rate
	Layer objects will somehow connect with this, and, as it ticks forward, simulate ongoing occurrences
		For highly-variable layers like cities, we'll need to do something to prevent them from slowing simulation to a crawl
		Perhaps have them drop aspects of simulation over time?
			e.g., only have layout grow and change for the first 150 years
			and have it simulate again based on new parameters for 10 years of every 100?
		Alternatively, do a more abstract simulation of characteristics (population, trade, etc)
		Then have it do more detailed simulation near generation end, or even on-demand.
	"""
