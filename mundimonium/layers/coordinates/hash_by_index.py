from itertools import count


class HashByIndex:
	_MAX_INDEX = 0x7fffffffffffffff  # 2**(64-1)-1 == max signed 64-bit integer
	_hash_index = 0

	@classmethod
	def _next_hash(cls) -> int:
		cls._hash_index = (
				(cls._hash_index + 1) & cls._MAX_INDEX)
		return hash((cls._hash_index,))

	@classmethod
	def hash_index(cls) -> int:
		return cls._hash_index

	@classmethod
	def skip_first(cls, hash_count: int) -> None:
		cls._hash_index = hash_count & cls._MAX_INDEX

	def __new__(cls, *args, **kwargs):
		instance = super().__new__(cls)
		instance._hash = HashByIndex._next_hash()
		return instance

	def __eq__(self, other: "HashByIndex") -> bool:
		return self is other

	def __hash__(self) -> int:
		return self._hash
