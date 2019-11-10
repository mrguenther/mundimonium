from itertools import count


class HashByIndex:
	_MAX_INDEX = 0x7fffffffffffffff  # 2**(64-1)-1 == max signed 64-bit integer
	_hash_index = 0

	@staticmethod
	def _next_hash() -> int:
		HashByIndex._hash_index = (
				(HashByIndex._hash_index + 1) & HashByIndex._MAX_INDEX)
		return hash((HashByIndex._hash_index,))

	@staticmethod
	def hash_index() -> int:
		return HashByIndex._hash_index

	@staticmethod
	def skip_first(hash_count) -> None:
		HashByIndex._hash_index = hash_count & HashByIndex._MAX_INDEX

	def __new__(cls, *args, **kwargs):
		instance = super().__new__(cls, *args, **kwargs)
		instance._hash = HashByIndex._next_hash()
		return instance

	def __eq__(self, other: "HashByIndex") -> bool:
		return self is other

	def __hash__(self) -> int:
		return self._hash
