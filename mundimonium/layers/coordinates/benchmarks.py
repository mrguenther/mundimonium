import numpy as np
import timeit

from typing import List

def entropy(data: List[int], nbits: int = 64):
	data_len = len(data)
	return np.sum(np.fromiter(
		(-p * np.log2(p) if p > 0 else 0
			for p in (
				sum((x >> i) & 1
					for x in data
				) / data_len
				for i in range(nbits)
			)
		), dtype=float))

benchmarks = [
	# lambda: [hash(bytes(i)) for i in range(-5,5)],
	lambda: [hash(str(i)) for i in range(-5,5)],
	lambda: [hash(hex(i)) for i in range(-5,5)],
	lambda: [hash((i,)) for i in range(-5,5)],
	lambda: [hash(chr(i & 0xfffff)) for i in range(-5,5)],
	lambda: [hash(i + 0.1) for i in range(-5,5)],
	lambda: [hash(i) for i in range(-5,5)]
]

def main():
	setup = "from benchmarks import benchmarks"
	results = [None] * len(benchmarks)
	for i in range(len(benchmarks)):
		results[i] = timeit.timeit(f"benchmarks[{i}]()", setup=setup)
		print(f"{entropy(benchmarks[i]()):10.7f} {results[i]:10.7f}")

if __name__ == '__main__':
	main()
