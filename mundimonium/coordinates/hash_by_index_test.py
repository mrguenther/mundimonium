from mundimonium.coordinates.hash_by_index import HashByIndex


class Alpha(HashByIndex):
  pass


class Beta(HashByIndex):
  pass


def test_each_instance_gets_a_unique_hash():
  a, b, c = Alpha(), Alpha(), Alpha()
  assert hash(a) != hash(b) != hash(c)
  assert len({hash(a), hash(b), hash(c)}) == 3


def test_hash_is_stable_across_calls():
  a = Alpha()
  assert hash(a) == hash(a)


def test_hash_counter_is_shared_across_subclasses():
  # __new__ always advances the counter via `HashByIndex._next_hash()`
  # (the base class, not `cls`), so unrelated subclasses never collide.
  a = Alpha()
  b = Beta()
  assert hash(a) != hash(b)


def test_equality_is_identity_based():
  a1 = Alpha()
  a2 = Alpha()
  assert a1 == a1
  assert a1 != a2
  assert not (a1 == a2)


def test_instances_work_as_dict_and_set_keys():
  a1, a2 = Alpha(), Alpha()
  s = {a1, a2, a1}
  assert len(s) == 2
  d = {a1: "first", a2: "second"}
  assert d[a1] == "first"
  assert d[a2] == "second"


def test_hash_index_reflects_the_shared_counter():
  before = HashByIndex.hash_index()
  Alpha()
  Beta()
  after = HashByIndex.hash_index()
  assert after == before + 2


def test_skip_first_advances_the_counter():
  HashByIndex.skip_first(1000)
  assert HashByIndex.hash_index() == 1000
  a = Alpha()
  assert HashByIndex.hash_index() == 1001
  assert hash(a) == hash((1001,))


def test_max_index_wraps_around():
  HashByIndex.skip_first(HashByIndex._MAX_INDEX)
  a = Alpha()
  # (_MAX_INDEX + 1) & _MAX_INDEX == 0
  assert HashByIndex.hash_index() == 0
  assert hash(a) == hash((0,))
