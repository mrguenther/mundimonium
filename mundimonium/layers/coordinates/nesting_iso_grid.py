from mundimonium.layers.coordinates.isometric \
    import IsometricGrid, IsometricPoint


class IsoGridSector:
  def __init__(self, parent_grid: IsoGridSectorTable):
    self._parent_grid = parent_grid


class IsoGridSectorTable(IsometricGrid):
  def __init__(self):
    super().__init__()
    # TODO: Replace this with a list of size resolution**2, initializing
    # each element as None: `self._sectors = [None] * resolution**2`
    self._sectors = {
        self._hash_sector_indices(b, s, d): None
        for b in range(self.resolution)
        for s in range(self.resolution - b)
        # for d in range(max(0, self.resolution - b - s - 2),
        #     self.resolution - b - s)}
        # for d in range(self.resolution - b - s - 1,
        #     min(self.resolution, self.resolution - b - s + 1))}
        for d in range(self.resolution - b - s)}
    for key in self._sectors.keys():
      print(f"{key:0>5d}")

  def __getitem__(self, near_point: IsometricPoint) -> IsoGridSector | None:
    return self._sectors[self._hash_point(near_point)]

  def __setitem__(
      self, near_point: IsometricPoint, sector: IsoGridSector | None) -> None:
    self._sectors[self._hash_point(near_point)] = sector

  def get_or_insert(
      self, near_point: IsometricPoint, sector: IsoGridSector
  ) -> IsoGridSector:
    key = self._hash_point(near_point)
    if self._sectors[key] is None:
      self._sectors[key] = sector

    return self._sectors[key]

  def get_or_emplace(
      self,
      near_point: IsometricPoint,
      sector_type: type[IsoGridSector],
      *args,
      **kwargs) -> IsoGridSector:
    key = self._hash_point(near_point)
    if self._sectors[key] is None:
      self._sectors[key] = sector_type(self, *args, **kwargs)

    return self._sectors[key]

  def _hash_sector_indices(
      self, b_index: int, s_index: int, d_index: int) -> int:
    # TODO: Optimize this to generate a list index instead of an arbitrary
    # dict key. The list will have size resolution**2, or the sum of 2n-1
    # for n in the range 1..resolution.

    resolution = self.resolution

    if b_index < 0 or s_index < 0 or d_index < 0 or \
        b_index >= resolution or s_index >= resolution or \
        d_index >= resolution:
      raise KeyError(
          "The provided IsometricPoint lies outside of its grid.")

    # return hash((b_index, s_index, d_index))
    return b_index * 10000 + s_index * 100 + d_index

  def _hash_point(self, point: IsometricPoint) -> int:
    if point.grid is not self:
      raise KeyError("The provided IsometricPoint is not on this grid.")

    scale_factor = self.resolution / self.altitude

    return self._hash_sector_indices(
        int(point.b * scale_factor),
        int(point.s * scale_factor),
        int(point.d * scale_factor),
    )

  @property
  def resolution(self):
    raise NotImplementedError()


class NestingIsoGrid(IsoGridSectorTable):
  def __init__(
      self, resolution: int | None,
      parent_sector: IsoGridSector | None = None):
    self._sectors = None
    self._resolution = resolution

    if resolution is None:
      IsometricGrid.__init__(self)
    else:
      IsoGridSectorTable.__init__(self)

    self._parent_sector = parent_sector

  def get_or_emplace(
      self,
      near_point: IsometricPoint,
      sector_type: type[IsoGridSector] | None = None,
      *args,
      **kwargs) -> IsoGridSector:
    return super().get_or_emplace(
        self, near_point,
        self.default_sector_type if sector_type is None else sector_type,
        *args, **kwargs)

  @property
  def resolution(self):
    return self._resolution

  @property
  def altitude(self):
    return 1

  @property
  def default_sector_type(self):
    return IsoGridSector


def main():
  grid = NestingIsoGrid(4)
  p1 = IsometricPoint(grid, 0.2, 0.2)
  p2 = IsometricPoint(grid, 0.5, 0.5)

  print(grid._sectors)
  print()
  print(grid.get_or_emplace(p1))
  print()
  print(grid._sectors)
  print()
  print(grid.get_or_emplace(p2))
  print()
  print(grid._sectors)


if __name__ == "__main__":
  main()
