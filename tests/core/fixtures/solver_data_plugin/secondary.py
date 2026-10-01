from . import primary


class Options(primary.Options):
    pass


class Solver(primary.Solver, options_cls=Options):
    pass
