import philote_mdo.general as pmdo

class Paraboloid(pmdo.ExplicitDiscipline):
    """
    Basic two-dimensional paraboloid example (explicit) discipline.
    """

    def setup(self):
        self.add_input("x", shape=(1,), units="m")
        self.add_input("y", shape=(1,), units="m")

        self.add_output("f_xy", shape=(1,), units="m**2")

    # def setup_partials(self):
    #     self.declare_partials("f_xy", "x")
    #     self.declare_partials("f_xy", "y")

    def compute(self, inputs, outputs):
        x = inputs["x"]
        y = inputs["y"]

        outputs["f_xy"] = (x - 3.0) ** 2 + x * y + (y + 4.0) ** 2 - 3.0

    # def compute_partials(self, inputs, partials):
    #     x = inputs["x"]
    #     y = inputs["y"]

    #     partials["f_xy", "x"] = 2.0 * x - 6.0 + y
    #     partials["f_xy", "y"] = 2.0 * y + 8.0 + x