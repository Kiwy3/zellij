# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-10-03T22:41:19+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


import unittest
from zellij.core import (
    MixedSearchspace,
    ContinuousSearchspace,
    DiscreteSearchspace,
    ArrayVar,
    FloatVar,
    IntVar,
    CatVar,
    Loss,
    MockModel,
    Variable,
)


class TestMixedSearchspace(unittest.TestCase):
    def setUp(self):

        self.values = ArrayVar(
            IntVar("int_1", 0, 8),
            IntVar("int_2", 4, 45),
            FloatVar("float_1", 2, 12),
            CatVar("cat_1", ["Hello", 87, 2.56]),
        )

        self.loss = Loss(save=False, verbose=False)(MockModel())
        self.sp = MixedSearchspace(self.values, self.loss)

    def test_creation(self):
        """test_creation

        Test create of a MixedSearchspace

        """

        with self.assertRaises(AssertionError):
            sp = MixedSearchspace(self.values, self.loss)
            self.fail("Assertion not raised for wrong loss creation")

        with self.assertRaises(AssertionError):
            sp = MixedSearchspace(IntVar("int_1", 0, 8), self.loss)
            self.fail("Assertion not raised for wrong values creation")

        sp = MixedSearchspace(self.values, self.loss)

        self.assertIsInstance(sp, MixedSearchspace)
        self.assertTrue(len(self.sp) == 4)

    def test_random_attribute(self):
        """Test random_attribute method"""

        # One
        self.assertIsInstance(self.sp.random_attribute()[0], Variable)

        # Multiple
        r_a = self.sp.random_attribute(size=20)
        for elem in r_a:
            self.assertIsInstance(elem, Variable)

        # Excluding type
        r_a = self.sp.random_attribute(size=20, exclude=IntVar)
        for elem in r_a:
            self.assertFalse(
                isinstance(elem, IntVar),
                "Error in excluding a type in random_attribute",
            )

        r_a = self.sp.random_attribute(size=20, exclude=[IntVar, CatVar])
        for elem in r_a:
            self.assertIsInstance(
                elem,
                FloatVar,
                "Error in excluding list of types in random_attribute",
            )

        # Excluding Variable
        r_a = self.sp.random_attribute(size=20, exclude=self.values[2])
        for elem in r_a:
            self.assertFalse(
                isinstance(elem, FloatVar),
                "Error in excluding a Variable in random_attribute",
            )

        r_a = self.sp.random_attribute(size=20, exclude=self.values[0:2])
        for elem in r_a:
            self.assertFalse(
                isinstance(elem, IntVar),
                "Error in excluding a list of Variable in random_attribute",
            )

        # Excluding index
        r_a = self.sp.random_attribute(size=20, exclude=2)
        for elem in r_a:
            self.assertFalse(
                isinstance(elem, FloatVar),
                "Error in excluding an index in random_attribute",
            )

        r_a = self.sp.random_attribute(size=20, exclude=[0, 1])
        for elem in r_a:
            self.assertFalse(
                isinstance(elem, IntVar),
                "Error in excluding a list of indexes in random_attribute",
            )

    def test_random_point(self):
        self.assertTrue(
            len(self.sp.random_point(10)) == 10, "Wrong output size"
        )

    def test_subspace(self):
        lo = [0, 30, 10, "Hello"]
        up = [5, 40, 12, 87]
        new = self.sp.subspace(lo, up)
        self.assertIsInstance(new.values, ArrayVar)
        self.assertTrue(len(new) == 4)

        self.assertIsInstance(new.values[0], IntVar)
        self.assertTrue(new.values[0].low_bound == 0)
        self.assertTrue(new.values[0].up_bound == 6)
        self.assertIsInstance(new.values[1], IntVar)
        self.assertTrue(new.values[1].low_bound == 30)
        self.assertTrue(new.values[1].up_bound == 41)
        self.assertIsInstance(new.values[2], FloatVar)
        self.assertTrue(new.values[2].low_bound == 10)
        self.assertTrue(new.values[2].up_bound == 12)
        self.assertIsInstance(new.values[3], CatVar)
        self.assertTrue(new.values[3].features[0] == "Hello")
        self.assertTrue(new.values[3].features[1] == 87)
        self.assertTrue(len(new.values[3].features) == 2)


class TestContinuousSearchspace(unittest.TestCase):
    def setUp(self):

        self.values = ArrayVar(
            FloatVar("float_1", 0, 5),
            FloatVar("float_2", 10, 15),
        )

        self.loss = Loss(save=False, verbose=False)(MockModel())
        self.sp = ContinuousSearchspace(self.values, self.loss)

    def test_creation(self):
        """Test creation of a ContinuousSearchspace"""

        with self.assertRaises(AssertionError):
            sp = ContinuousSearchspace(self.values, self.loss)
            self.fail("Assertion not raised for wrong loss creation")

        with self.assertRaises(AssertionError):
            sp = ContinuousSearchspace(FloatVar("float_1", 0, 8), self.loss)
            self.fail("Assertion not raised for wrong values creation")

        with self.assertRaises(AssertionError):
            sp = ContinuousSearchspace(
                ArrayVar(FloatVar("float_1", 0, 8), IntVar("int_1", 0, 8)),
                self.loss,
            )
            self.fail("Assertion not raised for not FloatVar creation")

        sp = ContinuousSearchspace(self.values, self.loss)

        self.assertIsInstance(sp, ContinuousSearchspace)
        self.assertTrue(len(self.sp) == 2)

    def test_random_attribute(self):
        """Test random_attribute method"""

        # One
        self.assertIsInstance(self.sp.random_attribute()[0], FloatVar)

        # Multiple
        r_a = self.sp.random_attribute(size=20)
        for elem in r_a:
            self.assertIsInstance(elem, FloatVar)

    def test_random_point(self):
        self.assertTrue(
            len(self.sp.random_point(10)) == 10, "Wrong output size"
        )

    def test_subspace(self):
        lo = [3, 12]
        up = [4, 13]
        new = self.sp.subspace(lo, up)
        self.assertIsInstance(new.values, ArrayVar)
        self.assertTrue(len(new) == 2)

        self.assertIsInstance(new.values[0], FloatVar)
        self.assertTrue(new.values[0].low_bound == 3)
        self.assertTrue(new.values[0].up_bound == 4)
        self.assertIsInstance(new.values[1], FloatVar)
        self.assertTrue(new.values[1].low_bound == 12)
        self.assertTrue(new.values[1].up_bound == 13)


class TestDiscreteSearchspace(unittest.TestCase):
    def setUp(self):

        self.values = ArrayVar(
            IntVar("int_1", 0, 5),
            IntVar("int2", 10, 15),
        )

        self.loss = Loss(save=False, verbose=False)(MockModel())
        self.sp = DiscreteSearchspace(self.values, self.loss)

    def test_creation(self):
        """Test creation of a DiscreteSearchspace"""

        with self.assertRaises(AssertionError):
            sp = DiscreteSearchspace(self.values, self.loss)
            self.fail("Assertion not raised for wrong loss creation")

        with self.assertRaises(AssertionError):
            sp = Searchspace(label, type, [["l"], [-5, 5], [-5, 5], [-5, -5]], neighborhood)
            self.fail("Assertion not raised for wrong values creation")

        with self.assertRaises(AssertionError):
            sp = Searchspace(label, type, values, [0.5, 1, 0.5, 5])
            self.fail("Assertion not raised for wrong neighborhood creation")

        with self.assertRaises(AssertionError):
            sp = Searchspace(label, type, values, 5)
            self.fail("Assertion not raised for wrong neighborhood value creation")

        sp = DiscreteSearchspace(self.values, self.loss)

        self.assertEqual(sp.neighborhood, [1.0, 1, -1, -1], "Wrong neighborhood creation")

    def setUp(self):
        labels = ["a", "b", "c", "d"]
        types = ["R", "D", "C", "K"]
        values = [[-5, 5], [-5, 5], ["v1", "v2", "v3"], 2]
        neighborhood = [0.5, 1, -1, -1]

        self.sp = Searchspace(labels, types, values, neighborhood)

    def test_random_attribute(self):
        self.assertIn(self.sp.random_attribute(), self.sp.label, "Wrong Random attribute with size=1, replace=True, exclude=None")

        self.assertIn(self.sp.random_attribute(exclude="a"), self.sp.label[1:], "Wrong Random attribute with exclusion")

        with self.assertRaises(ValueError):
            self.sp.random_attribute(size=10, replace=False, exclude="a")

    def test_random_value(self):
        self.assertTrue(
            isinstance(self.sp.random_value("a")[0], float) and (-5 <= self.sp.random_value("a")[0] <= 5), "Wrong Random real value with size=1, replace=True, exclude=None"
        )

        self.assertTrue(
            isinstance(self.sp.random_value("b")[0], int) and (-5 <= self.sp.random_value("b")[0] <= 5), "Wrong Random int value with size=1, replace=True, exclude=None"
        )

        self.assertIn(self.sp.random_value("c")[0], self.sp.values[2], "Wrong Random categorical value with size=1, replace=True, exclude=None")

        self.assertIn(self.sp.random_value("c", exclude="v1")[0], self.sp.values[2][1:], "Wrong Random attribute with exclusion")

        with self.assertRaises(ValueError):
            self.sp.random_value("c", size=10, replace=False, exclude="v1")

    def test_get_real_neighbor(self):
        neighbor = self.sp._get_real_neighbor(4, 0)
        self.assertTrue(isinstance(neighbor, float) and -5 <= neighbor <= 5 and neighbor != 4, "Wrong real neighbor generation")

    def test_get_discrete_neighbor(self):
        neighbor = self.sp._get_discrete_neighbor(4, 1)
        self.assertTrue(isinstance(neighbor, int) and -5 <= neighbor <= 5 and neighbor != 4, "Wrong discrete neighbor generation")

    def test_get_categorical_neighbor(self):
        neighbor = self.sp._get_categorical_neighbor("v2", 2)
        self.assertTrue(neighbor in ["v1", "v3"], "Wrong categorical neighbor generation")

    def test_get_neighbor(self):

        neighbor = self.sp.get_neighbor([4, 4, "v2", 2])[0]

        self.assertTrue(
            (isinstance(neighbor[0], float) or isinstance(neighbor[0], int))
            and isinstance(neighbor[1], int)
            and neighbor[2] in ["v1", "v2", "v3"]
            and neighbor[3] == 2
            and -5 <= neighbor[0] <= 5
            and -5 <= neighbor[1] <= 5
            and (neighbor[0] != 4 or neighbor[1] != 4 or neighbor[2] != "v2"),
            "Wrong neighbor generation",
        )

        neighbor = self.sp.get_neighbor([-5, -5, "v1", 2])[0]

        self.assertTrue(
            (isinstance(neighbor[0], float) or isinstance(neighbor[0], int))
            and isinstance(neighbor[1], int)
            and neighbor[2] in ["v1", "v2", "v3"]
            and neighbor[3] == 2
            and -5 <= neighbor[0] <= 5
            and -5 <= neighbor[1] <= 5
            and (neighbor[0] != -5 or neighbor[1] != -5 or neighbor[2] != "v1"),
            "Wrong neighbor generation for lower bounds",
        )

        neighbor = self.sp.get_neighbor([5, 5, "v3", 2])[0]
        self.assertTrue(
            (isinstance(neighbor[0], float) or isinstance(neighbor[0], int))
            and isinstance(neighbor[1], int)
            and neighbor[2] in ["v1", "v2", "v3"]
            and neighbor[3] == 2
            and -5 <= neighbor[0] <= 5
            and -5 <= neighbor[1] <= 5
            and (neighbor[0] != 5 or neighbor[1] != 5 or neighbor[2] != "v3"),
            "Wrong neighbor generation for upper bounds",
        )

    def test_random_point(self):
        self.assertTrue(
            (isinstance(point[0], float) or isinstance(point[0], int))
            and isinstance(point[1], int)
            and point[2] in ["v1", "v2", "v3"]
            and point[3] == 2
            and -5 <= point[0] <= 5
            and -5 <= point[1] <= 5,
            "Wrong random point generation",
        )

    def test_convert(self):

        converted = self.sp.convert_to_continuous([[4, 4, "v2", 2]])[0]
        reconverted = self.sp.convert_to_continuous([converted], reverse=True)[0]

        self.assertEqual(converted, [0.9, 0.9, 0.3333333333333333, 1], "Wrong convertion to continuous")
        self.assertEqual(reconverted, [4, 4, "v2", 2], "Wrong convertion to mixed")

        converted = self.sp.convert_to_continuous([[5, 5, "v3", 2]])[0]
        reconverted = self.sp.convert_to_continuous([converted], reverse=True)[0]

        self.assertEqual(converted, [1, 1, 0.6666666666666666, 1], "Wrong convertion to continuous of the upper bounds")
        self.assertEqual(reconverted, [5, 5, "v3", 2], "Wrong convertion to mixed of the upper bounds")

        converted = self.sp.convert_to_continuous([[-5, -5, "v1", 2]])[0]
        reconverted = self.sp.convert_to_continuous([converted], reverse=True)[0]

        self.assertEqual(converted, [0, 0, 0, 1], "Wrong convertion to continuous of the lower bounds")
        self.assertEqual(reconverted, [-5, -5, "v1", 2], "Wrong convertion to mixed of the lower bounds")

    def test_general_convert(self):
        spc = self.sp.general_convert()

        self.assertEqual(spc.label, self.sp.label, "Wrong general convertion for labels")
        self.assertTrue(all(x == "R" for x in spc.types), "Wrong general convertion for types")
        self.assertTrue(all(x == [0, 1] for x in spc.values), "Wrong general convertion for values")

        self.assertEqual(spc.neighborhood, [0.05, 0.1, 1, 1], "Wrong general convertion of the neighborhood")

    def test_subspace(self):
        lo = [-2, 2, "v2", 1]
        up = [2, 2, "v3", 3]
        spc = self.sp.subspace(lo, up)

        self.assertEqual(spc.values, [[-2.0, 2.0], 2, ["v2", "v3"], 2], "Wrong subspacing for values")
        self.assertEqual(spc.types, ["R", "K", "C", "K"], "Wrong subspacing for types")
        self.assertEqual(spc.neighborhood, [0.2, -1, -1, -1], "Wrong subspacing for neighborhood")

        lo = [-5, -5, "v3", 1]
        up = [-4.99, -4, "v1", 1]
        spc = self.sp.subspace(lo, up)

        self.assertEqual(spc.values, [[-5.0, -4.99], [-5, -4], ["v1", "v2", "v3"], 2], "Wrong subspacing for lower bounds - values")
        self.assertEqual(spc.types, ["R", "D", "C", "K"], "Wrong subspacing for lower bounds - types")
        self.assertEqual(spc.neighborhood, [0.0004999999999999894, 1, -1, -1], "Wrong subspacing for lower bounds - neighborhood")

        lo = [4.99, 4, "v1", 1]
        up = [5, 5, "v1", 1]
        spc = self.sp.subspace(lo, up)

        self.assertEqual(spc.values, [[4.99, 5.0], [4, 5], "v1", 2], "Wrong subspacing for upper bounds")
        self.assertEqual(spc.types, ["R", "D", "K", "K"], "Wrong subspacing for upper bounds - types")
        self.assertEqual(spc.neighborhood, [0.0004999999999999894, 1, -1, -1], "Wrong subspacing for upper bounds - neighborhood")


if __name__ == "__main__":
    unittest.main()
