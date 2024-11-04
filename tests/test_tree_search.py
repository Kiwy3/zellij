# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-10-03T22:41:21+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


import unittest
from unittest.mock import Mock
from zellij.strategies.tools import (
    Breadth_first_search,
    Depth_first_search,
    Best_first_search,
    Beam_search,
    Diverse_best_first_search,
    Cyclic_best_first_search,
    Epsilon_greedy_search,
)

tree_search_algorithm = [
    Breadth_first_search,
    Depth_first_search,
    Best_first_search,
    Beam_search,
    Diverse_best_first_search,
    Cyclic_best_first_search,
    Epsilon_greedy_search,
]


class TestTreeSearch(unittest.TestCase):
    def setUp(self):

        self.element = Mock()
        self.element.score = None
        self.element.father = "GOD"
        self.element.min_score = float("inf")
        self.element.level = 0

        self.child1 = Mock()
        self.child1.score = 5
        self.child1.father = self.element
        self.child1.min_score = 5
        self.child1.level = 1

        self.child2 = Mock()
        self.child2.score = 4
        self.child2.father = self.element
        self.child2.min_score = 4
        self.child2.level = 1

        self.child3 = Mock()
        self.child3.score = 3
        self.child3.father = self.element
        self.child3.min_score = 3
        self.child3.level = 1

    def test_tree_search_algorithms(self):

<<<<<<< Updated upstream
        for t in tree_search_algorithm:
=======
        for t in tree_search_algorithm.values():
>>>>>>> Stashed changes
            ts = t([self.element], 10)

            self.assertTrue(
                len(ts.open) == 1,
                f"Error on length of open list for object creation,\nFor {ts.__class__.__name__}",
            )

            go, h = ts.get_next()

            self.assertEqual(
                h,
                [self.element],
                f"Error when get next for initialisation,\nFor {ts.__class__.__name__}",
            )
            self.assertTrue(
                go,
                f"Error on continue for initialisation,\nFor {ts.__class__.__name__}",
            )
            self.assertTrue(
                len(ts.open) == 0,
                f"Error on length of open list for initialisation,\nFor {ts.__class__.__name__}",
            )

            ts.add(self.child1)
            ts.add(self.child2)
            ts.add(self.child3)

            self.assertTrue(
                len(ts.next_frontier) == 3,
                f"Error on length of open list when adding children,\nFor {ts.__class__.__name__}",
            )

            go, h = ts.get_next()
            self.assertIn(
                h,
                [[self.child1], [self.child2], [self.child3]],
                f"Error when get next for children,\nFor {ts.__class__.__name__}",
            )
            self.assertTrue(
                go,
                f"Error on continue for children,\nFor {ts.__class__.__name__}",
            )

            go, h = ts.get_next()
            self.assertIn(h, [[self.child1], [self.child2], [self.child3]])
            self.assertTrue(
                go,
                f"Error on continue for children,\nFor {ts.__class__.__name__}",
            )

            go, h = ts.get_next()
            self.assertIn(h, [[self.child1], [self.child2], [self.child3]])
            self.assertTrue(
                go,
                f"Error on continue for children,\nFor {ts.__class__.__name__}",
            )

            go, h = ts.get_next()
            self.assertEqual(h, -1)
            self.assertFalse(
                go,
                f"Error on ending for children,\nFor {ts.__class__.__name__}",
            )
<<<<<<< Updated upstream


if __name__ == "__main__":
    unittest.main()
=======
>>>>>>> Stashed changes
