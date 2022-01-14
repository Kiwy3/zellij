from zellij.core.metaheuristic import Metaheuristic
from zellij.strategies.utils.cooling import Cooling
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd


class Simulated_annealing(Metaheuristic):

    """Simulated_annealing

    Simulated_annealing (SA) is an exploitation strategy allowing to do hill climbing by starting from\
    an initial solution and iteratively moving to next one,\
     better than the previous one, or slightly worse to escape from local optima.

    It uses a cooling schedule which partially drives the acceptance probability. This is the probability\
    to accept a worse solution according to the temperature, the best solution found so far and the actual solution.

    Attributes
    ----------

    max_iter : int
        Maximum iterations of the inner loop. Determine how long the algorithm should sampled neighbors of a solution,\
        before decreasing the temperature.

    T_0 : float
        Initial temperature of the cooling schedule.\
         Higher temperature leads to higher acceptance of a worse solution. (more exploration)

    T_end : float
        Temperature threshold. When reached the temperature is violently increased proportionately to\
        T_0. It allows to periodically easily escape from local optima.

    n_peaks : int
        Maximum number of crossed threshold according to T_end. The temperature will be increased\
        <n_peaks> times.

    red_rate : float
        Reduction rate of the initial temperature.
        Each time the threshold is crossed, temperature is increased by <red_rate>*<T_0>.


    Methods
    -------

    run(self, n_process=1)
        Runs Genetic_algorithm

    decrease_temperature(self, T)
        Linear cooling schedule. Deprecated. Must be replaced by a cooling schedule object.

    number_of_iterations(self)
        Determine the number of iterations with the actual parameters.

    show(filename=None)
        Plots results

    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is
    LossFunc : Describes what a loss function is in Zellij
    Searchspace : Describes what a loss function is in Zellij
    """

    # Initialize simulated annealing
    def __init__(self, loss_func, search_space, f_calls, cooling, max_iter, verbose=True):

        """__init__(self,loss_func, search_space, f_calls, cooling, max_iter, verbose=True)

        Initialize Genetic_algorithm class

        Parameters
        ----------
        loss_func : Loss
            Loss function to optimize. must be of type f(x)=y

        search_space : Searchspace
            Search space object containing bounds of the search space.

        f_calls : int
            Maximum number of loss_func calls

        cooling : Cooling
            Cooling schedule used to determine the probability of acceptance.

        max_iter : int
            Maximum iterations of the inner loop. Determine how long the algorithm should sampled neighbors of a solution,\
            before decreasing the temperature.

        save : boolean, optional
            if True save results into a file

        verbose : boolean, default=True
            Algorithm verbosity


        """

        super().__init__(loss_func, search_space, f_calls, verbose)

        # Max iteration after each temperature decrease
        self.max_iter = max_iter

        # Cooling schedule
        self.cooling = cooling

        self.n_scores = []
        self.n_best = []

        self.record_temp = [self.cooling.cool()]
        self.record_proba = [0]

        self.file_created = False

    # RUN SA
    def run(self, X0, Y0, n_process=1):

        """run(self,shift=1, n_process=1)

        Parameters
        ----------
        X0 : list[float]
            Initial solution
        Y0 : {int, float}
            Score of the initial solution
        shift : int, default=1
            Determine the starting point of the chaotic map.
        n_process : int, default=1
            Determine the number of best solution found to return.

        Returns
        -------
        best_sol : list[float]
            Returns a list of the <n_process> best found points to the continuous format

        best_scores : list[float]
            Returns a list of the <n_process> best found scores associated to best_sol

        """
        self.loss_func.file_created = False

        self.X0, self.Y0 = X0, Y0

        self.n_best.append(X0)
        self.n_scores.append(Y0)

        print("Simulated Annealing starting")
        print(X0, Y0)

        # Determine the number of iteration according to the function parameters
        print("Determining number of iterations")
        nb_iteration = self.cooling.iterations() * self.max_iter
        print("Number of iterations: " + str(nb_iteration))

        # Initialize variable for simulated annealing
        # Best solution so far
        X = X0[:]

        # Best solution in the neighborhood
        X_p = X[:]

        # Current solution
        Y = X[:]

        # Initialize score
        cout_X = Y0
        cout_X_p = Y0

        T_actu = self.cooling.Tcurrent

        # Simulated annealing starting
        while T_actu and self.loss_func.calls < self.f_calls:
            iteration = 0
            while iteration < self.max_iter and self.loss_func.calls < self.f_calls:

                neighbors = self.search_space.get_neighbor(X, size=n_process)
                loss_values = self.loss_func(neighbors, temperature=self.record_temp[-1], probability=self.record_proba[-1])

                index_min = np.argmin(loss_values)
                Y = neighbors[index_min][:]
                cout_Y = loss_values[index_min]

                # Compute previous cost minus new cost
                delta = cout_Y - cout_X

                out = "\nNew model score: " + str(cout_Y) + "\nOld model score: " + str(cout_X) + "\nBest model score: " + str(cout_X_p)

                # If a better model is found do...
                if delta < 0:
                    X = Y[:]
                    cout_X = cout_Y
                    if cout_Y < cout_X_p:

                        # Print if best model is found
                        out += "\nBest model found: /!\ Yes /!\ "

                        X_p = X[:]
                        cout_X_p = cout_X

                    else:
                        out += "\nBest model found: No "

                    self.record_proba.append(0)

                else:
                    out += "\nBest model found: No "
                    p = np.random.uniform(0, 1)
                    emdst = np.exp(-delta / T_actu)

                    self.record_proba.append(emdst)

                    out += "\nEscaping :  p<exp(-df/T) -->" + str(p) + "<" + str(emdst)
                    if p <= emdst:
                        X = Y[:]
                        cout_X = cout_Y
                    else:
                        Y = X[:]

                iteration += 1

                out += "\nITERATION:" + str(self.loss_func.calls) + "/" + str(nb_iteration)
                out += "\n==============================================================\n"

                self.record_temp.append(T_actu)

                if self.verbose:
                    print(out)

                # Save file
                if self.loss_func.save:
                    if not self.file_created:
                        self.sa_save = os.path.join(self.loss_func.outputs_path, "sa_best.csv")
                        with open(self.sa_save, "w") as f:
                            f.write(",".join(e for e in self.search_space.labels) + ",loss,temperature,probability\n")
                            f.write(",".join(str(e) for e in self.X0) + "," + str(self.Y0) + "," + str(self.cooling.T0) + ",0\n")
                            self.file_created = True

                    with open(self.sa_save, "a") as f:
                        f.write(",".join(str(e) for e in X) + "," + str(cout_X) + "," + str(self.record_temp[-1]) + "," + str(self.record_proba[-1]) + "\n")

                self.n_scores.append(cout_X)
                self.n_best.append(X)

            T_actu = self.cooling.cool()

        # print the best solution from the simulated annealing
        print("Best parameters: " + str(X_p) + " score: " + str(cout_X_p))
        print("Simulated Annealing ending")

        # return self.n_best,self.n_scores

    def show(self, filepath="", save=False):

        """show(self, filename=None)

        Plots solutions and scores evaluated during the optimization

        Parameters
        ----------
        filepath : str, default=""
            If a filepath is given, the method will read files insidethe folder and will try to plot contents.

        save : boolean, default=False
            Save figures
        """

        data_all, all_scores = super().show(filepath, save)

        if filepath:

            path_sa = os.path.join(filepath, "outputs", "sa_best.csv")
            data_sa = pd.read_table(path_sa, sep=",", decimal=".")
            sa_scores = data_sa["loss"].to_numpy()

            temperature = data_sa["temperature"].to_numpy()
            probability = data_sa["probability"].to_numpy()

        else:
            data_sa = self.n_best
            sa_scores = np.array(self.n_scores)

            temperature = np.array(self.record_temp)
            probability = np.array(self.record_proba)

        argmin = np.argmin(sa_scores)
        f, (l1, l2) = plt.subplots(2, 2, figsize=(19.2, 14.4))

        ax1, ax2 = l1
        ax3, ax4 = l2

        ax1.plot(list(range(len(sa_scores))), sa_scores, "-")
        argmin = np.argmin(sa_scores)
        ax1.scatter(argmin, sa_scores[argmin], color="red", label="Best score: " + str(sa_scores[argmin]))
        ax1.scatter(0, sa_scores[0], color="green", label="Initial score: " + str(sa_scores[0]))

        ax1.set_title("Simulated annealing")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Score")
        ax1.legend(loc="upper right")

        if len(all_scores) < 100:
            s = 5
        else:
            s = 2500 / len(all_scores)

        ax2.scatter(list(range(len(all_scores))), all_scores, s=s)
        ax2.scatter(argmin, sa_scores[argmin], color="red", label="Best score: " + str(sa_scores[argmin]))
        ax2.scatter(0, sa_scores[0], color="green", label="Initial score: " + str(sa_scores[0]))

        ax2.set_title("All evaluated solutions")
        ax2.set_xlabel("Solution ID")
        ax2.set_ylabel("Score")
        ax2.legend(loc="upper right")

        ax3.plot(list(range(len(sa_scores))), temperature, "-")
        argmin = np.argmin(sa_scores)
        ax3.scatter(argmin, temperature[argmin], color="red", label="Best score: " + str(temperature[argmin]))

        ax3.set_title("Temperature decrease")
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Temperature")
        ax3.legend(loc="upper right")

        if len(sa_scores) < 100:
            s = 5
        else:
            s = 2500 / len(all_scores)

        ax4.scatter(list(range(len(sa_scores))), probability, s=s)
        argmin = np.argmin(sa_scores)
        ax4.scatter(argmin, probability[argmin], color="red", label="Best score: " + str(probability[argmin]))

        ax4.set_title("Escaping probability")
        ax4.set_xlabel("Iteration")
        ax4.set_ylabel("Probability")
        ax4.legend(loc="upper right")

        if save:
            save_path = os.path.join(self.loss_func.plots_path, f"sa_summary.png")

            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
