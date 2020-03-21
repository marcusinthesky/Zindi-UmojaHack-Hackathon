# -*- coding: utf-8 -*-

""" base module
"""

# Author: Olivier Garcia <o.garcia.dev@gmail.com>
# License: BSD 3 clause

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin


class CA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, row_labels=None, col_labels=None,
                 stats=True):
        self.n_components = n_components
        self.row_labels = row_labels
        self.col_labels = col_labels
        self.stats = stats
            
    def fit(self, X, y=None):
        # Initializations
        self.n_ = np.sum(X)
        self.r_ = np.sum(X, axis=1).reshape(-1, 1)
        self.c_ = np.sum(X, axis=0).reshape(1, -1)
        
        # Stats initializations
        self.row_contrib_ = None
        self.col_contrib_ = None
        self.row_cos2_ = None
        self.col_cos2_ = None
        
        # Compute SVD
        self._compute_svd(X)
        
        # Set row labels
        nrows = X.shape[0]
        self.row_labels_ = self.row_labels
        if (self.row_labels_ is None) or (len(self.row_labels_) != nrows):
            self.row_labels_ = ["row" + str(x) for x in np.arange(0, nrows)]
        
        # Set column labels
        ncols = X.shape[1]
        self.col_labels_ = self.col_labels
        if (self.col_labels_ is None) or (len(self.col_labels_) != ncols):
            self.col_labels_ = ["col" + str(x) for x in np.arange(0, ncols)]
        self.col_labels_short_ = self.col_labels_
        self.model_ = "base"

        return self
    
    def transform(self, X, y=None):
        x = np.sum(X, axis=1).astype(float)
        return ((np.diag(x ** (-1)).dot(X)).dot((self.eig_[0]
                ** (-0.5)).reshape(1, -1)
                * self.col_coord_))

    def _compute_svd(self, X):
        # SVD        
        E = (self.r_.dot(self.c_)) / self.n_
        R = (1.0 / np.sqrt(self.n_)) * (X - E) / np.sqrt(E)
        U, lambdas, V = np.linalg.svd(R, full_matrices=False)
        
        # Eigen values first step
        eigen_values = lambdas ** 2
        eigen_values_percent = 100 * eigen_values / np.sum(eigen_values)
        eigen_values_percent_cumsum = np.cumsum(eigen_values_percent)
        
        # Set n_components_
        self.n_components_ = self.n_components
        if self.n_components_ is None:
            self.n_components_ = len(eigen_values) - 1
        elif (self.n_components_ >= 0) and (self.n_components_ < 1):
            i = 0
            threshold = 100 * self.n_components_
            while eigen_values_percent_cumsum[i] < threshold:
                i = i + 1
            self.n_components_ = i
        elif ((self.n_components_ >= 1)
                and (self.n_components_ <= len(eigen_values) -1)
                and (isinstance(self.n_components_, int))):
            self.n_components_ = int(np.trunc(self.n_components_))
        elif ((self.n_components_ >= 1)
                and (self.n_components_ <= len(eigen_values) -1)
                and (isinstance(self.n_components_, float))):
            self.n_components_ = int(np.floor(self.n_components_))
        else:
            self.n_components_ = len(eigen_values) - 1
        
        # Eigen values second step
        self.eig_ = np.array([eigen_values[:self.n_components_],
                            eigen_values_percent[:self.n_components_],
                            eigen_values_percent_cumsum[:self.n_components_]])
        
        # Factor coordinates
        row_coord = (U / np.sqrt(self.r_ / self.n_)) * (lambdas.reshape(1, -1))
        col_coord = ((V.T / np.sqrt(self.c_.T / self.n_))
                    * (lambdas.reshape(1, -1)))
        self.row_coord_ = row_coord[:, :self.n_components_]
        self.col_coord_ = col_coord[:, :self.n_components_]
        
        # Compute stats
        if self.stats:
            self._compute_stats(X)

    def _compute_stats(self, X):
        # Contributions
        row_contrib = 100 * ((self.r_ / self.n_)
                          * (self.row_coord_ ** 2)
                          * (1 / self.eig_[0].T))
        col_contrib = 100 * ((self.c_.T / self.n_)
                          * (self.col_coord_ ** 2)
                          * (1 / self.eig_[0].T))
        self.row_contrib_ = row_contrib[:, :self.n_components_]
        self.col_contrib_ = col_contrib[:, :self.n_components_]
        
        # Cos2
        row_cos2 = ((self.row_coord_ ** 2)
                    / (np.sum((self.n_ / self.c_)
                        * ((X / self.r_) - (self.c_ / self.n_))**2, axis=1)
                        .reshape(-1, 1)))
        col_cos2 = ((self.col_coord_ ** 2)
                    / (np.sum((self.n_ / self.r_)
                        * ((X / self.c_) - (self.r_ / self.n_))**2, axis=0))
                        .reshape(-1, 1))
        self.row_cos2_ = row_cos2[:, :self.n_components_]
        self.col_cos2_ = col_cos2[:, :self.n_components_]

    def plot_eigenvalues(self, type="absolute", figsize=None):
        """ Plot the eigen values graph
        
        Parameters
        ----------
        type : string
            Select the graph to plot :
                - If "absolute" : plot the eigenvalues.
                - If "percentage" : plot the percentage of variance.
                - If "cumulative" : plot the cumulative percentage of
                  variance.
        figsize : tuple of integers or None
            Width, height of the figure in inches.
            If not provided, defaults to rc figure.figsize

        Returns
        -------
        None
        """
        plt.figure(figsize=figsize)
        if type == "absolute":
            plt.bar(np.arange(1, self.eig_[0].shape[0] + 1), self.eig_[0],
                    color="steelblue", align="center")
            plt.xlabel("Axis")
            plt.ylabel("Eigenvalue")
        elif type == "percentage":
            plt.bar(np.arange(1, self.eig_[1].shape[0] + 1), self.eig_[1],
                    color="steelblue", align="center")
            plt.xlabel("Axis")
            plt.ylabel("Percentage of variance")
        elif type == "cumulative":
            plt.bar(np.arange(1, self.eig_[2].shape[0] + 1), self.eig_[2],
                    color="steelblue", align="center")
            plt.xlabel("Axis")
            plt.ylabel("Cumulative percentage of variance")
        else:
            raise Exception("Error : 'type' variable must be 'absolute' or \
                            'percentage' or 'cumulative'")
        plt.title("Scree plot")
        plt.show()

    def mapping(self, num_x_axis, num_y_axis, short_labels=True, figsize=None):
        """ Plot the Factor map for rows and columns simultaneously
        
        Parameters
        ----------
        num_x_axis : int
            Select the component to plot as x axis.
        
        num_y_axis : int
             Select the component to plot as y axis.
        
        short_labels : bool
            Useful only for MCA, which inherits from Base class
            (so, useful if model_ == "mca").
            -> If model == "mca" :
                - If short_labels is True, then the column labels have
                  no prefixes.
                - If short_labels is False, the column labels are
                  prefixed by the names of the variables.
                    -> If categories from different variables have the
                       same code, prefixing them by the names of the
                       variables avoids confusion.
        
        figsize : tuple of integers or None
            Width, height of the figure in inches.
            If not provided, defaults to rc figure.figsize

        Returns
        -------
        None
        """
        plt.figure(figsize=figsize)
        if self.model_ == "mca" and short_labels:
            col_labels = self.col_labels_short_
        else:
            col_labels = self.col_labels_
        plt.scatter(self.row_coord_[:, num_x_axis - 1],
                    self.row_coord_[:, num_y_axis - 1],
                    marker=".", color="white")
        plt.scatter(self.col_coord_[:, num_x_axis - 1],
                    self.col_coord_[:, num_y_axis - 1],
                    marker=".", color="white")
        for i in np.arange(0, self.row_coord_.shape[0]):
            plt.text(self.row_coord_[i, num_x_axis - 1],
                     self.row_coord_[i, num_y_axis - 1],
                     self.row_labels_[i],
                     horizontalalignment="center", verticalalignment="center",
                     color="red")
        for i in np.arange(0, self.col_coord_.shape[0]):
            plt.text(self.col_coord_[i, num_x_axis - 1],
                     self.col_coord_[i, num_y_axis - 1],
                     col_labels[i],
                     horizontalalignment="center", verticalalignment="center",
                     color="blue")
        plt.title("Factor map")
        plt.xlabel("Dim " + str(num_x_axis) + " ("
                    + str(np.around(self.eig_[1, num_x_axis - 1], 2)) + "%)")
        plt.ylabel("Dim " + str(num_y_axis) + " ("
                    + str(np.around(self.eig_[1, num_y_axis - 1], 2)) + "%)")
        plt.axvline(x=0, linestyle="--", linewidth=0.5, color="k")
        plt.axhline(y=0, linestyle="--", linewidth=0.5, color="k")
        plt.show()
    
    def mapping_row(self, num_x_axis, num_y_axis, figsize=None):
        """ Plot the Factor map for rows only
        
        Parameters
        ----------
        num_x_axis : int
            Select the component to plot as x axis.
        
        num_y_axis : int
             Select the component to plot as y axis.
        
        figsize : tuple of integers or None
            Width, height of the figure in inches.
            If not provided, defaults to rc figure.figsize

        Returns
        -------
        None
        """
        plt.figure(figsize=figsize)
        plt.scatter(self.row_coord_[:, num_x_axis - 1],
                    self.row_coord_[:, num_y_axis - 1],
                    marker=".", color="white")
        for i in np.arange(0, self.row_coord_.shape[0]):
            plt.text(self.row_coord_[i, num_x_axis - 1],
                     self.row_coord_[i, num_y_axis - 1],
                     self.row_labels_[i],
                     horizontalalignment="center", verticalalignment="center",
                     color="red")
        plt.title("Factor map for rows")
        plt.xlabel("Dim " + str(num_x_axis) + " ("
                    + str(np.around(self.eig_[1, num_x_axis - 1], 2)) + "%)")
        plt.ylabel("Dim " + str(num_y_axis) + " ("
                    + str(np.around(self.eig_[1, num_y_axis - 1], 2)) + "%)")
        plt.axvline(x=0, linestyle="--", linewidth=0.5, color="k")
        plt.axhline(y=0, linestyle="--", linewidth=0.5, color="k")
        plt.show()
    
    def mapping_col(self, num_x_axis, num_y_axis, short_labels=True,
                    figsize=None):
        """ Plot the Factor map for columns only
        
        Parameters
        ----------
        num_x_axis : int
            Select the component to plot as x axis.
        
        num_y_axis : int
             Select the component to plot as y axis.
        
        short_labels : bool
            Useful only for MCA, which inherits from Base class
            (so, useful if model_ == "mca").
            -> If model == "mca" :
                - If short_labels is True, then the column labels have
                  no prefixes.
                - If short_labels is False, the column labels are
                  prefixed by the names of the variables.
                    -> If categories from different variables have the
                       same code, prefixing them by the names of the
                       variables avoids confusion.
        
        figsize : tuple of integers or None
            Width, height of the figure in inches.
            If not provided, defaults to rc figure.figsize

        Returns
        -------
        None
        """
        plt.figure(figsize=figsize)
        if self.model_ == "mca" and short_labels:
            col_labels = self.col_labels_short_
        else:
            col_labels = self.col_labels_
        plt.scatter(self.col_coord_[:, num_x_axis - 1],
                    self.col_coord_[:, num_y_axis - 1],
                    marker=".", color="white")
        for i in np.arange(0, self.col_coord_.shape[0]):
            plt.text(self.col_coord_[i, num_x_axis - 1],
                     self.col_coord_[i, num_y_axis - 1],
                     col_labels[i],
                     horizontalalignment="center", verticalalignment="center",
                     color="blue")
        plt.title("Factor map for columns")
        plt.xlabel("Dim " + str(num_x_axis) + " ("
                    + str(np.around(self.eig_[1, num_x_axis - 1], 2)) + "%)")
        plt.ylabel("Dim " + str(num_y_axis) + " ("
                    + str(np.around(self.eig_[1, num_y_axis - 1], 2)) + "%)")
        plt.axvline(x=0, linestyle="--", linewidth=0.5, color="k")
        plt.axhline(y=0, linestyle="--", linewidth=0.5, color="k")
        plt.show()
    
    def plot_row_contrib(self, num_axis, nb_values=None, figsize=None):
        """ Plot the row contributions graph
            
        For the selected axis, the graph represents the row
        contributions sorted in descending order.            
        
        Parameters
        ----------
        num_axis : int
            Select the axis for which the row contributions are plotted.
        
        nb_values : int
             Set the maximum number of values to plot.
             If nb_values is None : all the values are plotted.
        
        figsize : tuple of integers or None
            Width, height of the figure in inches.
            If not provided, defaults to rc figure.figsize
        
        Returns
        -------
        None
        """
        plt.figure(figsize=figsize)
        n_rows = len(self.row_labels_)
        n_labels = len(self.row_labels_)
        if (nb_values is not None) and (nb_values < n_labels):
            n_labels = nb_values
        limit = n_rows - n_labels
        contribs = self.row_contrib_[:, num_axis - 1]
        contribs_sorted = np.sort(contribs)[limit:n_rows]
        labels = pd.Series(self.row_labels_)[np.argsort(contribs)]\
                                                        [limit:n_rows]
        r = range(n_labels)
        bar_width = 0.5
        plt.yticks([ri + bar_width / 2 for ri in r], labels)
        plt.barh(r, contribs_sorted, height=bar_width, color="steelblue",
                 align="edge")
        plt.title("Rows contributions")
        plt.xlabel("Contributions (%)")
        plt.ylabel("Rows")
        plt.show()
    
    def plot_col_contrib(self, num_axis, nb_values=None, short_labels=True,
                         figsize=None):
        """ Plot the column contributions graph
            
        For the selected axis, the graph represents the column
        contributions sorted in descending order.            
        
        Parameters
        ----------
        num_axis : int
            Select the axis for which the column contributions are
            plotted.
        
        nb_values : int
             Set the maximum number of values to plot.
             If nb_values is None : all the values are plotted.
        
        short_labels : bool
            Useful only for MCA, which inherits from Base class
            (so, useful if model_ == "mca").
            -> If model == "mca" :
                - If short_labels is True, then the column labels have
                  no prefixes.
                - If short_labels is False, the column labels are
                  prefixed by the names of the variables.
                    -> If categories from different variables have the
                       same code, prefixing them by the names of the
                       variables avoids confusion.
        
        figsize : tuple of integers or None
            Width, height of the figure in inches.
            If not provided, defaults to rc figure.figsize

        Returns
        -------
        None
        """
        plt.figure(figsize=figsize)
        n_cols = len(self.col_labels_)
        n_labels = len(self.col_labels_)
        if self.model_ == "mca" and short_labels:
            col_labels = self.col_labels_short_
        else:
            col_labels = self.col_labels_
        if (nb_values is not None) and (nb_values < n_labels):
            n_labels = nb_values
        limit = n_cols - n_labels
        contribs = self.col_contrib_[:, num_axis - 1]
        contribs_sorted = np.sort(contribs)[limit:n_cols]
        labels = pd.Series(col_labels)[np.argsort(contribs)][limit:n_cols]
        r = range(n_labels)
        bar_width = 0.5
        plt.yticks([ri + bar_width / 2 for ri in r], labels)
        plt.barh(r, contribs_sorted, height=bar_width, color="steelblue",
                 align="edge")
        plt.title("Columns contributions")
        plt.xlabel("Contributions (%)")
        plt.ylabel("Columns")
        plt.show()

    def plot_row_cos2(self, num_axis, nb_values=None, figsize=None):
        """ Plot the row cosines graph
            
        For the selected axis, the graph represents the row cosines
        sorted in descending order.            
        
        Parameters
        ----------
        num_axis : int
            Select the axis for which the row cosines are plotted.
        
        nb_values : int
             Set the maximum number of values to plot.
             If nb_values is None : all the values are plotted.
        
        figsize : tuple of integers or None
            Width, height of the figure in inches.
            If not provided, defaults to rc figure.figsize
        
        Returns
        -------
        None
        """
        plt.figure(figsize=figsize)
        n_rows = len(self.row_labels_)
        n_labels = len(self.row_labels_)
        if (nb_values is not None) and (nb_values < n_labels):
            n_labels = nb_values
        limit = n_rows - n_labels
        cos2 = self.row_cos2_[:, num_axis - 1]
        cos2_sorted = np.sort(cos2)[limit:n_rows]
        labels = pd.Series(self.row_labels_)[np.argsort(cos2)][limit:n_rows]
        r = range(n_labels)
        bar_width = 0.5
        plt.yticks([ri + bar_width / 2 for ri in r], labels)
        plt.barh(r, cos2_sorted, height=bar_width, color="steelblue",
                 align="edge")
        plt.title("Rows cos2")
        plt.xlabel("Cos2")
        plt.ylabel("Rows")
        plt.show()

    def plot_col_cos2(self, num_axis, nb_values=None, short_labels=True,
                      figsize=None):
        """ Plot the column cosines graph
            
        For the selected axis, the graph represents the column cosines
        sorted in descending order.            
        
        Parameters
        ----------
        num_axis : int
            Select the axis for which the column cosines are plotted.
        
        nb_values : int
             Set the maximum number of values to plot.
             If nb_values is None : all the values are plotted.
        
        short_labels : bool
            Useful only for MCA, which inherits from Base class
            (so, useful if model_ == "mca").
            -> If model == "mca" :
                - If short_labels is True, then the column labels have
                  no prefixes.
                - If short_labels is False, the column labels are
                  prefixed by the names of the variables.
                    -> If categories from different variables have the
                       same code, prefixing them by the names of the
                       variables avoids confusion.
        
        figsize : tuple of integers or None
            Width, height of the figure in inches.
            If not provided, defaults to rc figure.figsize

        Returns
        -------
        None
        """
        plt.figure(figsize=figsize)
        n_cols = len(self.col_labels_)
        n_labels = len(self.col_labels_)
        if self.model_ == "mca" and short_labels:
            col_labels = self.col_labels_short_
        else:
            col_labels = self.col_labels_
        if (nb_values is not None) and (nb_values < n_labels):
            n_labels = nb_values
        limit = n_cols - n_labels
        cos2 = self.col_cos2_[:, num_axis - 1]
        cos2_sorted = np.sort(cos2)[limit:n_cols]
        labels = pd.Series(col_labels)[np.argsort(cos2)][limit:n_cols]
        r = range(n_labels)
        bar_width = 0.5
        plt.yticks([ri + bar_width / 2 for ri in r], labels)
        plt.barh(r, cos2_sorted, height=bar_width, color="steelblue",
                 align="edge")
        plt.title("Columns cos2")
        plt.xlabel("Cos2")
        plt.ylabel("Columns")
        plt.show()

    def row_topandas(self):
        """ Export row data (row coordinates, row contributions,
        row cosines) into a pandas.DataFrame
        
        Parameters
        ----------
        None

        Returns
        -------
        Y : pandas.DataFrame, shape (n_rows, 3 * n_components_)
            - Column indices from 0 to (n_components_ - 1) :
              row coordinates.
            - Column indices from n_components_ to (2*n_components_-1) : 
              row contributions.
            - Column indices from (2 * n_components_) to 
              (3 * n_components_ - 1) : row cosines.
        """
        if self.stats:
            ndim = self.eig_[0].shape[0]
            nrows = self.row_coord_.shape[0]
            
            # Columns labels
            col = ["row_coord_dim" + str(x + 1) for x in np.arange(0, ndim)]
            col.extend(["row_contrib_dim" + str(x + 1)
                        for x in np.arange(0, ndim)])
            col.extend(["row_cos2_dim" + str(x + 1)
                        for x in np.arange(0, ndim)])
            
            # Row labels
            if ((self.row_labels_ is not None)
                    and (len(self.row_labels_) == nrows)):
                ind = self.row_labels_
            else:
                ind = ["row" + str(x) for x in np.arange(0, nrows)]
            
            return pd.DataFrame(np.c_[self.row_coord_,
                                      self.row_contrib_,
                                      self.row_cos2_],
                                      index=ind, columns=col)
        
        else:
            raise Exception("Error : self.stats attribute set at \'False\'")

    def col_topandas(self):
        """ Export column data (column coordinates,
        column contributions, column cosines) into a pandas.DataFrame
        
        Parameters
        ----------
        None

        Returns
        -------
        Y : pandas.DataFrame, shape (n_columns, 3 * n_components_)
            - Column indices from 0 to (n_components_ - 1) :
              column coordinates
            - Column indices from n_components_ to (2*n_components_-1) : 
              column contributions.
            - Column indices from (2 * n_components_) to 
              (3 * n_components_ - 1) : column cosines.

        """
        if self.stats:
            ndim = self.eig_[0].shape[0]
            ncols = self.col_coord_.shape[0]
            
            # Columns labels
            col = ["col_coord_dim" + str(x + 1) for x in np.arange(0, ndim)]
            col.extend(["col_contrib_dim" + str(x + 1)
                        for x in np.arange(0, ndim)])
            col.extend(["col_cos2_dim" + str(x + 1)
                        for x in np.arange(0, ndim)])
            
            # Row labels

            if ((self.col_labels_ is not None)
                    and (len(self.col_labels_) == ncols)):
                ind = self.col_labels_
            else:
                ind = ["col" + str(x) for x in np.arange(0, ncols)]
            
            return pd.DataFrame(np.c_[self.col_coord_,
                                      self.col_contrib_,
                                      self.col_cos2_],
                                      index=ind, columns=col)
        
        else:
            raise Exception("Error : self.stats attribute set at \'False\'")