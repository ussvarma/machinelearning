{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Support_vector_regression.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/ussvarma/machinelearning/blob/main/Support_vector_regression.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "m3PAEPRDRLA3"
      },
      "source": [
        "# Support Vector Regression (SVR)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "0VCUAVIjRdzZ"
      },
      "source": [
        "## Importing the libraries"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "56oRF-QfSDzC"
      },
      "source": [
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import pandas as pd"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "fXVXoFWtSF4_"
      },
      "source": [
        "## Importing the dataset"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "dataset=pd.read_csv(\"Position_Salaries.csv\")\n",
        "x=dataset.iloc[:,1:-1].values\n",
        "y=dataset.iloc[:,-1].values\n"
      ],
      "metadata": {
        "id": "fc4ea_zl2DSw"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "fAWJV6gpiTYM",
        "outputId": "cb30be92-47ed-46d4-b4f9-749290c24e72",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "source": [
        "print(x)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[ 1]\n",
            " [ 2]\n",
            " [ 3]\n",
            " [ 4]\n",
            " [ 5]\n",
            " [ 6]\n",
            " [ 7]\n",
            " [ 8]\n",
            " [ 9]\n",
            " [10]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "P1CzeAyRiU3c",
        "outputId": "3b2d84f0-6a01-4409-e1d2-532314f5f59b",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "source": [
        "print(y)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[  45000   50000   60000   80000  110000  150000  200000  300000  500000\n",
            " 1000000]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "_8Ny1GfPiV3m",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "dafcaf1d-6324-46ed-f834-2d5c6d27dfd2"
      },
      "source": [
        "y=y.reshape(len(y),1)\n",
        "print(y) #converted y feature into 2dimensional"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[  45000]\n",
            " [  50000]\n",
            " [  60000]\n",
            " [  80000]\n",
            " [ 110000]\n",
            " [ 150000]\n",
            " [ 200000]\n",
            " [ 300000]\n",
            " [ 500000]\n",
            " [1000000]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "YS8FeLHYS-nI"
      },
      "source": [
        "## Feature Scaling"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "PGeAlD1HTDI1"
      },
      "source": [
        "from sklearn.preprocessing import StandardScaler\n",
        "sc_x=StandardScaler()\n",
        "sc_y=StandardScaler()\n",
        "x=sc_x.fit_transform(x)\n",
        "y=sc_y.fit_transform(y)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "nXa8Z9FgjFTQ",
        "outputId": "c0d235a3-0e58-4782-c187-6e07d21ea824",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "source": [
        "print(x)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[-1.5666989 ]\n",
            " [-1.21854359]\n",
            " [-0.87038828]\n",
            " [-0.52223297]\n",
            " [-0.17407766]\n",
            " [ 0.17407766]\n",
            " [ 0.52223297]\n",
            " [ 0.87038828]\n",
            " [ 1.21854359]\n",
            " [ 1.5666989 ]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "i7Oo2kybjGr2",
        "outputId": "71e20c0d-8aee-4a92-b7f5-d8c4b8b642b1",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "source": [
        "print(y)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[-0.72004253]\n",
            " [-0.70243757]\n",
            " [-0.66722767]\n",
            " [-0.59680786]\n",
            " [-0.49117815]\n",
            " [-0.35033854]\n",
            " [-0.17428902]\n",
            " [ 0.17781001]\n",
            " [ 0.88200808]\n",
            " [ 2.64250325]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "eiU6D2QFRjxY"
      },
      "source": [
        "## Training the SVR model on the whole dataset"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "y6R4rt_GRz15",
        "outputId": "c93379ed-2cf7-437d-9981-bab609ca6b8a",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "source": [
        "from sklearn.svm import SVR\n",
        "regressor=SVR(kernel=\"rbf\")\n",
        "regressor.fit(x,y)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/sklearn/utils/validation.py:993: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
            "  y = column_or_1d(y, warn=True)\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "SVR()"
            ]
          },
          "metadata": {},
          "execution_count": 57
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "deDnDr8UR5vq"
      },
      "source": [
        "## Predicting a new result"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ib89-Fq8R8v-",
        "outputId": "6f0538b9-5a33-4eea-9274-45c6cc125ded",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "source": [
        "sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(-1,1)) \n",
        "#as predict object gives 1d array , we are converting it into 2d array"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[170370.0204065]])"
            ]
          },
          "metadata": {},
          "execution_count": 93
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "zzedFlUISSu_"
      },
      "source": [
        "## Visualising the SVR results"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "OWPRGsKpSW9U",
        "outputId": "eb6c4809-57b6-470b-e6f7-777b4029d244",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 295
        }
      },
      "source": [
        "plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y),color=\"red\")\n",
        "plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x).reshape(-1,1)), color = 'blue')\n",
        "plt.title(\"SVR\")\n",
        "plt.xlabel(\"Position\")\n",
        "plt.ylabel(\"salary\")\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "UahPVNlJSZ-K"
      },
      "source": [
        "## Visualising the SVR results (for higher resolution and smoother curve)"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "7fkhPL7RSd2X",
        "outputId": "c572c6ad-1b38-4c31-ab8c-c3cec800feda",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 295
        }
      },
      "source": [
        "x_grid=np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1)\n",
        "x_grid=x_grid.reshape(-1,1) #converting into 2D\n",
        "plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y),color=\"red\")\n",
        "plt.plot(x_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid)).reshape(-1,1)), color = 'blue')\n",
        "plt.title('Truth or Bluff (SVR)')\n",
        "plt.xlabel('Position level')\n",
        "plt.ylabel('Salary')\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Disadvantage of svr model:\n",
        "It cannot handle outliers\n",
        "\n",
        "compulsory feature scaling should be done"
      ],
      "metadata": {
        "id": "cYMMb9fybm9J"
      }
    }
  ]
}