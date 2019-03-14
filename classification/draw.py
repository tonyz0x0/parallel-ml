# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import json
import argparse
from LogisticRegression import readBeta
from operator import itemgetter

"""
Parameter example:
  --opt 1
"""

def readPlot(input):
    with open(input, 'r') as fh:
        return json.load(fh)

def drawChart1(json1, json2, x, y):
    if json1:
        plt.plot(json1[x], json1[y], 'r-x', label=json1['name'])
    if json2:
        plt.plot(json2[x], json2[y], 'b-^', label=json2['name'])
    plt.legend()
    plt.xlabel('Time(s)')
    plt.ylabel(y)
    plt.show()
    
def drawChart2(json1, json2, json3, x, y):
    if json1 and json2 and json3:
        plt.plot(json1[x], json1[y], 'r-x', label=json1['name'])
        plt.plot(json2[x], json2[y], 'b-^', label=json2['name'])
        plt.plot(json3[x], json3[y], 'g-*', label=json3['name'])
    plt.legend()
    plt.xlabel('Time(s)')
    plt.ylabel(y)
    plt.show()

def drawChart3(top10positive, top10negative):
    if top10positive and top10negative:
        plt.bar([k for (k, v) in top10positive], [v for (k, v) in top10positive])
        plt.title('Top 10 features with positive values')
        plt.show()
        plt.bar([k for (k, v) in top10negative], [v for (k, v) in top10negative])
        plt.title('Top 10 features with negative values')
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot draw tool',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--opt', default=0, type=int,
                        help='Choose which chart to draw')
    args = parser.parse_args()

    input1 = 'plot_data_non_parallel.json'
    input2 = 'plot_data_parallel_0.json'
    input3 = 'plot_data_parallel_0.5.json'
    input4 = 'plot_data_parallel_1.0.json'
    input5 = 'beta'

    loaded_json1 = readPlot(input1)
    loaded_json1['name'] = 'non-parallel'
    loaded_json2 = readPlot(input2)
    loaded_json2['name'] = 'parallel'
    loaded_json3 = readPlot(input3)
    loaded_json3['name'] = r'$\lambda = 0.5$'
    loaded_json4 = readPlot(input4)
    loaded_json4['name'] = r'$\lambda = 1.0$'

    """
    Draw 4 plots between non-parallel and parallel JSON files
    """
    if args.opt == 1:
        drawChart1(loaded_json1, loaded_json2, 'period', 'gradNorm')
        drawChart1(loaded_json1, loaded_json2, 'period', 'acc')
        drawChart1(loaded_json1, loaded_json2, 'period', 'pre')
        drawChart1(loaded_json1, loaded_json2, 'period', 'rec')

    """
    Draw 1 plot with three different lambda values for parallel JSON files
    """
    if args.opt == 2:
        loaded_json2['name'] = r'$\lambda = 0.0$'

        drawChart2(loaded_json2, loaded_json3, loaded_json4, 'period', 'acc')

    """
    Draw 1 bar with 10 most positive values and 10 most negative values of features
    when there is the highest accuracy in a specific lambda and iteration.
    """

    if args.opt == 3:
        # Find out the highest accuracy
        print max(loaded_json3['acc']), loaded_json3['acc'].index(max(loaded_json3['acc'])) + 1

        # Read beta file
        features = readBeta(input5)
        features = sorted(features.items(), key=itemgetter(1))
        top10positive = features[-10:]
        top10negative = features[:10]

        drawChart3(top10positive, top10negative)