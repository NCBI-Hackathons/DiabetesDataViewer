import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt, mpld3


def return_graph(data_frame, attributes):
    '''Return pie chart of attributes.'''
    group = attributes[0]
    attributes.append('PatientID')

    df_patients = data_frame.loc[:,attributes]

    a = df_patients.groupby(group).count()
    a.sort_values(by='PatientID', inplace=True)
    cutoff = a/a.sum() * 100
    cutoff[cutoff > 5.0] = 0
    cutoff = .09 / cutoff
    cutoff[cutoff == np.inf] = 0


    plt.gca().axis("equal")
    pie = plt.pie(a, startangle=0,explode = cutoff['PatientID'].tolist(), autopct='%1.1f%%', pctdistance = 1.3)
    labels=a.index.unique()
    plt.title('Pie Chart Demonstration', weight='bold', size=14)
    plt.legend(pie[0],labels, bbox_to_anchor=(1,.75), loc="center right", fontsize=10,
                 bbox_transform=plt.gcf().transFigure)
    plt.subplots_adjust(left=0.0, bottom=0.1, right=0.85)
    mpld3.fig_to_html(plt.gcf())
