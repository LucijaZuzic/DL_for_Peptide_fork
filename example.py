import numpy as np
import matplotlib.pyplot as plt


category_names = ['Strongly disagree', 'Disagree',
                  'Neither agree nor disagree', 'Agree', 'Strongly agree']
results = {
    'Question 1': [10, 15, 17, 32, 26],
    'Question 2': [26, 22, 29, 10, 13],
    'Question 3': [35, 37, 7, 2, 19],
    'Question 4': [32, 11, 9, 15, 33],
    'Question 5': [21, 29, 5, 5, 40],
    'Question 6': [8, 19, 5, 30, 38]
}


def survey(results):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_names = ["TN", "FP", "FN", "TP"]
    light_color = ["#FFF2CC", "#F8CECC", "#DAE8FC", "#D5E8D4"]
    dark_color = ["#D6B656", "#B85450", "#6C8EBF", "#82B366"]

    fig, ax = plt.subplots(figsize=(10, len(results) / 2))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())
    if len(results) == 5:
        ax.set_ylabel("Model")
    else:
        ax.set_ylabel("Model and seed")
    for i, (colname, color_l, color_d) in enumerate(zip(category_names, light_color, dark_color)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=color_l, edgecolor=color_d)
        xcenters = starts + widths / 2

        for y, (x, c) in enumerate(zip(xcenters, widths)):
            if int(c) > 0:
                ax.text(x, y, str(int(c)), ha='center', va='center',
                        color='black')
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    return fig, ax


#survey(results, category_names)
#plt.show()