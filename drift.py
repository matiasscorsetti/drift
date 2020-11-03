import pandas as pd
import numpy as np
from skmultiflow.drift_detection.adwin import ADWIN
from sklearn.cluster import MiniBatchKMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.model_selection import StratifiedShuffleSplit
from math import sqrt
import weightedstats as ws
import plotly.express as px
from warnings import warn


def drift_detection(data_stream,
                    delta, # delta in adwin detection
                    drift_detector=None,
                    ):
    """ data change detection """

    if drift_detector is None:
        drift_detector = ADWIN(delta=delta)

    results = pd.DataFrame(columns=["sqrt", "change"])
    
    for index, row in data_stream.iteritems():
         drift_detector.add_element(row)
         result = pd.DataFrame(data={
                                     "sqrt": sqrt(drift_detector.variance), 
                                     "change": 1 if drift_detector.detected_change()==True else 0,
                                     },
                               index=[index],
                               )
         results = results.append(result)

    return results, drift_detector


def drift_detect_multicols(df,
                           delta,
                           drift_detector_dict=None,
                           ):
    """ data change detection for each column """

    df_results = pd.DataFrame()

    if drift_detector_dict is None:
        drift_detector_dict = {}
        for col in df.columns:
            results, drift_detector = drift_detection(data_stream=df[col],
                                                      delta=delta)
            results.columns = [i + '_' + col for i in results.columns]
            df_results = pd.concat([df_results, results], axis=1)
            drift_detector_dict[str(col) + "_detector"] = drift_detector

    else:
        for col, drift_detector in zip(df.columns, drift_detector_dict.values()):
            results, drift_detector = drift_detection(data_stream=df[col],
                                                      delta=delta,
                                                      drift_detector=drift_detector,
                                                      )
            results.columns = [i + '_' + col for i in results.columns]
            df_results = pd.concat([df_results, results], axis=1)

    return df_results, drift_detector_dict


def n_optimal_clusters(df,
                       k,
                       n_clusters_without_method,
                       ):
    ''' train a clustering model choosing the optimal number of clusters '''

    visualizer = KElbowVisualizer(model=MiniBatchKMeans(),
                                  k=k,
                                  locate_elbow=True,
                                  )
    visualizer.fit(df)
    visualizer.finalize()

    if visualizer.elbow_value_:
        return visualizer.elbow_value_

    else:
        visualizer = KElbowVisualizer(model=MiniBatchKMeans(),
                                      k=k,
                                      locate_elbow=True,
                                      metric='calinski_harabasz',
                                      showbool=False,
                                      )
        visualizer.fit(df)
        visualizer.finalize()

        if visualizer.elbow_value_:
            return visualizer.elbow_value_

        else:
            return n_clusters_without_method


def cluster_sampling(df,
                     sample_size,
                     k,
                     n_clusters_without_method,
                     cluster_model=None,
                     ):
    '''train a clustering model and take a stratified sample '''

    if cluster_model is None:
        n_clusters = n_optimal_clusters(df=df,
                                        k=k,
                                        n_clusters_without_method=n_clusters_without_method,
                                        )
        cluster_model = MiniBatchKMeans(n_clusters=n_clusters)

    y = cluster_model.fit_predict(df)

    if len(np.unique(y)) > 1:

        try:
            print("cluster sampling")
            sss = StratifiedShuffleSplit(n_splits=1, test_size=sample_size)

            for train_index, test_index in sss.split(df.values, y):
               df_results = df.iloc[test_index]

            return df_results, cluster_model

        except ValueError as ex:
            warn(str(ex))
            print("Conglomerate sampling will not be applied. A random sample will be selected")

            return df.sample(frac=sample_size), cluster_model


    else:
        print("Conglomerate sampling will not be applied. A random sample will be selected")
        return df.sample(frac=sample_size), cluster_model


def drift_df_multicols(df,
                      delta=0.99,
                      drift_detector_dict=None,
                      cluster_model=None,
                      sample_size=0.1,
                      k=(1,9),
                      n_clusters_without_method=5,
                      ):
    ''' build drift dataframe '''

    if float(sample_size) != 1.0:
        df, cluster_model = cluster_sampling(df=df,
                                             sample_size=sample_size,
                                             k=k,
                                             n_clusters_without_method=n_clusters_without_method,
                                             cluster_model=cluster_model,
                                             )
    else:
        print("sampling does not apply")

    df_results, drift_detector_dict = drift_detect_multicols(df=df,
                                                             delta=delta,
                                                             drift_detector_dict=drift_detector_dict,
                                                             )

    return df_results, drift_detector_dict, cluster_model


def drift_average_calculate(df_results,
                            list_of_weights=None, # array with the weights of each column
                            ):
    ''' calculate the deviation average or weighted average '''

    columns = [col for col in df_results.columns if col.startswith("change_")]
    totals = df_results[columns].sum()
    percentage = totals / df_results.shape[0] * 100

    if list_of_weights:
        return totals, percentage, ws.weighted_mean(percentage, list_of_weights)
    else:
        return totals, percentage, ws.mean(percentage)


def plot_drift(df,
               ):
    ''' plot variance and change detect for each column '''
    
    sqrt_columns = [col for col in df.columns if col.startswith("sqrt_")]
    change_columns = [col for col in df.columns if col.startswith("change_")]
    rows = len(sqrt_columns)
    fig_list = []

    for sqrt, change in zip(sqrt_columns, change_columns):
        trace = px.scatter(x=df.reset_index().index,
                           y=df[sqrt],
                           color=df[change],
                           title=sqrt, 
                           color_discrete_map={1:"red", 0:"blue"},
                           )
        trace.update_layout(xaxis_title="X - index number",
                            yaxis_title="Y - sqrt",
                            )
        fig_list.append(trace)

    return fig_list


class DriftEstimator(object):
    ''' drift estimator for multiple columns using cluster sampling and weight weights
        - https://github.com/matiasscorsetti
    '''

    def __init__(self, delta=0.99, sample_size=0.1, k=(1,9), n_clusters_without_method=5, list_of_weights=None):
        self.delta = delta
        self.sample_size = sample_size
        self.k = k
        self.n_clusters_without_method = n_clusters_without_method
        self.list_of_weights = list_of_weights


    def fit(self, df):
        df_results, drift_detector_dict, cluster_model = drift_df_multicols(df=df,
                                                                            delta=self.delta,
                                                                            sample_size=self.sample_size,
                                                                            k=self.k,
                                                                            n_clusters_without_method=self.n_clusters_without_method,
                                                                            )
        self.df_results = df_results
        self.drift_detector_dict = drift_detector_dict
        self.cluster_model = cluster_model

        totals, percentage, drift_average = drift_average_calculate(df_results=self.df_results,
                                                                    list_of_weights=self.list_of_weights,
                                                                    )
        self.totals = totals
        self.percentage = percentage
        self.drift_average = drift_average

        self.plots = plot_drift(df=self.df_results)
        
    def predict(self, df, sample_size=1):
        df_results_predict, drift_detector_dict_predict, cluster_model_predict = drift_df_multicols(df=df,
                                                                                                    delta=self.delta,
                                                                                                    drift_detector_dict=self.drift_detector_dict,
                                                                                                    cluster_model=self.cluster_model,
                                                                                                    sample_size=sample_size,
                                                                                                    k=self.k,
                                                                                                    n_clusters_without_method=self.n_clusters_without_method,
                                                                                                    )
        self.df_results_predict = df_results_predict
        self.drift_detector_dict_predict = drift_detector_dict_predict
        self.cluster_model_predict = cluster_model_predict

        totals_predict, percentage, drift_average_predict = drift_average_calculate(df_results=self.df_results_predict,
                                                                                    list_of_weights=self.list_of_weights,
                                                                                    )
        self.totals_predict = totals_predict
        self.percentage_predict = percentage
        self.drift_average_predict = drift_average_predict

        self.plots_predict = plot_drift(df=self.df_results_predict)

        return drift_average_predict