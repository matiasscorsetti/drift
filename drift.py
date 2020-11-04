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
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy

def drift_detection(data_stream,
                    delta, # delta in adwin detection
                    predict=False,
                    drift_detector=None,
                    ):
    """ data change detection """

    if predict is False:
        drift_detector = ADWIN(delta=delta)

    else:
        copy_drift_detector = deepcopy(drift_detector)

    results = pd.DataFrame(columns=["value", "sqrt", "change"])
    
    for index, row in data_stream.iteritems():
         drift_detector.add_element(row)
         change = drift_detector.detected_change()
         result = pd.DataFrame(data={
                                     "value": row,
                                     "sqrt": sqrt(drift_detector.variance), 
                                     "change": 1 if change==True else 0,
                                     },
                               index=[index],
                               )

         if change==True and predict==True:
             drift_detector = deepcopy(copy_drift_detector)

         results = results.append(result)

    return results, drift_detector


def drift_detect_multicols(df,
                           delta,
                           predict=False,
                           drift_detector_dict=None,
                           scaler_dict=None,
                           ):
    """ data change detection for each column """

    df_results = pd.DataFrame()

    if predict is False:
        drift_detector_dict = {}
        scaler_dict = {}

        for col in df.columns:
            results, drift_detector = drift_detection(data_stream=df[col],
                                                      delta=delta)
            scaler = MinMaxScaler()
            results = pd.DataFrame(scaler.fit_transform(results), columns=[i + '_' + col for i in results.columns])
            df_results = pd.concat([df_results, results], axis=1)
            drift_detector_dict[str(col) + "_detector"] = drift_detector
            scaler_dict[str(col) + "_scaler"] = scaler

    else:
        for col, drift_detector, scaler in zip(df.columns, drift_detector_dict.values(), scaler_dict.values()):
            results, drift_detector = drift_detection(data_stream=df[col],
                                                      delta=delta,
                                                      predict=True,
                                                      drift_detector=drift_detector,
                                                      )
            results = pd.DataFrame(scaler.transform(results), columns=[i + '_' + col for i in results.columns])
            df_results = pd.concat([df_results, results], axis=1)

    return df_results, drift_detector_dict, scaler_dict


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
                      predict=False,
                      drift_detector_dict=None,
                      scaler_dict=None,
                      cluster_model=None,
                      scaler_df=None,
                      sample_size=0.1,
                      k=(1,9),
                      n_clusters_without_method=5,
                      ):
    ''' build drift dataframe '''

    if scaler_df is None:
        scaler_df = MinMaxScaler()

    df = pd.DataFrame(scaler_df.fit_transform(df), columns=df.columns)

    if float(sample_size) != 1.0:
        df, cluster_model = cluster_sampling(df=df,
                                             sample_size=sample_size,
                                             k=k,
                                             n_clusters_without_method=n_clusters_without_method,
                                             cluster_model=cluster_model,
                                             )
    else:
        print("sampling does not apply")

    df_results, drift_detector_dict, scaler_dict = drift_detect_multicols(df=df,
                                                                          delta=delta,
                                                                          predict=predict,
                                                                          drift_detector_dict=drift_detector_dict,
                                                                          scaler_dict=scaler_dict,
                                                                          )

    return df_results, drift_detector_dict, scaler_dict, cluster_model, scaler_df


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
    
    values_columns = [col for col in df.columns if col.startswith("value_")]
    sqrt_columns = [col for col in df.columns if col.startswith("sqrt_")]
    change_columns = [col for col in df.columns if col.startswith("change_")]
    rows = len(values_columns)
    fig_list = []

    for value, sqrt, change in zip(values_columns, sqrt_columns, change_columns):
        fig = px.scatter(x=df.reset_index().index,
                           y=df[sqrt],
                           color=df[change].astype(int).astype(str),
                           title=sqrt, 
                           color_discrete_map={"1":'#EF553B', "0":'#636EFA'},
                           )
        fig.update_layout(legend_title_text='change detect')
        fig2 = px.line(x=df.reset_index().index, y=df[value], color_discrete_sequence=['rgba(0, 204, 150, .25)'])
        fig2['data'][0]['showlegend'] = True
        fig2['data'][0]['name'] = 'scaled value'
        fig.add_trace(fig2.data[0])
        fig.update_layout(xaxis_title="X - index number",
                          yaxis_title="Y - sqrt",
                          )
        fig_list.append(fig)

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
        df_results, drift_detector_dict, scaler_dict, cluster_model, scaler_df = drift_df_multicols(df=df,
                                                                                         delta=self.delta,
                                                                                         predict=False,
                                                                                         sample_size=self.sample_size,
                                                                                         k=self.k,
                                                                                         n_clusters_without_method=self.n_clusters_without_method,
                                                                                         )
        self.df_results = df_results
        self.drift_detector_dict = drift_detector_dict
        self.scaler_dict = scaler_dict
        self.cluster_model = cluster_model
        self.scaler_df = scaler_df

        totals, percentage, drift_average = drift_average_calculate(df_results=self.df_results,
                                                                    list_of_weights=self.list_of_weights,
                                                                    )
        self.totals = totals
        self.percentage = percentage
        self.drift_average = drift_average

        self.plots = plot_drift(df=self.df_results)
        
    def predict(self, df, sample_size=1):
        df_results_predict, drift_detector_dict_predict, scaler_dict, cluster_model_predict, scaler_df_predict = drift_df_multicols(df=df,
                                                                                                                 delta=self.delta,
                                                                                                                 predict=True,
                                                                                                                 drift_detector_dict=self.drift_detector_dict,
                                                                                                                 scaler_dict=self.scaler_dict,
                                                                                                                 cluster_model=self.cluster_model,
                                                                                                                 scaler_df=self.scaler_df,
                                                                                                                 sample_size=sample_size,
                                                                                                                 k=self.k,
                                                                                                                 n_clusters_without_method=self.n_clusters_without_method,
                                                                                                                 )
        self.df_results_predict = df_results_predict

        totals_predict, percentage, drift_average_predict = drift_average_calculate(df_results=self.df_results_predict,
                                                                                    list_of_weights=self.list_of_weights,
                                                                                    )
        self.totals_predict = totals_predict
        self.percentage_predict = percentage
        self.drift_average_predict = drift_average_predict

        self.plots_predict = plot_drift(df=self.df_results_predict)

        return drift_average_predict