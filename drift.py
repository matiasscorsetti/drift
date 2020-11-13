import pandas as pd
import numpy as np
from skmultiflow.drift_detection.adwin import ADWIN
from sklearn.cluster import MiniBatchKMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import weightedstats as ws
from warnings import warn
from sklearn.preprocessing import MinMaxScaler
from copy import copy
import plotly.express as px
from plotly.subplots import make_subplots


def drift_detection(data_stream,
                    delta, # delta in adwin detection, delta - the desired false positive rate
                    predict=False,
                    drift_detector=None,
                    ):
    """ data change detection """

    if predict==False:
        drift_detector = ADWIN(delta=delta)
        results = pd.DataFrame(columns=["value", "sqrt", "change"])

        for index, row in data_stream.iteritems():
            drift_detector.add_element(row)
            change = drift_detector.detected_change()
            variance = drift_detector.variance
            sqrt = np.sqrt(abs(variance))
            if np.isnan(sqrt):
                print("complex number", variance)
                sqrt = 0.0001
                print(sqrt)

            result = pd.DataFrame(data={
                                        "value": row,
                                        "sqrt": sqrt, 
                                        "change": 1 if change==True else 0,
                                        },
                                index=[index],
                                )
            results = results.append(result)

        return results, drift_detector

    else:
        copy_drift_detector = copy(drift_detector)
        results = pd.DataFrame(columns=["value", "sqrt", "change"])

        for index, row in data_stream.iteritems():
             drift_detector.add_element(row)
             change = drift_detector.detected_change()
             variance = drift_detector.variance
             sqrt = np.sqrt(abs(variance))
             if np.isnan(sqrt):
                print("complex number", variance)
                sqrt =  0.0001
                print(sqrt)

             result = pd.DataFrame(data={
                                         "value": row,
                                         "sqrt": sqrt,
                                         "change": 1 if change==True else 0,
                                         },
                                   index=[index],
                                   )

             if change==True and predict==True:
                 drift_detector = copy(copy_drift_detector)

             results = results.append(result)
        
        return results


def drift_detect_multicols(df,
                           delta,
                           predict=False,
                           drift_detector_dict=None,
                           scaler_dict=None,
                           ):
    """ data change detection for each column """

    df_results = pd.DataFrame()

    if predict==False:
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

        return df_results, drift_detector_dict, scaler_dict

    else:
        for col, drift_detector, scaler in zip(df.columns, drift_detector_dict.values(), scaler_dict.values()):
            results = drift_detection(data_stream=df[col],
                                                      delta=delta,
                                                      predict=True,
                                                      drift_detector=drift_detector,
                                                      )
            results = pd.DataFrame(scaler.transform(results), columns=[i + '_' + col for i in results.columns])
            df_results = pd.concat([df_results, results], axis=1)

        return df_results


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
            X_train, X_test, y_train, y_test = train_test_split(df.values, y, stratify=y, test_size=sample_size)
            df_results = pd.DataFrame(X_test, columns=df.columns)

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
                      sample_size=None,
                      k=(1,9),
                      n_clusters_without_method=5,
                      ):
    ''' build drift dataframe '''

    if scaler_df is None:
        scaler_df = MinMaxScaler()

    df = pd.DataFrame(scaler_df.fit_transform(df), columns=df.columns)

    if sample_size is not None:

        if sample_size > df.shape[0] and predict == True:
             print("resample")
             df = resample(df, n_samples=sample_size)
        
        else:
            df, cluster_model = cluster_sampling(df=df,
                                                 sample_size=sample_size,
                                                 k=k,
                                                 n_clusters_without_method=n_clusters_without_method,
                                                 cluster_model=cluster_model,
                                                 )
    else:
        print("sampling does not apply")

    if predict == True:
         df_results = drift_detect_multicols(df=df,
                                            delta=delta,
                                            predict=True,
                                            drift_detector_dict=drift_detector_dict,
                                            scaler_dict=scaler_dict,
                                            )

         return df_results

     
    else:
        df_results, drift_detector_dict, scaler_dict = drift_detect_multicols(df=df,
                                                                              delta=delta,
                                                                              predict=False,
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

    describe = pd.Series()
    describe = describe.append(percentage)

    columns = [col for col in df_results.columns if col.startswith("value_")]
    mean = df_results[columns].mean()
    mean.index = [index.replace("value_", "mean_") for index in mean.index]
    describe = describe.append(mean)

    std = df_results[columns].std()
    std.index = [index.replace("value_", "std_") for index in std.index]
    describe = describe.append(std)

    if list_of_weights:
        return totals, percentage, ws.weighted_mean(percentage, list_of_weights), describe
    else:
        return totals, percentage, ws.mean(percentage), describe


def plot_drift(df,
               color_discrete_sequence=['rgba(0, 204, 150, .20)'],
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
                           color=np.where(df[change]==1, "drift", "not drift"),
                           title=sqrt, 
                           color_discrete_map={"drift":'#EF553B', "not drift":'#636EFA'},
                           )
        fig.update_layout(legend_title_text='change detect')
        fig2 = px.line(x=df.reset_index().index, y=df[value], color_discrete_sequence=color_discrete_sequence)
        fig2['data'][0]['showlegend'] = True
        fig2['data'][0]['name'] = 'scaled value'
        fig.add_trace(fig2.data[0])
        fig.update_layout(xaxis_title="X - index number",
                          yaxis_title="Y - sqrt",
                          )
        fig_list.append(fig)

    return fig_list


def multiplots(df_results,
               plots,
               plots_predict=None,
               ):

    rows = len(plots)
    
    if plots_predict is None:
        subplot_titles = [y for col in df_results.columns for y in [(col[6:22] + " train")] if col.startswith("value_")]
        fig = make_subplots(rows=rows, cols=1, subplot_titles=subplot_titles)

        for row in range(rows):
            for plot in plots[row]["data"]:
                fig.add_trace(plot, row=row+1, col=1)

    else:
        subplot_titles = [y for col in df_results.columns for y in [(col[6:22] + " train"), (col[6:22] + " predict")] if col.startswith("value_")]
        fig = make_subplots(rows=rows, cols=2, subplot_titles=subplot_titles)

        for row in range(rows):
            for plot in plots[row]["data"]:
                fig.add_trace(plot, row=row+1, col=1)

        for row in range(rows):
            for plot in plots_predict[row]["data"]:
                fig.add_trace(plot, row=row+1, col=2)
        
    fig.update_layout(height=600, width=800, title_text="Drift", showlegend=False)

    return fig


def change_plot(describe_fit,
                describe_predict):
    describe_fit = pd.DataFrame(describe_fit, columns=["values"]).reset_index()
    describe_fit["type"] = "fit"

    describe_predict = pd.DataFrame(describe_predict, columns=["values"]).reset_index()
    describe_predict["type"] = "predict"

    describe_totals = pd.concat([describe_fit, describe_predict], axis=0)
    describe_totals = describe_totals.rename(columns={"index": "metrics"})

    describe_totals["col_name"] = describe_totals["metrics"].str.replace("change_", "").str.replace("mean_", "").str.replace("std_", "")
    describe_totals['metrics'] = describe_totals.apply(lambda x: x['metrics'].replace("_" + x['col_name'], ""), axis=1)

    fig = px.bar(describe_totals, x="metrics", y="values", color="type",  facet_row="col_name", 
                 title="Describe Drift of Each Column", barmode="group",
                 color_discrete_map={"fit":'rgba(0, 204, 150, .90)', "predict":'rgba(171, 99, 250, 0.90)'},)

    return fig

class DriftEstimator(object):
    ''' Drift estimator for multiple columns using cluster sampling and weight weights.

        - https://github.com/matiasscorsetti/drift

        It is based on an ADWIN (ADaptive WINdowing) model for each column of a dataframe.
        ADWIN is an adaptive sliding window algorithm for detecting changes, \
        and keep up-to-date statistics on a data stream. ADWIN allows algorithms not adapted for drifting data, \
        be resistant to this phenomenon.

        The general idea is to keep statistics from a variable size window while detecting concept drift.

        The algorithm will decide the size of the window by cutting the statistics window at different points \
        and analyze the average of some statistics in these two windows. 
        If the absolute value of the difference between \
        the two averages exceed a predefined threshold, the change is detected at that point and all data before that point \
        is discarded.

        When training the model, the size of the resulting dataset is saved \
        (if a sample was performed in the training, the sample size determines the dataframe size, see "size" attributes) \
        the results should be evaluated at the dataframe level in general or per column (and not at the row level)
        
        Always automatically adjusts the size of the input dataframe to the size of the dataset used in training.

        parameters:
            - delta: delta in adwin detection, delta - the desired false positive rate (default: delta=0.1)
            - sample_size: sample size using cluster sampling. if sample_size = None a sample is not applied (default: sample_size=0.1)
            - k: the values ​​of k to calculate the number of optimal clusters in the stratified sampling (default: k=(1,9))
            - n_clusters_without_method: n clusters when an optimum is not detected using the elbow method, 
                                        to perform a stratified sampling (default: n_clusters_without_method=5)
            - list_of_weights: list of weights to weight each column in the total score (default: list_of_weights=None)


        Based on: https://scikit-multiflow.readthedocs.io/en/stable/api/generated/skmultiflow.drift_detection.ADWIN.html#skmultiflow.drift_detection.ADWIN
    '''

    def __init__(self, delta=0.1, sample_size=0.1, k=(1,9), n_clusters_without_method=5, list_of_weights=None):
        self.delta = delta
        self.sample_size = sample_size
        self.k = k
        self.n_clusters_without_method = n_clusters_without_method
        self.list_of_weights = list_of_weights


    def fit(self, df):
        ''' 
        train the model with a dataframe

        parameters: Pandas dataframe

        '''
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

        totals, percentage, drift_average, describe_fit = drift_average_calculate(df_results=self.df_results,
                                                                    list_of_weights=self.list_of_weights,
                                                                    )
        self.totals = totals
        self.percentage = percentage
        self.drift_average = drift_average
        self.describe_fit = describe_fit
        self.size = df_results.shape[0]

    def plot_train(self):

        plots = plot_drift(df=self.df_results)
        plot = multiplots(df_results=self.df_results,
                               plots=plots,
                               )
        return plot
        

    def predict(self, df):
        ''' 
        predict if a dataframe has drift

        size of the input dataframe is always automatically adjusted to the size of the dataset used in the training 
        (if a sampling was done in the training, the sample size determines the dataframe size, see attributes "size")

        parameters: Pandas Dataframe

        return: drift score

        '''
        self.df_results_predict = None
        self.totals_predict = None
        self.percentage_predict = None
        self.drift_average_predict = None
        self.describe_predict = None

        df_results_predict = drift_df_multicols(df=df,
                                                delta=self.delta,
                                                predict=True,
                                                drift_detector_dict=self.drift_detector_dict,
                                                scaler_dict=self.scaler_dict,
                                                cluster_model=self.cluster_model,
                                                scaler_df=self.scaler_df,
                                                sample_size=self.size,
                                                k=self.k,
                                                n_clusters_without_method=self.n_clusters_without_method,
                                                )
        self.df_results_predict = df_results_predict

        totals_predict, percentage, drift_average_predict, describe_predict = drift_average_calculate(df_results=self.df_results_predict,
                                                                                    list_of_weights=self.list_of_weights,
                                                                                    )
        self.totals_predict = totals_predict
        self.percentage_predict = percentage
        self.drift_average_predict = drift_average_predict
        self.describe_predict = describe_predict

        return drift_average_predict

    def plot_predict(self):

        plots = plot_drift(df=self.df_results)
        plots_predict = plot_drift(df=self.df_results_predict,
                                    color_discrete_sequence=['rgba(171, 99, 250, 0.4)'],
                                    )
        plot = multiplots(df_results=self.df_results,
                          plots=plots,
                          plots_predict=plots_predict,
                          )
        return plot

    def plot_describe(self):
        
        plot = change_plot(describe_fit=self.describe_fit,
                       describe_predict=self.describe_predict,
                       )

        return plot