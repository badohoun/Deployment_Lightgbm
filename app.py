import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair  as alt
import plotly.express as px
from PIL import Image
import pickle
import lightgbm as lgb
import shap
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
icon = Image.open("Inetum_logo.jpg")

st.set_page_config(
   page_title="Data Application",
   page_icon = icon,
   layout = "centered",
   initial_sidebar_state ="auto"

)


image = Image.open("Infographie-leNouvel-Observatoire-riques-routiers-06-02-01.jpg")

#st.image(image)



audio_file = open("Nouvel enregistrement 36.m4a", "rb")
audio_bytes = audio_file.read()
#st.audio(audio_bytes)




# Load a dataset

@st.cache
def load_data():
    df = pd.read_csv("matrix_trajet_resample_5s.csv")
    return df
df = load_data()


# define application

def main():
    page = st.sidebar.selectbox(
        "Select a Page",
        [
        "HomePage",
        "Line Plot",
        "Bar",
        "Scatter",
        "Scatter Matrix",
        "Heatmap",
        "Horizontal bar",
        "Stacked Bar  Chart",
        "Box Plot",
        "Concatenate",
        "Model Building",
        "Map Prediction 26",
        "Map Prediction 13",
        "Map Prediction 18",
        "Explicability of model"

        ]
    )
    if page == "HomePage":
        #st.header("Data Application")
        """

        # Building phone position prediction application :
         Please select a page on the left
        """



        st.image(image)
        st.audio(audio_bytes)

        st.balloons()
        st.header("Input Features")
        st.write(df)
    elif page == "Vega Lite":
        st.header("Vega Lite")
        vegalite()
    elif page == "Line Plot":
        st.header("Line Plot")
        visualize_line()
    elif page == "Bar":
        st.header("Bar")
        bar()
    elif page == "Scatter":
        st.header("Scatter")
        scatter()
    elif page == "Area Chart":
        st.header("Area Chart")
        area_chart()
    elif page == "Scatter Matrix":
        st.header("Scatter Matrix")
        scatter_matrix()
    elif page == "Heatmap":
        st.header("Heatmap")
        heatmap()
    elif page == "Horizontal bar":
        horizontal_bar()
    elif page == "Grouped Bar":
        st.header("Grouped Bar")
        grouped_bar()
    elif page == "Stacked Bar  Chart":
        st.header("Stacked Bar Chart")
        stacked_bar()
    elif page ==  "Box Plot":
        st.header("Box Plot")
        boxplot()
    elif page == "Concatenate":
        st.header("Concatenate")
        concatenate()
    elif page == "Model Building":
        st.header("Model Building")
        model_building()
    elif page == "Map Prediction 26":
        st.header("Map Prediction 26")
        prediction_map_26()
    elif page == "Map Prediction 13":
        st.header("Map Prediction 13")
        prediction_map_13()
    elif page == "Map Prediction 18":
        st.header("Map Prediction 18")
        prediction_map_18()
    elif page == "Explicabilite du model":
        st.header("check features shift the decision positively or negatively)")
        shap_summary_plot()




def visualize_line():
    df_copy = df.copy()

    df_copy['datetime']=pd.to_datetime(df_copy['datetime'])
    line  = (
        alt.Chart(df_copy)
        .mark_line()
        .encode(x = "datetime" , y = "activity")
        .properties(width=650, height=500)
        .interactive()
    )
    st.altair_chart(line)
def bar():
    bar_data = df.sort_values(by = "datetime" , ascending = True)
    #bar_data = bar_data.head(80)
    sort = st.checkbox("Sort")
    if sort:
        chart =(
            alt.Chart(bar_data).mark_bar()
            .encode(
                x = alt.X('activity:N', sort='-y'),
                y='z_median'

            )
            .properties(width = 650 , height=500)
            .interactive()
        )

    else :
        chart = (
        alt.Chart(bar_data).mark_bar().encode(
            x = "activity",
            y="z_median"
        )
        .properties(width = 650 , height = 500)
        .interactive()




    )


    st.altair_chart(chart)


def scatter():
    toogle = st.checkbox("Toogle Scatter")
    if toogle:
        chart = (
        alt.Chart(df, background="maroon").mark_point().encode(
            x = "datetime",
            y="activity",
            size = "x_min",
            color = "x_min",
            tooltip = ["x_max", "x_min" , "x_median" , "norm_median"]
            )
            .properties(width = 650 , height=500)
            )
    else :
          chart = (
          alt.Chart(df).mark_point().encode(
              x = "datetime",
              y="activity",
              size = "x_min",
              color = "x_min",
              tooltip = ["x_max", "x_min" , "x_median" , "norm_median"]
              )
              .properties(width = 850 , height=950)
              )




    st.altair_chart(chart)

def scatter_matrix():
    chart = (
        alt.Chart(df).mark_circle().encode(
            alt.X(alt.repeat("column"), type="quantitative"),
            alt.Y(alt.repeat("column"), type="quantitative"),
            color = "activity"
        )
        .properties(width=300 , height = 100)
        .repeat(
            row=["y_min" , "y_max" , "y_median"],
            column = ["y_median", "y_max", "y_min"]
        )
        .interactive()
    )
    st.altair_chart(chart)



def heatmap():
    chart = (
        alt.Chart(df).mark_rect().encode(
            x="datetime",
            y="x_max",
            color="activity",
            tooltip=["activity" , "y_max"]
        )
        .interactive()
        .properties(width=650,height=500)

    )
    st.altair_chart(chart)
def horizontal_bar():
    bar_data = df.sort_values(by = "datetime" , ascending = True)
    bar_data = bar_data.head(80)
    chart = (
            alt.Chart(bar_data).mark_bar()
            .encode(
                x = "datetime",
                y= "activity"
            )
            .properties(width = 650 , height = 500)
            .interactive()
        )
    st.altair_chart(chart)
def stacked_bar():
    toogle = st.checkbox("Toogle this")
    if toogle :
        chart = (
         alt.Chart(df).mark_bar().encode(
             y= "datetime",
             x="sum(x_min)",
             color = "activity",
             tooltip=["x_min"]
        )
        .properties(height=500 , width=700)
        .interactive()
     )
    else :
         chart = (
          alt.Chart(df).mark_bar().encode(
              x= "datetime",
              y="x_min",
              color = "activity",
              tooltip=["x_min"]
            )
            .properties(height=500 , width=700)
            .interactive()
    )
    st.altair_chart(chart)
def boxplot():
    chart = (
        alt.Chart(df).mark_boxplot().encode(
            x="activity",
            y="x_min"

        )
        .interactive()
        .properties(height=500 , width=700)
    )
    st.altair_chart(chart)
def concatenate():
    scatter = (
         alt.Chart(df).mark_point().encode(
             x = "x_min",
             y="activity:N"
         )
         .properties(width=250 , height=250)
    )
    chart =(
         alt.concat(
             scatter.encode(color="y_min"),
             scatter.encode(color="z_min")

         )
         .resolve_scale(color="independent")
    )
    st.altair_chart(chart)
def model_building():

    st.sidebar.markdown("""
    [Example CSV input file](https://github.com/badohoun/Phone_Position_Prediction_Heroku/blob/main/test_gab.csv)
    """)
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file" , type = ["csv"])

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def user_input_features():
            x_min = st.sidebar.slider('x_min', -1.8 , 0.8 , 0.16)
            x_max = st.sidebar.slider('x_max' , -0.07 , 2.3 , 0.5)
            x_std = st.sidebar.slider('x_std', 0.00049 , 0.6, 0.07)
            x_median = st.sidebar.slider('x_median', -0.9 , 0.9 , 0.3)
            y_min =  st.sidebar.slider('y_min', 4.29 , 0.1 , -0.18)
            y_max = st.sidebar.slider('y_max', -1.0 , 2.0 ,  0.24)
            y_std = st.sidebar.slider('y_std', 0.0004 , 2.0 ,  0.075)
            y_median = st.sidebar.slider('y_median', -1.09 , 1.01 ,  0.082)
            z_min = st.sidebar.slider('z_min', -2.05 , 1.0 ,  0.03)
            z_max = st.sidebar.slider('z_max', 0.53 ,3.0 ,  0.44)
            z_std = st.sidebar.slider('z_std', 0.0007 ,0.6 ,0.072)
            z_median = st.sidebar.slider('z_median', 0.6 ,1.0 ,0.2)
            norm_min = st.sidebar.slider('norm_min', 0.04 ,1.0 ,0.9)
            norm_max = st.sidebar.slider('norm_max', 0.9 ,4.4,1.0)
            norm_std = st.sidebar.slider('norm_std', 0.0007 ,0.9,0.07)
            norm_median = st.sidebar.slider('norm_median', 0.83 ,1.25,1.004)


            data = {'x_min':x_min,
                    'x_max' : x_max,
                    'x_std': x_std,
                    'x_median':x_median,
                    'y_min':y_min,
                    'y_max':y_max,
                    'y_std':y_std,
                    'y_median':y_median,
                    'z_min': z_min,
                    'z_max': z_max,
                    'z_std': z_std,
                    'z_median': z_median,
                    'norm_min': norm_min,
                    'norm_max': norm_max,
                    'norm_std': norm_std,
                    'norm_median':norm_median}
            features = pd.DataFrame(data , index = [0])
            return features
        input_df = user_input_features()
    @st.cache
    def load_data1():
        accelerometer_data_path = 'poche_arriere_df_resample_10.csv'
        data = pd.read_csv(accelerometer_data_path)
        return data
    df2 = load_data1()
    df2 = df2.drop(labels = ['datetime'], axis = 1)

    st.subheader('User Input Parameters')

    if uploaded_file is not None :
        st.write(input_df)
    else:

        st.write('Awaiting CSV file to be uploaded . Currently using example input parameters ')
        st.write(input_df)


    # Reads the saved classification model
    load_Phone_Position_prediction_model = pickle.load(open('bestlightgbm2_model.pkl', 'rb'))


    # Apply a model to make predictions
    prediction1 = load_Phone_Position_prediction_model.predict(input_df)
    prediction2 = load_Phone_Position_prediction_model.predict(df2)




    st.header('Prediction User Input Parameters')

    st.write(prediction1)
    st.header('Prediction of Activity Poche arriere')
    st.write(prediction2)


def prediction_map_26():
    @st.cache
    def load_data2():
        submission_data_path = 'submission_lighgbm_26_5s_azure_databricks.csv'
        data = pd.read_csv(submission_data_path)
        return data
    df3 = load_data2()
    px.set_mapbox_access_token('pk.eyJ1IjoiZXNwZXJvODQiLCJhIjoiY2tpeTZreWY0MWEzejJzbWUwN3ZkcG4zbCJ9.bZlLDZmnmbbFcUy2PabzKw')
    st.write(
        px.scatter_mapbox(
            df3 , lat="latitude" , lon="longitude" , color = "Activity",
            color_continuous_scale=px.colors.sequential.Rainbow,
            hover_data={'datetime_acc':True,
                        'datetime':True,
                        'speed':True,
                        'accuracy':True,
                        'altitude':True,
                        'bearing':True,
                        'battery':False,
                        'ischarging':True,
                        'deviceid':False,
                        'dateentry':False,
                        'latitude': False,
                        'longitude':False,
                        'Activity':True},
            size = 'speed',
            size_max=7,zoom=10,height=700,width=950

        )
    )




def prediction_map_13():
    @st.cache
    def load_data3():
        submission_data_path = 'submission_lighgbm_13_5s_azure_databricks.csv'
        data = pd.read_csv(submission_data_path)
        return data
    df3 = load_data3()
    px.set_mapbox_access_token('pk.eyJ1IjoiZXNwZXJvODQiLCJhIjoiY2tpeTZreWY0MWEzejJzbWUwN3ZkcG4zbCJ9.bZlLDZmnmbbFcUy2PabzKw')
    st.write(
        px.scatter_mapbox(
            df3 , lat="latitude" , lon="longitude" , color = "Activity",
            color_continuous_scale=px.colors.sequential.Rainbow,
            hover_data={'datetime_acc':True,
                        'datetime':True,
                        'speed':True,
                        'accuracy':True,
                        'altitude':True,
                        'bearing':True,
                        'battery':False,
                        'ischarging':True,
                        'deviceid':False,
                        'dateentry':False,
                        'latitude': False,
                        'longitude':False,
                        'Activity':True},
            size = 'speed',
            size_max=7,zoom=10,height=700,width=950

        )
    )
def prediction_map_18():
    @st.cache
    def load_data3():
        submission_data_path = 'submission_lightgbm_18_5s.csv'
        data = pd.read_csv(submission_data_path)
        return data
    df3 = load_data3()
    px.set_mapbox_access_token('pk.eyJ1IjoiZXNwZXJvODQiLCJhIjoiY2tpeTZreWY0MWEzejJzbWUwN3ZkcG4zbCJ9.bZlLDZmnmbbFcUy2PabzKw')
    st.write(
        px.scatter_mapbox(
            df3 , lat="latitude" , lon="longitude" , color = "Activity",
            color_continuous_scale=px.colors.sequential.Rainbow,
            hover_data={'dateEntry':False,
                        'datetime':True,
                        'speed':True,
                        'accuracy':True,
                        'altitude':True,
                        'bearing':True,
                        'token':False,
                        'id':False,
                        'deviceId':False,
                        'dateEntry':False,
                        'latitude': False,
                        'longitude':False,
                        'Activity':True},
            size = 'speed',
            size_max=7,zoom=10,height=700,width=950

        )
    )
def shap_summary_plot():
    @st.cache
    def load_data4():
        submission_data_path = 'df.csv'
        data = pd.read_csv(submission_data_path)
        return data
    df4 = load_data4()
    X = df4.drop(labels = ['datetime', 'activity'] , axis =1)
    y = df4.activity

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2 , random_state=42, stratify=y)
    np.random.seed(42)
    learning_rate = 0.46
    is_unbalance=True
    n_estimators=6000
    n_jobs=1
    num_class=3
    num_leaves=100
    objective = 'multiclass'
    boosting_type = 'dart'


    #!pip install lightgbm
    import lightgbm as lgb
    model = lgb.LGBMClassifier(learning_rate = learning_rate , is_unbalance=is_unbalance, n_estimators=n_estimators, n_jobs=n_jobs, num_class=num_class,num_leaves=num_leaves, objective = objective, boosting_type = boosting_type)

    model.fit(X_train, y_train)



    explainer = shap.TreeExplainer(model)
    shap_values =explainer.shap_values(X_train)

    st.header('Feature Importance')
    plt.title('Feature importance based on SHAP values (Bar) ')

    shap.summary_plot(shap_values , X_train , plot_type="bar")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(bbox_inches = 'tight')
    acc = model.score(X_test, y_test)
    st.subheader("Accuracy")
    st.write('Accuracy: ', acc)
    pred_lightgbm = model.predict(X_test)
    st.subheader("Confusion matrix")
    cm=confusion_matrix(y_test,pred_lightgbm)
    st.write('Confusion matrix: ', cm)
    st.subheader("Accuracy")
    cr=classification_report(y_test, pred_lightgbm)
    st.subheader("Classification Report")
    st.text('Model Report:\n ' + cr)




if __name__ == "__main__":
    main()
