import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

def main():
    st.set_page_config(page_title="Pokemon Anomaly Detection", page_icon="📊", layout="centered")
    st.title("Pokemon Anomaly Detection")
    st.write("Machine Learning und Deep Learning")
    st.write("""
               - Die folgende Anwendung untersucht die Anomalien bei Pokemons
               - Der Datensatz enthält 781 Pokemons und deren Eigenschaften
               - Die Spalte is_legendary gibt an, ob ein Pokemon als legendär zu klassifizieren ist (Anomaly)
               """)
    st.markdown("""----""")
    with st.sidebar.header("💡 Algorithmus auswählen"):
        algo = st.sidebar.selectbox("Algorithmus auswählen", options=["K-Nearest-Neighbor", "One-Class SVM", "Isolation Forest"], index=0)
    with st.sidebar.subheader("train_test_split"):
        split_size = st.sidebar.slider("Aufteilen in Traings- und Testdaten (Standard: 30% Testdaten):", 0.1, 0.9, 0.3, 0.1)
    with st.sidebar.subheader("Parameter auswählen"):
            n_neighbors = st.sidebar.slider("n_neighbors für KNN auswählen:", 1, 10, 5, 1)
            nu = st.sidebar.slider("nu für One-Class SVM auswählen:", 0.0001, 0.9999, 0.5, 0.01)
            n_estimators = st.sidebar.slider("n_estimators für Isolation Forest auswählen:", 50, 500, 200, 10)
    st.title("👩‍💻 Tabelle")
    df=pd.read_csv(r"https://raw.githubusercontent.com/tobiarnold/outlier-detection/main/pokemon1.csv")
    df_show = df.style.format({"height_m": "{:,.1f}","weight_kg": "{:,.1f}"})
    st.dataframe(df_show)
    st.markdown("""----""")
    st.title("📊 Diagramme")
    try:
        fig = px.histogram(df, x="is_legendary", color="is_legendary",title="Verteilung der Klasse is_legendary (Legendäre Pokemons)")
        #fig.update_layout(xaxis_range=[0,1])
        fig.update_layout(xaxis = dict(tickmode = 'array',tickvals = [0, 1],ticktext = ["0","1"]))
        fig.update_layout(yaxis_title="Anzahl")
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.write("Verteilungsdiagramm kann nicht dargestellt werden, bitte Seite neu laden")
    df_leg=df[df["is_legendary"] == 0]
    df_no = df[df["is_legendary"] == 1]
    df_leg = df_leg.drop(columns =["name","experience_growth","is_legendary","weight_kg","height_m"])
    df_no=df_no.drop(columns =["name","experience_growth","is_legendary","weight_kg","height_m"])
    fig = go.Figure()
    for col in df_no:
        fig.add_trace(go.Box(y=df_no[col].values, name=df_no[col].name))
    fig.update_layout(title="Boxplots ausgewählter Spalten normaler Pokemons")
    st.plotly_chart(fig, use_container_width=True)
    fig = go.Figure()
    for col in df_leg:
        fig.add_trace(go.Box(y=df_leg[col].values, name=df_leg[col].name))
    fig.update_layout(title="Boxplots ausgewählter Spalten legendärer Pokemons")
    st.plotly_chart(fig, use_container_width=True)
    df_corr=df.corr().round(2)
    fig = px.imshow(df_corr, aspect="auto",title="Heatmap der Korrelationen",text_auto=True)
    #fig.update_layout(autosize=False,width=500,height=500)
    st.plotly_chart(fig, use_container_width=True)
    #st.dataframe(df_corr)
    st.markdown("""----""")
    st.title("👨‍🔬 Algorithmen zur Anomaly Detection")
    st.write("Du hast "+str(algo)+" gewählt.")
    X = df.drop(columns =["name","is_legendary"])
    y = df["is_legendary"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size, random_state=42)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    if algo =="K-Nearest-Neighbor":
        knn = KNeighborsClassifier(n_neighbors= n_neighbors)
        knn.fit(X_train,y_train)
        y_predictions = knn.predict(X_test)
        st.text("Model Report:\n "+classification_report(y_test, y_predictions))
        st.markdown("")
        y_predictions = pd.Series(y_predictions, name="Predicted Labels")
        y_test = y_test.to_frame().rename({"is_legendary": "Actual Labels"}, axis='columns')
        y_test = y_test.reset_index()
        y_test = y_test.drop(columns=["index"])
        y_predictions = y_predictions.to_frame()
        confusion_matrix = pd.concat([y_test, y_predictions], axis=1)
        confusion_matrix = pd.crosstab(confusion_matrix["Actual Labels"], confusion_matrix["Predicted Labels"],rownames=["Actual"], colnames=["Predicted"])
        try:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
            ax = sns.heatmap(confusion_matrix, annot=True,fmt=".0f")
            ax.set_title("Confusion Matrix")
            st.write(fig)
        except:
            st.write("Heatmap kann nicht dargestellt werden, bitte Seite neu laden")
    elif algo =="One-Class SVM":
        svm = OneClassSVM(nu=nu)
        svm.fit(X_train)
        y_predictions = svm.predict(X_test)
        y_predictions = [1 if i == -1 else 0 for i in y_predictions]
        st.text("Model Report:\n " + classification_report(y_test, y_predictions))
        st.markdown("")
        y_predictions = pd.Series(y_predictions, name="Predicted Labels")
        y_test = y_test.to_frame().rename({"is_legendary": "Actual Labels"}, axis='columns')
        y_test = y_test.reset_index()
        y_test = y_test.drop(columns=["index"])
        y_predictions = y_predictions.to_frame()
        confusion_matrix = pd.concat([y_test, y_predictions], axis=1)
        confusion_matrix = pd.crosstab(confusion_matrix["Actual Labels"], confusion_matrix["Predicted Labels"],
                                       rownames=["Actual"], colnames=["Predicted"])
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
        ax = sns.heatmap(confusion_matrix, annot=True,fmt=".0f")
        ax.set_title("Confusion Matrix")
        st.write(fig)
    elif algo =="Isolation Forest":
        iso=IsolationForest(n_estimators=n_estimators)
        iso.fit(X_train)
        y_predictions = iso.predict(X_test)
        y_predictions = [1 if i == -1 else 0 for i in y_predictions]
        st.text("Model Report:\n " + classification_report(y_test, y_predictions))
        st.markdown("")
        y_predictions = pd.Series(y_predictions, name="Predicted Labels")
        y_test = y_test.to_frame().rename({"is_legendary": "Actual Labels"}, axis='columns')
        y_test = y_test.reset_index()
        y_test = y_test.drop(columns=["index"])
        y_predictions = y_predictions.to_frame()
        confusion_matrix = pd.concat([y_test, y_predictions], axis=1)
        confusion_matrix = pd.crosstab(confusion_matrix["Actual Labels"], confusion_matrix["Predicted Labels"],
                                       rownames=["Actual"], colnames=["Predicted"])
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
        ax = sns.heatmap(confusion_matrix, annot=True,fmt=".0f")
        ax.set_title("Confusion Matrix")
        st.write(fig)

if __name__ == "__main__":
  main()
