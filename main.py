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
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report

def main():
    st.set_page_config(page_title="Pokemon Anomaly Detection", page_icon="📊", layout="centered")
    st.title("Pokemon Anomaly Detection")
    st.write("Machine Learning und Deep Learning")
    st.write("""
               - Die folgende Anwendung untersucht die Anomalien bei Pokemons
               - Der Datensatz enthält 781 Pokemons und deren Eigenschaften
               - Die Spalte is_legendary gibt an, ob ein Pokemon als legendär zu klassifizieren ist (Anomaly)
               ⚠️ bei Zugriff mit mobilen Geräten ist der Filter für die Algorithmen standardmäßig ausgeblendet und lässt sich mit dem Pfeil oben links wieder einblenden
               """)
    st.markdown("""----""")
    with st.sidebar.subheader("💡 Parameter auswählen:"):
        split_size = st.sidebar.slider("Aufteilen in Traings- und Testdaten (Standard 30% Testdaten):", 0.1, 0.9, 0.3, 0.1)
        n_neighbors = st.sidebar.slider("n_neighbors für KNN bzw. LOF auswählen:", 1, 10, 5, 1)
        nu = st.sidebar.slider("nu für One-Class SVM auswählen:", 0.0001, 0.9999, 0.5, 0.01)
        kernel=st.sidebar.selectbox("Kernel für One-Class SVM auswählen:",options=["linear", "poly", "rbf", "sigmoid"], index=0)
        n_estimators = st.sidebar.slider("n_estimators für Random Forest bzw. Isolation Forest auswählen:", 50, 500, 200, 10)
        max_depth=st.sidebar.slider("Tiefe für Decision Tree bzw. Random Forest auswählen:", 2, 50, 10, 1)
    st.title("👩‍💻 Tabelle")
    df=pd.read_csv(r"https://raw.githubusercontent.com/tobiarnold/outlier-detection/main/pokemon1.csv")
    def highlight_rows(row):
       value = row.loc["is_legendary]
       if value == 1:
           color = "#ffffa1"
       return ['background-color: {}'.format(color) for r in row]
    #style1 = (lambda x: "background-color : #ffffa1" if x > 0 else '')
    #df_show = df.style.format({"height_m": "{:,.1f}","weight_kg": "{:,.1f}"}).applymap(style1, subset=["is_legendary"])
    df_show = df.style.format({"height_m": "{:,.1f}","weight_kg": "{:,.1f}"}).apply(highlight_rows, axis=1)
    st.dataframe(df_show)
    st.markdown("""----""")
    st.title("📊 Diagramme")
    try:
        config = {"displayModeBar": False}
        fig = px.histogram(df, x="is_legendary", color="is_legendary",title="Verteilung der Klasse is_legendary (Legendäre Pokemons)")
        #fig.update_layout(xaxis_range=[0,1])
        fig.update_layout(xaxis = dict(tickmode = 'array',tickvals = [0, 1],ticktext = ["0","1"]))
        fig.update_layout(yaxis_title="Anzahl")
        st.plotly_chart(fig, use_container_width=True, config=config)
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
    st.plotly_chart(fig, use_container_width=True, config=config)
    fig = go.Figure()
    for col in df_leg:
        fig.add_trace(go.Box(y=df_leg[col].values, name=df_leg[col].name))
    fig.update_layout(title="Boxplots ausgewählter Spalten legendärer Pokemons")
    st.plotly_chart(fig, use_container_width=True, config=config)
    df_corr=df.corr().round(2)
    fig = px.imshow(df_corr, aspect="auto",title="Heatmap der Korrelationen",text_auto=True)
    #fig.update_layout(autosize=False,width=500,height=500)
    st.plotly_chart(fig, use_container_width=True, config=config)
    #st.dataframe(df_corr)
    st.markdown("""----""")
    st.title("👨‍🔬 Algorithmen zur Anomaly Detection")
    algo=st.selectbox("💡 Algorithmus auswählen", options=["K-Nearest-Neighbor (KNN)", "One-Class SVM", "Decision Tree","Random Forest","Isolation Forest","Local Outlier Factor (LOF)"], index=0)
    st.write("Du hast "+str(algo)+" gewählt.")
    st.write("Zur Skalierung der Daten wurde der Standard Scaler verwendet.")
    X = df.drop(columns =["name","is_legendary"])
    y = df["is_legendary"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size, random_state=42)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    try:
        if algo =="K-Nearest-Neighbor (KNN)":
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
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
            ax = sns.heatmap(confusion_matrix, annot=True,fmt=".0f")
            ax.set_title("Confusion Matrix")
            st.write(fig)
        elif algo =="One-Class SVM":
            svm = OneClassSVM(kernel=kernel, nu=nu)
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
        elif algo =="Decision Tree":
            dec=DecisionTreeClassifier(max_depth=max_depth)
            dec.fit(X_train,y_train)
            y_predictions = dec.predict(X_test)
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
        elif algo =="Random Forest":
            rf=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
            rf.fit(X_train,y_train)
            y_predictions = rf.predict(X_test)
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
        elif algo =="Local Outlier Factor (LOF)":
            lof=LocalOutlierFactor(n_neighbors=n_neighbors)
            lof.fit(X_train)
            y_predictions = lof.fit_predict(X_test)
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
    except:
        st.write("Bitte anderen Parameter wählen")

if __name__ == "__main__":
  main()
