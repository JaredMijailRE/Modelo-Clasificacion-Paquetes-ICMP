import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def explore_and_clean_data(file_path):
    # Cargar los datos
    data = pd.read_csv(file_path, low_memory=False)

    # Mostrar información general del dataset
    print("Información del dataset:")
    print(data.info())

    # Mostrar estadísticas descriptivas
    print("Estadísticas descriptivas:")
    print(data.describe(include='all'))

    # Identificar valores únicos en cada columna
    print("Valores únicos por columna:")
    for column in data.columns:
        print(f"{column}: {data[column].nunique()} valores únicos")

    # Limpiar columnas con tipos mixtos
    for column in data.select_dtypes(include=['object']).columns:
        data[column] = pd.to_numeric(data[column], errors='coerce')

    # Rellenar valores faltantes con la mediana
    data.fillna(data.median(numeric_only=True), inplace=True)

    return data


def train_model():
    # Explorar y limpiar los datos
    data = explore_and_clean_data('train.csv')

    # Eliminar filas con valores NaN en la columna 'label'
    data = data.dropna(subset=['label'])

    # Convertir la columna 'label' a valores numéricos
    data['label'] = data['label'].astype('category').cat.codes

    # Verificar si hay suficientes datos después de eliminar NaN
    if data.empty:
        raise ValueError("El conjunto de datos está vacío después de eliminar valores NaN en la columna 'label'.")

    # Inspeccionar la columna 'label' antes de eliminar NaN
    print("Valores únicos en la columna 'label':")
    print(data['label'].value_counts(dropna=False))

    # Inspeccionar más a fondo los valores en la columna 'label'
    print("Primeros 10 valores en la columna 'label':")
    print(data['label'].head(10))
    print("Cantidad de valores NaN en 'label':", data['label'].isna().sum())

    # Mostrar todos los valores únicos en la columna 'label' antes de cualquier procesamiento
    print("Todos los valores únicos en la columna 'label':")
    print(data['label'].unique())

    # Separar características (X) y etiquetas (y)
    X = data.drop(columns=['label'])
    y = data['label']

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar un modelo de bosque aleatorio
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Crear una matriz de confusión
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title("Matriz de Confusión")
    plt.show()

    # Mostrar importancia de características
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    feature_importances.nlargest(10).plot(kind='barh')
    plt.title("Importancia de Características")
    plt.show()

    # Guardar el modelo entrenado
    joblib.dump(model, 'icmp_model.pkl')


def main():
    print("Hello from modelo-clasificacion-paquetes-icmp!")


if __name__ == "__main__":
    train_model()
    main()
