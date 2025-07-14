from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Carga de artefactos guardados
encoder = joblib.load('ordinal_encoder.pkl')
scaler = joblib.load('robust_scaler.pkl')
pca = joblib.load('pca_model.pkl')
model = joblib.load('modelo_final.pkl')  # Aquí tu modelo predictivo final

app.logger.debug('Modelos y transformadores cargados correctamente.')

# Define las columnas categóricas que se deben codificar
categorical_cols_final = ['Sex', 'Title', 'Ticket']

# Define las características finales que usa el modelo (orden y nombres exactos)
features = ['Sex', 'Ticket', 'Age', 'Fare', 'Pclass', 'SibSp', 'Title', 'FamilySize']

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extraer datos del formulario (asumiendo que llegan en formato string)
        input_data = {}
        for feature in features:
            val = request.form.get(feature)
            if val is None:
                return jsonify({'error': f'Missing value for {feature}'}), 400
            input_data[feature] = val

        # Crear DataFrame con un solo registro
        df_input = pd.DataFrame([input_data])

        # Convertir numéricos a float
        num_cols = ['Age', 'Fare', 'Pclass', 'SibSp', 'FamilySize']
        for col in num_cols:
            df_input[col] = df_input[col].astype(float)

        # Codificar variables categóricas
        df_input[categorical_cols_final] = encoder.transform(df_input[categorical_cols_final])

        # Escalar
        X_scaled = scaler.transform(df_input[features])

        # PCA
        X_pca = pca.transform(X_scaled)

        # Predecir
        prediction = model.predict(X_pca)

        app.logger.debug(f'Predicción: {prediction[0]}')

        return jsonify({'prediccion': prediction[0]})

    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
