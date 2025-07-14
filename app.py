from flask import Flask , request, render_template,jsonify
import joblib
import pandas as pd
import logging

app= Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

model= joblib.load('modelo_insectos.pkl')
app.logger.debug('Modelo cargando correctamnete')


@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        abdomen = float(request.form['abdomen'])
        antena = float(request.form['antena'])
        
        data_df = pd.DataFrame([[abdomen,antena]], columns=['abdomen', 'antena'])
        app.logger.debug(f'DataFrame creado: {data_df}')
        
        prediction = model.predict(data_df)
        app.logger.debug(f'Prediccion: {prediction[0]}')
        
        return jsonify({'categoria': prediction[0]})
    
    except Exception as e:
        app.logger.error(f'Error en la prediccion: {str(e)}')
        return jsonify({'error': str(e)}),400
    
    

if __name__=='__main__':
    app.run(debug=True)
    
