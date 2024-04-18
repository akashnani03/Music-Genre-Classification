from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
import librosa

app = Flask(__name__)

# Load the model
model = load_model('model.h5')

@app.route('/')
def home():
    return render_template('index.html', prediction='')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return render_template('index.html', prediction='No file part')

    file = request.files['file']

    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return render_template('index.html', prediction='No selected file')

    # If the file is valid
    if file:
        try:
            # Load the audio file
            audio, sr = librosa.load(file, sr=None)

            # Extract features (e.g., mel spectrogram) using librosa
            mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
            mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)
            mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)

            # Reshape the mel spectrogram if necessary (e.g., if the last dimension is incorrect)
            if mel_spectrogram.shape[-1] != 129:
                mel_spectrogram = mel_spectrogram[:, :, :129]

            # Make prediction
            prediction = model.predict(mel_spectrogram)

            # Convert prediction to genre label
            #genres = ['blues','classical','country','disco', 'hip-hop','jazz','metal','pop',  'reggae','rock']
            genres = ['metal', 'disco', 'classical','hiphop','jazz','country', 'pop', 'blues','reggae','rock']
            predicted_genre = genres[np.argmax(prediction)]

            return render_template('index.html', prediction=predicted_genre)
        except Exception as e:
            return render_template('index.html', prediction='Error predicting: {}'.format(e))

    return render_template('index.html', prediction='Something went wrong')

if __name__ == '__main__':
    # For development
    app.run(debug=True)
